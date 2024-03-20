import boto3
import gc
import geopandas as gpd
import json
import json
import os
import rasterio
import time
import torch


from app.lib.downloader import Downloader
from app.lib.infer import Infer
from app.lib.post_process import PostProcess

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from huggingface_hub import hf_hub_download

from multiprocessing import Pool, cpu_count

from rasterio.io import MemoryFile
from rasterio.merge import merge

from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

from shapely.geometry import shape

from skimage.morphology import disk, binary_closing
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

BUCKET_NAME = os.environ.get('BUCKET_NAME')
LAYERS = ['HLSS30', 'HLSL30']

MODELS = ['burn_scars', 'flood']
ROLE_ARN = os.environ.get('ROLE_ARN')
ROLE_NAME = os.environ.get('ROLE_NAME')


def assumed_role_session():
    # Assume the "notebookAccessRole" role we created using AWS CDK.
    client = boto3.client('sts')
    return boto3.session.Session()

def download_from_s3(s3_path, force):
    session = assumed_role_session()
    s3_connection = session.client('s3')
    splits = s3_path.split('/')
    bucket_name = splits[2]
    filename = splits[-1]
    key = s3_path.replace(f"s3://{bucket_name}/", '')
    if not(os.path.exists(key)) or force:
        intermediate_path = key.replace(filename, '')
        if intermediate_path and not(os.path.exists(intermediate_path)):
            os.makedirs(intermediate_path)
        s3_connection.download_file(bucket_name, key, key)
    return key

def update_config(config, model_path):
    with open(config, 'r') as config_file:
        config_details = config_file.read()
        updated_config = config_details.replace('f"{data_root}/models/Prithvi_100M.pt"', f"'{model_path}'")

    with open(config, 'w') as config_file:
        config_file.write(updated_config)


def load_model(config_path, model_path, force=False):
    config = download_from_s3(config_path, force)
    model_path = download_from_s3(model_path, force)
    update_config(config, model_path)
    infer = Infer(config, model_path)
    _ = infer.load_model()
    return infer


def download_files(infer_date, layer, bounding_box):
    downloader = Downloader(infer_date, layer)
    return downloader.download_tiles(bounding_box)


def save_cog(mosaic, profile, transform, filename):
    profile.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "dtype": 'float32',
            "count": 1,
        }
    )
    with rasterio.open(filename, 'w', **profile) as raster:
        raster.write(mosaic[0], 1)
    output_profile = cog_profiles.get('deflate')
    output_profile.update(dict(BIGTIFF="IF_SAFER"))
    output_profile.update(profile)

    # Dataset Open option (see gdalwarp `-oo` option)
    config = dict(
        GDAL_NUM_THREADS="ALL_CPUS",
        GDAL_TIFF_INTERNAL_MASK=True,
        GDAL_TIFF_OVR_BLOCKSIZE="512",
    )
    with MemoryFile() as memory_file:
        cog_translate(
            filename,
            memory_file.name,
            output_profile,
            config=config,
            quiet=True,
            in_memory=True,
        )
        connection = boto3.client('s3')
        connection.upload_fileobj(memory_file, BUCKET_NAME, filename)

    return f"s3://{BUCKET_NAME}/{filename}"


def post_process(detections, transform):
    contours, shape = PostProcess.prepare_contours(detections)
    detections = PostProcess.extract_shapes(detections, contours, transform, shape)
    detections = PostProcess.remove_intersections(detections)
    return PostProcess.convert_to_geojson(detections)


def subset_geojson(geojson, bounding_box):
    geom = [shape(i['geometry']) for i in geojson]
    geom = gpd.GeoDataFrame({'geometry': geom})
    bbox = {
        "type": "Polygon",
        "coordinates": [
            [
                [bounding_box[0], bounding_box[1]],
                [bounding_box[2], bounding_box[1]],
                [bounding_box[2], bounding_box[3]],
                [bounding_box[0], bounding_box[3]],
                [bounding_box[0], bounding_box[1]],
            ]
        ],
    }
    bbox = shape(bbox)
    bbox = gpd.GeoDataFrame({'geometry': [bbox]})
    return json.loads(geom.overlay(bbox, how='intersection').to_json())


def batch(tiles, spacing=40):
    length = len(tiles)
    for tile in range(0, length, spacing):
        yield tiles[tile : min(tile + spacing, length)]


@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {'Hello': 'World'}


@app.get('/models')
def list_models():
    response = jsonable_encoder(MODELS)
    return JSONResponse({'models': response})


@app.post('/infer')
async def infer_from_model(request: Request, background_tasks: BackgroundTasks):
    instances = await request.json()
    infer_date = instances['date']
    bounding_box = instances['bounding_box']
    
    final_geojson = infer(instances, infer_date, bounding_box, background_tasks)
    return JSONResponse(content=jsonable_encoder(final_geojson))


def infer(instances, infer_date, bounding_box, background_tasks):
    if 'config_path' not in instances or 'model_path' not in instances or 'model_type' not in instances:
        response = {'statusCode': 422, 'message': 'Either config_path file or model_path or model_type is missing.'}
        return JSONResponse(content=jsonable_encoder(response))
    
    config_path = instances['config_path']
    model_path = instances['model_path']
    model_type = instances['model_type']

    model = load_model(config_path, model_path, instances.get('force', False))
    all_tiles = list()
    geojson_list = list()

    geojson = {'type': 'FeatureCollection', 'features': []}

    for layer in LAYERS:
        tiles = download_files(infer_date, layer, bounding_box)
        for tile in tiles:
            tile_name = tile
            if model_type == 'burn_scars':
                tile_name = tile_name.replace('.tif', '_scaled.tif')
            all_tiles.append(tile_name)

    start_time = time.time()
    mosaic = []
    s3_link = ''
    if all_tiles:
        try:
            torch.cuda.synchronize()
            results = list()
            for tiles in batch(all_tiles):
                results.extend(model.infer(tiles))
            transforms = list()
            memory_files = list()
            del model
            torch.cuda.empty_cache()
            for index, tile in enumerate(all_tiles):
                with rasterio.open(tile) as raster:
                    profile = raster.profile
                memfile = MemoryFile()
                profile.update({'count': 1, 'dtype': 'float32'})
                with memfile.open(**profile) as memoryfile:
                    memoryfile.write(results[index], 1)
                memory_files.append(memfile.open())

            mosaic, transform = merge(memory_files)
            mosaic[0] = binary_closing(mosaic[0], disk(6))
            [memfile.close() for memfile in memory_files]
            prediction_filename = f"{start_time}-predictions.tif"

            background_tasks.add_task(
                save_cog, mosaic, raster.meta.copy(), transform, prediction_filename
            )
            s3_link = f"s3://{BUCKET_NAME}/{prediction_filename}"
            geojson = post_process(mosaic[0], transform)

            for geometry in geojson:
                updated_geometry = PostProcess.convert_geojson(geometry)
                geojson_list.append(updated_geometry)
            geojson = subset_geojson(geojson_list, bounding_box)
        except Exception as e:
            print('!!! infer error', infer_date, model_path, bounding_box, e)
            torch.cuda.empty_cache()
        print("!!! Infer Time:", time.time() - start_time)
    gc.collect()

    return {
        'predictions': geojson,
        's3_link': s3_link,
    }
