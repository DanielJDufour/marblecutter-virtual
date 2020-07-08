# coding=utf-8
from __future__ import absolute_import

import logging

from affine import Affine
from cachetools.func import lru_cache
from collections import defaultdict
from concurrent import futures
from flask import Flask, Markup, jsonify, redirect, render_template, request
from flask_cors import CORS
from marblecutter import Bounds, WEB_MERCATOR_CRS, get_resolution_in_meters, get_source, read_window
from marblecutter.catalogs import WGS84_CRS
from marblecutter.tiling import TILE_SHAPE
from marblecutter.mosaic import composite, get_pixels, MAX_WORKERS
from marblecutter import NoCatalogAvailable, tiling
from marblecutter.formats.optimal import Optimal
from marblecutter.transformations import Image
from marblecutter.web import bp, url_for
from marblecutter.utils import Bounds, Source, PixelCollection
from rasterio import warp
from rasterio.io import MemoryFile
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
import mercantile
import multiprocessing
import numpy as np
import jq
import json
import re
import requests
from shapely.geometry import box, Polygon
from werkzeug.datastructures import ImmutableMultiDict

try:
    from urllib.parse import urlparse, urlencode
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError
except ImportError:
    from urlparse import urlparse
    from urllib import urlencode
    from urllib2 import urlopen, Request, HTTPError

from .catalogs import VirtualCatalog

LOG = logging.getLogger(__name__)

IMAGE_TRANSFORMATION = Image()
IMAGE_FORMAT = Optimal()

TILE_SHAPE = (256, 256)

app = Flask("marblecutter-virtual")
app.register_blueprint(bp)
app.url_map.strict_slashes = False
CORS(app, send_wildcard=True)


@lru_cache()
def make_catalog(args):
    if args.get("url", "") == "":
        raise NoCatalogAvailable()

    try:
        return VirtualCatalog(
            args["url"],
            rgb=args.get("rgb"),
            nodata=args.get("nodata"),
            linear_stretch=args.get("linearStretch"),
            resample=args.get("resample"),
            expr=args.get("expr", None)
        )
    except Exception as e:
        LOG.exception(e)
        raise NoCatalogAvailable()


@app.route("/")
def index():
    return (render_template("index.html"), 200, {"Content-Type": "text/html"})


@app.route("/tiles/")
def meta():
    catalog = make_catalog(request.args)

    meta = {
        "bounds": catalog.bounds,
        "center": catalog.center,
        "maxzoom": catalog.maxzoom,
        "minzoom": catalog.minzoom,
        "name": catalog.name,
        "tilejson": "2.1.0",
        "tiles": [
            "{}{{z}}/{{x}}/{{y}}?{}".format(
                url_for("meta", _external=True, _scheme=""), urlencode(request.args)
            )
        ],
    }

    return jsonify(meta)


@app.route("/bounds/")
def bounds():
    catalog = make_catalog(request.args)

    return jsonify({"url": catalog.uri, "bounds": catalog.bounds})

@app.route("/test")
def test():
    return (
        render_template(
            "test.html",
            tilejson_url=Markup(
                url_for("meta", _external=True, _scheme="", **request.args)
            ),
        ),
        200,
        {"Content-Type": "text/html"},
    )


@app.route("/preview")
def preview():
    try:
        # initialize the catalog so this route will fail if the source doesn't exist
        make_catalog(request.args)
    except Exception:
        return redirect(url_for("index"), code=303)

    return (
        render_template(
            "preview.html",
            tilejson_url=Markup(
                url_for("meta", _external=True, _scheme="", **request.args)
            ),
            source_url=request.args["url"],
        ),
        200,
        {"Content-Type": "text/html"},
    )


@app.route("/stac/<int:z>/<int:x>/<int:y>")
@app.route("/stac/<int:z>/<int:x>/<int:y>@<int:scale>x")
def render_png_from_stac_catalog(z, x, y, scale=1):
    # example:
    # https://4reb3lh9m6.execute-api.us-west-2.amazonaws.com/stage/stac/search
    # /stac/16/16476/24074@2x?https://sat-api.developmentseed.org/collections/landsat-8-l1/items?bbox=%5B-20%2C-10%2C-1.99951171875%2C0.0%5D&limit=500
    # test tile:
    # http://localhost:8000/stac/16/16476/24074@2x?url=https%3A%2F%2F4reb3lh9m6.execute-api.us-west-2.amazonaws.com%2Fstage%2Fstac%2Fsearch
    # compare result to single geotiff version:
    # http://localhost:8000/tiles/16/16476/24074@2x?url=https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsyncarto-data-test%2Foutput%2F060801NE_COG.TIF
    stac_catalog_url = request.args.get("url", None)
    jq_filter = request.args.get("jq", None)
    print("jq_filter:", jq_filter)
    expr = request.args.get("expr", None)

    # asset_name_pattern = "([A-Za-z][A-Za-z0-9]+)(?:\[(\d+)\])?"
    asset_name_pattern = "(?P<asset>[A-Za-z][A-Za-z0-9]+)(?:\[(?P<name>\d+)\])?"
    print("expr:", expr)
    # parts of the expr that reference an asset by name
    matches = list(set(re.findall(asset_name_pattern, expr)))

    # list of assets and which band to take from it
    asset_bands = sorted(list(set([(asset, int(band) if band else 0) for asset, band in matches])))
    print("asset_bands:", asset_bands)

    # list of names of assets
    assets = list(set([asset for asset, band in matches]))
    print("assets:", assets)

    def repl(m):
        print("starting repl with m.string:", m.string)
        asset, band = m.groups()
        print("m.groups:", m.groups())
        band = int(band) if band else 0
        # add one to index number because bands index starts at 1, i.e. b1, b2, b3... 
        return 'b' + str(asset_bands.index((asset, band)) + 1)
    print("about to call re.sub with expr:", expr)
    source_expr = re.sub(asset_name_pattern, repl, expr) if expr else None
    print("source_expr:", source_expr)

    tile = mercantile.Tile(x, y, z)

    tile_bounds = mercantile.bounds(tile)
    tile_bbox = [tile_bounds.west, tile_bounds.south, tile_bounds.east, tile_bounds.north]

    parent_tile = mercantile.parent(tile)
    search_bounds = mercantile.bounds(parent_tile)

    # per https://github.com/radiantearth/stac-spec/blob/master/api-spec/filters.md
    search_bbox = [
        search_bounds.west,
        search_bounds.south,
        search_bounds.east,
        search_bounds.north
    ]
    print("search_bbox:", search_bbox)

    tile_polygon = box(*tile_bbox)

    params = {
        'bbox': str(search_bbox).replace(' ', ''),
        'limit': 500,
    }
    print("params:", params)
    response = requests.get(stac_catalog_url, params=params)
    assert response.status_code == 200
    try:
        features = response.json()['features']
    except:
        print("response.text:", response.text)
        return

    with open("features.json", "w") as f:
        f.write(json.dumps(features))
    LOG.info('{} number of features: {}'.format(response.url, len(features)))
    print("features[0]", features[0]['properties'])

    # filter to bbox's that actually overlap; sat-api elasticsearch
    # precision not good enough for our <1km tiles
    features = [feature for feature in features if box(*feature['bbox']).intersects(tile_polygon)]
    LOG.info('features left after bbox overlap filter: {}'.format(len(features)))

    if jq_filter: features = jq.compile(jq_filter).input(features).first()
    LOG.info('features left after jq filter: {}'.format(len(features)))

    canvas_bounds = Bounds(bounds=mercantile.xy_bounds(tile), crs=WEB_MERCATOR_CRS)
    print("canvas_bounds:", canvas_bounds)

    sources = []
    for fid, feature in enumerate(features):
        print("feature:", type(feature))

        props = feature.get('properties', {})

        # create the bounds
        feature_epsg = props.get('eo:epsg', None)
        print("feature_epsg:", feature_epsg)
        feature_crs: CRS = CRS.from_epsg(feature_epsg) if feature_epsg else None
        print("feature_crs:", feature_crs)

        # bounds in WGS84
        feature_bounds: BoundingBox = BoundingBox(*feature['bbox']) if 'bbox' in feature else None
        print("feature_bounds:", feature_bounds)

        if 'eo:bands' in props:
            name2res = dict([(b['name'], b['gsd']) for b in props['eo:bands']])
            print("name2res:", name2res)
        else:
            raise Exception("unknown resolution")

        # if not all same resolution, throw an error
        if len(set([name2res[name] for name in assets])) != 1:
            raise Exception("Uh Oh, more than one resolution")

        images = {}
        if assets:
            print('filtering by assets:', assets)
            for asset in assets:
                print("asset:", asset)
                images[asset] = feature['assets'][asset]['href']
        elif 'visual' in feature['assets']:
            images['visual'] = feature['assets']['visual']['href']

        # size of the tile, usually (256, 256)
        shape = tuple(map(int, Affine.scale(scale) * TILE_SHAPE))
        print("shape:", shape)

        # render tile at highest-available resolution
        resolution = min([name2res[key] for key in assets])
        print("mosaic resolution:", resolution)

        def read_pixels(asset: str) -> (str, PixelCollection):
            LOG.info("Getting pixels for " + asset)
            url = images[asset]
            source = Source(url=url, name=url, resolution=resolution)            
            # _, pixels = get_pixels(source, expand=True, canvas_bounds=canvas_bounds, shape=shape)
            with get_source(source.url) as src:
                window_data = read_window(src, canvas_bounds, shape, source)
                print("window_data:", window_data)         
            return (asset, window_data)

        with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            asset2pc = dict(list(executor.map(read_pixels, assets)))
        print("asset2pc:", asset2pc)

        # should do a check and make sure all values in asset2pixels have the same bounds
        # asset2pixels.values().all
        if not len(set([px.bounds for px in asset2pc.values()])) == 1:
            raise Exception("seems like there are more than one bounds in window data")

        ## map bands arithmetic to single file arithmetic
        windows = tuple([asset2pc[asset].data[band] for asset, band in asset_bands])
        stacked = np.ma.stack(windows)
        print("stacked:", type(stacked))
        print("stacked.shape:", stacked.shape)

        # should probs use pc bounds
        pixels = PixelCollection(stacked, canvas_bounds)
        print('pixels:', pixels)

        source = Source(
            url="",
            name=str(fid) + '{' + ','.join(assets) + '}',
            resolution=resolution,
            expr=source_expr,
            window_data=pixels,
            recipes={ "expr": source_expr, "imagery": True }
        )
        print("source:", source)
        sources.append(source)

    headers, data = tiling.render_tile_from_sources(
        tile,
        sources,
        format=IMAGE_FORMAT,
        transformation=IMAGE_TRANSFORMATION,
        scale=scale,
    )

    # # ???
    # # headers.update(catalog.headers)

    return data, 200, headers


@app.route("/tiles/<int:z>/<int:x>/<int:y>")
@app.route("/tiles/<int:z>/<int:x>/<int:y>@<int:scale>x")
def render_png(z, x, y, scale=1):
    catalog = make_catalog(request.args)
    tile = mercantile.Tile(x, y, z)

    headers, data = tiling.render_tile(
        tile,
        catalog,
        format=IMAGE_FORMAT,
        transformation=IMAGE_TRANSFORMATION,
        scale=scale,
    )

    headers.update(catalog.headers)

    return data, 200, headers
