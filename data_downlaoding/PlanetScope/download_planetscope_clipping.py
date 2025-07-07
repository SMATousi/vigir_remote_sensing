#!/usr/bin/env python
"""
Download PlanetScope PSScene imagery defined in a YAML config.

* Reads AOI/date/cloud limits from config.yaml
* Reads API key from environment variable PLANET_API_KEY
* Searches Planet Data API v1
* Activates & downloads analytic_sr assets
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List

import requests
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_bounds
from rasterio.warp import transform_geom
import yaml

API_ROOT = "https://api.planet.com/data/v1"
SEARCH_URL = f"{API_ROOT}/quick-search"


# ───────────────────────── Helpers ────────────────────────────────
def load_yaml(path: str | pathlib.Path = "config.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_api_key() -> str:
    api_key = os.getenv("PLANET_API_KEY")
    if not api_key:
        sys.exit("Error: environment variable PLANET_API_KEY is not set.")
    return api_key


def read_aoi(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a GeoJSON geometry dict from file or inline config."""
    file_path = cfg["aoi"].get("geojson_file")
    if file_path and pathlib.Path(file_path).exists():
        with open(file_path, "r") as f:
            geojson = json.load(f)
        if "type" in geojson and geojson["type"] == "FeatureCollection":
            return geojson["features"][0]["geometry"]
        if geojson.get("type") == "Feature":
            return geojson["geometry"]
        return geojson
    # fallback: inline geometry
    return cfg["aoi"]["geometry"]


def build_filter(cfg: Dict[str, Any]) -> Dict[str, Any]:
    geom_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": read_aoi(cfg),
    }
    date_filter = {
        "type": "DateRangeFilter",
        "field_name": "acquired",
        "config": {
            "gte": cfg["date_range"]["start"],
            "lte": cfg["date_range"]["end"],
        },
    }
    cloud_filter = {
        "type": "RangeFilter",
        "field_name": "cloud_cover",
        "config": {"lte": cfg["cloud_cover"]["max"]},
    }
    return {"type": "AndFilter", "config": [geom_filter, date_filter, cloud_filter]}


def search_items(cfg: Dict[str, Any], api_key: str) -> List[Dict[str, Any]]:
    payload = {"item_types": cfg["item_types"], "filter": build_filter(cfg)}
    r = requests.post(SEARCH_URL, auth=(api_key, ""), json=payload)
    r.raise_for_status()
    return r.json().get("features", [])


def activate(asset_url: str, api_key: str) -> Dict[str, Any]:
    """Poll until an asset becomes active and return its JSON."""
    while True:
        asset = requests.get(asset_url, auth=(api_key, "")).json()
        status = asset["status"]
        print("Asset status: ", status)
        if status == "active":
            return asset
        if status == "inactive":
            requests.post(asset["_links"]["activate"], auth=(api_key, ""))
        time.sleep(2)


def download_asset(url: str, target: pathlib.Path, chunk: int = 8192) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        tmp = target.with_suffix(".part")
        with open(tmp, "wb") as f:
            for piece in r.iter_content(chunk):
                f.write(piece)
        tmp.rename(target)


def clip_to_geometry(src_path: pathlib.Path,
                     dst_path: pathlib.Path,
                     geom_geojson: Dict[str, Any]) -> None:
    """Clip src_path to geom_geojson and save to dst_path."""
    with rasterio.open(src_path) as src:
        # re-project geometry from EPSG:4326 to the image CRS
        geom_img_crs = transform_geom(
            "EPSG:4326", src.crs, geom_geojson, precision=6
        )
        out_image, out_transform = mask(src, [geom_img_crs], crop=True)
        meta = src.meta.copy()
        meta.update(
            {
                "height": out_image.shape[1],
                "width":  out_image.shape[2],
                "transform": out_transform,
            }
        )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **meta) as dst:
        dst.write(out_image)


def process_item(item: Dict[str, Any], cfg: Dict[str, Any], api_key: str) -> None:
    item_id   = item["id"]
    item_type = item["properties"]["item_type"]
    assets_url = f"{API_ROOT}/item-types/{item_type}/items/{item_id}/assets/"
    assets = requests.get(assets_url, auth=(api_key, "")).json()

    for asset_type in cfg["asset_types"]:
        if asset_type not in assets:
            print(f"✗ {asset_type} not offered for {item_id}")
            continue

        out_dir   = pathlib.Path(cfg["output"]["directory"]) / item_id
        full_tif  = out_dir / f"{asset_type}.tif"
        clip_tif  = out_dir / f"{asset_type}_clip.tif"

        if clip_tif.exists() and not cfg["output"]["overwrite"]:
            print(f"• {clip_tif} exists — skipping")
            return

        # ── download full scene ───────────────────────────────────
        asset_json = activate(assets[asset_type]["_links"]["_self"], api_key)
        print(f"↓ downloading {item_id}:{asset_type}")
        download_asset(asset_json["location"], full_tif, cfg["download"]["chunk_size"])

        # ── optional clipping step ────────────────────────────────
        if cfg.get("clip_to_aoi", True):
            geom = read_aoi(cfg)
            print(f"✂ clipping {full_tif.name} to AOI …")
            clip_to_geometry(full_tif, clip_tif, geom)
            print(f"✓ saved clipped raster → {clip_tif}")
            if not cfg["output"]["keep_full_scene"]:
                full_tif.unlink(missing_ok=True)   # delete big original
        else:
            print(f"✦ clipping disabled; kept full scene at {full_tif}")



# ────────────────────────── Main ─────────────────────────────────
def main() -> None:
    cfg = load_yaml()
    api_key = get_api_key()

    items = search_items(cfg, api_key)
    # print(items)
    print(f"Found {len(items)} scenes.")

    # parallel downloads
    with cf.ThreadPoolExecutor(max_workers=cfg["download"]["max_parallel"]) as pool:
        futures = [
            pool.submit(process_item, item, cfg, api_key) for item in items
        ]
        for fut in cf.as_completed(futures):
            fut.result()


if __name__ == "__main__":
    main()
