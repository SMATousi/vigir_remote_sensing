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
from datetime import datetime
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
    """Return a GeoJSON geometry dict from file or inline config.
    
    Supports:
    - FeatureCollection with one or more features
    - Single Feature objects
    - Direct geometry objects
    - MultiPolygon and Polygon geometries
    """
    file_path = cfg["aoi"].get("geojson_file")
    if file_path:
        # Expand user home directory if present
        expanded_path = pathlib.Path(file_path).expanduser()
        
        if not expanded_path.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {expanded_path}")
            
        try:
            with open(expanded_path, "r") as f:
                geojson = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in GeoJSON file {expanded_path}: {e}")
        
        # Handle FeatureCollection
        if geojson.get("type") == "FeatureCollection":
            features = geojson.get("features", [])
            if not features:
                raise ValueError("FeatureCollection contains no features")
            
            # If multiple features, we could either:
            # 1. Use only the first feature (current behavior)
            # 2. Union all geometries (more complex)
            # For now, use first feature but warn if multiple
            if len(features) > 1:
                print(f"Warning: FeatureCollection contains {len(features)} features. Using the first feature only.")
                print("If you need to use all features, consider merging them into a single MultiPolygon.")
            
            geometry = features[0].get("geometry")
            if not geometry:
                raise ValueError("First feature has no geometry")
            return geometry
            
        # Handle single Feature
        elif geojson.get("type") == "Feature":
            geometry = geojson.get("geometry")
            if not geometry:
                raise ValueError("Feature has no geometry")
            return geometry
            
        # Handle direct geometry object
        elif geojson.get("type") in ["Polygon", "MultiPolygon", "Point", "LineString", "MultiPoint", "MultiLineString"]:
            return geojson
            
        else:
            raise ValueError(f"Unsupported GeoJSON type: {geojson.get('type')}")
    
    # Fallback: inline geometry from config
    if "geometry" not in cfg["aoi"]:
        raise ValueError("No geojson_file specified and no inline geometry found in config")
    
    return cfg["aoi"]["geometry"]


def generate_year_date_ranges(cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate date ranges for each year in the specified range."""
    date_range_cfg = cfg["date_range"]
    
    # Check if it's the old single date format
    if "start" in date_range_cfg and "end" in date_range_cfg:
        return [{"start": date_range_cfg["start"], "end": date_range_cfg["end"]}]
    
    # New multi-year format
    years_cfg = date_range_cfg["years"]
    day_cfg = date_range_cfg["day_timeframe"]
    time_cfg = date_range_cfg["time_of_day"]
    
    start_year = years_cfg["start_year"]
    end_year = years_cfg["end_year"]
    start_month = day_cfg["start_month"]
    start_day = day_cfg["start_day"]
    end_month = day_cfg["end_month"]
    end_day = day_cfg["end_day"]
    start_time = time_cfg["start"]
    end_time = time_cfg["end"]
    
    date_ranges = []
    for year in range(start_year, end_year + 1):
        start_date = f"{year:04d}-{start_month:02d}-{start_day:02d}T{start_time}"
        end_date = f"{year:04d}-{end_month:02d}-{end_day:02d}T{end_time}"
        date_ranges.append({"start": start_date, "end": end_date})
    
    return date_ranges


def build_filter(cfg: Dict[str, Any], date_range: Dict[str, str]) -> Dict[str, Any]:
    """Build filter for a specific date range."""
    geom_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": read_aoi(cfg),
    }
    date_filter = {
        "type": "DateRangeFilter",
        "field_name": "acquired",
        "config": {
            "gte": date_range["start"],
            "lte": date_range["end"],
        },
    }
    cloud_filter = {
        "type": "RangeFilter",
        "field_name": "cloud_cover",
        "config": {"lte": cfg["cloud_cover"]["max"]},
    }
    return {"type": "AndFilter", "config": [geom_filter, date_filter, cloud_filter]}


def search_items(cfg: Dict[str, Any], api_key: str, date_range: Dict[str, str]) -> List[Dict[str, Any]]:
    """Search for items within a specific date range."""
    payload = {"item_types": cfg["item_types"], "filter": build_filter(cfg, date_range)}
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
    """Download an asset from url to target path."""
    target.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(target, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk):
            f.write(chunk)

# ───────────────────────── CRS resolver ───────────────────────────
def _download_text(url: str, api_key: str, chunk: int = 8192) -> str:
    r = requests.get(url, auth=(api_key, ""), stream=True)
    r.raise_for_status()
    buf = []
    for c in r.iter_content(chunk_size=chunk):
        if c:
            buf.append(c)
    return b"".join(buf).decode("utf-8", errors="replace")


def resolve_planet_epsg(item: Dict[str, Any], assets: Dict[str, Any], api_key: str) -> str | None:
    """
    Try hard to obtain the scene's projected CRS from Planet metadata.

    Priority:
      1) item['properties']['epsg_code'] if present
      2) XML/JSON metadata asset for this scene (parse EPSG or UTM WKT)
      3) Heuristic UTM from centroid lon/lat (last resort)
    """
    # 1) Direct property
    epsg = (item.get("properties") or {}).get("epsg_code")
    if isinstance(epsg, (int, str)):
        try:
            epsg_int = int(str(epsg))
            return f"EPSG:{epsg_int}"
        except Exception:
            pass  # fall through

    # 2) Look for a metadata-like asset and parse
    # Common names vary by item/asset versions; try several:
    metadata_keys = [
        "analytic_sr_xml", "analytic_xml", "metadata", "metadata_xml",
        "udm2_xml", "basic_analytic_xml", "basic_xml"
    ]
    for k in metadata_keys:
        if k in assets and "location" in assets[k]:
            try:
                txt = _download_text(assets[k]["location"], api_key=api_key)
                # Very permissive EPSG parse: look for 'EPSG:32xxx/327xx' or 'AUTHORITY["EPSG","32xxx"]'
                m = re.search(r"EPSG[:\"]\s*([0-9]{4,6})", txt)
                if m:
                    return f"EPSG:{int(m.group(1))}"
                # Try extracting UTM zone from WKT text if EPSG tag not explicit
                # e.g., PROJCS["WGS 84 / UTM zone 14N"]
                m2 = re.search(r"UTM zone\s+(\d{1,2})([NS])", txt, re.IGNORECASE)
                if m2:
                    zone = int(m2.group(1))
                    hemi = m2.group(2).upper()
                    return f"EPSG:{326 if hemi=='N' else 327}{zone:02d}"
            except Exception:
                pass  # try next option

    # 3) UTM heuristic (last resort). Use item geometry centroid.
    try:
        geom = item.get("geometry") or {}
        # Compute centroid coarsely from bbox if available, else assume the first coordinate
        bbox = (item.get("bbox") or [])
        if len(bbox) == 4:
            lon = 0.5 * (bbox[0] + bbox[2])
            lat = 0.5 * (bbox[1] + bbox[3])
        else:
            # Very rough: pick first coordinate in geometry
            coords = geom.get("coordinates")
            # Handle Polygon/MultiPolygon minimalistically
            while isinstance(coords, list) and len(coords) and isinstance(coords[0], list):
                coords = coords[0]
            lon, lat = coords if isinstance(coords, (list, tuple)) and len(coords) >= 2 else (None, None)

        if lon is not None and lat is not None:
            zone = int((lon + 180) // 6) + 1
            return f"EPSG:{326 if lat >= 0 else 327}{zone:02d}"
    except Exception:
        pass

    return None


def fix_raster_crs(raster_path: pathlib.Path, target_crs: str) -> None:
    """
    Assign a (missing) CRS to a raster without altering data/transform.
    Does nothing if the file already has a CRS.
    """
    with rasterio.open(raster_path) as src:
        if src.crs is not None:
            print(f"✓ {raster_path.name} already has CRS: {src.crs}")
            return
        data = src.read()
        meta = src.meta.copy()

    if not target_crs:
        raise ValueError(f"No CRS present on {raster_path} and no target_crs provided.")

    meta["crs"] = target_crs
    # Ensure required keys are intact
    for k in ("transform", "dtype", "count", "driver", "width", "height"):
        if k not in meta:
            raise ValueError(f"Missing '{k}' in raster metadata for {raster_path}")

    temp_path = raster_path.with_suffix(".tmp.tif")
    try:
        with rasterio.open(temp_path, "w", **meta) as dst:
            dst.write(data)
            # carry over tags (if any)
            try:
                with rasterio.open(raster_path) as src_old:
                    dst.update_tags(**src_old.tags())
            except Exception:
                pass
        temp_path.replace(raster_path)
        print(f"🔧 Assigned CRS {target_crs} → {raster_path.name}")
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise



def clip_to_geometry(src_path: pathlib.Path,
                     dst_path: pathlib.Path,
                     geom_geojson: Dict[str, Any]) -> None:
    """
    Clip src_path to geom_geojson and save to dst_path.

    Requires the source to have a valid CRS. If CRS is missing, fix it first.
    """
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError(
                f"{src_path.name} has no CRS; call fix_raster_crs(...) first with the correct EPSG."
            )

        raster_crs = src.crs
        # AOI is assumed in EPSG:4326 (config); reproject to raster CRS
        geom_img_crs = transform_geom("EPSG:4326", raster_crs, geom_geojson, precision=6)

        out_image, out_transform = mask(src, [geom_img_crs], crop=True)
        meta = src.meta.copy()
        meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": raster_crs
        })

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **meta) as dst:
        dst.write(out_image)



def process_item(item: Dict[str, Any], cfg: Dict[str, Any], api_key: str, year: int = None) -> None:
    """Process a single item (with optional year for directory organization)."""
    item_id   = item["id"]
    item_type = item["properties"]["item_type"]
    assets_url = f"{API_ROOT}/item-types/{item_type}/items/{item_id}/assets/"
    assets = requests.get(assets_url, auth=(api_key, "")).json()

    for asset_type in cfg["asset_types"]:
        if asset_type not in assets:
            print(f"✗ {asset_type} not offered for {item_id}")
            continue

        # Organize by year if provided
        base_dir = pathlib.Path(cfg["output"]["directory"])
        if year:
            out_dir = base_dir / f"year_{year}" / item_id
        else:
            out_dir = base_dir / item_id
            
        full_tif  = out_dir / f"{asset_type}.tif"
        clip_tif  = out_dir / f"{asset_type}_clip.tif"

        if clip_tif.exists() and not cfg["output"]["overwrite"]:
            print(f"• {clip_tif} exists — skipping")
            return

        # ── download full scene ───────────────────────────────────
        asset_json = activate(assets[asset_type]["_links"]["_self"], api_key)
        print(f"↓ downloading {item_id}:{asset_type}")
        download_asset(asset_json["location"], full_tif, cfg["download"]["chunk_size"])
        

        with rasterio.open(full_tif) as src:
            has_crs = src.crs is not None

        if not has_crs:
            resolved = resolve_planet_epsg(item, assets, api_key=api_key)
            if not resolved:
                raise RuntimeError(
                    f"Could not resolve EPSG for {item_id} ({asset_type}). "
                    f"Inspect metadata assets and set CRS manually."
                )
            fix_raster_crs(full_tif, target_crs=resolved)

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

    # Generate date ranges for all years
    date_ranges = generate_year_date_ranges(cfg)
    print(f"Processing {len(date_ranges)} date range(s)...")
    
    all_items = []
    
    # Search for items in each date range
    for i, date_range in enumerate(date_ranges):
        print(f"\nSearching date range {i+1}/{len(date_ranges)}: {date_range['start']} to {date_range['end']}")
        items = search_items(cfg, api_key, date_range)
        print(f"Found {len(items)} scenes for this date range.")
        
        # Extract year from date range for directory organization
        year = int(date_range['start'][:4])
        
        # Add year information to items for processing
        for item in items:
            item['_year'] = year
        
        all_items.extend(items)
    
    print(f"\nTotal scenes found across all date ranges: {len(all_items)}")
    
    if not all_items:
        print("No scenes found. Exiting.")
        return
    
    # parallel downloads
    with cf.ThreadPoolExecutor(max_workers=cfg["download"]["max_parallel"]) as pool:
        futures = [
            pool.submit(process_item, item, cfg, api_key, item.get('_year')) for item in all_items
        ]
        for fut in cf.as_completed(futures):
            fut.result()


if __name__ == "__main__":
    main()
