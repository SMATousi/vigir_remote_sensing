#!/usr/bin/env python
"""
PlanetScope ROI Cloud Coverage via UDM2 (PSScene)

For each month (Jan–Sep by default), this script:
  1) Finds PSScene items intersecting your AOI.
  2) Activates & downloads their `ortho_udm2` mask.
  3) Clips the UDM2 to your AOI and computes cloud coverage % within the AOI.
  4) Aggregates monthly stats using the ROI-based cloud % rather than the scene-level cloud_cover.

Outputs:
  - A CSV with per-scene ROI cloud % for auditing.
  - A summary plot comparing monthly totals and thresholds (<5%, ==0%).
"""

from __future__ import annotations
import os
import sys
import json
import time
import pathlib
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import requests
import yaml
import pandas as pd
import matplotlib.pyplot as plt

import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import numpy as np
from rasterio.warp import transform_geom
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom
import numpy as np
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from shapely.validation import make_valid
# ----------------------------
# CONFIGURABLE UDM2 CLASSES
# ----------------------------
# Adjust these if your UDM2 coding differs.
# Commonly: 1=clear/valid, 2=cloud, 3=cloud shadow, 4=snow/ice, 5=haze
CLEAR_VALUES = {1}
CLOUDY_VALUES = {2, 3, 4, 5}

API_ROOT = "https://api.planet.com/data/v1"
SEARCH_URL = f"{API_ROOT}/quick-search"

# Item & asset constants
UDM2_ASSET_KEY = "ortho_udm2"  # the asset we activate/download per PSScene item

# I/O
DOWNLOAD_DIR = pathlib.Path("udm2_cache")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def load_yaml(path: str | pathlib.Path = "config.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_api_key() -> str:
    api_key = os.getenv("PLANET_API_KEY")
    if not api_key:
        sys.exit("Error: environment variable PLANET_API_KEY is not set.")
    return api_key

def read_aoi(cfg: Dict[str, Any]) -> Dict[str, Any]:
    file_path = cfg["aoi"].get("geojson_file")
    if file_path:
        expanded = pathlib.Path(file_path).expanduser()
        if not expanded.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {expanded}")
        with open(expanded, "r") as f:
            gj = json.load(f)
        if gj.get("type") == "FeatureCollection":
            feats = gj.get("features", [])
            if not feats:
                raise ValueError("FeatureCollection contains no features")
            return feats[0]["geometry"]
        if gj.get("type") == "Feature":
            return gj["geometry"]
        return gj
    if "geometry" not in cfg["aoi"]:
        raise ValueError("No geojson_file specified and no inline geometry found in config")
    return cfg["aoi"]["geometry"]

def build_filter_for_month(cfg: Dict[str, Any], year: int, month: int) -> Dict[str, Any]:
    geom_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": read_aoi(cfg),
    }
    start_date = f"{year:04d}-{month:02d}-01T00:00:00Z"
    end_date = f"{(year + (month==12)):04d}-{(1 if month==12 else month+1):02d}-01T00:00:00Z"
    date_filter = {
        "type": "DateRangeFilter",
        "field_name": "acquired",
        "config": {"gte": start_date, "lt": end_date},
    }
    return {"type": "AndFilter", "config": [geom_filter, date_filter]}

def search_items_for_month(cfg: Dict[str, Any], api_key: str, year: int, month: int) -> List[Dict[str, Any]]:
    payload = {"item_types": cfg["item_types"], "filter": build_filter_for_month(cfg, year, month)}
    items: List[Dict[str, Any]] = []
    next_url = SEARCH_URL
    while next_url:
        if next_url == SEARCH_URL:
            r = requests.post(next_url, auth=(api_key, ""), json=payload)
        else:
            r = requests.get(next_url, auth=(api_key, ""))
        r.raise_for_status()
        data = r.json()
        items.extend(data.get("features", []))
        next_url = data.get("_links", {}).get("_next")
    return items

def activate_asset(api_key: str, item_type: str, item_id: str, asset_key: str) -> Optional[str]:
    """
    Activates asset and returns a download location URL when ready.
    """
    assets_url = f"{API_ROOT}/item-types/{item_type}/items/{item_id}/assets"
    r = requests.get(assets_url, auth=(api_key, ""))
    r.raise_for_status()
    assets = r.json()

    if asset_key not in assets:
        return None

    asset_info = assets[asset_key]
    # If not active, activate
    if asset_info["status"] != "active":
        activate_url = asset_info["_links"]["activate"]
        ra = requests.post(activate_url, auth=(api_key, ""))
        # Activation is async; we will poll
        if ra.status_code not in (200, 202):
            return None

        # Poll for activation
        for _ in range(60):  # up to ~5 min (60 * 5s)
            time.sleep(5)
            r2 = requests.get(assets_url, auth=(api_key, ""))
            r2.raise_for_status()
            assets2 = r2.json()
            if assets2[asset_key]["status"] == "active":
                return assets2[asset_key]["location"]
        return None
    else:
        return asset_info["location"]

def download_asset(location_url: str, api_key: str, outpath: pathlib.Path) -> pathlib.Path:
    with requests.get(location_url, auth=(api_key, ""), stream=True) as r:
        r.raise_for_status()
        with open(outpath, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return outpath

def to_multi_polygon(geom: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize any Polygon/MultiPolygon Feature geometry to a unified MultiPolygon-like dict for rasterio.mask.
    """
    g = shape(geom)
    if g.geom_type in ("Polygon", "MultiPolygon"):
        # unify if multiple features (user might pass FC later)
        if g.geom_type == "Polygon":
            return mapping(g)
        else:
            return mapping(unary_union([g]))
    # If line/point slipped in, buffer slightly (rare)
    return mapping(g.buffer(0.0000001))

def _normalize_geom_to_raster_crs(geom_geojson: Dict[str, Any], src_crs: Any, dst_crs: Any) -> Dict[str, Any]:
    """
    Reproject a GeoJSON geometry from src_crs to dst_crs and ensure it's valid.
    Returns a GeoJSON geometry in dst_crs.
    """
    # Reproject (preserve geometry type, densify to better respect curves)
    reproj = transform_geom(src_crs, dst_crs, geom_geojson, precision=6)


    # Validate/repair with shapely (handles self-intersections, ring orientation, etc.)
    g = shape(reproj)
    try:
        g = make_valid(g)  # Shapely >= 2.0
    except Exception:
        # Fallback: buffer(0) trick for older shapely
        g = g.buffer(0)

    # Merge multiparts just in case
    if g.geom_type == "GeometryCollection":
        # keep only areal parts
        polys = [geom for geom in g.geoms if geom.geom_type in ("Polygon", "MultiPolygon")]
        if polys:
            g = unary_union(polys)
    elif g.geom_type not in ("Polygon", "MultiPolygon"):
        # If user passed points/lines, create a tiny buffer so it has area
        g = g.buffer(0.00001)

    return mapping(g)


def compute_roi_cloud_percent(udm2_path: pathlib.Path, roi_geom_wgs84: Dict[str, Any]) -> Optional[float]:
    """
    ROI cloud % from UDM2 (bands 1..6 exclusive classes; band 6=cloud).
    Reprojects the WGS84 AOI to the raster's CRS before masking.

      valid = any(B1..B6 == 1)
      cloudy = (B6 == 1)
      roi_cloud_percent = 100 * cloudy / valid
    """
    try:
        with rasterio.open(udm2_path) as src:
            # Planet/GeoJSON AOIs are WGS84 (EPSG:4326). Reproject to the raster CRS.
            raster_crs = src.crs
            if raster_crs is None:
                raise RuntimeError("Raster has no CRS; cannot reproject AOI.")

            roi_in_raster_crs = _normalize_geom_to_raster_crs(
                roi_geom_wgs84, src_crs="EPSG:4326", dst_crs=raster_crs
            )

            # Clip; set all_touched=True to be inclusive at edges (optional)
            data, _ = rio_mask(src, [roi_in_raster_crs], crop=True, nodata=0, all_touched=True)

            if data.shape[0] < 6:
                print(f"[warn] UDM2 has {data.shape[0]} bands (<6): {udm2_path}")
                return None

            b1_clear  = data[0]
            b2_snow   = data[1]
            b3_shadow = data[2]
            b4_lhaze  = data[3]
            b5_hhaze  = data[4] if data.shape[0] >= 5 else 0
            b6_cloud  = data[5]

            # Optional strictness: exclude UDM 'unusable' pixels (band 8)
            # b8_unusable = data[7] if data.shape[0] >= 8 else None

            valid_mask = (
                (b1_clear == 1) |
                (b2_snow == 1) |
                (b3_shadow == 1) |
                (b4_lhaze == 1) |
                (b5_hhaze == 1) |
                (b6_cloud == 1)
            )

            # If excluding unusable pixels, uncomment:
            # if b8_unusable is not None:
            #     valid_mask = valid_mask & (b8_unusable == 0)

            valid_px = int(np.count_nonzero(valid_mask))
            if valid_px == 0:
                return None

            cloudy_mask = (b6_cloud == 1) & valid_mask
            cloudy_px = int(np.count_nonzero(cloudy_mask))

            print(f"Valid pixels: {valid_px}")
            print(f"Cloudy pixels: {cloudy_px}")

            return 100.0 * cloudy_px / valid_px



    except ValueError as ve:
        # This is the common "Input shapes do not overlap raster" case
        print(f"[warn] No overlap after reprojection for {udm2_path}: {ve}")
        return None
    except Exception as e:
        print(f"[warn] Failed to compute ROI cloud from {udm2_path}: {e}")
        return None



def month_name(m: int) -> str:
    return ["January","February","March","April","May","June","July","August","September"][m-1]

def get_monthly_statistics_roi(cfg: Dict[str, Any], api_key: str, year: int) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    months = list(range(1, 10))  # Jan–Sep
    roi_geom = read_aoi(cfg)

    per_scene_rows: List[Dict[str, Any]] = []
    monthly_stats: List[Dict[str, Any]] = []

    for m in months:
        print(f"\nAnalyzing {month_name(m)} {year}...")
        items = search_items_for_month(cfg, api_key, year, m)
        total_count = 0
        lt5 = 0
        eq0 = 0

        for feat in items:
            item_type = feat["properties"]["item_type"] if "item_type" in feat["properties"] else feat["id"].split("_")[0]
            item_id = feat["id"]
            total_count += 1

            # Activate + download UDM2
            loc = activate_asset(api_key, item_type, item_id, UDM2_ASSET_KEY)
            if not loc:
                print(f"  [skip] No {UDM2_ASSET_KEY} for {item_id}")
                continue

            out_tif = DOWNLOAD_DIR / f"{item_id}_udm2.tif"
            if not out_tif.exists():
                try:
                    download_asset(loc, api_key, out_tif)
                except Exception as e:
                    print(f"  [warn] download failed for {item_id}: {e}")
                    continue

            roi_cloud = compute_roi_cloud_percent(out_tif, roi_geom)
            if roi_cloud is None:
                # No overlap or no valid pixels; keep but flag
                per_scene_rows.append({
                    "item_id": item_id,
                    "acquired": feat["properties"].get("acquired"),
                    "roi_cloud_percent": None,
                    "note": "no valid pixels in ROI (or failed parse)"
                })
                continue

            per_scene_rows.append({
                "item_id": item_id,
                "acquired": feat["properties"].get("acquired"),
                "roi_cloud_percent": round(roi_cloud, 3),
                "note": ""
            })

            if roi_cloud < 5.0:
                lt5 += 1
            if roi_cloud == 0.0:
                eq0 += 1

        # Aggregate for the month
        month_stat = {
            "Month": month_name(m),
            "Year": year,
            "Total_Images": total_count,
            "Images_Under_5_Percent_Cloud_ROI": lt5,
            "Images_Zero_Cloud_ROI": eq0,
            "Percent_Under_5_Cloud_ROI": round((lt5 / total_count * 100) if total_count else 0, 1),
            "Percent_Zero_Cloud_ROI": round((eq0 / total_count * 100) if total_count else 0, 1),
        }
        monthly_stats.append(month_stat)

        print(f"  Total scenes (intersecting search): {total_count}")
        print(f"  ROI < 5% cloud: {lt5} ({month_stat['Percent_Under_5_Cloud_ROI']}%)")
        print(f"  ROI = 0% cloud: {eq0} ({month_stat['Percent_Zero_Cloud_ROI']}%)")

    per_scene_df = pd.DataFrame(per_scene_rows)
    return monthly_stats, per_scene_df

def create_visualization(statistics: List[Dict[str, Any]], output_path: str) -> None:
    df = pd.DataFrame(statistics)
    # Simple two-panel plot (no seaborn; keep lightweight)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('PlanetScope Availability (ROI-based cloud % via UDM2)', fontsize=14, fontweight='bold')

    ax1.bar(df['Month'], df['Total_Images'])
    ax1.set_title('Total Scenes by Month')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)

    x = range(len(df))
    ax2.plot(df['Month'], df['Percent_Under_5_Cloud_ROI'], marker='o', label='< 5% ROI cloud')
    ax2.plot(df['Month'], df['Percent_Zero_Cloud_ROI'], marker='s', label='== 0% ROI cloud')
    ax2.set_title('ROI-based Usable Percentages')
    ax2.set_ylabel('Percent (%)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()

    plt.tight_layout()
    png_path = f"{output_path}_roi_udm2.png"
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    print(f"✓ Visualization saved: {png_path}")
    plt.show()

def main() -> None:
    print("PlanetScope ROI Cloud Coverage via UDM2")
    print("=" * 50)
    cfg = load_yaml()
    api_key = get_api_key()

    # Determine year
    year = None
    if "date_range" in cfg:
        dr = cfg["date_range"]
        if "years" in dr:
            year = dr["years"]["start_year"]
        elif "start" in dr:
            year = int(dr["start"][:4])
    if not year:
        year = datetime.utcnow().year
        print(f"No year specified in config, using current year: {year}")

    print(f"Item types: {cfg['item_types']}")
    print(f"Asset used for ROI clouds: {UDM2_ASSET_KEY}")
    print("Computing ROI cloud percentages from UDM2 (clear vs cloudy classes are configurable).")

    monthly_stats, per_scene_df = get_monthly_statistics_roi(cfg, api_key, year)

    # Save per-scene audit
    audit_csv = f"psscene_roi_udm2_{year}.csv"
    per_scene_df.to_csv(audit_csv, index=False)
    print(f"✓ Per-scene ROI cloud stats saved to: {audit_csv}")

    # Summary viz
    output_base = f"planetscope_roi_udm2_{year}"
    create_visualization(monthly_stats, output_base)

    print("\nDone.")

if __name__ == "__main__":
    main()
