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
import seaborn as sns

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
        for _ in range(300):  # up to ~5 min (60 * 5s)
            time.sleep(1)
            r2 = requests.get(assets_url, auth=(api_key, ""))
            r2.raise_for_status()
            assets2 = r2.json()
            if assets2[asset_key]["status"] == "active":
                return assets2[asset_key]["location"]
        return None
    else:
        return asset_info["location"]

def batch_activate_assets(api_key: str, items: List[Dict[str, Any]], asset_key: str) -> Dict[str, Optional[str]]:
    """
    Batch activate assets for all items and return a mapping of item_id -> download_url.
    This function requests activation for all assets first, then polls for completion.
    """
    print(f"\nBatch activating {asset_key} assets for {len(items)} items...")
    
    # Step 1: Collect asset info and request activation for inactive assets
    activation_requests = []
    already_active = {}
    
    for feat in items:
        item_type = feat["properties"]["item_type"] if "item_type" in feat["properties"] else feat["id"].split("_")[0]
        item_id = feat["id"]
        
        assets_url = f"{API_ROOT}/item-types/{item_type}/items/{item_id}/assets"
        try:
            r = requests.get(assets_url, auth=(api_key, ""))
            r.raise_for_status()
            assets = r.json()
            
            if asset_key not in assets:
                print(f"  [skip] No {asset_key} asset for {item_id}")
                continue
                
            asset_info = assets[asset_key]
            if asset_info["status"] == "active":
                already_active[item_id] = asset_info["location"]
                print(f"  [active] {item_id} already active")
            else:
                # Request activation
                activate_url = asset_info["_links"]["activate"]
                ra = requests.post(activate_url, auth=(api_key, ""))
                if ra.status_code in (200, 202):
                    activation_requests.append((item_id, item_type, assets_url))
                    print(f"  [requested] Activation requested for {item_id}")
                else:
                    print(f"  [failed] Activation request failed for {item_id}: {ra.status_code}")
                    
        except Exception as e:
            print(f"  [error] Failed to process {item_id}: {e}")
            continue
    
    print(f"\nActivation requested for {len(activation_requests)} assets")
    print(f"Already active: {len(already_active)} assets")
    
    # Step 2: Poll for activation completion
    result_urls = already_active.copy()
    pending_activations = activation_requests.copy()
    
    max_polls = 0  # up to ~5 min total
    poll_count = 0
    
    while pending_activations and poll_count < max_polls:
        poll_count += 1
        if poll_count % 30 == 0:  # Progress update every 30 seconds
            print(f"  Polling... {len(pending_activations)} assets still pending (attempt {poll_count}/{max_polls})")
        
        time.sleep(1)
        completed_this_round = []
        
        for item_id, item_type, assets_url in pending_activations:
            try:
                r = requests.get(assets_url, auth=(api_key, ""))
                r.raise_for_status()
                assets = r.json()
                
                if assets[asset_key]["status"] == "active":
                    result_urls[item_id] = assets[asset_key]["location"]
                    completed_this_round.append((item_id, item_type, assets_url))
                    print(f"  [ready] {item_id} activation complete")
                    
            except Exception as e:
                print(f"  [error] Polling failed for {item_id}: {e}")
                completed_this_round.append((item_id, item_type, assets_url))  # Remove from pending
        
        # Remove completed items from pending list
        for completed_item in completed_this_round:
            pending_activations.remove(completed_item)
    
    if pending_activations:
        print(f"\n[warning] {len(pending_activations)} assets did not activate within timeout:")
        for item_id, _, _ in pending_activations:
            print(f"  - {item_id}")
    
    print(f"\nBatch activation complete: {len(result_urls)} assets ready for download")
    return result_urls

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
        total_count = len(items)
        
        if total_count == 0:
            print(f"  No items found for {month_name(m)} {year}")
            month_stat = {
                "Month": month_name(m),
                "Year": year,
                "Total_Images": 0,
                "Images_Under_5_Percent_Cloud_ROI": 0,
                "Images_Zero_Cloud_ROI": 0,
                "Percent_Under_5_Cloud_ROI": 0,
                "Percent_Zero_Cloud_ROI": 0,
            }
            monthly_stats.append(month_stat)
            continue
        
        print(f"  Found {total_count} items, batch activating {UDM2_ASSET_KEY} assets...")
        
        # Batch activate all assets for this month
        asset_urls = batch_activate_assets(api_key, items, UDM2_ASSET_KEY)
        
        print(f"\nProcessing {len(asset_urls)} activated assets...")
        
        lt5 = 0
        eq0 = 0
        processed_count = 0

        for feat in items:
            item_id = feat["id"]
            
            # Check if asset was successfully activated
            if item_id not in asset_urls or asset_urls[item_id] is None:
                print(f"  [skip] No activated {UDM2_ASSET_KEY} for {item_id}")
                per_scene_rows.append({
                    "item_id": item_id,
                    "acquired": feat["properties"].get("acquired"),
                    "roi_cloud_percent": None,
                    "note": "asset activation failed"
                })
                continue

            # Download the asset
            out_tif = DOWNLOAD_DIR / f"{item_id}_udm2.tif"
            if not out_tif.exists():
                try:
                    download_asset(asset_urls[item_id], api_key, out_tif)
                    print(f"  [downloaded] {item_id}")
                except Exception as e:
                    print(f"  [warn] download failed for {item_id}: {e}")
                    per_scene_rows.append({
                        "item_id": item_id,
                        "acquired": feat["properties"].get("acquired"),
                        "roi_cloud_percent": None,
                        "note": f"download failed: {e}"
                    })
                    continue
            else:
                print(f"  [cached] {item_id} (already downloaded)")

            # Compute ROI cloud percentage
            roi_cloud = compute_roi_cloud_percent(out_tif, roi_geom)
            processed_count += 1
            
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

        print(f"\n  Month Summary for {month_name(m)} {year}:")
        print(f"  Total scenes found: {total_count}")
        print(f"  Successfully processed: {processed_count}")
        print(f"  ROI < 5% cloud: {lt5} ({month_stat['Percent_Under_5_Cloud_ROI']}%)")
        print(f"  ROI = 0% cloud: {eq0} ({month_stat['Percent_Zero_Cloud_ROI']}%)")

    per_scene_df = pd.DataFrame(per_scene_rows)
    return monthly_stats, per_scene_df

# def create_visualization(statistics: List[Dict[str, Any]], output_path: str) -> None:
#     df = pd.DataFrame(statistics)
#     # Simple two-panel plot (no seaborn; keep lightweight)
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
#     fig.suptitle('PlanetScope Availability (ROI-based cloud % via UDM2)', fontsize=14, fontweight='bold')

#     ax1.bar(df['Month'], df['Total_Images'])
#     ax1.set_title('Total Scenes by Month')
#     ax1.set_ylabel('Count')
#     ax1.tick_params(axis='x', rotation=45)

#     x = range(len(df))
#     ax2.plot(df['Month'], df['Percent_Under_5_Cloud_ROI'], marker='o', label='< 5% ROI cloud')
#     ax2.plot(df['Month'], df['Percent_Zero_Cloud_ROI'], marker='s', label='== 0% ROI cloud')
#     ax2.set_title('ROI-based Usable Percentages')
#     ax2.set_ylabel('Percent (%)')
#     ax2.tick_params(axis='x', rotation=45)
#     ax2.legend()

#     plt.tight_layout()
#     png_path = f"{output_path}_roi_udm2.png"
#     plt.savefig(png_path, dpi=200, bbox_inches='tight')
#     print(f"✓ Visualization saved: {png_path}")
#     plt.show()


def create_visualization(statistics: List[Dict[str, Any]], output_path: str) -> None:
    """Create visualizations of the asset availability statistics."""
    if not statistics:
        print("[warning] No statistics data available for visualization")
        return
        
    df = pd.DataFrame(statistics)
    
    # Check if required columns exist
    required_columns = ['Month', 'Total_Images', 'Images_Under_5_Percent_Cloud_ROI', 'Images_Zero_Cloud_ROI', 
                       'Percent_Under_5_Cloud_ROI', 'Percent_Zero_Cloud_ROI']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"[error] Missing required columns in data: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PlanetScope Asset Availability Statistics (ROI-based)', fontsize=16, fontweight='bold')
    
    year = statistics[0]["Year"] if statistics else "Unknown"
    
    # Plot 1: Total Images by Month (Bar Chart)
    ax1.bar(df['Month'], df['Total_Images'], color='skyblue', alpha=0.8)
    ax1.set_title('Total Images Available by Month', fontweight='bold')
    ax1.set_ylabel('Number of Images')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(df['Total_Images']):
        if v > 0:  # Only add label if there are images
            ax1.text(i, v + max(df['Total_Images']) * 0.01, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Cloud Coverage Comparison (Grouped Bar Chart)
    x = range(len(df['Month']))
    width = 0.35
    
    ax2.bar([i - width/2 for i in x], df['Images_Under_5_Percent_Cloud_ROI'], 
            width, label='< 5% Cloud (ROI)', color='lightgreen', alpha=0.8)
    ax2.bar([i + width/2 for i in x], df['Images_Zero_Cloud_ROI'], 
            width, label='0% Cloud (ROI)', color='darkgreen', alpha=0.8)
    
    ax2.set_title('Images by Cloud Coverage Threshold (ROI-based)', fontweight='bold')
    ax2.set_ylabel('Number of Images')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Month'], rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Percentage of Usable Images (Line Plot)
    ax3.plot(df['Month'], df['Percent_Under_5_Cloud_ROI'], 
             marker='o', linewidth=2, markersize=8, label='< 5% Cloud (ROI)', color='orange')
    ax3.plot(df['Month'], df['Percent_Zero_Cloud_ROI'], 
             marker='s', linewidth=2, markersize=8, label='0% Cloud (ROI)', color='red')
    
    ax3.set_title('Percentage of Usable Images by Month (ROI-based)', fontweight='bold')
    ax3.set_ylabel('Percentage (%)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Set y-axis limits with safety check
    max_percent = max(df['Percent_Under_5_Cloud_ROI'].max(), df['Percent_Zero_Cloud_ROI'].max())
    if max_percent > 0:
        ax3.set_ylim(0, max_percent + 5)
    else:
        ax3.set_ylim(0, 10)
    
    # Plot 4: Summary Statistics (Horizontal Bar Chart)
    total_images = df['Total_Images'].sum()
    total_low_cloud = df['Images_Under_5_Percent_Cloud_ROI'].sum()
    total_no_cloud = df['Images_Zero_Cloud_ROI'].sum()
    
    categories = ['Total Images\n(Jan-Sep)', 'Images < 5% Cloud\n(Jan-Sep)', 'Images 0% Cloud\n(Jan-Sep)']
    values = [total_images, total_low_cloud, total_no_cloud]
    colors = ['lightblue', 'lightgreen', 'darkgreen']
    
    bars = ax4.barh(categories, values, color=colors, alpha=0.8)
    ax4.set_title('Summary Statistics (ROI-based)', fontweight='bold')
    ax4.set_xlabel('Number of Images')
    
    # Add value labels on bars with safety check
    for i, (bar, value) in enumerate(zip(bars, values)):
        width = bar.get_width()
        if total_images > 0:
            percentage = round(value/total_images*100, 1) if i > 0 else 100
        else:
            percentage = 0
        
        label_x = width + max(values) * 0.01 if max(values) > 0 else 1
        ax4.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{value}\n({percentage}%)', ha='left', va='center', fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the plot
    plot_path = f"{output_path}_visualization.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {plot_path}")
    
    # Also save as PDF for better quality
    pdf_path = f"{output_path}_visualization.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ High-quality PDF saved to: {pdf_path}")
    
    # Show the plot
    plt.show()
    
    # Print summary to console
    print(f"\n" + "="*60)
    print(f"SUMMARY STATISTICS FOR {year}")
    print(f"="*60)
    print(f"Total images (Jan-Sep): {total_images}")
    if total_images > 0:
        print(f"Images < 5% cloud: {total_low_cloud} ({round(total_low_cloud/total_images*100, 1)}%)")
        print(f"Images 0% cloud: {total_no_cloud} ({round(total_no_cloud/total_images*100, 1)}%)")
    else:
        print(f"Images < 5% cloud: {total_low_cloud} (0.0%)")
        print(f"Images 0% cloud: {total_no_cloud} (0.0%)")
    
    if statistics:
        best_month_total = max(statistics, key=lambda x: x["Total_Images"])
        worst_month_total = min(statistics, key=lambda x: x["Total_Images"])
        best_month_clear = max(statistics, key=lambda x: x["Images_Zero_Cloud_ROI"])
        
        print(f"\nBest month for total images: {best_month_total['Month']} ({best_month_total['Total_Images']} images)")
        print(f"Worst month for total images: {worst_month_total['Month']} ({worst_month_total['Total_Images']} images)")
        print(f"Best month for clear images: {best_month_clear['Month']} ({best_month_clear['Images_Zero_Cloud_ROI']} clear images)")
    print(f"="*60)

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
