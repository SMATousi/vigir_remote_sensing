#!/usr/bin/env python
"""
PlanetScope Asset Availability Statistics

Analyzes PlanetScope asset availability by month for January through September.
Reports three key statistics:
1. Total number of images available
2. Number of images after < 5% cloud coverage restriction
3. Number of images with absolute NO cloud coverage (0%)

Uses the same config.yaml file as the download script for region and asset type settings.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple
import json
import pathlib

import requests
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


API_ROOT = "https://api.planet.com/data/v1"
SEARCH_URL = f"{API_ROOT}/quick-search"


def load_yaml(path: str | pathlib.Path = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_api_key() -> str:
    """Get Planet API key from environment variable."""
    api_key = os.getenv("PLANET_API_KEY")
    if not api_key:
        sys.exit("Error: environment variable PLANET_API_KEY is not set.")
    return api_key


def read_aoi(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a GeoJSON geometry dict from file or inline config."""
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
            
            if len(features) > 1:
                print(f"Warning: FeatureCollection contains {len(features)} features. Using the first feature only.")
            
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


def build_filter_for_month(cfg: Dict[str, Any], year: int, month: int, max_cloud_cover: float = None) -> Dict[str, Any]:
    """Build filter for a specific month and year with optional cloud cover limit."""
    geom_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": read_aoi(cfg),
    }
    
    # Create date range for the entire month
    start_date = f"{year:04d}-{month:02d}-01T00:00:00Z"
    if month == 12:
        end_date = f"{year+1:04d}-01-01T00:00:00Z"
    else:
        end_date = f"{year:04d}-{month+1:02d}-01T00:00:00Z"
    
    date_filter = {
        "type": "DateRangeFilter",
        "field_name": "acquired",
        "config": {
            "gte": start_date,
            "lt": end_date,
        },
    }
    
    filters = [geom_filter, date_filter]
    
    # Add cloud cover filter if specified
    if max_cloud_cover is not None:
        cloud_filter = {
            "type": "RangeFilter",
            "field_name": "cloud_cover",
            "config": {"lte": max_cloud_cover},
        }
        filters.append(cloud_filter)
    
    return {"type": "AndFilter", "config": filters}


def search_items_for_month(cfg: Dict[str, Any], api_key: str, year: int, month: int, max_cloud_cover: float = None) -> List[Dict[str, Any]]:
    """Search for items within a specific month with optional cloud cover limit."""
    payload = {
        "item_types": cfg["item_types"], 
        "filter": build_filter_for_month(cfg, year, month, max_cloud_cover)
    }
    
    all_items = []
    next_url = SEARCH_URL
    
    while next_url:
        if next_url == SEARCH_URL:
            # First request
            r = requests.post(next_url, auth=(api_key, ""), json=payload)
        else:
            # Subsequent requests (pagination)
            r = requests.get(next_url, auth=(api_key, ""))
        
        r.raise_for_status()
        data = r.json()
        
        features = data.get("features", [])
        all_items.extend(features)
        
        # Check for next page
        links = data.get("_links", {})
        next_url = links.get("_next")
    
    return all_items


def get_monthly_statistics(cfg: Dict[str, Any], api_key: str, year: int) -> List[Dict[str, Any]]:
    """Get statistics for each month from January to September."""
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September"]
    
    statistics = []
    
    for month_num, month_name in enumerate(months, 1):
        print(f"\nAnalyzing {month_name} {year}...")
        
        # Get total number of images (no cloud cover filter)
        total_items = search_items_for_month(cfg, api_key, year, month_num)
        total_count = len(total_items)
        
        # Get images with < 5% cloud coverage
        low_cloud_items = search_items_for_month(cfg, api_key, year, month_num, max_cloud_cover=0.05)
        low_cloud_count = len(low_cloud_items)
        
        # Get images with 0% cloud coverage
        no_cloud_items = search_items_for_month(cfg, api_key, year, month_num, max_cloud_cover=0.0)
        no_cloud_count = len(no_cloud_items)
        
        month_stats = {
            "Month": month_name,
            "Year": year,
            "Total_Images": total_count,
            "Images_Under_5_Percent_Cloud": low_cloud_count,
            "Images_Zero_Cloud": no_cloud_count,
            "Percent_Under_5_Cloud": round((low_cloud_count / total_count * 100) if total_count > 0 else 0, 1),
            "Percent_Zero_Cloud": round((no_cloud_count / total_count * 100) if total_count > 0 else 0, 1)
        }
        
        statistics.append(month_stats)
        
        print(f"  Total images: {total_count}")
        print(f"  Images < 5% cloud: {low_cloud_count} ({month_stats['Percent_Under_5_Cloud']}%)")
        print(f"  Images 0% cloud: {no_cloud_count} ({month_stats['Percent_Zero_Cloud']}%)")
    
    return statistics


def create_visualization(statistics: List[Dict[str, Any]], output_path: str) -> None:
    """Create visualizations of the asset availability statistics."""
    df = pd.DataFrame(statistics)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PlanetScope Asset Availability Statistics', fontsize=16, fontweight='bold')
    
    year = statistics[0]["Year"] if statistics else "Unknown"
    
    # Plot 1: Total Images by Month (Bar Chart)
    ax1.bar(df['Month'], df['Total_Images'], color='skyblue', alpha=0.8)
    ax1.set_title('Total Images Available by Month', fontweight='bold')
    ax1.set_ylabel('Number of Images')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(df['Total_Images']):
        ax1.text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Cloud Coverage Comparison (Grouped Bar Chart)
    x = range(len(df['Month']))
    width = 0.35
    
    ax2.bar([i - width/2 for i in x], df['Images_Under_5_Percent_Cloud'], 
            width, label='< 5% Cloud', color='lightgreen', alpha=0.8)
    ax2.bar([i + width/2 for i in x], df['Images_Zero_Cloud'], 
            width, label='0% Cloud', color='darkgreen', alpha=0.8)
    
    ax2.set_title('Images by Cloud Coverage Threshold', fontweight='bold')
    ax2.set_ylabel('Number of Images')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Month'], rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Percentage of Usable Images (Line Plot)
    ax3.plot(df['Month'], df['Percent_Under_5_Cloud'], 
             marker='o', linewidth=2, markersize=8, label='< 5% Cloud', color='orange')
    ax3.plot(df['Month'], df['Percent_Zero_Cloud'], 
             marker='s', linewidth=2, markersize=8, label='0% Cloud', color='red')
    
    ax3.set_title('Percentage of Usable Images by Month', fontweight='bold')
    ax3.set_ylabel('Percentage (%)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_ylim(0, max(df['Percent_Under_5_Cloud'].max(), df['Percent_Zero_Cloud'].max()) + 5)
    
    # Plot 4: Summary Statistics (Horizontal Bar Chart)
    total_images = df['Total_Images'].sum()
    total_low_cloud = df['Images_Under_5_Percent_Cloud'].sum()
    total_no_cloud = df['Images_Zero_Cloud'].sum()
    
    categories = ['Total Images\n(Jan-Sep)', 'Images < 5% Cloud\n(Jan-Sep)', 'Images 0% Cloud\n(Jan-Sep)']
    values = [total_images, total_low_cloud, total_no_cloud]
    colors = ['lightblue', 'lightgreen', 'darkgreen']
    
    bars = ax4.barh(categories, values, color=colors, alpha=0.8)
    ax4.set_title('Summary Statistics', fontweight='bold')
    ax4.set_xlabel('Number of Images')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        width = bar.get_width()
        percentage = round(value/total_images*100, 1) if i > 0 else 100
        ax4.text(width + total_images*0.01, bar.get_y() + bar.get_height()/2, 
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
    print(f"Images < 5% cloud: {total_low_cloud} ({round(total_low_cloud/total_images*100, 1)}%)")
    print(f"Images 0% cloud: {total_no_cloud} ({round(total_no_cloud/total_images*100, 1)}%)")
    
    if statistics:
        best_month_total = max(statistics, key=lambda x: x["Total_Images"])
        worst_month_total = min(statistics, key=lambda x: x["Total_Images"])
        best_month_clear = max(statistics, key=lambda x: x["Images_Zero_Cloud"])
        
        print(f"\nBest month for total images: {best_month_total['Month']} ({best_month_total['Total_Images']} images)")
        print(f"Worst month for total images: {worst_month_total['Month']} ({worst_month_total['Total_Images']} images)")
        print(f"Best month for clear images: {best_month_clear['Month']} ({best_month_clear['Images_Zero_Cloud']} clear images)")
    print(f"="*60)


def main() -> None:
    """Main function to run the asset availability analysis."""
    print("PlanetScope Asset Availability Statistics")
    print("=" * 50)
    
    # Load configuration
    cfg = load_yaml()
    api_key = get_api_key()
    
    # Determine year to analyze
    # Try to get year from config, otherwise use current year
    year = None
    if "date_range" in cfg:
        date_range_cfg = cfg["date_range"]
        if "years" in date_range_cfg:
            year = date_range_cfg["years"]["start_year"]
        elif "start" in date_range_cfg:
            year = int(date_range_cfg["start"][:4])
    
    if not year:
        year = datetime.now().year
        print(f"No year specified in config, using current year: {year}")
    
    print(f"Analyzing asset availability for {year}")
    print(f"Region: {cfg['aoi'].get('geojson_file', 'Inline geometry')}")
    print(f"Item types: {cfg['item_types']}")
    print(f"Asset types: {cfg['asset_types']}")
    
    # Get statistics
    statistics = get_monthly_statistics(cfg, api_key, year)
    
    # Create visualization
    output_base = f"planetscope_availability_{year}"
    create_visualization(statistics, output_base)
    
    print(f"\n✓ Analysis complete! Check the generated visualization files.")


if __name__ == "__main__":
    main()
