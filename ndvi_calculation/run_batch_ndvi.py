#!/usr/bin/env python3
"""
Simple script to run batch NDVI processing on PlanetScope data.
"""

from batch_ndvi_processor import process_planetscope_directory
from pathlib import Path

# Define paths
script_dir = Path(__file__).parent
data_dir = script_dir.parent / "data_downlaoding" / "PlanetScope" / "Ellis_planet_downloads" / "finalized"
output_dir = script_dir / "Ellis_field_total_ndvi_outputs"

print("Starting batch NDVI processing...")
print(f"Input directory: {data_dir}")
print(f"Output directory: {output_dir}")

# Process all TIFF files
process_planetscope_directory(str(data_dir), str(output_dir))
