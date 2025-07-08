#!/usr/bin/env python3
"""
Batch NDVI processor for PlanetScope imagery.

This script processes all TIFF files in the PlanetScope data directory and 
calculates normalized NDVI maps for each file.
"""

import os
import sys
from pathlib import Path
from ndvi_calculator import NDVICalculator


def process_planetscope_directory(data_dir: str, output_dir: str = None):
    """
    Process all TIFF files in the PlanetScope directory and calculate normalized NDVI.
    
    Args:
        data_dir (str): Path to PlanetScope data directory
        output_dir (str): Output directory for NDVI results (optional)
    """
    # Convert to Path objects
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return
    
    # Set up output directory
    if output_dir is None:
        output_path = data_path.parent.parent / "ndvi_calculation" / "ndvi_outputs"
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize NDVI calculator with PlanetScope band configuration
    # PlanetScope 4-band: Blue(1), Green(2), Red(3), NIR(4)
    calculator = NDVICalculator(red_band_idx=3, nir_band_idx=4)
    
    # Find all TIFF files
    tiff_files = list(data_path.rglob("*_clip.tif")) + list(data_path.rglob("*_clip.tiff"))
    
    if not tiff_files:
        print(f"No TIFF files found in {data_dir}")
        return
    
    print(f"Found {len(tiff_files)} TIFF files to process")
    print(f"Output directory: {output_path}")
    print("=" * 60)
    
    successful = 0
    failed = 0
    
    for i, tiff_file in enumerate(tiff_files, 1):
        print(f"\n[{i}/{len(tiff_files)}] Processing: {tiff_file.name}")
        
        try:
            # Extract year from parent directory name (e.g., 20160804_232118_0d06 -> 2016)
            parent_folder = tiff_file.parent.name
            year = parent_folder[:4]  # Extract first 4 characters as year
            
            # Create output filename using the year
            output_filename = f"{year}_ndvi.tif"
            output_file = output_path / output_filename
            
            # Ensure output subdirectory exists
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            # Process the file
            raw_ndvi, normalized_ndvi, metadata = calculator.process_tiff(
                input_path=str(tiff_file),
                output_path=str(output_file),
                normalization_method='percentile',
                save_raw_ndvi=False
            )
            
            print(f"✓ Successfully processed: {output_filename}")
            successful += 1
            
        except Exception as e:
            print(f"✗ Error processing {tiff_file.name}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Processing complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed: {failed} files")
    print(f"Results saved to: {output_path}")


def main():
    """Command-line interface for batch NDVI processing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch process PlanetScope TIFF files to calculate normalized NDVI'
    )
    parser.add_argument(
        'data_dir', 
        help='Path to PlanetScope data directory'
    )
    parser.add_argument(
        '-o', '--output-dir', 
        help='Output directory for NDVI results (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    # Process the directory
    process_planetscope_directory(args.data_dir, args.output_dir)


if __name__ == "__main__":
    # Default behavior: process the PlanetScope directory relative to this script
    script_dir = Path(__file__).parent
    default_data_dir = script_dir.parent / "data_downlaoding" / "PlanetScope"
    
    if len(sys.argv) == 1:
        # No arguments provided, use default directory
        if default_data_dir.exists():
            print(f"No arguments provided. Processing default directory: {default_data_dir}")
            process_planetscope_directory(str(default_data_dir))
        else:
            print(f"Default directory {default_data_dir} not found.")
            print("Usage: python batch_ndvi_processor.py <data_directory>")
            sys.exit(1)
    else:
        # Use command-line interface
        main()
