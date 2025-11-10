#!/usr/bin/env python3
"""
Simple batch processor for vegetation indices calculation across multiple folders.

This script processes a "folder of folders" structure where each subfolder contains
TIFF files that need vegetation indices calculated. It uses the existing 
vegetation_indices_calculator.py to process each folder.

Usage:
    python batch_folder_processor.py /path/to/parent/folder --indices NDVI EXG --output /path/to/output
    python batch_folder_processor.py /path/to/parent/folder --indices ALL --save-raw
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List

# Import the vegetation indices calculator
from vegetation_indices_calculator import VegetationIndicesCalculator, batch_process_folder


def process_folder_of_folders(parent_folder: str, output_base_dir: str, 
                            indices: List[str], calculator: VegetationIndicesCalculator,
                            normalization_method: str = 'min_max', save_raw: bool = False):
    """
    Process a folder containing multiple subfolders, each with TIFF files.
    
    Args:
        parent_folder: Path to parent folder containing subfolders
        output_base_dir: Base directory for all outputs
        indices: List of indices to calculate
        calculator: VegetationIndicesCalculator instance
        normalization_method: Normalization method to use
        save_raw: Whether to save raw index values
    """
    parent_path = Path(parent_folder)
    
    if not parent_path.exists():
        raise FileNotFoundError(f"Parent folder not found: {parent_folder}")
    
    if not parent_path.is_dir():
        raise ValueError(f"Path is not a directory: {parent_folder}")
    
    # Find all subdirectories
    subdirs = [d for d in parent_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        print(f"No subdirectories found in {parent_folder}")
        return
    
    print(f"Found {len(subdirs)} subdirectories to process:")
    for subdir in subdirs:
        print(f"  - {subdir.name}")
    
    # Process each subdirectory
    for i, subdir in enumerate(subdirs, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING FOLDER {i}/{len(subdirs)}: {subdir.name}")
        print(f"{'='*80}")
        
        try:
            # Create output directory for this subfolder
            output_dir = os.path.join(output_base_dir, subdir.name)
            
            # Process all TIFF files in this subfolder
            batch_process_folder(
                input_folder=str(subdir),
                output_base_dir=output_dir,
                indices=indices,
                calculator=calculator,
                normalization_method=normalization_method,
                save_raw=save_raw
            )
            
            print(f"Successfully completed processing folder: {subdir.name}")
            
        except Exception as e:
            print(f"Error processing folder {subdir.name}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("ALL FOLDERS PROCESSING COMPLETED!")
    print(f"Results saved to: {output_base_dir}")
    print(f"{'='*80}")


def main():
    """Command-line interface for batch folder processing."""
    parser = argparse.ArgumentParser(
        description='Process multiple folders containing TIFF files for vegetation indices calculation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subfolders with NDVI calculation
  python batch_folder_processor.py /data/satellite_images --indices NDVI
  
  # Process with multiple indices and save raw values
  python batch_folder_processor.py /data/satellite_images --indices NDVI EXG NDWI --save-raw
  
  # Process all available indices with custom output location
  python batch_folder_processor.py /data/satellite_images --indices ALL --output /results/vegetation_indices
        """
    )
    
    parser.add_argument('parent_folder', help='Parent folder containing subfolders with TIFF files')
    parser.add_argument('-o', '--output', help='Output directory path (default: parent_folder + "_vegetation_indices")')
    parser.add_argument('--indices', nargs='+', 
                       choices=['NDWI', 'NDVI', 'TDVI', 'NDRE', 'NGRDI', 'ClGreen', 'ClRedEdge', 'GNDVI', 'EXG', 'ALL'],
                       default=['NDVI'], help='Indices to calculate (default: NDVI). Use ALL to calculate all available indices.')
    parser.add_argument('--green-band', type=int, default=4, help='Green band index (1-indexed, default: 4)')
    parser.add_argument('--red-band', type=int, default=6, help='Red band index (1-indexed, default: 6)')
    parser.add_argument('--nir-band', type=int, default=8, help='NIR band index (1-indexed, default: 8)')
    parser.add_argument('--red-edge-band', type=int, default=7, help='Red edge band index (1-indexed, default: 7)')
    parser.add_argument('--blue-band', type=int, default=2, help='Blue band index (1-indexed, default: 2)')
    parser.add_argument('--normalize', choices=['min_max', 'percentile', 'z_score'], 
                       default='min_max', help='Normalization method (default: min_max)')
    parser.add_argument('--save-raw', action='store_true', help='Also save raw index values')
    
    args = parser.parse_args()
    
    # Handle 'ALL' option for indices
    if 'ALL' in args.indices:
        args.indices = ['NDWI', 'NDVI', 'TDVI', 'NDRE', 'NGRDI', 'ClGreen', 'ClRedEdge', 'GNDVI', 'EXG']
        print(f"Calculating all available indices: {', '.join(args.indices)}")
    
    # Set default output directory
    output_dir = args.output or f"{args.parent_folder}_vegetation_indices"
    
    # Initialize calculator
    calculator = VegetationIndicesCalculator(
        green_band=args.green_band,
        red_band=args.red_band,
        nir_band=args.nir_band,
        red_edge_band=args.red_edge_band,
        blue_band=args.blue_band
    )
    
    try:
        process_folder_of_folders(
            parent_folder=args.parent_folder,
            output_base_dir=output_dir,
            indices=args.indices,
            calculator=calculator,
            normalization_method=args.normalize,
            save_raw=args.save_raw
        )
        
        print("\nBatch folder processing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
