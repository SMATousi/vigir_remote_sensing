#!/usr/bin/env python3
"""
Example script showing how to use the batch folder processor programmatically.

This demonstrates processing a folder structure like:
/data/satellite_data/
├── field_1/
│   ├── image1.tif
│   ├── image2.tif
│   └── ...
├── field_2/
│   ├── image1.tif
│   ├── image2.tif
│   └── ...
└── field_3/
    ├── image1.tif
    ├── image2.tif
    └── ...
"""

from vegetation_indices_calculator import VegetationIndicesCalculator
from batch_folder_processor import process_folder_of_folders


def example_usage():
    """Example of how to use the batch folder processor programmatically."""
    
    # Define paths
    parent_folder = "/path/to/your/satellite_data"  # Change this to your actual path
    output_folder = "/path/to/your/output"          # Change this to your desired output path
    
    # Initialize the calculator with your band configuration
    # These are the default PlanetScope band indices
    calculator = VegetationIndicesCalculator(
        green_band=4,    # Green band
        red_band=6,      # Red band  
        nir_band=8,      # NIR band
        red_edge_band=7, # Red edge band
        blue_band=2      # Blue band
    )
    
    # Define which indices to calculate
    indices_to_calculate = ['NDVI', 'EXG', 'NDWI']  # Or use 'ALL' for all indices
    
    try:
        print("Starting batch processing of multiple folders...")
        
        # Process all subfolders
        process_folder_of_folders(
            parent_folder=parent_folder,
            output_base_dir=output_folder,
            indices=indices_to_calculate,
            calculator=calculator,
            normalization_method='min_max',  # Options: 'min_max', 'percentile', 'z_score'
            save_raw=True  # Set to False if you only want normalized outputs
        )
        
        print("Batch processing completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")


def example_with_all_indices():
    """Example processing all available vegetation indices."""
    
    parent_folder = "/path/to/your/satellite_data"
    output_folder = "/path/to/your/output_all_indices"
    
    calculator = VegetationIndicesCalculator()
    
    # Calculate all available indices
    all_indices = ['NDWI', 'NDVI', 'TDVI', 'NDRE', 'NGRDI', 'ClGreen', 'ClRedEdge', 'GNDVI', 'EXG']
    
    try:
        process_folder_of_folders(
            parent_folder=parent_folder,
            output_base_dir=output_folder,
            indices=all_indices,
            calculator=calculator,
            normalization_method='percentile',  # Use percentile normalization
            save_raw=True
        )
        
        print("All indices calculated successfully!")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Batch Folder Processor Examples")
    print("=" * 50)
    print("\n1. Basic usage example:")
    print("   Uncomment the line below and set your actual paths")
    # example_usage()
    
    print("\n2. All indices example:")
    print("   Uncomment the line below and set your actual paths")  
    # example_with_all_indices()
    
    print("\nTo use these examples:")
    print("1. Edit the folder paths in the functions above")
    print("2. Uncomment the function calls")
    print("3. Run this script")
    
    print("\nOr use the command line interface:")
    print("python batch_folder_processor.py /your/parent/folder --indices NDVI EXG --save-raw")
