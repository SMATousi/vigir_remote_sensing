#!/usr/bin/env python3
"""
Example script demonstrating the use of VegetationIndicesCalculator.

This script shows how to:
1. Calculate multiple vegetation indices from a single TIFF file
2. Batch process multiple TIFF files in a folder
3. Visualize the results
4. Save outputs with proper naming conventions

Author: Vegetation Indices Calculator
Date: 2024
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path to import our calculator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vegetation_indices_calculator import VegetationIndicesCalculator, batch_process_folder


def example_single_file_processing():
    """Example of processing a single TIFF file with multiple indices."""
    print("="*60)
    print("EXAMPLE 1: Single File Processing")
    print("="*60)
    
    # Initialize the calculator with band configuration
    # Adjust these band indices based on your TIFF file structure
    calculator = VegetationIndicesCalculator(
        green_band=1,    # Green band (1-indexed)
        red_band=2,      # Red band (1-indexed)
        nir_band=3,      # NIR band (1-indexed)
        red_edge_band=4, # Red edge band (1-indexed)
        blue_band=5      # Blue band (1-indexed)
    )
    
    # Example input file (replace with your actual file path)
    input_file = "example_multispectral.tif"
    
    # Check if file exists (for demonstration)
    if not os.path.exists(input_file):
        print(f"Note: Example file '{input_file}' not found.")
        print("Replace 'input_file' variable with your actual TIFF file path.")
        return
    
    # Define which indices to calculate
    indices_to_calculate = [
        'NDVI',      # Normalized Difference Vegetation Index
        'NDWI',      # Normalized Difference Water Index
        'GNDVI',     # Green NDVI
        'NGRDI',     # Normalized Green-Red Difference Index
        'ClGreen',   # Green Chlorophyll Index
        'EXG'        # Excess Green Index
    ]
    
    try:
        # Process the file
        results = calculator.process_tiff(
            input_path=input_file,
            indices=indices_to_calculate,
            output_dir="single_file_outputs",
            normalization_method='min_max',
            save_raw=True  # Save both raw and normalized values
        )
        
        # Visualize results
        calculator.visualize_indices(
            results=results,
            save_plot="single_file_visualization.png"
        )
        
        print(f"\nProcessed {len(indices_to_calculate)} indices successfully!")
        print("Outputs saved to: single_file_outputs/")
        
    except Exception as e:
        print(f"Error processing file: {e}")


def example_batch_processing():
    """Example of batch processing multiple TIFF files in a folder."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Processing")
    print("="*60)
    
    # Initialize the calculator
    calculator = VegetationIndicesCalculator(
        green_band=1,
        red_band=2,
        nir_band=3,
        red_edge_band=4,
        blue_band=5
    )
    
    # Example input folder (replace with your actual folder path)
    input_folder = "multispectral_images"
    output_base_dir = "batch_processing_outputs"
    
    # Check if folder exists (for demonstration)
    if not os.path.exists(input_folder):
        print(f"Note: Example folder '{input_folder}' not found.")
        print("Replace 'input_folder' variable with your actual folder path.")
        return
    
    # Define indices to calculate for all files
    indices_to_calculate = [
        'NDVI',      # Essential for vegetation analysis
        'NDWI',      # Water content analysis
        'NDRE',      # Red edge analysis (if red edge band available)
        'GNDVI',     # Alternative vegetation index
        'ClGreen',   # Chlorophyll content
        'EXG'        # Excess green for early growth detection
    ]
    
    try:
        # Batch process all TIFF files in the folder
        batch_process_folder(
            input_folder=input_folder,
            output_base_dir=output_base_dir,
            indices=indices_to_calculate,
            calculator=calculator,
            normalization_method='percentile',  # Use percentile normalization
            save_raw=False  # Save only normalized values to save space
        )
        
        print(f"\nBatch processing completed!")
        print(f"Results organized by source folder structure in: {output_base_dir}/")
        
    except Exception as e:
        print(f"Error in batch processing: {e}")


def example_custom_band_configuration():
    """Example showing how to handle different band configurations."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Band Configuration")
    print("="*60)
    
    # Example for different satellite data with different band arrangements
    
    # Configuration 1: Landsat-like (bands 2,3,4,5 for Blue,Green,Red,NIR)
    landsat_calculator = VegetationIndicesCalculator(
        green_band=2,    # Green
        red_band=3,      # Red
        nir_band=4,      # NIR
        red_edge_band=5  # If available
    )
    
    # Configuration 2: Sentinel-2 like (bands 3,4,8,5 for Green,Red,NIR,RedEdge)
    sentinel_calculator = VegetationIndicesCalculator(
        green_band=3,    # Green (B3)
        red_band=4,      # Red (B4)
        nir_band=8,      # NIR (B8)
        red_edge_band=5  # Red Edge (B5)
    )
    
    print("Landsat-like configuration:")
    print(f"  Green: Band {landsat_calculator.band_config['green']}")
    print(f"  Red: Band {landsat_calculator.band_config['red']}")
    print(f"  NIR: Band {landsat_calculator.band_config['nir']}")
    print(f"  Red Edge: Band {landsat_calculator.band_config['red_edge']}")
    
    print("\nSentinel-2 like configuration:")
    print(f"  Green: Band {sentinel_calculator.band_config['green']}")
    print(f"  Red: Band {sentinel_calculator.band_config['red']}")
    print(f"  NIR: Band {sentinel_calculator.band_config['nir']}")
    print(f"  Red Edge: Band {sentinel_calculator.band_config['red_edge']}")
    
    print("\nNote: Adjust band indices based on your specific satellite data!")


def example_index_selection_guide():
    """Guide for selecting appropriate indices for different applications."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Index Selection Guide")
    print("="*60)
    
    applications = {
        "General Vegetation Health": ['NDVI', 'GNDVI'],
        "Crop Monitoring": ['NDVI', 'NDRE', 'ClGreen'],
        "Water Stress Detection": ['NDWI', 'NDVI'],
        "Chlorophyll Content": ['ClGreen', 'ClRedEdge'],
        "Early Growth Detection": ['GNDVI', 'NGRDI'],
        "Precision Agriculture": ['NDVI', 'NDRE', 'ClGreen', 'GNDVI'],
        "Forest Monitoring": ['NDVI', 'NDWI', 'ClGreen'],
        "Drought Assessment": ['NDWI', 'NDVI', 'TDVI']
    }
    
    print("Recommended indices for different applications:\n")
    for application, indices in applications.items():
        print(f"{application}:")
        for index in indices:
            print(f"  - {index}")
        print()
    
    print("Index Descriptions:")
    descriptions = {
        'NDVI': 'Standard vegetation index, good for overall vegetation health',
        'NDWI': 'Water content indicator, useful for irrigation and drought monitoring',
        'GNDVI': 'Green-based vegetation index, sensitive to chlorophyll content',
        'NDRE': 'Red edge index, sensitive to chlorophyll and nitrogen content',
        'NGRDI': 'Green-red difference, good for early growth stages',
        'ClGreen': 'Chlorophyll index using green band',
        'ClRedEdge': 'Chlorophyll index using red edge band',
        'TDVI': 'Transformed NDVI, enhanced contrast for vegetation analysis'
    }
    
    for index, description in descriptions.items():
        print(f"{index}: {description}")


def main():
    """Run all examples."""
    print("Vegetation Indices Calculator - Examples")
    print("="*60)
    
    # Run examples
    example_single_file_processing()
    example_batch_processing()
    example_custom_band_configuration()
    example_index_selection_guide()
    
    print("\n" + "="*60)
    print("COMMAND LINE USAGE EXAMPLES")
    print("="*60)
    
    print("\n1. Calculate NDVI for a single file:")
    print("   python vegetation_indices_calculator.py input.tif --indices NDVI")
    
    print("\n2. Calculate multiple indices:")
    print("   python vegetation_indices_calculator.py input.tif --indices NDVI NDWI GNDVI")
    
    print("\n3. Batch process a folder:")
    print("   python vegetation_indices_calculator.py /path/to/folder --batch --indices NDVI NDRE")
    
    print("\n4. Custom band configuration:")
    print("   python vegetation_indices_calculator.py input.tif --green-band 2 --red-band 3 --nir-band 4")
    
    print("\n5. Save raw values and visualize:")
    print("   python vegetation_indices_calculator.py input.tif --save-raw --visualize")
    
    print("\n6. Use percentile normalization:")
    print("   python vegetation_indices_calculator.py input.tif --normalize percentile")
    
    print("\n7. Calculate ALL available indices:")
    print("   python vegetation_indices_calculator.py input.tif --indices ALL")
    
    print("\n8. Batch process with all indices:")
    print("   python vegetation_indices_calculator.py /path/to/folder --batch --indices ALL --save-raw")
    
    print("\n" + "="*60)
    print("Examples completed! Modify the file paths and run individual examples.")
    print("="*60)


if __name__ == "__main__":
    main()
