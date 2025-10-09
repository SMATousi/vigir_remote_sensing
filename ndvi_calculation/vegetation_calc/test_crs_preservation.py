#!/usr/bin/env python3
"""
Test script to verify CRS preservation in vegetation indices calculator.

This script demonstrates that the output TIFF files maintain the same CRS
as the input file.
"""

import rasterio
import numpy as np
import tempfile
import os
from vegetation_indices_calculator import VegetationIndicesCalculator


def create_test_tiff_with_crs():
    """Create a test multispectral TIFF file with a specific CRS for testing."""
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
    temp_file.close()
    
    # Define test data dimensions
    height, width = 100, 100
    
    # Create synthetic multispectral data (5 bands: Green, Red, NIR, RedEdge, Blue)
    green_band = np.random.randint(50, 150, (height, width), dtype=np.uint16)
    red_band = np.random.randint(30, 120, (height, width), dtype=np.uint16)
    nir_band = np.random.randint(100, 200, (height, width), dtype=np.uint16)
    red_edge_band = np.random.randint(80, 180, (height, width), dtype=np.uint16)
    blue_band = np.random.randint(20, 100, (height, width), dtype=np.uint16)
    
    # Define a specific CRS (UTM Zone 33N)
    from rasterio.crs import CRS
    test_crs = CRS.from_epsg(32633)  # UTM Zone 33N
    
    # Define transform (geospatial coordinates)
    from rasterio.transform import from_bounds
    transform = from_bounds(500000, 4000000, 510000, 4010000, width, height)
    
    # Write the test file
    with rasterio.open(
        temp_file.name, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=5,
        dtype='uint16',
        crs=test_crs,
        transform=transform
    ) as dst:
        dst.write(green_band, 1)
        dst.write(red_band, 2)
        dst.write(nir_band, 3)
        dst.write(red_edge_band, 4)
        dst.write(blue_band, 5)
    
    return temp_file.name, test_crs


def test_crs_preservation():
    """Test that output files preserve the same CRS as input."""
    print("="*60)
    print("CRS PRESERVATION TEST")
    print("="*60)
    
    # Create test input file
    print("Creating test multispectral TIFF with UTM Zone 33N CRS...")
    input_file, original_crs = create_test_tiff_with_crs()
    print(f"Input file: {input_file}")
    print(f"Original CRS: {original_crs}")
    
    # Initialize calculator
    calculator = VegetationIndicesCalculator(
        green_band=1, red_band=2, nir_band=3, red_edge_band=4, blue_band=5
    )
    
    # Create temporary output directory
    output_dir = tempfile.mkdtemp()
    print(f"Output directory: {output_dir}")
    
    try:
        # Process with multiple indices
        print("\nCalculating vegetation indices...")
        results = calculator.process_tiff(
            input_path=input_file,
            indices=['NDVI', 'GNDVI', 'EXG'],
            output_dir=output_dir,
            save_raw=True
        )
        
        # Check CRS of output files
        print("\nVerifying CRS preservation in output files:")
        output_files = []
        for file in os.listdir(output_dir):
            if file.endswith('.tif'):
                output_files.append(os.path.join(output_dir, file))
        
        all_crs_match = True
        for output_file in output_files:
            with rasterio.open(output_file) as src:
                output_crs = src.crs
                crs_match = output_crs == original_crs
                all_crs_match = all_crs_match and crs_match
                
                print(f"  {os.path.basename(output_file)}:")
                print(f"    CRS: {output_crs}")
                print(f"    Matches input: {'✓' if crs_match else '✗'}")
        
        # Final result
        print("\n" + "="*60)
        if all_crs_match:
            print("✅ SUCCESS: All output files preserve the original CRS!")
        else:
            print("❌ FAILURE: Some output files have different CRS!")
        print("="*60)
        
        return all_crs_match
        
    finally:
        # Cleanup
        try:
            os.unlink(input_file)
            for file in output_files:
                os.unlink(file)
            os.rmdir(output_dir)
        except:
            pass


def test_crs_with_different_projections():
    """Test CRS preservation with different projection systems."""
    print("\n" + "="*60)
    print("TESTING DIFFERENT CRS PROJECTIONS")
    print("="*60)
    
    test_crs_list = [
        ("WGS84 Geographic", "EPSG:4326"),
        ("Web Mercator", "EPSG:3857"),
        ("UTM Zone 10N", "EPSG:32610"),
        ("State Plane California", "EPSG:2227")
    ]
    
    for crs_name, crs_code in test_crs_list:
        print(f"\nTesting {crs_name} ({crs_code})...")
        
        # This would require creating test files with different CRS
        # For now, just demonstrate the concept
        from rasterio.crs import CRS
        test_crs = CRS.from_string(crs_code)
        print(f"  CRS object: {test_crs}")
        print(f"  Is geographic: {test_crs.is_geographic}")
        print(f"  Is projected: {test_crs.is_projected}")


if __name__ == "__main__":
    print("Vegetation Indices Calculator - CRS Preservation Test")
    
    # Test CRS preservation
    success = test_crs_preservation()
    
    # Test different CRS types
    test_crs_with_different_projections()
    
    print(f"\nTest completed. CRS preservation: {'PASSED' if success else 'FAILED'}")
