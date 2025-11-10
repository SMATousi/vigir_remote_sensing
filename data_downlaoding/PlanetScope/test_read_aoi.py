#!/usr/bin/env python
"""
Test script to verify the enhanced read_aoi function works with the new GeoJSON format.
"""

import json
import sys
from pathlib import Path

# Add the current directory to path to import the function
sys.path.insert(0, str(Path(__file__).parent))

from download_planetscope_clipping import read_aoi

def test_read_aoi():
    """Test the read_aoi function with the new GeoJSON format."""
    
    # Test configuration that points to our test GeoJSON file
    test_config = {
        "aoi": {
            "geojson_file": "test_outline.geojson"
        }
    }
    
    try:
        # Test reading the GeoJSON file
        print("Testing read_aoi function with FeatureCollection GeoJSON...")
        geometry = read_aoi(test_config)
        
        print("✓ Successfully read GeoJSON file!")
        print(f"Geometry type: {geometry['type']}")
        print(f"Number of coordinate arrays: {len(geometry['coordinates'])}")
        
        # Verify it's the expected MultiPolygon
        if geometry['type'] == 'MultiPolygon':
            print("✓ Correctly identified as MultiPolygon")
            coords = geometry['coordinates'][0][0]  # First polygon, first ring
            print(f"First coordinate: [{coords[0][0]:.6f}, {coords[0][1]:.6f}]")
            print(f"Number of points in polygon: {len(coords)}")
        else:
            print(f"⚠ Expected MultiPolygon, got {geometry['type']}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing read_aoi: {e}")
        return False

def test_with_missing_file():
    """Test error handling when file doesn't exist."""
    test_config = {
        "aoi": {
            "geojson_file": "nonexistent_file.geojson"
        }
    }
    
    try:
        print("\nTesting with missing file...")
        geometry = read_aoi(test_config)
        print("✗ Should have raised FileNotFoundError")
        return False
    except FileNotFoundError as e:
        print(f"✓ Correctly raised FileNotFoundError: {e}")
        return True
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_with_inline_geometry():
    """Test fallback to inline geometry."""
    test_config = {
        "aoi": {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-92.25, 38.92],
                    [-92.25, 38.90],
                    [-92.28, 38.90],
                    [-92.28, 38.92],
                    [-92.25, 38.92]
                ]]
            }
        }
    }
    
    try:
        print("\nTesting with inline geometry...")
        geometry = read_aoi(test_config)
        print("✓ Successfully read inline geometry!")
        print(f"Geometry type: {geometry['type']}")
        return True
    except Exception as e:
        print(f"✗ Error with inline geometry: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Enhanced read_aoi Function ===\n")
    
    success = True
    success &= test_read_aoi()
    success &= test_with_missing_file() 
    success &= test_with_inline_geometry()
    
    print(f"\n=== Test Results ===")
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    
    sys.exit(0 if success else 1)
