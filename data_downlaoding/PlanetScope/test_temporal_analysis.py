#!/usr/bin/env python
"""
Test script for temporal analysis functionality
"""

import sys
import pathlib
from datetime import datetime

# Add the current directory to Python path
sys.path.append(str(pathlib.Path(__file__).parent))

from asset_availability_statistics_udm import parse_filename_timestamp

def test_timestamp_parsing():
    """Test the timestamp parsing function with sample filenames."""
    
    test_filenames = [
        "20250103_162900_70_24b2_udm2.tif",
        "20250215_171208_05_2511_udm2.tif", 
        "20250830_173153_84_24f8_udm2.tif",
        "invalid_filename.tif",
        "20250230_251070_99_9999_udm2.tif"  # Invalid date/time
    ]
    
    print("Testing timestamp parsing function:")
    print("=" * 50)
    
    for filename in test_filenames:
        result = parse_filename_timestamp(filename)
        
        if result:
            dt = result['datetime']
            print(f"✓ {filename}")
            print(f"  Date: {dt.strftime('%Y-%m-%d')}")
            print(f"  Time: {dt.strftime('%H:%M:%S')} UTC")
            print(f"  Day of month: {result['day_of_month']}")
            print(f"  Hour of day: {result['hour_of_day']}")
        else:
            print(f"✗ {filename} - Failed to parse")
        print()

if __name__ == "__main__":
    test_timestamp_parsing()
