# NDVI Calculator

A comprehensive Python tool for calculating and normalizing NDVI (Normalized Difference Vegetation Index) from multispectral TIFF files using rasterio.

## Features

- **NDVI Calculation**: Computes NDVI for each pixel using the formula: `(NIR - Red) / (NIR + Red)`
- **Multiple Normalization Methods**: 
  - Min-Max scaling (0-1 range)
  - Percentile-based normalization
  - Z-score standardization
- **Flexible Band Selection**: Configurable red and NIR band indices
- **Output Options**: Save raw and/or normalized NDVI as TIFF files
- **Visualization**: Built-in plotting capabilities
- **Command-line Interface**: Easy-to-use CLI for batch processing

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### As a Python Module

```python
from ndvi_calculator import NDVICalculator

# Initialize calculator (assumes band 1 = Red, band 2 = NIR)
calculator = NDVICalculator(red_band_idx=1, nir_band_idx=2)

# Process a TIFF file
raw_ndvi, normalized_ndvi, metadata = calculator.process_tiff(
    input_path='path/to/your/multispectral.tif',
    output_path='path/to/output/ndvi.tif',
    normalization_method='min_max',
    save_raw_ndvi=True
)

# Visualize results
calculator.visualize_ndvi(raw_ndvi, normalized_ndvi, save_plot='ndvi_plot.png')
```

### Command Line Interface

```bash
# Basic usage
python ndvi_calculator.py input.tif -o output.tif

# With custom band indices and normalization
python ndvi_calculator.py input.tif -o output.tif --red-band 3 --nir-band 4 --normalize percentile

# Save raw NDVI and create visualization
python ndvi_calculator.py input.tif -o output.tif --save-raw --visualize

# Save visualization plot
python ndvi_calculator.py input.tif -o output.tif --save-plot ndvi_results.png
```

### Command Line Options

- `input`: Input TIFF file path (required)
- `-o, --output`: Output TIFF file path
- `--red-band`: Red band index (1-indexed, default: 1)
- `--nir-band`: NIR band index (1-indexed, default: 2)
- `--normalize`: Normalization method (`min_max`, `percentile`, `z_score`, default: `min_max`)
- `--save-raw`: Also save raw NDVI values
- `--visualize`: Show visualization plots
- `--save-plot`: Save visualization plot to file

## Normalization Methods

1. **Min-Max Scaling** (`min_max`): Scales values to [0, 1] range
2. **Percentile Normalization** (`percentile`): Uses 2nd and 98th percentiles for robust scaling
3. **Z-Score Standardization** (`z_score`): Centers data around mean with unit variance

## Output Files

- `*_normalized.tif`: Normalized NDVI values
- `*_raw.tif`: Raw NDVI values (if `--save-raw` is used)
- Visualization plots (if requested)

## Requirements

- Python 3.7+
- numpy >= 1.21.0
- rasterio >= 1.3.0
- matplotlib >= 3.5.0

## NDVI Interpretation

NDVI values typically range from -1 to +1:
- **Negative values**: Water bodies, snow, clouds
- **0 to 0.2**: Bare soil, rock, sand
- **0.2 to 0.4**: Sparse vegetation, grassland
- **0.4 to 0.8**: Moderate to dense vegetation
- **0.8 to 1.0**: Very dense, healthy vegetation
