# Vegetation Indices Calculator & Analysis Tools

A comprehensive Python toolkit for calculating multiple vegetation indices, NDVI analysis, spatial clustering, temporal analysis, and evaluating clustering results against yield maps.

## Features

### Vegetation Indices Calculator
- **9 Vegetation Indices**: Calculate multiple indices from multispectral imagery
  - **NDVI**: Normalized Difference Vegetation Index `(NIR - Red) / (NIR + Red)`
  - **NDWI**: Normalized Difference Water Index `(Green - NIR) / (Green + NIR)`
  - **GNDVI**: Green NDVI `(NIR - Green) / (NIR + Green)`
  - **NDRE**: Normalized Difference Red Edge Index `(NIR - RedEdge) / (NIR + RedEdge)`
  - **NGRDI**: Normalized Green-Red Difference Index `(Green - Red) / (Green + Red)`
  - **TDVI**: Transformed NDVI `sqrt((NIR - Red) / (NIR + Red) + 0.5)`
  - **ClGreen**: Green Chlorophyll Index `(NIR / Green) - 1`
  - **ClRedEdge**: Red-edge Chlorophyll Index `(NIR / RedEdge) - 1`
  - **EXG**: Excess Green Index `2*Green - (Red + Blue)`
- **Batch Processing**: Process entire folders of TIFF files with organized output structure
- **Flexible Band Configuration**: Support for different satellite data (Landsat, Sentinel-2, etc.)
- **Multiple Normalization Methods**: Min-max, percentile-based, and z-score standardization
- **Comprehensive Visualization**: Multi-index plotting and comparison
- **CRS Preservation**: Output files maintain the same Coordinate Reference System as input files
- **Spatial Reference Integrity**: Preserves geotransform, bounds, and projection information

### NDVI Calculator (Legacy)
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

### Vegetation Indices Calculator

#### As a Python Module

```python
from vegetation_indices_calculator import VegetationIndicesCalculator

# Initialize calculator with band configuration
calculator = VegetationIndicesCalculator(
    green_band=1,     # Green band index (1-indexed)
    red_band=2,       # Red band index (1-indexed)
    nir_band=3,       # NIR band index (1-indexed)
    red_edge_band=4,  # Red edge band index (1-indexed)
    blue_band=5       # Blue band index (1-indexed)
)

# Calculate multiple indices for a single file
results = calculator.process_tiff(
    input_path='path/to/multispectral.tif',
    indices=['NDVI', 'NDWI', 'GNDVI', 'NDRE', 'EXG'],
    output_dir='output_folder',
    normalization_method='min_max',
    save_raw=True
)

# Visualize all calculated indices
calculator.visualize_indices(results, save_plot='vegetation_indices.png')
```

#### Batch Processing

```python
from vegetation_indices_calculator import batch_process_folder, VegetationIndicesCalculator

# Initialize calculator
calculator = VegetationIndicesCalculator(
    green_band=1, red_band=2, nir_band=3, red_edge_band=4, blue_band=5
)

# Process all TIFF files in a folder
batch_process_folder(
    input_folder='path/to/tiff/folder',
    output_base_dir='batch_outputs',
    indices=['NDVI', 'NDWI', 'GNDVI', 'EXG'],
    calculator=calculator,
    normalization_method='percentile',
    save_raw=False
)
```

#### Command Line Interface

```bash
# Calculate single index for one file
python vegetation_indices_calculator.py input.tif --indices NDVI

# Calculate multiple indices
python vegetation_indices_calculator.py input.tif --indices NDVI NDWI GNDVI NDRE EXG

# Batch process entire folder
python vegetation_indices_calculator.py /path/to/folder --batch --indices NDVI NDRE EXG

# Custom band configuration (e.g., for Landsat data)
python vegetation_indices_calculator.py input.tif --green-band 2 --red-band 3 --nir-band 4 --red-edge-band 5 --blue-band 1

# Use percentile normalization and save raw values
python vegetation_indices_calculator.py input.tif --normalize percentile --save-raw

# Visualize results
python vegetation_indices_calculator.py input.tif --indices NDVI GNDVI --visualize --save-plot results.png

# Calculate ALL available indices
python vegetation_indices_calculator.py input.tif --indices ALL

# Batch process with all indices
python vegetation_indices_calculator.py /path/to/folder --batch --indices ALL --save-raw
```

#### Available Indices and Applications

| Index | Best For | Description |
|-------|----------|-------------|
| NDVI | General vegetation health | Standard vegetation index |
| NDWI | Water stress, irrigation | Water content indicator |
| GNDVI | Chlorophyll content | Green-based vegetation index |
| NDRE | Nitrogen status, crop health | Red edge vegetation index |
| NGRDI | Early growth detection | Green-red difference |
| TDVI | Enhanced vegetation contrast | Transformed NDVI |
| ClGreen | Chlorophyll estimation | Green chlorophyll index |
| ClRedEdge | Advanced chlorophyll analysis | Red edge chlorophyll index |
| EXG | Early growth, weed detection | Excess green for plant segmentation |

#### Spatial Reference System Handling

The vegetation indices calculator automatically preserves all spatial reference information from input files:

- **CRS Preservation**: Output files maintain the exact same Coordinate Reference System as the input
- **Geotransform**: Spatial positioning and pixel size are preserved
- **Bounds**: Geographic extent remains unchanged
- **NoData Handling**: Proper handling of invalid/missing pixels

**Supported CRS Types:**
- Geographic coordinate systems (e.g., WGS84, NAD83)
- Projected coordinate systems (e.g., UTM, State Plane)
- Custom projections and local coordinate systems

**Verification:**
The calculator displays CRS information during processing and confirms preservation in output files.

### NDVI Calculator (Legacy)

#### As a Python Module

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

## Clustering vs Yield Map Evaluation

The `clustering_yield_evaluation.py` tool evaluates clustering results against yield map masks using classification metrics.

### Features

- **Multiple Metrics**: Calculates precision, recall, F1-score, and IoU (Intersection over Union)
- **Flexible Yield Ranges**: Create masks from yield maps using custom yield ranges
- **Comprehensive Analysis**: Evaluates each cluster individually against the yield mask
- **Visualization**: Generates bar charts showing metrics for each cluster
- **Export Results**: Saves detailed results to CSV files with metadata

### Usage

#### Command Line Interface

```bash
# Evaluate clustering results against yield map
python clustering_yield_evaluation.py clustering_results.tif yield_map.tif 150 200

# With custom output directory and prefix
python clustering_yield_evaluation.py clustering_results.tif yield_map.tif 150 200 \
    --output_dir results --prefix high_yield_evaluation
```

#### As a Python Module

```python
from clustering_yield_evaluation import ClusteringYieldEvaluator

# Initialize evaluator
evaluator = ClusteringYieldEvaluator('clustering_results.tif', 'yield_map.tif')

# Load data
evaluator.load_data()

# Evaluate clusters for high yield areas (150-200 bu/acre)
results_df = evaluator.evaluate_all_clusters(yield_min=150, yield_max=200)

# Save results
evaluator.save_results(results_df, 'evaluation_results.csv', 150, 200)
evaluator.create_visualization(results_df, 'evaluation_plot.png')
evaluator.print_summary(results_df)
```

### Metrics Explained

- **Precision**: What fraction of pixels predicted as belonging to a cluster actually fall within the yield range?
- **Recall**: What fraction of pixels within the yield range are correctly identified by the cluster?
- **F1-Score**: Harmonic mean of precision and recall (balanced measure)
- **IoU**: Intersection over Union - measures overlap between cluster and yield mask

### Output Files

- `*_results.csv`: Detailed metrics for each cluster
- `*_visualization.png`: Bar charts showing all metrics by cluster
- Console output with summary statistics and top-performing clusters

## NDVI Interpretation

NDVI values typically range from -1 to +1:
- **Negative values**: Water bodies, snow, clouds
- **0 to 0.2**: Bare soil, rock, sand
- **0.2 to 0.4**: Sparse vegetation, grassland
- **0.4 to 0.8**: Moderate to dense vegetation
- **0.8 to 1.0**: Very dense, healthy vegetation
