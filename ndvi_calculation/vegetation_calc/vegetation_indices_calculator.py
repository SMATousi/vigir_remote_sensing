import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
import warnings
import argparse
import os
import glob
from pathlib import Path


class VegetationIndicesCalculator:
    """
    A class for calculating various vegetation indices from multispectral TIFF files.
    
    Supported indices:
    - NDWI: Normalized Difference Water Index
    - NDVI: Normalized Difference Vegetation Index  
    - TDVI: Transformed Normalized Difference Vegetation Index
    - NDRE: Normalized Difference Red Edge Index
    - NGRDI: Normalized Green-Red Difference Index
    - ClGreen: Green Chlorophyll Index
    - ClRedEdge: Red-edge Chlorophyll Index
    - GNDVI: Green NDVI
    
    Attributes:
        band_config (dict): Configuration mapping for band indices
    """
    
    def __init__(self, green_band: int = 1, red_band: int = 2, 
                 nir_band: int = 3, red_edge_band: int = 4, blue_band: int = 5):
        """
        Initialize the Vegetation Indices Calculator.
        
        Args:
            green_band (int): Band index for green channel (1-indexed)
            red_band (int): Band index for red channel (1-indexed) 
            nir_band (int): Band index for NIR channel (1-indexed)
            red_edge_band (int): Band index for red edge channel (1-indexed)
            blue_band (int): Band index for blue channel (1-indexed)
        """
        self.band_config = {
            'green': green_band,
            'red': red_band,
            'nir': nir_band,
            'red_edge': red_edge_band,
            'blue': blue_band
        }
        
        # Define available indices and their calculation methods
        self.available_indices = {
            'NDWI': self._calculate_ndwi,
            'NDVI': self._calculate_ndvi,
            'TDVI': self._calculate_tdvi,
            'NDRE': self._calculate_ndre,
            'NGRDI': self._calculate_ngrdi,
            'ClGreen': self._calculate_clgreen,
            'ClRedEdge': self._calculate_clrededge,
            'GNDVI': self._calculate_gndvi,
            'EXG': self._calculate_exg
        }
    
    def read_tiff(self, file_path: str) -> Tuple[rasterio.DatasetReader, dict]:
        """
        Read a TIFF file and return the dataset and metadata.
        
        Args:
            file_path (str): Path to the TIFF file
            
        Returns:
            Tuple[rasterio.DatasetReader, dict]: Dataset and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        dataset = rasterio.open(file_path)
        
        # Validate that we have enough bands
        max_band_needed = max(self.band_config.values())
        if dataset.count < max_band_needed:
            raise ValueError(f"File has {dataset.count} bands, but need at least {max_band_needed} bands")
        
        metadata = {
            'width': dataset.width,
            'height': dataset.height,
            'count': dataset.count,
            'dtype': dataset.dtypes[0],
            'crs': dataset.crs,
            'transform': dataset.transform,
            'bounds': dataset.bounds,
            'nodata': dataset.nodata
        }
        
        # Validate CRS preservation
        if dataset.crs is None:
            print("Warning: Input file has no CRS information. Output files will also have no CRS.")
        else:
            print(f"Input CRS: {dataset.crs}")
        
        return dataset, metadata
    
    def _read_bands(self, dataset: rasterio.DatasetReader, 
                   bands_needed: List[str], window: Optional[Window] = None) -> Dict[str, np.ndarray]:
        """
        Read specified bands from the dataset.
        
        Args:
            dataset: Rasterio dataset
            bands_needed: List of band names needed
            window: Optional window to read
            
        Returns:
            Dict mapping band names to arrays
        """
        bands = {}
        for band_name in bands_needed:
            if band_name in self.band_config:
                band_idx = self.band_config[band_name]
                bands[band_name] = dataset.read(band_idx, window=window).astype(np.float64)
            else:
                raise ValueError(f"Unknown band: {band_name}")
        return bands
    
    def _calculate_ndwi(self, bands: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate Normalized Difference Water Index: (GREEN - NIR) / (GREEN + NIR)"""
        green = bands['green']
        nir = bands['nir']
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            numerator = green - nir
            denominator = green + nir
            ndwi = np.divide(numerator, denominator, 
                           out=np.zeros_like(numerator), 
                           where=denominator != 0)
        
        ndwi[denominator == 0] = np.nan
        return ndwi
    
    def _calculate_ndvi(self, bands: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate Normalized Difference Vegetation Index: (NIR - RED) / (NIR + RED)"""
        red = bands['red']
        nir = bands['nir']
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            numerator = nir - red
            denominator = nir + red
            ndvi = np.divide(numerator, denominator, 
                           out=np.zeros_like(numerator), 
                           where=denominator != 0)
        
        ndvi[denominator == 0] = np.nan
        return ndvi
    
    def _calculate_tdvi(self, bands: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate Transformed NDVI: sqrt((NIR - RED) / (NIR + RED) + 0.5)"""
        red = bands['red']
        nir = bands['nir']
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            numerator = nir - red
            denominator = nir + red
            ndvi_part = np.divide(numerator, denominator, 
                                out=np.zeros_like(numerator), 
                                where=denominator != 0)
            
            # Add 0.5 and take square root
            tdvi = np.sqrt(ndvi_part + 0.5)
        
        tdvi[denominator == 0] = np.nan
        return tdvi
    
    def _calculate_ndre(self, bands: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate Normalized Difference Red Edge Index: (NIR - RED_EDGE) / (NIR + RED_EDGE)"""
        red_edge = bands['red_edge']
        nir = bands['nir']
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            numerator = nir - red_edge
            denominator = nir + red_edge
            ndre = np.divide(numerator, denominator, 
                           out=np.zeros_like(numerator), 
                           where=denominator != 0)
        
        ndre[denominator == 0] = np.nan
        return ndre
    
    def _calculate_ngrdi(self, bands: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate Normalized Green-Red Difference Index: (GREEN - RED) / (GREEN + RED)"""
        green = bands['green']
        red = bands['red']
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            numerator = green - red
            denominator = green + red
            ngrdi = np.divide(numerator, denominator, 
                            out=np.zeros_like(numerator), 
                            where=denominator != 0)
        
        ngrdi[denominator == 0] = np.nan
        return ngrdi
    
    def _calculate_clgreen(self, bands: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate Green Chlorophyll Index: (NIR / GREEN) - 1"""
        green = bands['green']
        nir = bands['nir']
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            clgreen = np.divide(nir, green, 
                              out=np.zeros_like(nir), 
                              where=green != 0) - 1
        
        clgreen[green == 0] = np.nan
        return clgreen
    
    def _calculate_clrededge(self, bands: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate Red-edge Chlorophyll Index: (NIR / RED_EDGE) - 1"""
        red_edge = bands['red_edge']
        nir = bands['nir']
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            clrededge = np.divide(nir, red_edge, 
                                out=np.zeros_like(nir), 
                                where=red_edge != 0) - 1
        
        clrededge[red_edge == 0] = np.nan
        return clrededge
    
    def _calculate_gndvi(self, bands: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate Green NDVI: (NIR - GREEN) / (NIR + GREEN)"""
        green = bands['green']
        nir = bands['nir']
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            numerator = nir - green
            denominator = nir + green
            gndvi = np.divide(numerator, denominator, 
                            out=np.zeros_like(numerator), 
                            where=denominator != 0)
        
        gndvi[denominator == 0] = np.nan
        return gndvi
    
    def _calculate_exg(self, bands: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate Excess Green Index: 2*GREEN - (RED + BLUE)"""
        green = bands['green']
        red = bands['red']
        blue = bands['blue']
        
        # Calculate EXG: 2*G - (R + B)
        exg = 2 * green - (red + blue)
        
        return exg
    
    def calculate_index(self, dataset: rasterio.DatasetReader, index_name: str,
                       window: Optional[Window] = None) -> np.ndarray:
        """
        Calculate a specific vegetation index.
        
        Args:
            dataset: Rasterio dataset
            index_name: Name of the index to calculate
            window: Optional window to read
            
        Returns:
            np.ndarray: Calculated index values
        """
        if index_name not in self.available_indices:
            raise ValueError(f"Unknown index: {index_name}. Available: {list(self.available_indices.keys())}")
        
        # Determine which bands are needed for this index
        bands_needed = self._get_bands_needed(index_name)
        
        # Read the required bands
        bands = self._read_bands(dataset, bands_needed, window)
        
        # Calculate the index
        return self.available_indices[index_name](bands)
    
    def _get_bands_needed(self, index_name: str) -> List[str]:
        """Get the list of bands needed for a specific index."""
        band_requirements = {
            'NDWI': ['green', 'nir'],
            'NDVI': ['red', 'nir'],
            'TDVI': ['red', 'nir'],
            'NDRE': ['red_edge', 'nir'],
            'NGRDI': ['green', 'red'],
            'ClGreen': ['green', 'nir'],
            'ClRedEdge': ['red_edge', 'nir'],
            'GNDVI': ['green', 'nir'],
            'EXG': ['green', 'red', 'blue']
        }
        return band_requirements.get(index_name, [])
    
    def normalize_index(self, index_values: np.ndarray, 
                       method: str = 'min_max', 
                       percentile_range: Tuple[float, float] = (2, 98)) -> np.ndarray:
        """
        Normalize index values across the field.
        
        Args:
            index_values: Raw index values
            method: Normalization method ('min_max', 'percentile', 'z_score')
            percentile_range: Percentile range for percentile normalization
            
        Returns:
            np.ndarray: Normalized index values
        """
        # Remove NaN values for calculations
        valid_mask = ~np.isnan(index_values)
        valid_values = index_values[valid_mask]
        
        if len(valid_values) == 0:
            return index_values.copy()
        
        normalized = index_values.copy()
        
        if method == 'min_max':
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
            if max_val > min_val:
                normalized[valid_mask] = (valid_values - min_val) / (max_val - min_val)
            else:
                normalized[valid_mask] = 0.5  # All values are the same
                
        elif method == 'percentile':
            p_low, p_high = percentile_range
            low_val = np.percentile(valid_values, p_low)
            high_val = np.percentile(valid_values, p_high)
            if high_val > low_val:
                normalized[valid_mask] = np.clip((valid_values - low_val) / (high_val - low_val), 0, 1)
            else:
                normalized[valid_mask] = 0.5
                
        elif method == 'z_score':
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            if std_val > 0:
                normalized[valid_mask] = (valid_values - mean_val) / std_val
            else:
                normalized[valid_mask] = 0
                
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def process_tiff(self, input_path: str, indices: List[str],
                    output_dir: Optional[str] = None,
                    normalization_method: str = 'min_max',
                    save_raw: bool = False) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Complete processing pipeline: read TIFF, calculate indices, normalize values.
        
        Args:
            input_path: Path to input TIFF file
            indices: List of indices to calculate
            output_dir: Directory to save output files
            normalization_method: Normalization method
            save_raw: Whether to save raw index values
            
        Returns:
            Dict mapping index names to (raw, normalized) arrays
        """
        # Read the input file
        dataset, metadata = self.read_tiff(input_path)
        
        results = {}
        
        for index_name in indices:
            print(f"Calculating {index_name}...")
            
            # Calculate the index
            raw_index = self.calculate_index(dataset, index_name)
            
            # Normalize the index
            normalized_index = self.normalize_index(raw_index, normalization_method)
            
            # Store results
            results[index_name] = (raw_index, normalized_index)
            
            # Print statistics
            self._print_statistics(index_name, raw_index, normalized_index)
            
            # Save outputs if output directory is specified
            if output_dir:
                self._save_index_outputs(raw_index, normalized_index, metadata, 
                                       input_path, output_dir, index_name, save_raw)
        
        dataset.close()
        return results
    
    def _print_statistics(self, index_name: str, raw_index: np.ndarray, normalized_index: np.ndarray):
        """Print statistics about the index calculation."""
        valid_raw = raw_index[~np.isnan(raw_index)]
        valid_norm = normalized_index[~np.isnan(normalized_index)]
        
        print(f"\n{index_name} Statistics:")
        print(f"  Valid pixels: {len(valid_raw):,} / {raw_index.size:,}")
        if len(valid_raw) > 0:
            print(f"  Raw {index_name} - Min: {np.min(valid_raw):.4f}, Max: {np.max(valid_raw):.4f}, Mean: {np.mean(valid_raw):.4f}")
            print(f"  Normalized {index_name} - Min: {np.min(valid_norm):.4f}, Max: {np.max(valid_norm):.4f}, Mean: {np.mean(valid_norm):.4f}")
    
    def _save_index_outputs(self, raw_index: np.ndarray, normalized_index: np.ndarray,
                          metadata: dict, input_path: str, output_dir: str, 
                          index_name: str, save_raw: bool):
        """Save index outputs to TIFF files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output metadata - preserving CRS and spatial reference
        out_meta = {
            'driver': 'GTiff',
            'height': metadata['height'],
            'width': metadata['width'],
            'count': 1,
            'dtype': 'float32',
            'crs': metadata['crs'],  # Preserve original CRS
            'transform': metadata['transform'],  # Preserve original geotransform
            'compress': 'lzw',
            'nodata': np.nan  # Use NaN for nodata in float32 outputs
        }
        
        # Confirm CRS preservation
        if metadata['crs'] is not None:
            print(f"  Preserving CRS: {metadata['crs']}")
        
        # Get base filename without extension and folder name
        input_path_obj = Path(input_path)
        base_name = input_path_obj.stem
        folder_name = input_path_obj.parent.name
        
        # Create filename with folder name included
        filename_prefix = f"{folder_name}_{base_name}"
        
        # Save normalized index
        norm_path = os.path.join(output_dir, f"{filename_prefix}_{index_name}_normalized.tif")
        with rasterio.open(norm_path, 'w', **out_meta) as dst:
            dst.write(normalized_index.astype(np.float32), 1)
        print(f"  Normalized {index_name} saved to: {norm_path}")
        
        # Save raw index if requested
        if save_raw:
            raw_path = os.path.join(output_dir, f"{filename_prefix}_{index_name}_raw.tif")
            with rasterio.open(raw_path, 'w', **out_meta) as dst:
                dst.write(raw_index.astype(np.float32), 1)
            print(f"  Raw {index_name} saved to: {raw_path}")
    
    def visualize_indices(self, results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                         save_plot: Optional[str] = None):
        """
        Create visualization plots for multiple indices.
        
        Args:
            results: Dictionary mapping index names to (raw, normalized) arrays
            save_plot: Path to save the plot
        """
        n_indices = len(results)
        if n_indices == 0:
            return
        
        # Create subplots - 2 columns (raw and normalized) for each index
        fig, axes = plt.subplots(n_indices, 2, figsize=(12, 4 * n_indices))
        
        if n_indices == 1:
            axes = axes.reshape(1, -1)
        
        for i, (index_name, (raw_values, norm_values)) in enumerate(results.items()):
            # Raw index
            im1 = axes[i, 0].imshow(raw_values, cmap='RdYlGn')
            axes[i, 0].set_title(f'Raw {index_name}')
            axes[i, 0].axis('off')
            plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            # Normalized index
            im2 = axes[i, 1].imshow(norm_values, cmap='RdYlGn')
            axes[i, 1].set_title(f'Normalized {index_name}')
            axes[i, 1].axis('off')
            plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(save_plot, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_plot}")
        
        plt.show()


def batch_process_folder(input_folder: str, output_base_dir: str, 
                        indices: List[str], calculator: VegetationIndicesCalculator,
                        normalization_method: str = 'min_max', save_raw: bool = False):
    """
    Batch process all TIFF files in a folder and organize outputs by source folder.
    
    Args:
        input_folder: Path to folder containing TIFF files
        output_base_dir: Base directory for outputs
        indices: List of indices to calculate
        calculator: VegetationIndicesCalculator instance
        normalization_method: Normalization method to use
        save_raw: Whether to save raw index values
    """
    # Find all TIFF files
    tiff_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    tiff_files = []
    
    for pattern in tiff_patterns:
        tiff_files.extend(glob.glob(os.path.join(input_folder, '**', pattern), recursive=True))
    
    if not tiff_files:
        print(f"No TIFF files found in {input_folder}")
        return
    
    print(f"Found {len(tiff_files)} TIFF files to process")
    
    for i, tiff_file in enumerate(tiff_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {i}/{len(tiff_files)}: {os.path.basename(tiff_file)}")
        print(f"{'='*60}")
        
        try:
            # Get relative path from input folder to maintain directory structure
            rel_path = os.path.relpath(tiff_file, input_folder)
            rel_dir = os.path.dirname(rel_path)
            
            # Create output directory based on source folder structure
            if rel_dir:
                output_dir = os.path.join(output_base_dir, rel_dir)
            else:
                output_dir = output_base_dir
            
            # Process the file
            results = calculator.process_tiff(
                input_path=tiff_file,
                indices=indices,
                output_dir=output_dir,
                normalization_method=normalization_method,
                save_raw=save_raw
            )
            
            print(f"Successfully processed: {os.path.basename(tiff_file)}")
            
        except Exception as e:
            print(f"Error processing {tiff_file}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Batch processing completed!")
    print(f"Results saved to: {output_base_dir}")
    print(f"{'='*60}")


def main():
    """Command-line interface for the vegetation indices calculator."""
    parser = argparse.ArgumentParser(description='Calculate vegetation indices from TIFF files')
    parser.add_argument('input', help='Input TIFF file or folder path')
    parser.add_argument('-o', '--output', help='Output directory path')
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
    parser.add_argument('--batch', action='store_true', help='Process all TIFF files in input folder')
    parser.add_argument('--visualize', action='store_true', help='Show visualization plots')
    parser.add_argument('--save-plot', help='Save visualization plot to file')
    
    args = parser.parse_args()
    
    # Handle 'ALL' option for indices
    if 'ALL' in args.indices:
        args.indices = ['NDWI', 'NDVI', 'TDVI', 'NDRE', 'NGRDI', 'ClGreen', 'ClRedEdge', 'GNDVI', 'EXG']
        print(f"Calculating all available indices: {', '.join(args.indices)}")
    
    # Initialize calculator
    calculator = VegetationIndicesCalculator(
        green_band=args.green_band,
        red_band=args.red_band,
        nir_band=args.nir_band,
        red_edge_band=args.red_edge_band,
        blue_band=args.blue_band
    )
    
    try:
        if args.batch:
            # Batch processing mode
            if not os.path.isdir(args.input):
                print(f"Error: {args.input} is not a directory")
                return 1
            
            output_dir = args.output or f"{args.input}_vegetation_indices"
            
            batch_process_folder(
                input_folder=args.input,
                output_base_dir=output_dir,
                indices=args.indices,
                calculator=calculator,
                normalization_method=args.normalize,
                save_raw=args.save_raw
            )
        else:
            # Single file processing mode
            if not os.path.isfile(args.input):
                print(f"Error: {args.input} is not a file")
                return 1
            
            output_dir = args.output or os.path.dirname(args.input)
            
            # Process the single TIFF file
            results = calculator.process_tiff(
                input_path=args.input,
                indices=args.indices,
                output_dir=output_dir,
                normalization_method=args.normalize,
                save_raw=args.save_raw
            )
            
            # Visualize if requested
            if args.visualize or args.save_plot:
                calculator.visualize_indices(results, args.save_plot)
        
        print("\nVegetation indices calculation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
