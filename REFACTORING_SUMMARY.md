# Code Refactoring Summary

## Overview
This document summarizes the code refactoring performed to eliminate duplicated code in the GC-splus repository.

## Changes Made

### 1. New Utility Modules Created

#### `programs/magnitude_utils.py`
- **Purpose**: Centralized location for magnitude and color difference calculations
- **Functions**:
  - `calculate_color_differences(data, e, s, f1, f2, f3)`: Standard version that checks both start and end of ID strings
  - `calculate_color_differences_simple(data, e, f1, f2, f3)`: Simplified version that only checks the end of ID strings
- **Impact**: Eliminates duplicate `filter_mag` function implementations across 9+ files

#### `programs/fits_utils.py`
- **Purpose**: Reusable utilities for FITS image operations
- **Functions**:
  - `read_position_file(position_file)`: Read coordinates from position.reg files
  - `crop_fits_image(hdu, crop_coord, crop_radius, pix_scale, update_crval)`: Crop FITS images with configurable options
  - `save_cropped_fits(hdu, input_filename, output_suffix)`: Save cropped FITS files
- **Impact**: Provides reusable components for FITS image manipulation

### 2. Exception Module Consolidation

#### `programs/exceptions.py` (Updated)
- **Changes**: Merged all exception classes from both `exceptions.py` and `exceptions1.py`
- **Classes now included**:
  - `MAGALException` and `MAGALCLIError` (original)
  - `BGPEException`, `BGPECLIError`, `HDF5dbException`, `ReadFilterException` (from exceptions1.py)
- **Modernization**: Updated to Python 3 compatible print statements

#### `programs/exceptions1.py` (Updated)
- **Changes**: Now a compatibility wrapper that imports from `exceptions.py`
- **Purpose**: Maintains backward compatibility for existing code
- **Impact**: No code using `exceptions1.py` needs to be modified

### 3. FITS Image Cutting Script Consolidation

#### `programs/cut-images-fits.py` (Updated)
- **Changes**: Now supports both use cases through command-line parameters
- **New parameters**:
  - `--suffix`: File suffix (default: `_swp`, can use `.fits`)
  - `--crop-radius`: Crop radius in arcseconds (default: 6.90)
  - `--update-crval`: Update CRVAL1/CRVAL2 to crop center
  - `--output-suffix`: Output file suffix
- **Impact**: Single parameterized script replaces two nearly identical scripts

#### `programs/cut-images-fits-1.py` (Updated)
- **Changes**: Now a wrapper that calls `cut-images-fits.py` with appropriate parameters
- **Purpose**: Maintains backward compatibility
- **Impact**: Existing workflows using this script continue to work

### 4. Color Diagram Scripts Refactored

The following scripts were refactored to use `magnitude_utils`:

1. `color-diagram-SPLUS.py`
2. `color-diagram-JPLUS.py`
3. `color-diagram-JPAS.py`
4. `color-diagram-JPAS-E0.py`
5. `color-diagram-JPAS-E02.py`
6. `color-diagram-JPLUS_filk.py`
7. `SPLUS-diagram.py`
8. `JPLUS-diagram.py`
9. `JPAS-diagram.py`

**Changes in each file**:
- Added `from magnitude_utils import calculate_color_differences` (or `calculate_color_differences_simple`)
- Replaced the entire local `filter_mag` function implementation with a simple wrapper that calls the utility function
- Example:
  ```python
  # Before (repeated in every file):
  def filter_mag(e, s, f1, f2, f3):
      col, col0 = [], []
      if data['id'].endswith(e):
          if data['id'].startswith(str(s)):
              filter1 = data[f1]
              filter2 = data[f2]
              filter3 = data[f3]
              diff = filter1 - filter2
              diff0 = filter1 - filter3
              col.append(diff)
              col0.append(diff0)
      return col, col0
  
  # After (in each file):
  from magnitude_utils import calculate_color_differences
  
  def filter_mag(e, s, f1, f2, f3):
      return calculate_color_differences(data, e, s, f1, f2, f3)
  ```

## Benefits

1. **Reduced Code Duplication**: Hundreds of lines of duplicate code eliminated
2. **Easier Maintenance**: Bug fixes and improvements only need to be made once in the utility modules
3. **Better Code Organization**: Related functionality grouped together
4. **Backward Compatibility**: All existing scripts continue to work without modification
5. **Improved Testability**: Utility functions can be tested independently

## Testing

All modified Python files pass syntax validation:
- `python -m py_compile` succeeds for all updated files
- No syntax errors introduced

## Migration Guide

For new scripts:
- Import from `magnitude_utils` directly: `from magnitude_utils import calculate_color_differences`
- Import from `fits_utils` for FITS operations: `from fits_utils import crop_fits_image`
- Import from `exceptions` for exception classes: `from exceptions import MAGALException`

For existing scripts:
- No changes needed - all modifications maintain backward compatibility
- Consider migrating to use utility functions directly for cleaner code

## Files Not Modified

The following files with similar patterns were NOT modified as they have unique implementations:
- `diagram-color-alh.py`
- `diagram-color-alh-t.py`
- `diagram-color-alh-other-comb.py`

These files use hardcoded filter names and have different logic, so they were left as-is to avoid introducing bugs.
