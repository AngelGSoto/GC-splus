'''
Cutting images from FITS files (wrapper for backward compatibility).

This script now delegates to cut-images-fits.py with appropriate parameters.
To use the old behavior of cut-images-fits-1.py, run:
    cut-images-fits.py <source> --suffix=".fits" --crop-radius=80.0 --update-crval --output-suffix="-crop-findingchart.fits"
'''
from __future__ import print_function
import sys
import subprocess
import os

# This script is deprecated - redirect to the consolidated version
print("Note: cut-images-fits-1.py is deprecated. Using cut-images-fits.py with appropriate parameters.")
print("To directly use the new script with this behavior, run:")
print('  cut-images-fits.py <source> --suffix=".fits" --crop-radius=80.0 --update-crval --output-suffix="-crop-findingchart.fits"\n')

# Build command line arguments for the consolidated script
# Use the same Python interpreter that's running this script
script_path = os.path.join(os.path.dirname(__file__), 'cut-images-fits.py')
args = [sys.executable, script_path]
args.extend(sys.argv[1:])  # Pass through positional arguments

# Add default parameters that match cut-images-fits-1.py behavior
args.extend([
    '--suffix=.fits',
    '--crop-radius=80.0',
    '--update-crval',
    '--output-suffix=-crop-findingchart.fits'
])

# Execute the consolidated script
subprocess.call(args)
