'''
Created on Feb 23, 2012
@author: william

This module is now deprecated and imports from exceptions.py for backward compatibility.
All exception classes have been consolidated into exceptions.py.
'''

# Import all exception classes and functions from the consolidated exceptions module
from exceptions import (
    printException,
    strException,
    MAGALException,
    MAGALCLIError,
    BGPEException,
    BGPECLIError,
    HDF5dbException,
    ReadFilterException
)

__all__ = [
    'printException',
    'strException',
    'MAGALException',
    'MAGALCLIError',
    'BGPEException',
    'BGPECLIError',
    'HDF5dbException',
    'ReadFilterException'
]
