"""
Exception handling utilities for GC-splus project.

Created on Feb 23, 2012
@author: william
Updated for Python 3 compatibility
"""

import sys
import traceback


def printException(e, stream=sys.stdout):
    """
    Print exception details to the specified stream.
    
    Args:
        e: The exception to print
        stream: Output stream (default: sys.stdout)
    """
    print(''.join(strException(e)), file=stream)

    if hasattr(e, 'cause') and getattr(e, 'cause') is not None:
        print("Caused by:", file=stream, end=' ')
        print(''.join(e.cause), file=stream)

def strException(e):
    """
    Convert exception to a formatted string representation.
    
    Adapted from chimera observatory control system
    (http://code.google.com/p/chimera)
    
    Args:
        e: The exception to format
        
    Returns:
        List of strings representing the formatted exception traceback
    """
    try:
        exc_type, exc_value, exc_tb = sys.exc_info()
        local_tb = traceback.format_exception(exc_type, exc_value, exc_tb)
        return local_tb
    finally:
        # Clean up cycle to traceback, to allow proper GC
        del exc_type, exc_value, exc_tb

# Exceptions Hierarchy

class BGPEException(Exception):
    """Base exception class for BGPE-related errors."""
    
    def __init__(self, msg="", *args):
        """
        Initialize BGPE exception.
        
        Args:
            msg: Error message
            *args: Additional arguments
        """
        super().__init__(msg, *args)

        if not all(sys.exc_info()):
            self.cause = None
        else:
            self.cause = strException(sys.exc_info()[1])


class BGPECLIError(Exception):
    """Generic exception to raise and log different fatal errors on CLI programs."""
    
    def __init__(self, msg):
        """
        Initialize CLI error.
        
        Args:
            msg: Error message
        """
        super().__init__(type(self))
        self.msg = f"ERROR: {msg}"
    
    def __str__(self):
        return self.msg


class HDF5dbException(BGPEException):
    """Exception for HDF5 database operations."""
    pass


class ReadFilterException(BGPEException):
    """Exception for filter reading operations."""
    pass
