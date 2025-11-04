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
    
    This function should be called from within an exception handler
    to ensure sys.exc_info() has valid exception context.
    
    Args:
        e: The exception to print
        stream: Output stream (default: sys.stdout)
    """
    exc_info = sys.exc_info()
    if exc_info[0] is not None:
        print(''.join(traceback.format_exception(*exc_info)), file=stream)
    else:
        print(f"{type(e).__name__}: {e}", file=stream)

    if getattr(e, 'cause', None) is not None:
        print("Caused by:", file=stream, end=' ')
        print(''.join(e.cause), file=stream)

def strException(e):
    """
    Convert exception to a formatted string representation.
    
    This function should be called from within an exception handler
    to ensure sys.exc_info() has valid exception context.
    
    Adapted from chimera observatory control system
    (http://code.google.com/p/chimera)
    
    Args:
        e: The exception to format
        
    Returns:
        List of strings representing the formatted exception traceback,
        or a simple string representation if no exception context is available
    """
    try:
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_type is not None:
            local_tb = traceback.format_exception(exc_type, exc_value, exc_tb)
            return local_tb
        else:
            # No active exception context, return simple representation
            return [f"{type(e).__name__}: {e}\n"]
    finally:
        # Clean up cycle to traceback, to allow proper GC
        try:
            del exc_type, exc_value, exc_tb
        except NameError:
            pass  # Variables were not assigned

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

        exc_info = sys.exc_info()
        if exc_info[0] is not None:
            self.cause = strException(exc_info[1])
        else:
            self.cause = None


class BGPECLIError(Exception):
    """Generic exception to raise and log different fatal errors on CLI programs."""
    
    def __init__(self, msg):
        """
        Initialize CLI error.
        
        Args:
            msg: Error message
        """
        self.msg = f"ERROR: {msg}"
        super().__init__(self.msg)
    
    def __str__(self):
        return self.msg


class HDF5dbException(BGPEException):
    """Exception for HDF5 database operations."""
    pass


class ReadFilterException(BGPEException):
    """Exception for filter reading operations."""
    pass
