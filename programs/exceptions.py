'''
Created on Feb 23, 2012
@author: william

Consolidated exception handling module.
This module merges exceptions from both exceptions.py and exceptions1.py
to eliminate duplication.

Note: Updated to use Python 3 print function syntax for consistency with
other modules in the repository that use 'from __future__ import print_function'.
'''

import sys
import traceback

# Note: app import commented out as it may not be available in all contexts
# import app


def printException(e, stream=sys.stdout):
    """
    Print exception with traceback to stream.
    
    Note: Uses Python 3 print function. For Python 2 compatibility,
    ensure 'from __future__ import print_function' is used.
    """
    print(''.join(strException(e)), file=stream)

    if hasattr(e, 'cause') and getattr(e, 'cause') is not None:
        print("Caused by:", file=stream)
        print(''.join(e.cause), file=stream)


def strException(e):
    """
    Get exception traceback as a string.
    Adapted from chimera observatory control system
    (http://code.google.com/p/chimera)
    """
    try:
        exc_type, exc_value, exc_tb = sys.exc_info()
        local_tb = traceback.format_exception(exc_type, exc_value, exc_tb)
        return local_tb
    finally:
        # clean up cycle to traceback, to allow proper GC
        del exc_type, exc_value, exc_tb


#    Exceptions Hierarchy

class MAGALException(Exception):
    """Base exception class for MAGAL-related errors."""
    
    def __init__(self, msg="", *args):
        Exception.__init__(self, msg, *args)

        if not all(sys.exc_info()):
            self.cause = None
        else:
            self.cause = strException(sys.exc_info()[1])


class MAGALCLIError(Exception):
    """Generic exception to raise and log different fatal errors on scripts."""

    def __init__(self, msg):
        super(MAGALCLIError).__init__(type(self))
        self.msg = "ERROR: %s" % msg

    def __str__(self):
        return self.msg

    def __unicode__(self):
        return self.msg


# BGPE exceptions (consolidated from exceptions1.py)
class BGPEException(Exception):
    """Base exception class for BGPE-related errors."""
    
    def __init__(self, msg="", *args):
        Exception.__init__(self, msg, *args)

        if not all(sys.exc_info()):
            self.cause = None
        else:
            self.cause = strException(sys.exc_info()[1])


class BGPECLIError(Exception):
    """Generic exception to raise and log different fatal errors on CLI programs."""
    
    def __init__(self, msg):
        super(BGPECLIError).__init__(type(self))
        self.msg = "ERROR: %s" % msg
    
    def __str__(self):
        return self.msg
    
    def __unicode__(self):
        return self.msg


class HDF5dbException(BGPEException):
    """Exception for HDF5 database errors."""
    pass


class ReadFilterException(BGPEException):
    """Exception for filter reading errors."""
    pass
