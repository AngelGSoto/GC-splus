'''
Created on Feb 23, 2012
@author: william
'''

import sys
import traceback
#from app import app #cambie esto
import app
#from web_tool import app as app

#import logging

def printException(e, stream=sys.stdout):
    print >> stream, ''.join(strException(e))

    if hasattr(e, 'cause') and getattr(e, 'cause') is not None:
        print >> stream, "Caused by:",
        print >> stream, ''.join(e.cause)


def strException(e):
    # Adapted from chimera observatory control system
    #      (http://code.google.com/p/chimera)
    try:
        exc_type, exc_value, exc_tb = sys.exc_info()
        local_tb = traceback.format_exception(exc_type, exc_value, exc_tb)
        return local_tb
    finally:
        # clean up cycle to traceback, to allow proper GC
        del exc_type, exc_value, exc_tb


#    Exceptions Hierarchy

class MAGALException(Exception):
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


#class HDF5dbException(MAGALException):
#     pass
#
#
#class ReadFilterException(MAGALException):
#     pass
