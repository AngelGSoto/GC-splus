"""
Created on Jul 17, 2012
@author: william
This module defines the logging format. Adapted from chimera.
"""

import logging

# Configure logging once at module level to avoid repeated configuration
_logging_configured = False

def logger(name):
    global _logging_configured
    if not _logging_configured:
        logging.basicConfig()
        _logging_configured = True
    l = logging.getLogger(name)
    l.setLevel(logging.root.level)
    return l
