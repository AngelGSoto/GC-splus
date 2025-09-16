"""
Created on Jul 17, 2012
@author: william
This module defines the logging format. Adapted from chimera.
"""

import logging


def logger(name):
    logging.basicConfig()
    l = logging.getLogger(name)
    l.setLevel(logging.root.level)
    return l
