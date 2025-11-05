'''
Utility functions for magnitude and filter calculations.

This module contains shared functions used across multiple diagram scripts
to calculate color differences from photometric filter data.

Main Functions:
--------------
calculate_color_differences(data, e, s, f1, f2, f3)
    Standard version that checks both the ending and starting of ID strings.
    Returns color differences (f1-f2, f1-f3) for matching data entries.

calculate_color_differences_simple(data, e, f1, f2, f3)
    Simplified version that only checks the ending of ID strings.
    Returns color differences (f1-f2, f1-f3) for matching data entries.
'''
from __future__ import print_function


def calculate_color_differences(data, e, s, f1, f2, f3):
    """
    Calculate color differences for filters.
    
    This is the common filter_mag function used across many diagram scripts.
    
    Parameters:
    -----------
    data : dict
        Data dictionary containing filter magnitudes and 'id' field
    e : str
        Ending string to match in data['id']
    s : str
        Starting string to match in data['id']
    f1, f2, f3 : str
        Filter names to use for color calculations
        
    Returns:
    --------
    tuple : (list, list)
        Two lists containing color differences (f1-f2, f1-f3)
    """
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


def calculate_color_differences_simple(data, e, f1, f2, f3):
    """
    Calculate color differences for filters (simplified version without starting string check).
    
    This is a variant used in some scripts that only checks the ending of the ID.
    
    Parameters:
    -----------
    data : dict
        Data dictionary containing filter magnitudes and 'id' field
    e : str
        Ending string to match in data['id']
    f1, f2, f3 : str
        Filter names to use for color calculations
        
    Returns:
    --------
    tuple : (list, list)
        Two lists containing color differences (f1-f2, f1-f3)
    """
    col, col0 = [], []
    if data['id'].endswith(e):
        filter1 = data[f1]
        filter2 = data[f2]
        filter3 = data[f3]
        diff = filter1 - filter2
        diff0 = filter1 - filter3
        col.append(diff)
        col0.append(diff0)
    
    return col, col0
