"""
Utility functions for color diagram generation.
This module provides optimized data processing functions to reduce code duplication
and improve performance across color diagram scripts.
"""
import json
from collections import defaultdict


class FilterDataCollector:
    """
    Efficiently collect and organize filter magnitude data from JSON files.
    Replaces repetitive filter_mag calls and manual data appending loops.
    """
    
    def __init__(self, filters_config):
        """
        Initialize the collector with filter configurations.
        
        Parameters
        ----------
        filters_config : list of tuples
            Each tuple contains (suffix, prefix, index) to identify data categories
            Example: [("HPNe", "", 0), ("CV", "", 1), ...]
        """
        self.filters_config = filters_config
        self.data_x = defaultdict(list)
        self.data_y = defaultdict(list)
    
    def process_file(self, data, f1, f2, f3):
        """
        Process a single JSON data dict and extract magnitude differences.
        
        Parameters
        ----------
        data : dict
            JSON data containing 'id' and filter magnitude values
        f1, f2, f3 : str
            Filter names for calculating color differences
        """
        for suffix, prefix, idx in self.filters_config:
            if data['id'].endswith(suffix):
                if not prefix or data['id'].startswith(str(prefix)):
                    try:
                        filter1 = data[f1]
                        filter2 = data[f2]
                        filter3 = data[f3]
                        diff_x = filter1 - filter2
                        diff_y = filter1 - filter3
                        self.data_x[idx].append(diff_x)
                        self.data_y[idx].append(diff_y)
                    except KeyError:
                        # Skip if filters not found in data
                        pass
    
    def get_data_arrays(self):
        """
        Return collected data as lists indexed by configuration order.
        
        Returns
        -------
        tuple of (list, list)
            Two lists containing x and y data arrays for each filter configuration
        """
        max_idx = max((idx for _, _, idx in self.filters_config), default=-1)
        x_arrays = [self.data_x.get(i, []) for i in range(max_idx + 1)]
        y_arrays = [self.data_y.get(i, []) for i in range(max_idx + 1)]
        return x_arrays, y_arrays


def load_json_files_once(file_list):
    """
    Load all JSON files once and cache the data.
    Avoids repeated file I/O operations.
    
    Parameters
    ----------
    file_list : list
        List of JSON file paths
    
    Returns
    -------
    list of dict
        Parsed JSON data from all files
    """
    data_cache = []
    for file_name in file_list:
        try:
            with open(file_name) as f:
                data_cache.append(json.load(f))
        except (IOError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load {file_name}: {e}")
    return data_cache


def extend_lists_from_pairs(target_lists, source_pairs):
    """
    Efficiently extend target lists from source data pairs.
    Replaces manual zip and append loops.
    
    Parameters
    ----------
    target_lists : list of lists
        Target lists to extend, should be pairs [x_list, y_list, x_list, y_list, ...]
    source_pairs : list of tuples
        Source data as (x_data, y_data) pairs
    """
    for i, (x_data, y_data) in enumerate(source_pairs):
        if len(target_lists) > i * 2 + 1:
            target_lists[i * 2].extend(x_data)
            target_lists[i * 2 + 1].extend(y_data)
