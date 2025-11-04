import os
import numpy as np
#import atpy

from exceptions1 import ReadFilterException

class readfilterset(object):
    def __init__(self):
        pass
    
    def read(self, filterfile):
        if not os.path.exists(filterfile):
            raise Exception('File not found: %s' % filterfile)
        
        if filterfile.endswith('.filter'):
            dt = np.dtype([('ID_filter', 'S20'), ('wl', 'f'), ('transm', 'f')])
            self.filterset = np.loadtxt(filterfile, dtype=dt)
        else:
            raise Exception('Unsupported file format.')
              
    def uniform(self, dl=1):
        # Vectorized approach: build arrays more efficiently
        result_parts = []
        for fid in np.unique(self.filterset['ID_filter']):
            xx = self.filterset[self.filterset['ID_filter'] == fid]
            new_lambda = np.arange(xx['wl'].min(), xx['wl'].max(), 1.0)
            new_transm = np.interp(new_lambda, xx['wl'], xx['transm'])
            # Create structured array directly instead of appending tuples
            n_points = len(new_lambda)
            filter_data = np.empty(n_points, dtype=self.filterset.dtype)
            filter_data['ID_filter'] = fid
            filter_data['wl'] = new_lambda
            filter_data['transm'] = new_transm
            result_parts.append(filter_data)
        # Concatenate all parts at once instead of building list and converting
        self.filterset = np.concatenate(result_parts) if result_parts else np.array([], dtype=self.filterset.dtype)
            
    def calc_filteravgwls(self):
        # Vectorized calculation of filter average wavelengths
        unique_filters = np.unique(self.filterset['ID_filter'])
        avg = np.array([
            np.average(self.filterset[self.filterset['ID_filter'] == fid]['wl'])
            for fid in unique_filters
        ])
        self.filteravgwls = avg
