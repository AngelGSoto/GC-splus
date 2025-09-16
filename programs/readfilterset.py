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
        aux = []
        for fid in np.unique(self.filterset['ID_filter']):
            xx = self.filterset[self.filterset['ID_filter'] == fid]
            new_lambda = np.arange(xx['wl'].min(), xx['wl'].max(), 1.0)
            new_transm = np.interp(new_lambda, xx['wl'], xx['transm'])
            for i in range(len(new_lambda)):
                aux.append((fid, new_lambda[i], new_transm[i]))
        self.filterset = np.array(aux, dtype=self.filterset.dtype)
            
    def calc_filteravgwls(self):
        avg = []
        for fid in np.unique(self.filterset['ID_filter']):
            avg.append(np.average(self.filterset[self.filterset['ID_filter'] == fid]['wl']))
        self.filteravgwls = np.array(avg)
