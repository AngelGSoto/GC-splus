'''
Date:
Adapted for Luis A. Gutiérrez Soto
Created on Jul 17, 2012
by @author: william
'''

import numpy as np
from log import logger

def spec2filter(filter, obs_spec, model_spec=None, badpxl_tolerance=0.5):
    '''
    Converts a spectrum to AB magnitude given a filter bandpass.
    '''
    log = logger(__name__)
    
    # 1. Recortar el espectro al rango del filtro
    obs_cut = obs_spec[
        (obs_spec['wl'] >= np.min(filter['wl'])) & 
        (obs_spec['wl'] <= np.max(filter['wl']))
    ]
    
    # 2. Interpolar el filtro al grid del espectro observado
    wl_ = obs_cut['wl']
    transm_ = np.interp(wl_, filter['wl'], filter['transm'])
    
    # 3. Manejar modelo (si existe)
    if model_spec is not None:
        model_cut = model_spec[
            (model_spec['wl'] >= np.min(filter['wl'])) & 
            (model_spec['wl'] <= np.max(filter['wl']))
        ]
        if len(model_cut) == 0:
            model_spec = None
        else:
            model_cut['flux'] = np.interp(wl_, model_cut['wl'], model_cut['flux'], left=0.0, right=0.0)
    
    # 4. Verificar píxeles malos
    n = len(obs_cut)
    not_neg_pix = (obs_cut['flux'] >= 0)
    
    # ... (código de verificación de píxeles malos sin cambios)
    
    # 5. Calcular magnitud AB
    c = 2.99792458e18  # Velocidad de la luz en Å/s
    try:
        numerator = np.trapz(obs_cut['flux'] * transm_ * wl_, wl_)
        denominator = c * np.trapz(transm_ / wl_, wl_)
        m_ab = -2.5 * np.log10(numerator / denominator) - 48.60
    except:
        m_ab = np.inf
    
    # 6. Calcular error (si existe)
    if 'error' in obs_cut.dtype.names:
        e_ab = 1.085736 * np.sqrt(np.sum((transm_ * obs_cut['error'] * wl_)**2)) / numerator
    else:
        e_ab = 0.0
    
    return m_ab, e_ab

def spec2filterset(filterset, obs_spec, model_spec=None, badpxl_tolerance=0.5):
    '''
    Ejecuta spec2filter sobre un conjunto de filtros
    '''
    log = logger(__name__)
    filter_ids = np.unique(filterset['ID_filter'])
    mags = np.zeros(len(filter_ids), dtype=[('m_ab', '<f4'), ('e_ab', '<f4')])
    
    for i_filter in range(len(filter_ids)):
        filter = filterset[filterset['ID_filter'] == filter_ids[i_filter]]
        try:
            mags[i_filter]['m_ab'], mags[i_filter]['e_ab'] = spec2filter(
                filter, obs_spec, model_spec, badpxl_tolerance
            )
        except Exception as e:
            log.error(f"Error en filtro {filter_ids[i_filter]}: {str(e)}")
            mags[i_filter]['m_ab'] = np.inf
            mags[i_filter]['e_ab'] = np.inf
    
    return mags

# ... (Clases photoconv sin cambios)

class photoconv(object):
    """
    Spectrum to Photometry conversion class.
    """

    def __init__(self):
        self.log = logger(__name__)

    def fromStarlight(self, filterset, arq_in, arq_syn, starlight_version='starlightv4', badpxl_tolerance=0.5):
        """
        Converts automagically STARLIGHT input and output files into photometric magnitudes
        
        Parameters
        ----------
        filterset : object
            Filter transmission curves (see: magal.io.readfilterset).
        arq_in : string
            Starlight input filename (or atpy.TableSet(type='starlight_input') object)
        arq_syn : string
            Starlight synthesis filename (or atpy.TableSet(type=starlight_version) object)
        starlight_version : string, default = 'starlightv4'
            Starlight synthesis file version (Default: starlightv4)
        badpxl_tolerance : float, default: 0.5
            Bad pixel fraction tolerance on the spectral interval of each filter. (Default: 0.5)
        
        Returns
        -------
        m_ab : numpy.ndarray dtype = [('m_ab', '<f4'), ('e_ab', '<f4')]
        
        See Also
        --------
        fromSDSSfits, magal.io.readfilterset
        
        """
        
        try:  # Try to import pystarlight...
            import pystarlight.io
            import atpy
        except ImportError:
            MAGALException('Could not load pystarlight. Needed to convert from STARLIGHT')
        
        try:  # Check if it is an atpy or a filename
            obs_spec = arq_in.starlight_input.data.view(
                dtype=np.dtype([('wl', '<f8'), ('flux', '<f8'), ('error', '<f8'), ('flag', '<i8')]))
        except AttributeError:
            arq_in = atpy.Table(arq_in, type='starlight_input')
            obs_spec = arq_in.data.view(
                dtype=np.dtype([('wl', '<f8'), ('flux', '<f8'), ('error', '<f8'), ('flag', '<i8')]))
        
        try:  # Check if it is an atpy or a filename
            model_spec = arq_syn.spectra.data.view(dtype=np.dtype(
                [('wl', '<f8'), ('f_obs', '<f8'), ('flux', '<f8'), ('f_wei', '<f8'), ('Best_f_SSP', '<f8')]))
        except AttributeError:
            arq_syn = atpy.TableSet(arq_syn, type=starlight_version)
            model_spec = arq_syn.spectra.data.view(dtype=np.dtype(
                [('wl', '<f8'), ('f_obs', '<f8'), ('flux', '<f8'), ('f_wei', '<f8'), ('Best_f_SSP', '<f8')]))

        obs_spec['flux'] *= 1e-17
        obs_spec['error'] *= 1e-17
        model_spec['flux'] *= arq_syn.keywords['fobs_norm'] * 1e-17
        
        
        return spec2filterset(filterset, obs_spec, model_spec, badpxl_tolerance = badpxl_tolerance)

    def fromSDSSfits(self, filterset, fits, badpxl_tolerance = 0.5):
        ''' Converts automagically SDSS .fits spectrum files into photometric magnitudes
        
        Parameters
        ----------
        filterset : string or object
                    Filterset filename (or magal.io.readfilterset object)
        fits : string or object 
               SDSS .fits filename (or atpy.basetable.Table object)
        badpxl_tolerance : float 
                           Bad pixel fraction tolerance on the spectral interval of each filter. (Default: 0.5)
        
        
        Returns
        -------
        m_ab: numpy.ndarray dtype = [('m_ab', '<f8'), ('e_ab', '<f8')]
        
        See Also
        --------
        fromStarlight
        
        '''
        
        try:  # Try to import atpy
            import atpy
        except ImportError:
            MAGALException('Could not load atpy. Needed to convert from SDSS fits files')

        try:  # Is it already a atpy table?
            fits.data['loglam'] = 10 ** fits.data['loglam']
            self.obs_spec = fits.data.view(dtype=np.dtype(
                [('flux', '>f4'), ('wl', '>f4'), ('error', '>f4'), ('flag', '>i4'), ('or_mask', '>i4'), ('err', '>f4'),
                 ('sky', '>f4'), ('no', '>f4')]))
            self.model_spec = fits.data.view(dtype=np.dtype(
                [('no', '>f4'), ('wl', '>f4'), ('error', '>f4'), ('flag', '>i4'), ('or_mask', '>i4'), ('err', '>f4'),
                 ('sky', '>f4'), ('flux', '>f4')]))
        except AttributeError: # If doesn't work, read the file... 
            fits = atpy.Table(fits, hdu='COADD')
            fits.data['loglam'] = 10**fits.data['loglam']
            self.obs_spec = fits.data.view(dtype=np.dtype(
                [('flux', '>f4'), ('wl', '>f4'), ('error', '>f4'), ('flag', '>i4'), ('or_mask', '>i4'), ('err', '>f4'),
                 ('sky', '>f4'), ('no', '>f4')]))
            self.model_spec = fits.data.view(dtype=np.dtype(
                [('no', '>f4'), ('wl', '>f4'), ('error', '>f4'), ('flag', '>i4'), ('or_mask', '>i4'), ('err', '>f4'),
                 ('sky', '>f4'), ('flux', '>f4')]))
            
        
        self.obs_spec['flux'] *= 1e-17
        self.obs_spec['error'] *= 1e-17
        self.model_spec['flux'] *= 1e-17
        
        return spec2filterset(filterset, self.obs_spec, self.model_spec, badpxl_tolerance)
