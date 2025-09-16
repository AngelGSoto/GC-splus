"""
splus_photometry_final.py

Versión final del pipeline de fotometría­a para imágenes S-PLUS (incluye soporte para .fits.fz
comprimidos, weight maps, estimación robusta del fondo por anillo, cálculo de errores y calibración
simple contra Pan-STARRS/otro catálogo de referencia).

Uso básico:
    python splus_photometry_final.py --catalog catalogo_ngc5128.csv --images-dir /ruta/CenA01 

Salida: ngc5128_splus_photometry_3arcsec_calibrated.csv (por defecto)

Dependencias: astropy, numpy, pandas, photutils, astropy.stats, tqdm, scipy, astroquery (opcional)

Notas:
- Lee archivos comprimidos .fits.fz directamente con astropy.
- Asume que las weight maps (si existen) son inverse-variance por defecto. Si tu weight map tiene
  otra convención, usa el argumento --weight-is-invvar False y ajusta manualmente.
- Calibración: si se indica --auto-calibrate, el script intenta bajar Pan-STARRS (astroquery) y
  calcula ZP por filtro con una regresión simple (opcional tÃ©rmino de color).

Autor: Luis 
"""

import os
import glob
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus
from astropy.stats import sigma_clipped_stats
from tqdm import tqdm

# Opcional: astroquery para calibraciÃ³n automática
try:
    from astroquery.vizier import Vizier
    ASTROQUERY_AVAILABLE = True
except Exception:
    ASTROQUERY_AVAILABLE = False


class SPLUSPhotometry:
    def __init__(self, catalog_path, images_dir='../', filters=None,
                 default_zeropoints=None, pixel_scale=None,
                 weight_is_invvar=True, aper_arcsec=3.0,
                 ann_in_arcsec=6.0, ann_out_arcsec=9.0,
                 snr_min=3.0, match_radius_arcsec=1.0):
        self.catalog = pd.read_csv(catalog_path)
        # Si no pasa filtros, intentamos deducirlos por lista de archivos
        self.filters = filters
        self.images_dir = images_dir
        # Zeropoints por defecto (si quieres usarlos sin calibraciÃ³n)
        self.default_zeropoints = default_zeropoints or {}
        self.pixel_scale_override = pixel_scale
        self.weight_is_invvar = weight_is_invvar
        self.aper_arcsec = aper_arcsec
        self.ann_in_arcsec = ann_in_arcsec
        self.ann_out_arcsec = ann_out_arcsec
        self.snr_min = snr_min
        self.match_radius = match_radius_arcsec * u.arcsec

    def _find_filters_from_files(self):
        # Busca archivos tipo CenA01_FXXX.fits* en images_dir
        pattern = os.path.join(self.images_dir, 'CenA01_F*.fits*')
        files = sorted(glob.glob(pattern))
        filters = set()
        for f in files:
            base = os.path.basename(f)
            # esperar formato CenA01_F378.fits.fz o CenA01_F378.fits
            parts = base.split('_')
            if len(parts) >= 2 and parts[1].startswith('F'):
                fname = parts[1]
                # obtener Fxxx
                fcode = fname.split('.')[0]
                filters.add(fcode)
        return sorted(list(filters))

    def _get_pixel_scale_from_header(self, hdr):
        if self.pixel_scale_override is not None:
            return float(self.pixel_scale_override)
        if 'PIXSCALE' in hdr:
            return float(hdr['PIXSCALE'])
        if 'PIXSCL' in hdr:
            return float(hdr['PIXSCL'])
        if 'CDELT1' in hdr:
            return abs(float(hdr['CDELT1'])) * 3600.0
        if 'CD1_1' in hdr:
            return abs(float(hdr['CD1_1'])) * 3600.0
        return None

    def _open_image_and_weight(self, filter_name):
        # buscar archivo de imagen (preferir .fits.fz o .fits)
        patterns = [
            os.path.join(self.images_dir, f'CenA01_{filter_name}.fits.fz'),
            os.path.join(self.images_dir, f'CenA01_{filter_name}.fits'),
            os.path.join(self.images_dir, f'CenA01_{filter_name}.fz')
        ]
        image_fn = None
        for p in patterns:
            if os.path.exists(p):
                image_fn = p
                break
        if image_fn is None:
            raise FileNotFoundError(f"No se encontraron imagen para {filter_name} en {self.images_dir}")

        # weight map
        weight_patterns = [
            os.path.join(self.images_dir, f'CenA01_{filter_name}.weight.fits.fz'),
            os.path.join(self.images_dir, f'CenA01_{filter_name}.weight.fits'),
            os.path.join(self.images_dir, f'CenA01_{filter_name}.weight.fz')
        ]
        weight_fn = None
        for p in weight_patterns:
            if os.path.exists(p):
                weight_fn = p
                break

        # abrir imagen
        with fits.open(image_fn) as hdul:
            # si es un compressed image presentará COMPRESSED_IMAGE -> astropy maneja
            data = hdul[0].data.astype(float)
            hdr = hdul[0].header
            wcs = WCS(hdr)

        weight_data = None
        if weight_fn is not None:
            with fits.open(weight_fn) as hdulw:
                weight_data = hdulw[0].data.astype(float)

        return data, hdr, wcs, weight_data, image_fn, weight_fn

    def _weight_to_sigma(self, weight_data, fallback_value=1.0):
        # asumir default: weight = inverse variance
        if weight_data is None:
            return None
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.weight_is_invvar:
                sigma = 1.0 / np.sqrt(weight_data)
            else:
                # si weight no es inv var, intentar interpretar como exposure map -> sigma ~= sqrt(N)/gain
                sigma = 1.0 / np.sqrt(np.abs(weight_data))
        finite = np.isfinite(sigma)
        if not np.any(finite):
            sigma[:] = fallback_value
            return sigma
        sigma[~finite] = np.nanmax(sigma[finite])
        return sigma

    def measure_photometry(self, auto_calibrate=False, panstarrs_radius_deg=0.2, color_term=False):
        results = []
        # deducir filtros si necesario
        if self.filters is None:
            self.filters = self._find_filters_from_files()

        for filter_name in tqdm(self.filters, desc='Filters'):
            try:
                data, hdr, wcs, weight_data, image_fn, weight_fn = self._open_image_and_weight(filter_name)
                pixel_scale = self._get_pixel_scale_from_header(hdr) or 0.55
                exptime = float(hdr.get('EXPTIME', 1.0))
                sat = hdr.get('SATURATE', None)

                aper_pix = self.aper_arcsec / pixel_scale
                ann_in_pix = self.ann_in_arcsec / pixel_scale
                ann_out_pix = self.ann_out_arcsec / pixel_scale

                sigma_map = self._weight_to_sigma(weight_data)  # sigma per pixel if available

                # calcular pix positions para todo el catalogo
                coords = SkyCoord(ra=self.catalog['ra'].values * u.deg, dec=self.catalog['dec'].values * u.deg)
                x_all, y_all = wcs.world_to_pixel(coords)

                for i, src in self.catalog.iterrows():
                    x = x_all[i]
                    y = y_all[i]
                    # margen para anillo
                    margin = int(np.ceil(ann_out_pix)) + 1
                    if (x < margin or x >= data.shape[1] - margin or
                        y < margin or y >= data.shape[0] - margin):
                        continue

                    # construir aperturas
                    aperture = CircularAperture((x, y), r=aper_pix)
                    ann = CircularAnnulus((x, y), r_in=ann_in_pix, r_out=ann_out_pix)

                    # sumar flux
                    phot = aperture_photometry(data, aperture)
                    ann_phot = aperture_photometry(data, ann)
                    raw_sum = float(np.asarray(phot['aperture_sum'])[0])
                    ann_sum = float(np.asarray(ann_phot['aperture_sum'])[0])

                    # extraer pixeles del anillo para sigma-clipped stats
                    ann_mask = ann.to_mask(method='center')
                    if ann_mask is None:
                        continue
                    ann_data = ann_mask.multiply(data)
                    ann_vals = ann_data[ann_data != ann_mask.fill_value]
                    if ann_vals.size > 0:
                        bkg_mean, bkg_med, bkg_std = sigma_clipped_stats(ann_vals, sigma=3.0, maxiters=5)
                    else:
                        bkg_mean = ann_sum / ann.area()
                        bkg_std = 0.0

                    # fondo en la apertura
                    bkg_sum = bkg_mean * aperture.area()
                    bkg_error = bkg_std * np.sqrt(aperture.area())

                    # error de la suma por sigma_map
                    phot_mask = aperture.to_mask(method='center')
                    phot_sigma_vals = None
                    if sigma_map is not None:
                        phot_sigma = phot_mask.multiply(sigma_map)
                        phot_sigma_vals = phot_sigma[phot_sigma != phot_mask.fill_value]

                    if phot_sigma_vals is not None and phot_sigma_vals.size > 0:
                        phot_sum_err = np.sqrt(np.sum(phot_sigma_vals**2))
                    else:
                        phot_sum_err = bkg_error

                    final_sum = raw_sum - bkg_sum
                    final_error = np.sqrt(phot_sum_err**2 + bkg_error**2)

                    # comprobar final_sum
                    if (not np.isfinite(final_sum)) or final_sum <= 0 or final_error <= 0:
                        continue

                    # magnitud instrumental (dividiendo por EXPTIME)
                    m_inst = -2.5 * np.log10(final_sum / exptime)

                    snr = final_sum / final_error
                    if snr < self.snr_min:
                        continue

                    result = {
                        'source_id': src['id'],
                        'filter': filter_name,
                        'ra': src['ra'],
                        'dec': src['dec'],
                        'x': float(x),
                        'y': float(y),
                        'flux': float(final_sum),
                        'flux_err': float(final_error),
                        'm_inst': float(m_inst),
                        'snr': float(snr),
                        'aper_arcsec': self.aper_arcsec,
                        'aper_pix': float(aper_pix),
                        'exptime': float(exptime),
                        'image_fn': image_fn
                    }
                    results.append(result)

            except Exception as e:
                print(f"Error procesando filtro {filter_name}: {e}")
                continue

        results_df = pd.DataFrame(results)

        # calibrar si se pide
        if auto_calibrate:
            if not ASTROQUERY_AVAILABLE:
                print("astroquery no está disponible: no se puede auto-calibrar. Instala astroquery.")
            else:
                print("Intentando calibración automática con Pan-STARRS (si hay cobertura)...")
                zps = self._auto_calibrate_all_filters(results_df, radius_deg=panstarrs_radius_deg, color_term=color_term)
                # aplicar zp a magnitudes
                for f, zpinfo in zps.items():
                    mask = results_df['filter'] == f
                    if 'ZP' in zpinfo and np.isfinite(zpinfo['ZP']):
                        results_df.loc[mask, 'mag_cal'] = results_df.loc[mask, 'm_inst'] + zpinfo['ZP']
                        results_df.loc[mask, 'zp_used'] = float(zpinfo['ZP'])
                    else:
                        # si no hay zp, intentar usar default
                        if f in self.default_zeropoints:
                            results_df.loc[mask, 'mag_cal'] = results_df.loc[mask, 'm_inst'] + float(self.default_zeropoints[f])
                            results_df.loc[mask, 'zp_used'] = float(self.default_zeropoints[f])
                        else:
                            results_df.loc[mask, 'mag_cal'] = np.nan
                            results_df.loc[mask, 'zp_used'] = np.nan
        else:
            # usar defaults si existen
            if len(self.default_zeropoints) > 0:
                for f, zp in self.default_zeropoints.items():
                    mask = results_df['filter'] == f
                    results_df.loc[mask, 'mag_cal'] = results_df.loc[mask, 'm_inst'] + float(zp)
                    results_df.loc[mask, 'zp_used'] = float(zp)
            else:
                results_df['mag_cal'] = np.nan
                results_df['zp_used'] = np.nan

        return results_df

    def _query_panstarrs(self, center_coord, radius_deg=0.2):
        # consulta Pan-STARRS PS1 en VizieR (II/349/ps1)
        Vizier.ROW_LIMIT = 50000
        try:
            cat = Vizier.query_region(center_coord, radius=radius_deg * u.deg, catalog='II/349/ps1')
        except Exception as e:
            print(f"Error consultando Vizier/Pan-STARRS: {e}")
            return None
        if len(cat) == 0:
            return None
        df = cat[0].to_pandas()
        # renombrar columnas que nos interesan
        # RAJ2000, DEJ2000, gmag, rmag, imag
        if 'gmag' in df.columns and 'rmag' in df.columns:
            return df
        # si no viene con nombres esperados, devolver tabla sin cambios
        return df

    def _auto_calibrate_all_filters(self, results_df, radius_deg=0.2, color_term=False):
        zps = {}
        # group by image to pick a center for query
        for f in results_df['filter'].unique():
            df_f = results_df[results_df['filter'] == f]
            if len(df_f) < 6:
                print(f"Filtro {f}: pocas fuentes medidas ({len(df_f)}) -> no se calibra automáticamente")
                zps[f] = {'ZP': np.nan, 'rms': np.nan, 'n': len(df_f)}
                continue
            # elegir centro medio para consulta
            center = SkyCoord(ra=df_f['ra'].mean()*u.deg, dec=df_f['dec'].mean()*u.deg)
            ref = self._query_panstarrs(center, radius_deg=radius_deg)
            if ref is None or len(ref) == 0:
                print(f"Filtro {f}: no hay Pan-STARRS en la región ({center.to_string()})")
                zps[f] = {'ZP': np.nan, 'rms': np.nan, 'n': len(df_f)}
                continue
            # crossmatch: match sources measured to ref catalog
            ref_coords = SkyCoord(ra=ref['RAJ2000'].values*u.deg, dec=ref['DEJ2000'].values*u.deg)
            meas_coords = SkyCoord(ra=df_f['ra'].values*u.deg, dec=df_f['dec'].values*u.deg)
            idx, d2d, _ = meas_coords.match_to_catalog_sky(ref_coords)
            sep_mask = d2d < self.match_radius
            if not np.any(sep_mask):
                print(f"Filtro {f}: no se encontraron matches con Pan-STARRS dentro de {self.match_radius}")
                zps[f] = {'ZP': np.nan, 'rms': np.nan, 'n': len(df_f)}
                continue
            matched_meas = df_f.iloc[np.where(sep_mask)[0]]
            matched_ref = ref.iloc[idx[sep_mask]]

            # elegir banda de referencia: usar rmag si existe, si no usar gmag o imag
            mag_ref_col = None
            for c in ['rmag', 'gmag', 'imag']:
                if c in matched_ref.columns:
                    mag_ref_col = c
                    break
            if mag_ref_col is None:
                print(f"Filtro {f}: Pan-STARRS no trae g/r/i para los objetos matched.")
                zps[f] = {'ZP': np.nan, 'rms': np.nan, 'n': len(df_f)}
                continue

            m_inst = matched_meas['m_inst'].values
            m_ref = matched_ref[mag_ref_col].values

            # filtrar rango de magnitud
            mask_good = np.isfinite(m_ref) & np.isfinite(m_inst) & (m_ref > 14.0) & (m_ref < 19.0)
            if np.sum(mask_good) < 6:
                print(f"Filtro {f}: pocos matches buenos ({np.sum(mask_good)}) para calcular ZP")
                zps[f] = {'ZP': np.nan, 'rms': np.nan, 'n': np.sum(mask_good)}
                continue

            m_inst_sel = m_inst[mask_good]
            m_ref_sel = m_ref[mask_good]

            if color_term and ('gmag' in matched_ref.columns and 'rmag' in matched_ref.columns):
                color = (matched_ref['gmag'] - matched_ref['rmag']).values[mask_good]
                # ajustar m_ref - m_inst = ZP + CT * color
                A = np.vstack([np.ones_like(color), color]).T
                y = m_ref_sel - m_inst_sel
                coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
                ZP = float(coeffs[0])
                CT = float(coeffs[1])
                residuals = y - (ZP + CT * color)
                rms = float(np.std(residuals))
                zps[f] = {'ZP': ZP, 'CT': CT, 'rms': rms, 'n': int(np.sum(mask_good))}
                print(f"Filtro {f}: ZP={ZP:.3f}, CT={CT:.3f}, RMS={rms:.3f}, N={int(np.sum(mask_good))}")
            else:
                zp_vals = (m_ref_sel - m_inst_sel)
                ZP = float(np.median(zp_vals))
                rms = float(np.std(zp_vals))
                zps[f] = {'ZP': ZP, 'rms': rms, 'n': int(np.sum(mask_good))}
                print(f"Filtro {f}: ZP={ZP:.3f}, RMS={rms:.3f}, N={int(np.sum(mask_good))}")

        return zps


# ---------------------------
# CLI para ejecutar el script
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Pipeline fotometrétrico para S-PLUS (3" aperture)')
    p.add_argument('--catalog', required=True, help='CSV con columnas id,ra,dec')
    p.add_argument('--images-dir', default='.', help='Directorio con archivos CenA01_Fxxx.fits.fz y weight')
    p.add_argument('--filters', nargs='*', default=None, help='Lista de filtros a procesar (ej: F378 F395 ...)')
    p.add_argument('--auto-calibrate', action='store_true', help='Calibrar automáticamente con Pan-STARRS via astroquery')
    p.add_argument('--color-term', action='store_true', help='Incluir término de color en calibración (si hay colores)')
    p.add_argument('--output', default='ngc5128_splus_photometry_3arcsec_calibrated.csv', help='Nombre de fichero de salida CSV')
    p.add_argument('--weight-is-invvar', action='store_true', default=True, help='Indica que la weight map es inverse-variance (default True)')
    return p.parse_args()


def main():
    args = parse_args()
    sp = SPLUSPhotometry(args.catalog, images_dir=args.images_dir, filters=args.filters,
                         weight_is_invvar=args.weight_is_invvar)
    df = sp.measure_photometry(auto_calibrate=args.auto_calibrate, color_term=args.color_term)
    df.to_csv(args.output, index=False)
    print(f"Guardado: {args.output}. Filas: {len(df)}")

    # resumen por filtro
    for f in sorted(df['filter'].unique()):
        sub = df[df['filter'] == f]
        if len(sub) == 0:
            continue
        mean_snr = sub['snr'].mean()
        n = len(sub)
        zp_used = sub['zp_used'].dropna().unique()
        zp_str = zp_used[0] if len(zp_used) > 0 else 'none'
        print(f"{f}: N={n}, SNR_mean={mean_snr:.2f}, ZP_used={zp_str}")


if __name__ == '__main__':
    main()
