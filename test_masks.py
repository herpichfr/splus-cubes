# -*- coding: utf-8 -*-
"""
 tool to produce calibrated cubes from S-PLUS images
 Herpich F. R. herpich@usp.br - 2021-03-09

 based/copied/stacked from Kadu's scripts
"""
from __future__ import print_function, division

#import os
#import glob
#import itertools
#import warnings
#from getpass import getpass
import os.path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import astropy.units as u
from astropy.stats import sigma_clipped_stats
#from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
#from astropy.wcs import FITSFixedWarning
from astropy.wcs.utils import pixel_to_skycoord as pix2sky
from astropy.wcs.utils import skycoord_to_pixel as sky2pix
#from astropy.nddata.utils import Cutout2D
#import astropy.constants as const
#from astropy.visualization import SqrtStretch
#from astropy.visualization.mpl_normalize import ImageNormalize
#from tqdm import tqdm
#import splusdata
#import pandas as pd
#from scipy.interpolate import RectBivariateSpline
from regions import PixCoord, CirclePixelRegion
#from regions import CircleSkyRegion

import sewpy

#from photutils import datasets
from photutils import DAOStarFinder
from photutils import CircularAperture

workdir = '/home/herpich/Documents/pos-doc/t80s/scubes/STRIPE82-0059/'
pathtofits = workdir + 'NGC1087/NGC1087_STRIPE82-0059_600x600_det_scimas.fits'

f = fits.open(pathtofits)
gain = f[1].header['GAIN']
fwhm = f[1].header['PSFFWHM']

outfile = workdir + "NGC1087/sexcat.fits"
params = ["NUMBER", "X_IMAGE", "Y_IMAGE", "KRON_RADIUS", "ELLIPTICITY",
          "THETA_IMAGE", "A_IMAGE", "B_IMAGE", "MAG_AUTO", "FWHM_IMAGE",
          "CLASS_STAR"]
config = {
          "DETECT_TYPE": "CCD",
          "DETECT_MINAREA": 4,
          "DETECT_THRESH" : 1.1,
          "ANALYSIS_THRESH": 3.0,
          "FILTER": "Y",
          "FILTER_NAME": os.path.join(workdir, "NGC1087/sex_data/tophat_3.0_3x3.conv"),
          "DEBLEND_NTHRESH": 64,
          "DEBLEND_MINCONT": 0.0002,
          "CLEAN": "Y",
          "CLEAN_PARAM": 1.0,
          "MASK_TYPE": "CORRECT",
          "PHOT_APERTURES": 5.45454545,
          "PHOT_AUTOPARAMS": '3.0,1.82',
          "PHOT_PETROPARAMS": '2.0,2.73',
          "PHOT_FLUXFRAC": '0.2,0.5,0.7,0.9',
          "SATUR_LEVEL": 1947.8720989,
          "MAG_ZEROPOINT": 20,
          "MAG_GAMMA": 4.0,
          "GAIN": gain,
          "PIXEL_SCALE": 0.55,
          "SEEING_FWHM": fwhm,
          "STARNNW_NAME": os.path.join(workdir, 'NGC1087/sex_data/default.nnw'),
          "BACK_SIZE": 256,
          "BACK_FILTERSIZE": 7,
          "BACKPHOTO_TYPE": "LOCAL",
          "BACKPHOTO_THICK": 48,
          "CHECKIMAGE_TYPE": "SEGMENTATION",
          "CHECKIMAGE_NAME": "segmentation.fits"
          }

sew = sewpy.SEW(config=config, sexpath="sextractor", params=params)
sewcat = sew(pathtofits)
sewpos = np.transpose((sewcat['table']['X_IMAGE'], sewcat['table']['Y_IMAGE']))
#apertures2 = CircularAperture(sewpos, r=3.)
#radius = zip(sewcat['table']['A_IMAGE'], sewcat['table']['B_IMAGE'])
#sewregions = [CirclePixelRegion(center=PixCoord(x, y), radius=max(z))
#              for (x, y), z in zip(sewpos, radius)
#             ]
radius = sewcat['table']['FWHM_IMAGE'] / 0.55
mask = sewcat['table']['CLASS_STAR'] > 0.05
mask &= (sewcat['table']['X_IMAGE'] > 100) & (sewcat['table']['X_IMAGE'] < 500)
mask &= (sewcat['table']['Y_IMAGE'] > 50) & (sewcat['table']['Y_IMAGE'] < 550)
sewregions = [CirclePixelRegion(center=PixCoord(x, y), radius=z)
              for (x, y), z in zip(sewpos[mask], radius[mask])
             ]

fdata = f[1].data
mean, median, std = sigma_clipped_stats(fdata, sigma=3.0)
print(('mean', 'median', 'std'))
print((mean, median, std))

## starting to search the stars
#daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
#sources = daofind(fdata - median)
##for col in sources.colnames:
##    sources[col].info.format = '%.8g'  # for consistent table output
#
#print(sources)
#
#positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
#cutr = 0.04
#mask = (sources['roundness2'] > -cutr) & (sources['roundness2'] < cutr)
##apertures = CircularAperture(positions[mask], r=4.)
#apertures = CircularAperture(positions, r=4.)
##norm = ImageNormalize(stretch=SqrtStretch())

#########################
daofind = DAOStarFinder(fwhm=3.0, sharplo=0.5, sharphi=0.9,
                        roundlo=-0.25, roundhi=0.25, threshold=5.*std)
sources2 = daofind(fdata)
positions2 = np.transpose((sources2['xcentroid'], sources2['ycentroid']))
apertures2 = CircularAperture(positions2, r=3.)


#########################
wcs = WCS(f[1].header)
xp, yp = np.arange(fdata.shape[0]), np.arange(fdata.shape[1])
nsky = pix2sky(xp, yp, wcs, origin=0, mode=u'wcs')
ralims = np.min(nsky.ra), np.max(nsky.ra)
delims = np.min(nsky.dec), np.max(nsky.dec)

sext = fits.open(os.path.join(workdir, 'STRIPE82-0059.fits'))
sex_coords = SkyCoord(ra=sext[1].data['RA'], dec=sext[1].data['DEC'], unit=(u.deg, u.deg))
mask = (sex_coords.ra > ralims[0]) & (sex_coords.ra < ralims[1])
mask &= (sex_coords.dec > delims[0]) & (sex_coords.dec < delims[1])
mask &= sext[1].data['CLASS_STAR'] > 0.25

sex_coords = sex_coords[mask]
#radius = sext[1].data['KRON_RADIUS'][mask]
radius = (sext[1].data['FWHM'][mask] / 0.55)
#radius = [ max(a) for a in zip(sext[1].data['A'][mask], sext[1].data['B'][mask])]
sexpos = np.transpose(sky2pix(sex_coords, wcs, origin=0, mode='wcs'))
#sexpos = sky2pix(sex_coords, wcs, origin=0, mode='wcs')
#sexaper = CircularAperture(sexpos, r=5)
regions = [CirclePixelRegion(center=PixCoord(x, y), radius=z)
           for (x, y), z in zip(sexpos, radius)
           ]

plt.ion()
#plt.imshow(fdata, cmap='Greys', origin='lower', norm=norm,
#           interpolation='nearest')
ax1 = plt.subplot(221, projection=wcs)
ax1.imshow(fdata, cmap='Greys', origin='lower', vmin=-1, vmax=3.5)
#apertures.plot(color='blue', lw=1.5, alpha=0.5)
apertures2.plot(color='y', lw=1.5, alpha=0.5)
for region in regions:
    region.plot(ax=ax1, color='red')
for region in sewregions:
    region.plot(ax=ax1, color='g')
ax1.set_xlabel('RA')
ax1.set_ylabel('Dec')

s = fits.open(os.path.join(workdir, 'NGC1087/segmentation.fits'))
sdata = s[0].data
ax2 = plt.subplot(222, projection=wcs)
ax2.imshow(sdata, origin='lower', interpolation='none', cmap='nipy_spectral_r', vmin=1)
apertures2.plot(color='y', lw=1.5, alpha=0.5)
for region in regions:
    region.plot(ax=ax2, color='red')
for region in sewregions:
    region.plot(ax=ax2, color='g')
ax2.set_xlabel('RA')
#ax2.set_ylabel('Dec')

ax3 = plt.subplot(223)
img = mpimg.imread(os.path.join(workdir, 'NGC1087/NGC1087.png'))
ax3.imshow(img, origin='lower')


ax4 = plt.subplot(224, projection=wcs)
galnums = [1, 2, 3, 5, 7, 8, 29, 28, 27, 119, 71, 6, 107, 88, 4, 83, 26, 115,
           112, 4, 61, 80, 25]
gmask = np.zeros(sdata.shape)
for num in galnums:
    gmask[sdata == num] = 1
maskeddata = fdata * gmask
apertures2.plot(color='y', lw=1.5, alpha=0.5)
for region in regions:
    region.plot(ax=ax4, color='red')
for sregion in sewregions:
    sregion.plot(ax=ax4, color='g')
    mask = sregion.to_mask()
    maskeddata[mask.bbox.slices] *= 1 - mask.data
ax4.imshow(maskeddata, cmap='Greys', origin='lower', vmin=-1, vmax=3.5)

ax4.set_xlabel('RA')
ax4.set_ylabel('Dec')


plt.tight_layout()

plt.show()
