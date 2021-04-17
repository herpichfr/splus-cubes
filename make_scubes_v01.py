# -*- coding: utf-8 -*-
"""
 tool to produce calibrated cubes from S-PLUS images
 Herpich F. R. herpich@usp.br - 2021-03-09

 based/copied/stacked from Kadu's scripts
"""
from __future__ import print_function, division

import os
import glob
import itertools
import warnings
# from getpass import getpass

import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.io import fits, ascii
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
import astropy.constants as const
from astropy.stats import sigma_clipped_stats
from astropy.wcs import FITSFixedWarning
from astropy.wcs.utils import pixel_to_skycoord as pix2sky
from astropy.wcs.utils import skycoord_to_pixel as sky2pix
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from astropy.visualization import make_lupton_rgb
from tqdm import tqdm
import sewpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from regions import PixCoord, CirclePixelRegion

from photutils import DAOStarFinder
from photutils import CircularAperture

warnings.simplefilter('ignore', category=FITSFixedWarning)

lastversion = '0.1'
moddate = '2021-03-09'

initext = """
    ===================================================================
                    make_scubes - v%s - %s
             This version is not yet completely debugged
             In case of crashes, please send the log to:
                Herpich F. R. fabiorafaelh@gmail.com
    ===================================================================
    """ % (lastversion, moddate)


class SCubes(object):
    def __init__(self):
        """basic definitions"""

        # self.names = np.array(['NGC1087', 'NGC1090'])
        # self.coords = [['02:46:25.15', '-00:29:55.45'],
        #               ['02:46:33.916', '-00:14:49.35']]
        self.galaxies = np.array(['NGC1087'])
        self.coords = [['02:46:25.15', '-00:29:55.45']]
        self.tiles = ['STRIPE82-0059']
        self.sizes = np.array([100])
        self.work_dir: str = '/home/herpich/Documents/pos-doc/t80s/scubes/'
        # self.tile_dir = os.path.join(self.work_dir, self.fields[0])
        # self.gal_dir = os.path.join(self.work_dir, self.galaxies[0])
        self.data_dir = '/home/herpich/Documents/pos-doc/t80s/scubes/data/'
        self.zpcorr_dir = '/home/herpich/Documents/pos-doc/t80s/scubes/data/zpcorr_idr3/'

        # from Kadu's context
        self.ps = 0.55 * u.arcsec / u.pixel
        self.bands = ['U', 'F378', 'F395', 'F410', 'F430', 'G', 'F515', 'R',
                      'F660', 'I', 'F861', 'Z']
        self.narrow_bands = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.broad_bands = ['U', 'G', 'R', 'I', 'Z']
        self.bands_names = {'U': "$u$", 'F378': "$J378$", 'F395': "$J395$",
                            'F410': "$J410$", 'F430': "$J430$", 'G': "$g$",
                            'F515': "$J515$", 'R': "$r$", 'F660': "$J660$",
                            'I': "$i$", 'F861': "$J861$", 'Z': "$z$"}
        self.wave_eff = {"F378": 3770.0, "F395": 3940.0, "F410": 4094.0,
                         "F430": 4292.0, "F515": 5133.0, "F660": 6614.0,
                         "F861": 8611.0, "G": 4751.0, "I": 7690.0, "R": 6258.0,
                         "U": 3536.0, "Z": 8831.0}
        self.exptimes = {"F378": 660, "F395": 354, "F410": 177,
                         "F430": 171, "F515": 183, "F660": 870, "F861": 240,
                         "G": 99, "I": 138, "R": 120, "U": 681,
                         "Z": 168}

        ## matplotlib settings (from context)
        # plt.style.context("seaborn-paper")
        # plt.rcParams["text.usetex"] = True
        # plt.rcParams["font.family"] = "serif"
        # plt.rcParams['font.serif'] = 'Computer Modern'
        # plt.rcParams["xtick.direction"] = "in"
        # plt.rcParams["ytick.direction"] = "in"
        # plt.rcParams["xtick.minor.visible"] = True
        # plt.rcParams["ytick.minor.visible"] = True
        # plt.rcParams["xtick.top"] = True
        # plt.rcParams["ytick.right"] = True

    def check_infoot(self, save_output=False):
        pass
        #foot = pd.read_csv(os.path.join(self.data_dir, "tiles_new_status.csv"))
        #field_coords = SkyCoord(foot["RA"], foot["DEC"],
        #                        unit=(u.hourangle, u.degree))
        #inra = np.transpose(self.coords)[0]
        #inde = np.transpose(self.coords)[1]
        #c1 = SkyCoord(ra=inra, dec=inde, unit=(u.hour, u.deg), frame='icrs')
        #idx, d2d, d3d = c1.match_to_catalog_sky(field_coords)
        #max_sep = 1.0 * u.deg
        #sep_constraint = d2d < max_sep
        #status = foot['STATUS'][idx]
        #status[~sep_constraint] = -10
        #tiles = self.fields
        #tile[~sep_constraint] = '-'
        #for i in range(len(tile.values)):
        #    if len(tile.values[i].split('_')) > 1:
        #        tile.values[i] = tile.values[i].split('_')[0] + '-' + tile.values[i].split('_')[1]
        #result = zip(self.names, self.coords, tile)
        #print('matches found:')
        #for a in result:
        #    print(a)
        # cols = [self.names,
        #         np.transpose(self.coords)[0],
        #         np.transpose(self.coords)[1],
        #         np.array(self.fields)]
        # names = ['NAME', 'RA', 'DEC', 'TILE']
        # t = Table(cols, names=names)
        # if save_output:
        #     outname = self.work_dir + './objects_tilematched.csv'
        #     print('saving file', outname)
        #     t.write(outname, format='csv', overwrite=True)
        # return t

    def make_stamps_splus(self, redo=False, img_types=None, bands=None,
                          savestamps=True):
        """  Produces stamps of objects in S-PLUS from a table of names,
        coordinates.

        Parameters
        ----------
        names: np.array
            Array containing the name/id of the objects.

        coords: astropy.coordinates.SkyCoord
            Coordinates of the objects.

        size: np.array
            Size of the stamps (in pixels)

        outdir: str
            Path to the output directory. If not given, stamps are saved in the
            current directory.

        redo: bool
            Option to rewrite stamp in case it already exists.

        img_types: list
            List containing the image types to be used in stamps. Default is [
            "swp', "swpweight"] to save both the images and the weight images
            with uncertainties.

        bands: list
            List of bands for the stamps. Defaults produces stamps for all
            filters in S-PLUS. Options are 'U', 'F378', 'F395', 'F410', 'F430', 'G',
            'F515', 'R', 'F660', 'I', 'F861', and 'Z'.

        savestamps: boolean
            If True, saves the stamps in the directory outdir/object.
            Default is True.


        """
        names = np.atleast_1d(self.galaxies)
        sizes = np.atleast_1d(self.sizes)
        if len(sizes) == 1:
            sizes = np.full(len(names), sizes[0])
        sizes = sizes.astype(np.int)
        img_types = ["swp", "swpweight"] if img_types is None else img_types
        work_dir = os.getcwd() if self.work_dir is None else self.work_dir
        tile_dir = os.getcwd() if self.work_dir is None else os.path.join(self.work_dir, self.tiles[0])
        header_keys = ["OBJECT", "FILTER", "EXPTIME", "GAIN", "TELESCOP",
                       "INSTRUME", "AIRMASS"]
        bands = self.bands if bands is None else bands

        # Selecting tiles from S-PLUS footprint
        cols = [self.galaxies,
                np.transpose(self.coords)[0],
                np.transpose(self.coords)[1],
                np.array(self.tiles)]
        names = ['NAME', 'RA', 'DEC', 'TILE']
        fields = Table(cols, names=names)

        # Producing stamps
        for field in tqdm(fields, desc="Fields"):
            field_name = field["TILE"]
            fnames = [field['NAME']]
            fcoords = SkyCoord(ra=field['RA'], dec=field['DEC'],
                               unit=(u.hour, u.deg))  # self.coords[idx]
            fsizes = np.array(sizes)[fields['NAME'] == fnames]
            stamps = dict((k, []) for k in img_types)
            for img_type in tqdm(img_types, desc="Data types", leave=False,
                                 position=1):
                for band in tqdm(bands, desc="Bands", leave=False, position=2):
                    #tile_dir = os.path.join(tile_dir, field["TILE"], band)
                    fitsfile = os.path.join(tile_dir, "{}_{}.fits".format(
                        field["TILE"], img_type))
                    try:
                        header = fits.getheader(fitsfile)
                        data = fits.getdata(fitsfile)
                    except:  # os.path.isfile(os.path.join(tile_dir + field['TILE'] + '_' + band + '_' + img_type + '.fz')):
                        fzfile = os.path.join(tile_dir, field['TILE'] + '_' + band + '_' + img_type + '.fz')
                        f = fits.open(fzfile)[1]
                        header = f.header
                        data = f.data
                    else:
                        failedfile = os.path.join(tile_dir, "{}_{}.fits".format(field["TILE"], img_type))
                        Warning('file %s not found' % failedfile)
                    wcs = WCS(header)
                    xys = wcs.all_world2pix(fcoords.ra, fcoords.dec, 1)
                    for i, (name, size) in enumerate(tqdm(zip(fnames, fsizes),
                                                          desc="Galaxies", leave=False, position=3)):
                        galdir = os.path.join(work_dir, name)
                        output = os.path.join(galdir,
                                              "{0}_{1}_{2}_{3}x{3}_{4}.fits".format(
                                                  name, field_name, band, size, img_type))
                        if os.path.exists(output) and not redo:
                            continue
                        try:
                            cutout = Cutout2D(data, position=fcoords,
                                              size=size * u.pixel, wcs=wcs)
                        except ValueError:
                            continue
                        if np.all(cutout.data == 0):
                            continue
                        hdu = fits.ImageHDU(cutout.data)
                        for key in header_keys:
                            if key in header:
                                hdu.header[key] = header[key]
                        hdu.header["TILE"] = hdu.header["OBJECT"]
                        hdu.header["OBJECT"] = name
                        if img_type == "swp":
                            hdu.header["NCOMBINE"] = (header["NCOMBINE"], "Number of combined images")
                            hdu.header["EFFTIME"] = (header["EFECTIME"], "Effective exposed total time")
                        if "HIERARCH OAJ PRO FWHMMEAN" in header:
                            hdu.header["PSFFWHM"] = header["HIERARCH OAJ PRO FWHMMEAN"]
                        hdu.header["X0TILE"] = (xys[0].item(), "Location in tile")
                        hdu.header["Y0TILE"] = (xys[1].item(), "Location in tile")
                        hdu.header.update(cutout.wcs.to_header())
                        hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
                        if savestamps:
                            if not os.path.exists(galdir):
                                os.mkdir(galdir)
                            print('saving', output)
                            hdulist.writeto(output, overwrite=True)
                        else:
                            print('To be implemented')
                            # stamps[img_type].append(hdulist)
        # if savestamps:
        #    return
        # else:
        #    return stamps

    def make_det_stamp(self, savestamp: bool=True):
        """Cut the a stamp of the same size as the cube stamps to be used for the mask"""

        names = np.atleast_1d(self.galaxies)
        sizes = np.atleast_1d(self.sizes)
        if len(sizes) == 1:
            sizes = np.full(len(names), sizes[0])
        sizes = sizes.astype(np.int)
        outdir = os.getcwd() if self.work_dir is None else self.work_dir
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        tile_dir = os.getcwd() if self.work_dir is None else os.path.join(self.work_dir, self.tiles[0])
        header_keys = ["OBJECT", "FILTER", "EXPTIME", "GAIN", "TELESCOP",
                       "INSTRUME", "AIRMASS"]

        # Selecting tiles from S-PLUS footprint
        # fields = self.check_infoot() if self.fields is None else self.fields
        cols = [self.galaxies,
                np.transpose(self.coords)[0],
                np.transpose(self.coords)[1],
                np.array(self.tiles)]
        names = ['NAME', 'RA', 'DEC', 'TILE']
        fields = Table(cols, names=names)

        # Producing stamps
        for field in fields:
            field_name = field["TILE"]
            fnames = [field['NAME']]
            fcoords = SkyCoord(ra=field['RA'], dec=field['DEC'],
                               unit=(u.hour, u.deg))  # self.coords[idx]
            fsizes = np.array(sizes)[fields['NAME'] == fnames]
            for i, (name, size) in enumerate(zip(fnames, fsizes)):
                galdir = os.path.join(outdir, name)
                if not os.path.isdir(galdir):
                    os.makedirs(galdir)
                doutput = os.path.join(galdir,
                                       "{0}_{1}_{2}x{2}_{3}.fits".format(
                                           name, field_name, size, 'det_scimas'))
                if not os.path.isfile(doutput):
                    try:
                        d = fits.open(os.path.join(tile_dir, field_name + '_det_scimas.fits'))[0]
                    except:
                        d = fits.open(os.path.join(tile_dir, field_name + '_det_scimas.fits.fz'))[1]
                    dheader = d.header
                    ddata = d.data
                    wcs = WCS(dheader)
                    xys = wcs.all_world2pix(fcoords.ra, fcoords.dec, 1)
                    dcutout = Cutout2D(ddata, position=fcoords,
                                       size=size * u.pixel, wcs=wcs)
                    hdu = fits.ImageHDU(dcutout.data)
                    for key in header_keys:
                        if key in dheader:
                            hdu.header[key] = dheader[key]
                    hdu.header["TILE"] = hdu.header["OBJECT"]
                    hdu.header["OBJECT"] = name
                    if "HIERARCH OAJ PRO FWHMMEAN" in dheader:
                        hdu.header["PSFFWHM"] = dheader["HIERARCH OAJ PRO FWHMMEAN"]
                    hdu.header["X0TILE"] = (xys[0].item(), "Location in tile")
                    hdu.header["Y0TILE"] = (xys[1].item(), "Location in tile")
                    hdu.header.update(dcutout.wcs.to_header())
                    hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
                    if savestamp:
                        print('saving', doutput)
                        hdulist.writeto(doutput, overwrite=True)

    def get_zps(self, tile: str=None, tile_dir: str=None):
        """ Load all tables with zero points for iDR3. """
        _dir = self.data_dir if tile_dir is None else tile_dir
        tile = self.tiles[0] if tile is None else tile
        # tables = []
        # for fname in os.listdir(_dir):
        #    filename = os.path.join(_dir, fname)
        #    #data = np.genfromtxt(filename, dtype=None)
        #    data = ascii.read(filename)
        #    h = [s.replace("SPLUS_", "") for s in data.keys()]
        #    table = Table(data, names=h)
        #    tables.append(table)
        # zptable = np.vstack(tables)
        filename = os.path.join(_dir, tile + '_ZP.cat')
        data = ascii.read(filename)
        h = [s.replace("SPLUS_", "") for s in data.keys()]
        zptab = Table(data, names=h)

        return zptab

    def get_zp_correction(self):
        """ Get corrections of zero points for location in the field. """
        x0, x1, nbins = 0, 9200, 16
        xgrid = np.linspace(x0, x1, nbins + 1)
        zpcorr = {}
        for band in self.bands:
            corrfile = os.path.join(self.zpcorr_dir, 'SPLUS_' + band + '_offsets_grid.npy')
            corr = np.load(corrfile)
            zpcorr[band] = RectBivariateSpline(xgrid, xgrid, corr)

        return zpcorr

    def calibrate_stamps(self, galaxy: str=None):
        """
        Calibrate stamps
        """

        galaxy = self.galaxies[0] if galaxy is None else galaxy
        tile = os.listdir(self.work_dir + galaxy)[0].split('_')[1] if self.tiles is None else self.tiles[0]
        zps = self.get_zps(tile)
        zpcorr = self.get_zp_correction()
        stamps = sorted([_ for _ in os.listdir(self.work_dir + galaxy) if _.endswith("_swp.fits")])
        for stamp in stamps:
            filename = os.path.join(self.work_dir, galaxy, stamp)
            h = fits.getheader(filename, ext=1)
            h['TILE'] = tile
            filtername = h["FILTER"]
            zp = float(zps[filtername].data[0])
            x0 = h["X0TILE"]
            y0 = h["Y0TILE"]
            zp += round(zpcorr[filtername](x0, y0)[0][0], 5)
            fits.setval(filename, "MAGZP", value=zp,
                        comment="Magnitude zero point", ext=1)

    def run_sex(self, f: object, galaxy: str=None, tile: str=None, size: int=None):
        """ Run SExtractor to the detection stamp """
        #workdir = self.work_dir
        outdir = os.getcwd() if self.work_dir is None else self.work_dir
        galaxy = galaxy if self.galaxies is None else self.galaxies[0]
        tile = tile if self.tiles is None else self.tiles[0]
        size = size if self.sizes is None else self.sizes[0]
        galdir = os.path.join(outdir, galaxy)  # if self.gal_dir is None else self.gal_dir
        pathdetect = os.path.join(galdir, "{0}_{1}_{2}x{2}_{3}.fits".format(
            galaxy, tile, size, 'det_scimas'))
        pathtoseg = os.path.join(galdir, "{0}_{1}_{2}x{2}_{3}.fits".format(
            galaxy, tile, size, 'segmentation'))
        sexoutput = os.path.join(galdir, "{0}_{1}_{2}x{2}_{3}.fits".format(
            galaxy, tile, size, 'sexcat'))

        gain = f[1].header['GAIN']
        fwhm = f[1].header['PSFFWHM']
        wcs = WCS(f[1].header)
        fdata = f[1].data

        # calculating the limits for the stamp
        xp, yp = np.arange(fdata.shape[0]), np.arange(fdata.shape[1])
        nsky = pix2sky(xp, yp, wcs, origin=0, mode=u'wcs')
        ralims = np.min(nsky.ra), np.max(nsky.ra)
        delims = np.min(nsky.dec), np.max(nsky.dec)

        # output params for SExtractor
        params = ["NUMBER", "X_IMAGE", "Y_IMAGE", "KRON_RADIUS", "ELLIPTICITY",
                  "THETA_IMAGE", "A_IMAGE", "B_IMAGE", "MAG_AUTO", "FWHM_IMAGE",
                  "CLASS_STAR"]

        # configuration for SExtractor photometry
        config = {
                    "DETECT_TYPE": "CCD",
                    "DETECT_MINAREA": 4,
                    "DETECT_THRESH": 1.1,
                    "ANALYSIS_THRESH": 3.0,
                    "FILTER": "Y",
                    "FILTER_NAME": os.path.join(self.data_dir, "sex_data/tophat_3.0_3x3.conv"),
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
                    "STARNNW_NAME": os.path.join(self.data_dir, 'sex_data/default.nnw'),
                    "BACK_SIZE": 256,
                    "BACK_FILTERSIZE": 7,
                    "BACKPHOTO_TYPE": "LOCAL",
                    "BACKPHOTO_THICK": 48,
                    "CHECKIMAGE_TYPE": "SEGMENTATION",
                    "CHECKIMAGE_NAME": pathtoseg
        }

        sew = sewpy.SEW(config=config, sexpath="sextractor", params=params)
        sewcat = sew(pathdetect)
        sewcat["table"].write(sexoutput, format="fits", overwrite=True)

        return sewcat

    def run_DAOfinder(self, fdata: object):
        "calculate photometry using DAOfinder"

        mean, median, std = sigma_clipped_stats(fdata, sigma=3.0)
        print(('mean', 'median', 'std'))
        print((mean, median, std))
        daofind = DAOStarFinder(fwhm=4.0, sharplo=0.2, sharphi=0.9,
                                roundlo=-1, roundhi=1, threshold=5. * std)
        sources = daofind(fdata)
        return sources

    def make_Lupton_colorstamp(self, galaxy: str=None, tile: str=None, size: int=None):
        """Make Lupton colour image from stamps"""

        outdir = os.getcwd() if self.work_dir is None else self.work_dir
        galaxy = self.galaxies[0] if galaxy is None else galaxy
        tile = self.tiles[0] if tile is None else tile
        size = self.sizes[0] if size is None else size
        galdir = os.path.join(outdir, galaxy)
        bands = self.bands
        blues = ['U', 'F378', 'F395', 'F410', 'F515']
        greens = ['G', 'R', 'F660']
        reds = ['I', 'F861', 'Z']

        bimgs = [os.path.join(galdir, "{0}_{1}_{2}_{3}x{3}_swp.fits".format(
            galaxy, tile, band, size)) for band in blues]
        bdata = sum([fits.getdata(img) for img in bimgs])
        gimgs = [os.path.join(galdir, "{0}_{1}_{2}_{3}x{3}_swp.fits".format(
            galaxy, tile, band, size)) for band in greens]
        gdata = sum([fits.getdata(img) for img in gimgs])
        rimgs = [os.path.join(galdir, "{0}_{1}_{2}_{3}x{3}_swp.fits".format(
            galaxy, tile, band, size)) for band in reds]
        rdata = sum([fits.getdata(img) for img in rimgs])
        gal = os.path.join(galdir, galaxy + '.png')
        #rgb = make_lupton_rgb(rdata, gdata, bdata, Q=10, stretch=0.5, filename=gal)
        rgb = make_lupton_rgb(rdata, gdata, bdata, filename=gal)
        ax = plt.imshow(rgb, origin='lower')

        return rgb

    def calc_masks(self, galaxy: str=None, tile: str=None, size: int=None, savemask: bool=False,
                   savefig: bool=False, galindexes: list=[], maskstar: list=[]):
        """
        Calculate masks for S-PLUS stamps. Masks will use the catalogue of stars and from the
        SExtractor segmentation image. Segmentation regions need to be manually selected
        """
        outdir = os.getcwd() if self.work_dir is None else self.work_dir
        galaxy = self.galaxies[0] if galaxy is None else galaxy
        tile = self.tiles[0] if tile is None else tile
        size = self.sizes[0] if size is None else size
        galdir = os.path.join(outdir, galaxy)
        pathdetect: str = os.path.join(galdir, "{0}_{1}_{2}x{2}_{3}.fits".format(
            galaxy, tile, size, 'det_scimas'))
        #wmaps = glob.glob(galdir + '*_swpweight.fits')
        #for wmap in wmaps:
        #    w = fits.open(wmap)

        # get data
        f = fits.open(pathdetect)
        fdata = f[1].data
        wcs = WCS(f[1].header)

        sewcat = self.run_sex(f, galaxy=galaxy, tile=tile, size=size)
        sewpos = np.transpose((sewcat['table']['X_IMAGE'], sewcat['table']['Y_IMAGE']))
        radius = sewcat['table']['FWHM_IMAGE'] / 0.55
        sidelim = 50
        mask = sewcat['table']['CLASS_STAR'] > 0.05
        mask &= (sewcat['table']['X_IMAGE'] > sidelim)
        mask &= (sewcat['table']['X_IMAGE'] < fdata.shape[0] - sidelim)
        mask &= (sewcat['table']['Y_IMAGE'] > sidelim)
        mask &= (sewcat['table']['Y_IMAGE'] < fdata.shape[1] - sidelim)
        sewregions = [CirclePixelRegion(center=PixCoord(x, y), radius=z)
                      for (x, y), z in zip(sewpos[mask], radius[mask])]

        daocat = self.run_DAOfinder(fdata)
        daopos = np.transpose((daocat['xcentroid'], daocat['ycentroid']))
        daorad = 4 * (abs(daocat['sharpness']) + abs(daocat['roundness1']) + abs(daocat['roundness2']))
        #daoregions = CircularAperture(daopos, r=3.)
        daoregions = [CirclePixelRegion(center=PixCoord(x, y), radius=z)
                      for (x, y), z in zip(daopos, daorad)]

        #plt.figure(figsize=(10, 10))
        plt.rcParams['figure.figsize'] = (12.0, 10.0)
        plt.ion()

        ax1 = plt.subplot(221, projection=wcs)
        # make colour image
        rgb = self.make_Lupton_colorstamp(galaxy=galaxy, tile=tile, size=size)
        ax1.imshow(rgb, origin='lower')
        #daoregions.plot(color='y', lw=1.5, alpha=0.5)
        for dregion in daoregions:
            dregion.plot(ax=ax1, color='m')
        for sregion in sewregions:
            sregion.plot(ax=ax1, color='g')
        ax1.set_xlabel('RA')
        ax1.set_ylabel('Dec')

        ax2 = plt.subplot(222, projection=wcs)
        ax2.imshow(fdata, cmap='Greys_r', origin='lower', vmin=-0.1, vmax=3.5)
        #daoregions.plot(color='y', lw=1.5, alpha=0.5)
        for dregion in daoregions:
            dregion.plot(ax=ax2, color='m')
        for sregion in sewregions:
            sregion.plot(ax=ax2, color='g')
        ax2.set_xlabel('RA')
        ax2.set_ylabel('Dec')

        pathtoseg = os.path.join(galdir, "{0}_{1}_{2}x{2}_{3}.fits".format(
            galaxy, tile, size, 'segmentation'))
        s = fits.open(pathtoseg)
        sdata = s[0].data
        ax3 = plt.subplot(223, projection=wcs)
        ax3.imshow(sdata, origin='lower', interpolation='none', cmap='nipy_spectral_r', vmin=1)
        #daoregions.plot(color='y', lw=1.5, alpha=0.5)
        for dregion in daoregions:
            dregion.plot(ax=ax3, color='m')
        for sregion in sewregions:
            sregion.plot(ax=ax3, color='g')
        ax3.set_xlabel('RA')
        ax3.set_ylabel('Dec')
        #plt.canvas.mpl_connect('pick_event', self.pickSegReg)

        ax4 = plt.subplot(224, projection=wcs)
        #galindexes = [1, 2, 3, 5, 7, 8, 29, 28, 27, 119, 71, 6, 107, 88, 4, 83, 26, 115,
        #           112, 4, 61, 80, 25]
        galindexes = [] if galindexes is None else galindexes
        gmask = np.zeros(sdata.shape)
        for num in galindexes:
            gmask[sdata == num] = 1
        maskeddata = fdata * gmask
        #daoregions.plot(color='y', lw=1.5, alpha=0.5)
        for dregion in daoregions:
            dregion.plot(ax=ax4, color='m')
        for n, sregion in enumerate(sewregions):
            sregion.plot(ax=ax4, color='g')
            ax4.annotate(repr(n), (sregion.center.x, sregion.center.y), color='green')
            if n in maskstar:
                mask = sregion.to_mask()
                maskeddata[mask.bbox.slices] *= 1 - mask.data
        ax4.imshow(maskeddata, cmap='Greys_r', origin='lower', vmin=-0.1, vmax=3.5)

        ax4.set_xlabel('RA')
        ax4.set_ylabel('Dec')

        plt.subplots_adjust(wspace=.05, hspace=.2)
        #plt.tight_layout()

        fitsmask = s.copy()
        fitsmask[0].data = gmask
        fitsmask[0].header['IMGTYPE'] = ("MASK", "boolean mask")
        del fitsmask[0].header['EXPTIME']
        del fitsmask[0].header['FILTER']
        del fitsmask[0].header['GAIN']
        del fitsmask[0].header['PSFFWHM']
        if savemask:
            path2mask: str = os.path.join(galdir, "{0}_{1}_{2}x{2}_{3}.fits".format(
                galaxy, tile, size, 'mask'))
            print('saving mask', path2mask)
            fitsmask.writeto(path2mask, overwrite=True)
            indexesfile = open(os.path.join(galdir, "{}_{}_galindexes.txt".format(galaxy, tile)), 'w')
            print('saving masked indexes to', indexesfile)
            with indexesfile as ix:
                for num in galindexes:
                    ix.write(repr(num) + " ")
            ix.close()

        if savefig:
            path2fig: str = os.path.join(galdir, "{0}_{1}_{2}x{2}_{3}.png".format(
                galaxy, tile, size, 'maskMosaic'))
            print('saving fig', path2fig)
            plt.savefig(path2fig, format='png', dpi=180)

        return fitsmask

    def make_cubes(self, galdir: str=None, redo: bool=False, dodet: bool=True, get_mask: bool=True,
                   bands=None, specz="", photz="", bscale: float=1e-19):
        """ Get results from cutouts and join them into a cube. """

        #indir = os.path.join(self.work_dir, self.names[0]) if galdir is None else galdir
        #if not os.path.isdir(indir):
        #    os.mkdir(indir)
        galdir = os.path.join(self.work_dir, self.galaxies[0]) if galdir is None else galdir
        if not os.path.isdir(galdir):
            os.mkdir(galdir)
        filenames = glob.glob(galdir + '/*_swp*.fits')
        galcoords = SkyCoord(ra=self.coords[0][0], dec=self.coords[0][1], unit=(u.hour, u.deg))
        galaxy = "{}_{}".format(galcoords.ra.value, galcoords.dec.value) if self.galaxies[0] is None else self.galaxies[0]
        if redo:
            self.make_stamps_splus(redo=True)
            filenames = glob.glob(galdir + '/*_swp*.fits')
        if dodet:
            self.make_det_stamp()
        elif get_mask and not dodet:
            print('For mask detection image is required. Overwriting dodet=True')
            self.make_det_stamp()

        fields = set([_.split("_")[-4] for _ in filenames]) if self.tiles is None else self.tiles
        sizes = set([_.split("_")[-2] for _ in filenames]) if self.sizes is None else self.sizes
        bands = self.bands if bands is None else bands
        wave = np.array([self.wave_eff[band] for band in bands]) * u.Angstrom
        flam_unit = u.erg / u.cm / u.cm / u.s / u.AA
        fnu_unit = u.erg / u.s / u.cm / u.cm / u.Hz
        imtype = {"swp": "DATA", "swpweight": "WEIGHTS"}
        hfields = ["GAIN", "PSFFWHM", "DATE-OBS"]
        for tile, size in itertools.product(fields, sizes):
            cubename = os.path.join(galdir, "{0}_{1}_{2}x{2}_cube.fits".format(galaxy, tile,
                                                                   size))
            if os.path.exists(cubename) and not redo:
                print('cube exists!')
                continue
            # Loading and checking images
            imgs = [os.path.join(galdir, "{0}_{1}_{2}_{3}x{3}_swp.fits".format(
                galaxy, tile, band, size)) for band in bands]
            # Checking if images have calibration available
            headers = [fits.getheader(img, ext=1) for img in imgs]
            if not all(["MAGZP" in h for h in headers]):
                self.calibrate_stamps()
            headers = [fits.getheader(img, ext=1) for img in imgs]
            # Checking if weight images are available
            wimgs = [os.path.join(galdir, "{0}_{1}_{2}_{3}x{3}_swpweight.fits".format(
                galaxy, tile, band, size)) for band in bands]
            has_errs = all([os.path.exists(_) for _ in wimgs])
            # Making new header with WCS
            h = headers[0].copy()
            w = WCS(h)
            nw = WCS(naxis=3)
            nw.wcs.cdelt[:2] = w.wcs.cdelt
            nw.wcs.crval[:2] = w.wcs.crval
            nw.wcs.crpix[:2] = w.wcs.crpix
            nw.wcs.ctype[0] = w.wcs.ctype[0]
            nw.wcs.ctype[1] = w.wcs.ctype[1]
            try:
                nw.wcs.pc[:2, :2] = w.wcs.pc
            except:
                pass
            h.update(nw.to_header())
            # Performin calibration
            m0 = np.array([h["MAGZP"] for h in headers])
            gain = np.array([h["GAIN"] for h in headers])
            effexptimes = np.array([h["EFFTIME"] for h in headers])
            del h["FILTER"]
            del h["MAGZP"]
            del h["NCOMBINE"]
            del h["EFFTIME"]
            del h["GAIN"]
            del h["PSFFWHM"]
            f0 = np.power(10, -0.4 * (48.6 + m0))
            data = np.array([fits.getdata(img, 1) for img in imgs])
            fnu = data * f0[:, None, None] * fnu_unit
            flam = fnu * const.c / wave[:, None, None] ** 2
            flam = flam.to(flam_unit).value / bscale
            if has_errs:
                weights = np.array([fits.getdata(img, 1) for img in wimgs])
                dataerr = 1.0 / weights + np.clip(data, 0, np.infty) / gain[:, None, None]
                fnuerr = dataerr * f0[:, None, None] * fnu_unit
                flamerr = fnuerr * const.c / wave[:, None, None] ** 2
                flamerr = flamerr.to(flam_unit).value / bscale
            # Making table with metadata
            tab = []
            tab.append(bands)
            tab.append([self.wave_eff[band] for band in bands])
            tab.append(effexptimes)
            names = ["FILTER", "WAVE_EFF", "EXPTIME"]
            for f in hfields:
                if not all([f in h for h in headers]):
                    continue
                tab.append([h[f] for h in headers])
                names.append(f)
            tab = Table(tab, names=names)
            # Producing data cubes HDUs.
            hdus = [fits.PrimaryHDU()]
            hdu1 = fits.ImageHDU(flam, h)
            hdu1.header["EXTNAME"] = ("DATA", "Name of the extension")
            hdu1.header["SPECZ"] = (specz, "Spectroscopic redshift")
            hdu1.header["PHOTZ"] = (photz, "Photometric redshift")
            hdus.append(hdu1)
            if has_errs:
                hdu2 = fits.ImageHDU(flamerr, h)
                hdu2.header["EXTNAME"] = ("ERRORS", "Name of the extension")
                hdus.append(hdu2)
            for hdu in hdus:
                hdu.header["BSCALE"] = (bscale, "Linear factor in scaling equation")
                hdu.header["BZERO"] = (0, "Zero point in scaling equation")
                hdu.header["BUNIT"] = ("{}".format(flam_unit),
                                       "Physical units of the array values")
            if get_mask:
                path2mask: str = os.path.join(galdir, "{0}_{1}_{2}x{2}_{3}.fits".format(
                    galaxy, tile, size, 'mask'))
                if os.path.isfile(path2mask):
                    imagemask = fits.open(path2mask)
                else:
                    Warning('mask will be blank. Run interactively calc_masks() to determine true value of the galaxy')
                    imagemask = self.calc_masks(galaxy=galaxy, tile=tile, size=size,
                                                savemask=True, savefig=True)
                hdu3 = fits.ImageHDU(imagemask[0].data, imagemask[0].header)
                hdu3.header["EXTNAME"] = ("MASK", "Boolean mask of the galaxy")
                hdus.append(hdu3)
            thdu = fits.BinTableHDU(tab)
            hdus.append(thdu)
            thdu.header["EXTNAME"] = "METADATA"
            hdulist = fits.HDUList(hdus)
            print('writing cube', cubename)
            hdulist.writeto(cubename, overwrite=True)

if __name__ == "__main__":
    # print(initext)

    scubes = SCubes()

    # tile = scubes.check_infoot(save_output=True)
    # out = scubes.make_stamps_splus(redo=True, savestamps=False)
    # out = scubes.get_zps()
    # out = scubes.get_zp_correction()
    # out = scubes.calibrate_stamps()
    scubes.galaxies = np.array(['NGC1374'])
    scubes.coords = [['03:35:16.598', '-35:13:34.50']]
    scubes.tiles = ['SPLUS-s27s34']
    scubes.sizes = np.array([600])  # max(A, B) * 100
    #tile = scubes.check_infoot(save_output=False)
    # scubes.foot_dir = './'#'/home/luna/Documentos/usp/ic/fabio/aladin/'
    # scubes.zpcorr_dir = '/home/luna/Documentos/usp/ic/fabio/splus-cubes/data/zpcorr_idr3/'
    #scubes.sizes = np.array([600])  # max(A, B) * 100
    #scubes.galindexes = [1, 2, 3, 5, 7, 8, 29, 28, 27, 119, 71, 6, 107, 88, 4, 83,
    #                     26, 115, 112, 4, 61, 80, 25]
    #scubes.make_cubes(redo=False, specz=0.004755)
    #scubes.tiles_dir = './'
    #scubes.fields = ['SPLUS-s27a34']
    # scubes.make_det_stamp()
    #scubes.make_stamps_splus()
