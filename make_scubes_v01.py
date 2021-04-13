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
#from getpass import getpass

import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.io import fits, ascii
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
import astropy.constants as const
from tqdm import tqdm
from astropy.wcs import FITSFixedWarning
#import splusdata
import pandas as pd
from scipy.interpolate import RectBivariateSpline

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

class scubes(object):
    def __init__(self):

        #self.names = np.array(['NGC1087', 'NGC1090'])
        #self.coords = [['02:46:25.15', '-00:29:55.45'],
        #               ['02:46:33.916', '-00:14:49.35']]
        self.names = np.array(['NGC1087'])
        self.fields = None
        self.coords = [['02:46:25.15', '-00:29:55.45']]
        self.sizes = np.array([100])
        self.outdir = './'
        self.tiles_dir = './'
        self.foot_dir = '/home/herpich/Documents/pos-doc/t80s/Dropbox/myScripts/'
        self.data_dir = './data/'
        self.zpcorr_dir = '/home/herpich/Documents/pos-doc/t80s/Dropbox/myScripts/scubes/splus-fornax/data/zpcorr_idr3/'

        # from Kadu's context
        self.ps = 0.55 * u.arcsec / u.pixel
        self.bands = ['U', 'F378', 'F395', 'F410', 'F430', 'G', 'F515', 'R',
                      'F660', 'I', 'F861', 'Z']
        self.narrow_bands = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.broad_bands = ['U', 'G', 'R', 'I', 'Z']
        self.bands_names = {'U' : "$u$", 'F378': "$J378$", 'F395' : "$J395$",
                            'F410' : "$J410$", 'F430' : "$J430$", 'G' : "$g$",
                            'F515' : "$J515$", 'R' : "$r$", 'F660' : "$J660$",
                            'I' : "$i$", 'F861' : "$J861$", 'Z' : "$z$"}
        self.wave_eff = {"F378": 3770.0, "F395": 3940.0, "F410": 4094.0,
                         "F430": 4292.0, "F515": 5133.0, "F660": 6614.0,
                         "F861": 8611.0, "G": 4751.0, "I": 7690.0, "R": 6258.0,
                         "U": 3536.0, "Z": 8831.0}
        self.exptimes = {"F378": 660, "F395": 354, "F410": 177,
                         "F430": 171, "F515": 183, "F660": 870, "F861": 240,
                         "G": 99, "I": 138, "R": 120, "U": 681,
                         "Z": 168}

        ## matplotlib settings (from context)
        #plt.style.context("seaborn-paper")
        #plt.rcParams["text.usetex"] = True
        #plt.rcParams["font.family"] = "serif"
        #plt.rcParams['font.serif'] = 'Computer Modern'
        #plt.rcParams["xtick.direction"] = "in"
        #plt.rcParams["ytick.direction"] = "in"
        #plt.rcParams["xtick.minor.visible"] = True
        #plt.rcParams["ytick.minor.visible"] = True
        #plt.rcParams["xtick.top"] = True
        #plt.rcParams["ytick.right"] = True

    def check_infoot(self, save_output=False):
        foot = pd.read_csv(os.path.join(self.foot_dir, "tiles_new_status.csv"))
        field_coords =  SkyCoord(foot["RA"], foot["DEC"],
                                    unit=(u.hourangle, u.degree))
        inra = np.transpose(self.coords)[0]
        inde = np.transpose(self.coords)[1]
        c1 = SkyCoord(ra=inra, dec=inde, unit=(u.hour, u.deg), frame='icrs')
        idx, d2d, d3d = c1.match_to_catalog_sky(field_coords)
        max_sep = 1.0 * u.deg
        sep_constraint = d2d < max_sep
        status = foot['STATUS'][idx]
        status[~sep_constraint] = -10
        tile = foot['NAME'][idx]
        tile[~sep_constraint] = '-'
        for i in range(len(tile.values)):
            if len(tile.values[i].split('_')) > 1:
                tile.values[i] = tile.values[i].split('_')[0] + '-' + tile.values[i].split('_')[1]
        result = zip(self.names, self.coords, tile)
        print('matches found:')
        for a in result:
            print(a)
        cols = [self.names, inra, inde, tile, status]
        names = ['NAME', 'RA', 'DEC', 'TILE', 'STATUS']
        t = Table(cols, names=names)
        if save_output:
            outname = './objects_tilematched.csv'
            print('saving file', outname)
            t.write(outname, format='csv', overwrite=True)
        return t

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
        names = np.atleast_1d(self.names)
        sizes = np.atleast_1d(self.sizes)
        if len(sizes) == 1:
            sizes = np.full(len(names), sizes[0])
        sizes = sizes.astype(np.int)
        img_types = ["swp", "swpweight"] if img_types is None else img_types
        outdir = os.getcwd() if self.outdir is None else self.outdir
        tiles_dir = "/storage/share/all_coadded" if self.tiles_dir is None else self.tiles_dir
        header_keys = ["OBJECT", "FILTER", "EXPTIME", "GAIN", "TELESCOP",
                    "INSTRUME", "AIRMASS"]
        bands = self.bands if bands is None else bands

        # Selecting tiles from S-PLUS footprint
        fields = self.check_infoot()

        # Producing stamps
        for field in tqdm(fields, desc="Fields"):
            field_name = field["TILE"]
            fnames = [field['NAME']]
            fcoords = SkyCoord(ra=field['RA'], dec=field['DEC'],
                               unit=(u.hour, u.deg)) #self.coords[idx]
            fsizes = np.array(sizes)[fields['NAME'] == fnames]
            stamps = dict((k, []) for k in img_types)
            for img_type in tqdm(img_types, desc="Data types", leave=False,
                                position=1):
                for band in tqdm(bands, desc="Bands", leave=False, position=2):
                    tile_dir = os.path.join(tiles_dir, field["TILE"], band)
                    fitsfile = os.path.join(tile_dir, "{}_{}.fits".format(
                                            field["TILE"], img_type))
                    try:
                        header = fits.getheader(fitsfile)
                        data = fits.getdata(fitsfile)
                    except:
                        fzfile = tiles_dir + field['TILE'] + '_' + band + '_' + img_type + '.fz'
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
                        galdir = os.path.join(outdir, name)
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
                            #stamps[img_type].append(hdulist)
        #if savestamps:
        #    return
        #else:
        #    return stamps

    def make_det_stamp(self):
        names = np.atleast_1d(self.names)
        sizes = np.atleast_1d(self.sizes)
        if len(sizes) == 1:
            sizes = np.full(len(names), sizes[0])
        sizes = sizes.astype(np.int)
        outdir = os.getcwd() if self.outdir is None else self.outdir
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        tiles_dir = "/storage/share/all_coadded" if self.tiles_dir is None else self.tiles_dir
        header_keys = ["OBJECT", "FILTER", "EXPTIME", "GAIN", "TELESCOP",
                    "INSTRUME", "AIRMASS"]

        # Selecting tiles from S-PLUS footprint
        #fields = self.check_infoot() if self.fields is None else self.fields
        fields = self.check_infoot()

        # Producing stamps
        for field in fields:
            field_name = field["TILE"]
            fnames = [field['NAME']]
            fcoords = SkyCoord(ra=field['RA'], dec=field['DEC'],
                               unit=(u.hour, u.deg)) #self.coords[idx]
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
                        d = fits.open(tiles_dir + field_name + '_det_scimas.fits')[0]
                    except:
                        d = fits.open(tiles_dir + field_name + '_det_scimas.fits.fz')[1]
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
                    print('saving', doutput)
                    hdulist.writeto(doutput, overwrite=True)

    def get_zps(self, tile):
        """ Load all tables with zero points for iDR3. """
        _dir = self.data_dir
        #tables = []
        #for fname in os.listdir(_dir):
        #    filename = os.path.join(_dir, fname)
        #    #data = np.genfromtxt(filename, dtype=None)
        #    data = ascii.read(filename)
        #    h = [s.replace("SPLUS_", "") for s in data.keys()]
        #    table = Table(data, names=h)
        #    tables.append(table)
        #zptable = np.vstack(tables)
        fname = tile + '_ZP.cat'
        filename = os.path.join(_dir, fname)
        data = ascii.read(filename)
        h = [s.replace("SPLUS_", "") for s in data.keys()]
        zptab = Table(data, names=h)

        return zptab

    def get_zp_correction(self):
        """ Get corrections of zero points for location in the field. """
        x0, x1, nbins = 0, 9200, 16
        xgrid = np.linspace(x0, x1, nbins+1)
        zpcorr = {}
        for band in self.bands:
            corrfile = self.zpcorr_dir + 'SPLUS_' + band + '_offsets_grid.npy'
            corr = np.load(corrfile)
            zpcorr[band] = RectBivariateSpline(xgrid, xgrid, corr)

        return zpcorr

    def calibrate_stamps(self, obj=None):
        """
        Calibrate stamps
        """

        obj = self.names[0] if obj is None else obj
        tile = os.listdir(self.outdir + obj)[0].split('_')[1]
        zps = self.get_zps(tile)
        zpcorr = self.get_zp_correction()
        stamps = sorted([_ for _ in os.listdir(self.outdir + obj) if _.endswith("_swp.fits")])
        for stamp in stamps:
            filename = os.path.join(self.outdir, obj, stamp)
            h = fits.getheader(filename, ext=1)
            h['TILE'] = tile
            filtername = h["FILTER"]
            zp = float(zps[filtername].data[0])
            x0 = h["X0TILE"]
            y0 = h["Y0TILE"]
            zp += round(zpcorr[filtername](x0, y0)[0][0], 5)
            fits.setval(filename, "MAGZP", value=zp,
                        comment="Magnitude zero point", ext=1)

    def calc_masks(self):
        """ Calculate masks for S-PLUS images from the weight maps """
        indir = os.path.join(self.outdir, self.names[0])# if outdir is None else outdir
        outdir = os.path.join(self.outdir, self.names[0])# if outdir is None else outdir
        wmaps = glob.glob(indir + '*_swpweight.fits')
        for wmap in wmaps:
            w = fits.open(wmap)

    def make_cubes(self, indir=None, outdir=None, redo=False, dodet=False,
                   bands=None, specz="", photz="", bscale=1e-19):
        """ Get results from cutouts and join them in a cube. """

        indir = os.path.join(self.outdir, self.names[0]) if outdir is None else outdir
        if not os.path.isdir(indir):
            os.mkdir(indir)
        outdir = os.path.join(self.outdir, self.names[0]) if outdir is None else outdir
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        filenames = glob.glob(indir + '/*_swp*.fits')
        galaxy = self.names[0]
        if redo:
            self.make_stamps_splus(redo=True)
            filenames = glob.glob(indir + '/*_swp*.fits')
        if dodet:
            self.make_det_stamp()
        fields = set([_.split("_")[-4] for _ in filenames])
        sizes = set([_.split("_")[-2] for _ in filenames])
        bands = self.bands if bands is None else bands
        wave = np.array([self.wave_eff[band] for band in bands]) * u.Angstrom
        flam_unit = u.erg / u.cm / u.cm / u.s / u.AA
        fnu_unit = u.erg / u.s / u.cm / u.cm / u.Hz
        imtype = {"swp": "DATA", "swpweight": "WEIGHTS"}
        hfields = ["GAIN", "PSFFWHM", "DATE-OBS"]
        for field, size in itertools.product(fields, sizes):
            cubename = os.path.join(outdir, "{}_{}_{}.fits".format(galaxy, field,
                                                                size))
            if os.path.exists(cubename) and not redo:
                print('cube exists!')
                continue
            # Loading and checking images
            imgs = [os.path.join(indir, "{}_{}_{}_{}_swp.fits".format(galaxy,
                    field,  band, size)) for band in bands]
            # Checking if images have calibration available
            headers = [fits.getheader(img, ext=1) for img in imgs]
            if not all(["MAGZP" in h for h in headers]):
                self.calibrate_stamps()
            headers = [fits.getheader(img, ext=1) for img in imgs]
            # Checking if weight images are available
            wimgs = [os.path.join(indir, "{}_{}_{}_{}_swpweight.fits".format(
                    galaxy, field,  band, size)) for band in bands]
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
            flam = fnu * const.c / wave[:, None, None]**2
            flam = flam.to(flam_unit).value / bscale
            if has_errs:
                weights = np.array([fits.getdata(img, 1) for img in wimgs])
                dataerr = 1 / weights + np.clip(data, 0, np.infty) / gain[:, None, None]
                fnuerr= dataerr * f0[:, None, None] * fnu_unit
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
            thdu = fits.BinTableHDU(tab)
            hdus.append(thdu)
            thdu.header["EXTNAME"] = "METADATA"
            hdulist = fits.HDUList(hdus)
            print('writing cube', cubename)
            hdulist.writeto(cubename, overwrite=True)

if __name__ == "__main__":
    #print(initext)

    scubes = scubes()

    #tile = scubes.check_infoot(save_output=True)
    #out = scubes.make_stamps_splus(redo=True, savestamps=False)
    #out = scubes.get_zps()
    #out = scubes.get_zp_correction()
    #out = scubes.calibrate_stamps()
    scubes.names = np.array(['NGC1399'])
    scubes.coords = [['03:38:29.083', '-35:27:02.67']]
    scubes.foot_dir = '/home/luna/Documentos/usp/ic/fabio/aladin/'
    scubes.zpcorr_dir = '/home/luna/Documentos/usp/ic/fabio/splus-cubes/data/zpcorr_idr3/'
    scubes.sizes = np.array([2844]) # max(A, B) * 100
    #out = scubes.make_cubes(redo=False, specz=0.004755)
    scubes.tiles_dir = './'
    scubes.fields = ['SPLUS-s27a34']
    scubes.make_det_stamp()
