"""
sdgridder.py - Gridder for single dish telescope data

Copyright(C) 2021 by
Trey V. Wenger; tvwenger@gmail.com

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Changelog:
2021-02-15 Trey V. Wenger
"""

# TODO: handle NaNs in dot products? (a la smooth_regrid_spec)
# TODO: remove hardcoded GBT parameters
# TODO: correct final FWHM for Gaussian-Bessel kernel

import time
import multiprocessing as mp
import sparse
import numpy as np
from scipy.special import j1
from astropy.io import fits

# speed of light (m/s)
_C = 299792458.0

# Telescope diameter (m)
_DIAMETER = 100.0


def smooth_regrid_spec(data, old_velocity_axis, new_velocity_axis):
    """
    Smooth and re-grid spectra to a new velocity axis using
    sinc interpolation.

    Inputs:
        data :: N-D array of scalars
            The final axis should be the velocity axis
        old_velocity_axis :: 1-D array of scalars
            Current velocity axis
        new_velocity_axis :: 1-D array of scalars
            Regridded velocity axis

    Returns: newdata
        newdata :: N-D array of scalars
            The final axis is the regridded velocity axis
    """
    old_res = old_velocity_axis[1] - old_velocity_axis[0]
    new_res = new_velocity_axis[1] - new_velocity_axis[0]

    # catch decreasing velocity axis
    if old_res < 0.0:
        old_velocity_axis = old_velocity_axis[::-1]
        data = data[..., ::-1]
        old_res = np.abs(old_res)
    if new_res < old_res:
        raise ValueError("Cannot smooth to a finer resolution!")

    # construct sinc weights, and catch out of bounds
    sinc_wts = np.array(
        [
            np.sinc((v - old_velocity_axis) / new_res)
            if (old_velocity_axis[0] < v < old_velocity_axis[-1])
            else np.zeros(len(old_velocity_axis)) * np.nan
            for v in new_velocity_axis
        ],
        dtype="float32",
    )

    # normalize weights
    sinc_wts = (sinc_wts.T / np.nansum(sinc_wts, axis=1)).T

    # apply, handle NaNs
    isnan = np.isnan(data)
    data[isnan] = 0.0
    nan_weights = np.ones(data.shape, dtype=data.dtype)
    nan_weights[isnan] = 0.0
    data = np.tensordot(sinc_wts, data, axes=([1], [-1]))
    nan_weights = np.tensordot(sinc_wts, nan_weights, axes=([1], [-1]))

    # replace zero weights with nan
    nan_weights[nan_weights == 0.0] = np.nan
    data = data / nan_weights

    # move velocity axis back to end
    data = np.moveaxis(data, 0, -1)
    return data


# Global storage for convolution weights and data.
# By storing these data globally, we prevent the copying of
# large data to multiprocessing children (Unix has copy-on-write)
conv_weights = None
spec = None


def convolve(idx):
    return conv_weights.dot(spec[:, idx])


def gridder(
    files,
    outname,
    kernel="gaussian",
    start=None,
    end=None,
    width=None,
    gauss_fwhm=None,
    bessel_width=None,
    pixel_size=None,
    truncate=None,
    num_cpus=None,
    overwrite=False,
    verbose=False,
):
    """
    Create images from GBT SDFITS data

    Inputs:
        files :: list of strings
            SDFITS filenames
        outname :: string
            Output FITS images are saved like:
                <outname>.texp.fits
                <outname>.tsys.fits
                <outname>.cube.fits
        kernel :: string
            Convolution kernel. Either gaussian or gaussbessel
        start, end, width :: scalars (km/s)
            Regrid the spectral axis to this velocity grid.
            If None, use native.
        gauss_fwhm :: scalar (deg)
            FWHM of the convolution Gaussian. If None, use
            2 * sqrt(ln(2)/9) * HPBW for gaussian kernel or
            2.52 * 2 * sqrt(ln(2)/9) * HPBW for gaussbessel kernel
        bessel_width :: scalar (deg)
            Width of convolution Bessel function. If None,  use
            1.55 * HPBW / 3
        pixel_size :: scalar (deg)
            Pixel size. If None, use HPBW / 5
        truncate :: scalar (deg)
            Truncate kernel at this distance
        num_cpus :: integer
            Number of CPUs to use for multiprocessing. If None,
            use all available.
        overwrite :: boolean
            If True, overwrite output images
        verbose :: boolean
            If True, print information

    Returns: Nothing
    """
    start_time = time.time()
    global conv_weights
    global spec

    if verbose:
        print("Reading data...")
    # Check data shapes
    rest_freq = None
    old_velocity_axis = None
    num_positions = 0
    for fname in files:
        with fits.open(fname) as hdulist:
            num_positions += hdulist[1].data["crval2"].size
            spec_size = hdulist[1].data[0]["data"].size
            if old_velocity_axis is None:
                rest_freq = hdulist[1].data[0]["RESTFREQ"]
                freq_axis = np.arange(spec_size, dtype=np.float32)
                freq_axis -= hdulist[1].data[0]["CRPIX1"] - 1
                freq_axis *= hdulist[1].data[0]["CDELT1"]
                freq_axis += hdulist[1].data[0]["CRVAL1"]
                old_velocity_axis = (freq_axis - rest_freq) / rest_freq
                old_velocity_axis *= _C / 1000.0  # km/s
            else:
                if spec_size != old_velocity_axis.size:
                    raise ValueError("Spectral axis mismatch in %s", fname)
        if verbose:
            print(f"Found file: {fname}")

    # Loop over SDFITS files and get data and positions
    spec_size = old_velocity_axis.size
    spec = np.ones((num_positions, spec_size), dtype=np.float32) * np.nan
    glong = np.ones(num_positions, dtype=np.float32) * np.nan
    glat = np.ones(num_positions, dtype=np.float32) * np.nan
    texp = np.ones(num_positions, dtype=np.float32) * np.nan
    tsys = np.ones(num_positions, dtype=np.float32) * np.nan
    idx = 0
    for fname in files:
        with fits.open(fname) as hdulist:
            num = hdulist[1].data["crval2"].size
            spec[idx : idx + num] = hdulist[1].data["data"]  # K
            glong[idx : idx + num] = hdulist[1].data["crval2"]  # deg
            glat[idx : idx + num] = hdulist[1].data["crval3"]  # deg
            texp[idx : idx + num] = hdulist[1].data["exposure"]  # seconds
            tsys[idx : idx + num] = hdulist[1].data["tsys"]
            idx += num

    # Get re-gridded velocity axis
    regrid = not (start is None and end is None and width is None)
    new_velocity_axis = old_velocity_axis
    if start is None:
        start = old_velocity_axis[0]
    if end is None:
        end = old_velocity_axis[-1]
    if width is None:
        width = old_velocity_axis[1] - old_velocity_axis[0]
    if regrid:
        new_velocity_axis = np.arange(start, end + width, width)
    spec_size = new_velocity_axis.size

    # regrid
    if verbose:
        print("Regridding spectra...")
    if regrid:
        spec = smooth_regrid_spec(spec, old_velocity_axis, new_velocity_axis)

    # catch errors
    bad = (glong == 0.0) + (glat == 0.0) + (tsys == 0.0)
    spec[bad] = np.nan
    glong[bad] = np.nan
    glat[bad] = np.nan
    texp[bad] = np.nan
    tsys[bad] = np.nan

    # Get beam size (deg)
    beam_fwhm = np.rad2deg(1.2 * _C / (_DIAMETER * rest_freq))

    # Get convolution Gaussian FWHM (deg)
    if gauss_fwhm is None:
        if kernel == "gaussian":
            gauss_fwhm = 2.0 * np.sqrt(np.log(2.0) / 9) * beam_fwhm
        elif kernel == "gaussbessel":
            gauss_fwhm = 2.52 * 2.0 * np.sqrt(np.log(2.0) / 9) * beam_fwhm
    gauss_sigma = gauss_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # Get convolution Bessel function width (deg)
    if bessel_width is None:
        bessel_width = 1.55 * beam_fwhm / 3.0

    # Get FWHM resolution of final image
    final_fwhm = np.sqrt(beam_fwhm ** 2.0 + gauss_fwhm ** 2.0)

    # Get pixel size (deg)
    if pixel_size is None:
        pixel_size = beam_fwhm / 5.0
    if pixel_size > beam_fwhm / 3.0:
        raise ValueError("Pixel size is larger than HPBW / 3")

    # Support distance for convolution
    if truncate is None:
        support_distance = beam_fwhm
    else:
        support_distance = truncate

    # Generate image dimensions
    glong_size = np.nanmax(glong) - np.nanmin(glong)
    glat_size = np.nanmax(glat) - np.nanmin(glat)
    glong_start = np.nanmax(glong)
    glat_start = np.nanmin(glat)
    nx = int(np.ceil(glong_size / pixel_size)) + 1
    ny = int(np.ceil(glat_size / pixel_size)) + 1
    glong_axis = -np.arange(nx, dtype=np.float32) * pixel_size + glong_start
    glat_axis = np.arange(ny, dtype=np.float32) * pixel_size + glat_start

    # Generate spare matrix for the distance^2 between each
    # grid point and each data point.
    if verbose:
        print("Generating sparse distance matrix...")
    glong_diff = glong_axis[..., None] - glong
    remove = (np.abs(glong_diff) > support_distance) + np.isnan(glong_diff)
    glong_diff[remove] = np.inf
    glong_diff = sparse.COO(glong_diff, fill_value=np.inf)
    glat_diff = glat_axis[..., None] - glat
    remove = (np.abs(glat_diff) > support_distance) + np.isnan(glat_diff)
    glat_diff[remove] = np.inf
    glat_diff = sparse.COO(glat_diff, fill_value=np.inf)
    image_distance2 = (
        glong_diff[:, None, :] ** 2.0 + glat_diff[None, :, :] ** 2.0
    )

    # Evaluate the Gaussian convolution weights at each data point
    if verbose:
        print("Calculating convolution weights...")
    if kernel == "gaussian":
        if verbose:
            print("Using Gaussian kernel")
        conv_weights = np.exp(
            -image_distance2 / (np.float32(2.0 * gauss_sigma ** 2.0))
        )
    elif kernel == "gaussbessel":
        if verbose:
            print("Using Gaussian x Bessel kernel")
        # add small positive number to catch zeros in sparse matrix
        image_distance = np.sqrt(image_distance2) + np.float32(1.0e-32)
        conv_weights = j1(
            np.abs(image_distance) * np.float32(np.pi / bessel_width)
        )
        # replace j1(np.inf) = np.nan with 0.0
        conv_weights.fill_value = np.array(0.0)
        conv_weights = conv_weights / (
            np.abs(image_distance) * np.float32(np.pi / bessel_width)
        )
        conv_weights = conv_weights * np.exp(
            -image_distance2 / (np.float32(2.0 * gauss_sigma ** 2.0))
        )

    # ensure convolution weights are masked > support_distance
    conv_weights = conv_weights * (image_distance2 < support_distance ** 2.0)

    # Generate the texp image
    if verbose:
        print("Generating integration image...")
    texp_image = conv_weights.dot(texp)

    # Generate the tsys image
    if verbose:
        print("Generating system temperature image...")
    # Use np.divide to catch division by zero
    sum_conv_weights = np.nansum(conv_weights, axis=-1).todense()
    tsys_image = conv_weights.dot(tsys)
    tsys_image = np.divide(
        tsys_image,
        sum_conv_weights,
        out=np.ones((nx, ny)) * np.nan,
        where=sum_conv_weights != 0.0,
    )

    # Generate data weights, combine with convolution weights
    if verbose:
        print("Calculating data weights...")
    data_weights = texp / tsys ** 2.0
    isnan = np.isnan(data_weights)
    data_weights[isnan] = 0.0
    conv_weights = conv_weights * data_weights
    # add spectral axis
    sum_conv_weights = np.nansum(conv_weights, axis=-1).todense()[..., None]

    if verbose:
        print("Convolving...")
    if num_cpus is None:
        num_cpus = mp.cpu_count()
    with mp.Pool(num_cpus) as pool:
        chunksize = spec_size // num_cpus + 1
        result = pool.map_async(
            convolve, range(spec_size), chunksize=chunksize
        )
        pool.close()
        pool.join()
    image_cube = np.array(result.get(), dtype=np.float32)
    image_cube = np.moveaxis(image_cube, 0, -1)

    # Use np.divide to catch division by zero
    image_cube = np.divide(
        image_cube,
        sum_conv_weights,
        out=np.ones((nx, ny, spec_size)) * np.nan,
        where=sum_conv_weights != 0.0,
    )

    # Generate header for images
    hdr = fits.Header()
    hdr["NAXIS"] = 4
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    hdr["NAXIS3"] = 1
    hdr["NAXIS4"] = 1
    hdr["CTYPE1"] = "GLON-CAR"
    hdr["CUNIT1"] = "deg"
    hdr["CRVAL1"] = glong_start
    hdr["CRPIX1"] = 1
    hdr["CDELT1"] = -pixel_size
    hdr["CTYPE2"] = "GLAT-CAR"
    hdr["CUNIT2"] = "deg"
    hdr["CRVAL2"] = glat_start
    hdr["CRPIX2"] = 1
    hdr["CDELT2"] = pixel_size
    hdr["CTYPE3"] = "VELO"
    hdr["CUNIT3"] = "km/s"
    hdr["CRVAL3"] = 0.0
    hdr["CRPIX3"] = 1.0
    hdr["CDELT3"] = width
    hdr["CTYPE4"] = "STOKES"
    hdr["CRVAL4"] = 1.0
    hdr["CRPIX4"] = 1.0
    hdr["CDELT4"] = 1.0
    hdr["SPECSYS"] = "LSRK"
    hdr["VELREF"] = 257
    hdr["RESTFRQ"] = rest_freq
    hdr["TELESCOP"] = "NRAO_GBT"
    hdr["BMAJ"] = final_fwhm
    hdr["BMIN"] = final_fwhm
    hdr["BPA"] = 0.0
    hdr["COMMENT"] = "GBTGridder used these SDFITS files:"
    for fname in files:
        hdr["COMMENT"] = "   {0}".format(fname)

    # Save texp image
    hdr["BUNIT"] = ("s", "Effective Integration")
    fname = "{0}.texp.fits".format(outname)
    if verbose:
        print(f"Saving integration image to {fname}")
    # add Stokes axis
    hdu = fits.PrimaryHDU(texp_image[..., None].T, header=hdr)
    hdu.writeto(fname, overwrite=overwrite)

    # Save tsys image
    hdr["BUNIT"] = ("K", "Effective Tsys")
    fname = "{0}.tsys.fits".format(outname)
    if verbose:
        print(f"Saving system temperature image to {fname}")
    # add Stokes axis
    hdu = fits.PrimaryHDU(tsys_image[..., None].T, header=hdr)
    hdu.writeto(fname, overwrite=overwrite)

    # Save continuum image
    fname = "{0}.cont.fits".format(outname)
    if verbose:
        print(f"Saving data cube to {fname}")
    # add Stokes axis
    hdu = fits.PrimaryHDU(image_cube[..., None].T, header=hdr)
    hdu.writeto(fname, overwrite=overwrite)

    # Modify header for image cube
    hdr["NAXIS3"] = spec_size
    hdr["CRVAL3"] = new_velocity_axis[0]
    hdr["CRPIX3"] = 1.0
    hdr["CDELT3"] = width
    hdr["BUNIT"] = ("K", "Ta")

    # Save image cube
    fname = "{0}.cube.fits".format(outname)
    if verbose:
        print(f"Saving data cube to {fname}")
    # add Stokes axis
    hdu = fits.PrimaryHDU(image_cube[..., None].T, header=hdr)
    hdu.writeto(fname, overwrite=overwrite)

    end_time = time.time()
    if verbose:
        print(
            "Runtime: {0:.1f} minutes".format((end_time - start_time) / 60.0)
        )
