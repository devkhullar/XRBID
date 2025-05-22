###################################################################################
##########		Functions for automated photometric		########### 
##########		measurements and manipulation 			########### 
##########		Last update: Nov 26, 2024			########### 
##########		Update desc: Created script			########### 
###################################################################################

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, mean, median, log, log10, std
import pandas as pd
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.wcs import WCS
from astropy.io import fits
import time
import random

# from photutils import aperture_photometry
from photutils.aperture import aperture_photometry
from photutils.utils import calc_total_error
from photutils.detection import DAOStarFinder as DaoFind
from photutils.aperture import CircularAperture
from photutils.background import Background2D, MedianBackground

import os
cd = os.chdir
pwd = os.getcwd

from XRBID.WriteScript import WriteReg

from acstools import acszpt # for zeropoint retrieval
from XRBID.DataFrameMod import Find


file_dir = os.path.dirname(os.path.abspath(__file__))

# These files should be downloaded and the path to the file should be added to the file name. 
# ACS/WFC: https://www.stsci.edu/hst/instrumentation/acs/data-analysis/aperture-corrections
# WFC3/UVIS: https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-encircled-energy
# JWST NIRCam: https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions#NIRCamPointSpreadFunctions-Encircledenergy
ACS_EEFs = pd.read_csv(file_dir+"/ACS_WFC_EEFs.txt")          # Using new EEFs for ACS as of 7/19/23
WFC3_EEFs = pd.read_csv(file_dir+"/WFC3_UVIS1_EEFs.frame")    # Using new EEFs for WFC3 as of 7/19/23
short_EEFs = pd.read_csv(file_dir+"/Encircled_Energy_SW_ETCv2.csv") # EEFs for the short wavelength filter
long_EEFs = pd.read_csv(file_dir+"/Encircled_Energy_LW_ETCv2.csv")  # EEFs for the long wavelength filter

# WFC3 zeropoints from: https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2021/WFC3_ISR_2021-04.pdf
WFC3_UVIS1_zpt = pd.read_csv(file_dir+"/WFC3_UVIS1_zeropoints.txt")
WFC3_UVIS2_zpt = pd.read_csv(file_dir+"/WFC3_UVIS2_zeropoints.txt")

# Filters for JWST NIRCam 
long_filter = [ "F250M", "F277W", "F300M",
                "F322W2", "F323N", "F335M", "F356W",
                "F360M", "F405N", "F410M", "F430M",
                "F444W", "F460M", "F466N", "F470N"
                "F480M"]

short_filter = ["F070W", "F090W", "F115W", "F140M",
                "F150W2", "F150W", "F162M", "F164N",
                "F182M", "F187N", "F200W", "F210M",
                "F212N"]

###-----------------------------------------------------------------------------------------------------

def RunPhots(hdu, gal, instrument, filter, fwhm_arcs, pixtoarcs=False, zeropoint=False, EEF=False, sigma=3, threshold=3, sharplo=0.2, sharphi=1.0, roundlo=-1.0, roundhi=1.0, apcorr=0, aperr=0, num_stars=20, min_rad=3, max_rad=20, extended_rad=10, aperture_correction=True, reg_correction=False, suffix=""):

    """
    Generates the initial photometric files needed for the aperture currection and photometric analyses.
    Steps taken by this function include:
    (1) Finding objects in the hdu image using DaoFindObjects()
    (2) Generating a background-subtracted hdu image
    (3) Running the photometry for all DaoFind objects using the background-subtracted image.
        This includes a full aperture photometry between radii 1-30 pixels 
        (used for aperture corrections), the aperture photometry within 3 pixels 
        (used for source analysis), and the aperture photometry within the extended radius for clusters 
	(defaulted to 10 pixels).
    (4) Runs the interactive aperture correction code CorrectAp() on the full aperture photometry. 
	Applies this correction to both the minimum aperture photometry and the extended aperture photometry

    PARAMETERS
    ----------
    hdu 		[FITS]	: 	The fits image of the HST/JWST field
    gal  		[str]	: 	Name of the galaxy of interest
    instrument 		[str]	: 	Name of the instrument (ACS, WFC3, or NIRCAM)
    filter 		[str]	: 	Name of the hdu filter
    fwhm_arcs 		[float]	: 	FWHM of stars in the hdu image
    pixtoarcs   	[float] :   	Pixel to arcsecond conversion. Defaults to 0.05 for ACS, 0.03962 for WFC3;
	                                 0.031 for short wavelength and 0.063 for long wavelength filters for JWST NIRCam.
    zeropoint 		[float]	: 	Vega zeropoint magnitude, for converting photometry into Vega magnitudes. 
    				        Defaults to obtaining with Zeropoint() if none is given.
    EEF 		[float]	: 	The Encircled Energy Fraction at the maximum aperture pixel 
    			  	        radius (default 20) for the instrument/filter of interest.
    			  	        If none is given, will pull the ~20 pix EEF for the instrument given. 
    sigma       	[float] (3) :	The sigma used in DaoFind, which adjusts the sensitivity
    threshold 		[float] (3) :	The threshold used in DaoFind, which adjusts the sensitivity
    sharplo		[float]	(.2):	Lower limit for object sharpness read in by DAOStarFinder. 
    sharphi		[float]	(1) :	Upper limit for object sharpness read in by DAOStarFinder. 
    roundlo		[float] (-1):	Lower limit for object roundness read in by DAOStarFinder.
    roundhi		[float)	(1) :	Upper limit for object roundness read in by DAOStarFinder. 
    apcorr		[float] (0) :	In the event that aperture_correction is set to false, 
    				            user can input a manual aperture correction to the photometry, 
    			            	which will be saved in the photometry files.
    aperr 		[float] (0) : 	Manual input of the error on the aperture correction 
    				            (estimated as the standard deviation of the photometric 
    			            	difference between min_rad and max_rad in the aperture correction).
    num_stars 		[int]	    :	The number of target stars to use for the aperture correction.
    min_rad 		[float] (3) : 	The pixel radius of the minimum aperture size for a standard star
    max_rad 		[float] (20): 	The pixel radius of the maximum aperture size
    extended_rad 	[float] (10):	The pixel radius of the aperture size for extended sources (i.e. clusters)
    aperture_correction [bool]  :	If true, runs the aperture correction for the field. Defaults as True.
    reg_correction 	[list]   :	Pixel correction on the [x,y] to add to the region file, if you find the region file 
					created from photutils source extraction coordinates is misaligned with the HST 
					image (Typically a correction of [1,1] pixel is sufficient.) 
    suffix 		[str]	:	Additional suffix to add to the end of filenames, if applicable. 
    			            	Good for if multiple fields are used for a single filter.

    RETURNS
    -------
    apcorr 		[float]	    : 	Magnitude correction for point sources in the given HDU field.
    					Returns only if aperture_correction = True
    aperr 		[float]     : 	Error on the point source aperture correction.
    					Returns only if aperture_correction = True
    apcorr_ext 		[float]	    : 	Magnitude correction for extended sources in the given HDU field.
    					Returns only if aperture_correction = True
    aperr_ext 		[float]     : 	Error on the extended aperture correction.
    					Returns only if aperture_correction = True
    OTHER PRODUCTS
    --------------
    [GALAXY]_daofind_[FILTER]_[INSTRUMENT][SUFFIX]_img.reg: 
    	Region file for all daofind sources in the field in image coordinates.
    [GALAXY]_daofind_[FILTER]_[INSTRUMENT][SUFFIX]_fk5.reg: 
    	Region file for all daofind sources in the field in fk5 coordinates.
    photometry_[GALAXY]_[FILTER]_[INSTRUMENT]_full[SUFFIX].ecsv: 
    	Datafile containing the full 1-30 pixel aperture photometry of all sources in the field
    photometry_[GALAXY]_[FILTER]_[INSTRUMENT]_sources[SUFFIX].ecsv: 
    	Datafile containing the 3 pixel aperture photometry of all sources in the field
    photometry_[GALAXY]_[FILTER]_[INSTRUMENT]_extended[SUFFIX].ecsv: 
    	Datafile containing the extended pixel aperture photometry of all sources in the field
    """

    try: data = hdu['SCI',1].data
    except: data = hdu['PRIMARY',1].data
    
    # Setting up zeropoint, if not given
    if not zeropoint: zeropoint = Zeropoint(hdu, filter, instrument)
    print(f"Using Zeropoint {zeropoint}")
    
    # Setting up EEF, if not given
    # The WFC3 EEF is going to slightly underestimate the correction, because there is no 20 pix correction
    if not EEF: 
        if instrument.lower() == 'acs': EEF = ACS_EEFs[ACS_EEFs['Filter']==filter.upper()].reset_index()['20'][0]
        elif instrument.lower() == 'wfc3': EEF = WFC3_EEFs[WFC3_EEFs['Filter']==filter.upper()].reset_index()['20.75'][0]
        elif instrument.lower() == 'nircam':
			# Note that both the wavelengths for JWST use different conversion rates (check documentation above)
            if filter in short_filter: EEF = short_EEFs.at[14, filter] # 0.60'' as the default, approximating the 20 pixel radius 
            if filter in long_filter: EEF = long_EEFs.at[19, filter] # 1.60'' as the default, approximating the 20 pixel radius

    print(f"Using EEF: {EEF}")	
		
    # Setting up the pixel scale, if not given 
    if not pixtoarcs: 
        try: pixtoarcs = hdu['PRIMARY'].header['D001SCAL'] # Pull directly from header, if available
        except: 
            if instrument.lower() == 'acs': pixtoarcs = 0.05
            elif instrument.lower() == 'wfc3': pixtoarcs = 0.03962
            elif instrument.lower() == 'nircam': 
                if filter in long_filter: pixtoarcs = 0.063 
                if filter in short_filter: pixtoarcs = 0.031
    print(f"Using pixtoarcs {pixtoarcs}")
        
    # Identifying point sources with DaoFind
    print("Running DaoFind. This may take a while...")
    objects = DaoFindObjects(data,sigma=sigma,threshold=threshold, fwhm=fwhm_arcs, pixtoarcs=pixtoarcs, \
    			 savereg=False, sharplo=sharplo, sharphi=sharphi, roundlo=roundlo, roundhi=roundhi)
    			 
    # Saving region files
    # If given, this adds a pixel shift to the region file coordinates, because the region files appear shifted 
    # wrt the image. However, I assume the coordinates are correct when it comes to taking the photometry, 
    # since these are the coordinates DAOFind is operating on. 
    
    if reg_correction == False:
        xcoord_img = objects['xcentroid'].tolist()
        ycoord_img = objects['ycentroid'].tolist()
    else: 
        print("\nAdding",reg_correction,"pixel correction to x & y coordinates.\nPlease double-check accuracy of region file!")
        xcoord_img = [x+reg_correction[0] for x in objects['xcentroid'].tolist()]
        ycoord_img = [y+reg_correction[1] for y in objects['ycentroid'].tolist()]


    WriteReg(sources=[xcoord_img, ycoord_img], radius=3, coordsys="image", \
    	     outfile=gal+"_daofind_"+filter.lower()+"_"+instrument.lower()+suffix+"_img.reg", \
    	     label=objects["id"].tolist())
    		 
    if instrument.lower() in ['acs', 'wfc3']:
        wcs = WCS(hdu['PRIMARY'].header)
        xcoords_fk5, ycoords_fk5 = wcs.wcs_pix2world(xcoord_img, ycoord_img, 1)

    else: # if using JWST
        wcs = WCS(hdu['SCI'].header)
        xcoords_fk5, ycoords_fk5 = wcs.wcs_pix2world(xcoord_img, ycoord_img, 1)
    
    WriteReg(sources=[xcoords_fk5, ycoords_fk5], coordsys="fk5", \
                    outfile=gal+"_daofind_"+filter.lower()+"_"+instrument.lower()+suffix+"_fk5.reg", \
                    radius=0.15, radunit="arcsec", label=objects["id"].tolist())
             
    print("\n", len(objects), "sources found.")
    print("Background subtraction...")
    data_sub = SubtractBKG(data)

    positions = np.transpose((objects['xcentroid'], objects['ycentroid']))
	# Create apertures for aperture corrections
    ap_rads = [i for i in range(1,31)]
    ap_rads = [i for i in range(1,31)]
    apertures_full = [CircularAperture(positions, r=r) for r in ap_rads]
    apertures_source = CircularAperture(positions, r=min_rad) # 3px aperture photometry used for sources by default
    apertures_extended = CircularAperture(positions, r=extended_rad) # aperture photometry for clusters (default is 10 pixels)
	
    print("Photometry...")
	 # Generate aperture photometry with the background-subtracted data
	
	# Collects the photometry over the full range of the apertures needed for the aperture correction step
    starttime = time.time()
    phot_full = perform_photometry(data_sub, data, hdu, apertures_full, instrument, filter, type='full', gal=gal, suffix=suffix, calc_error=False, savefile=True)
    endtime = time.time()
    print("Time for full photometry:", (endtime-starttime)/60., "minutes")
	
    ### Errors are estimated using exposure time as the effective gain and the ###
    ### background-only image as the background noise			       ###
    
    # Collects the photometry that will be used as the 'true' photometry for the source, 
    # collected within an aperture of radius min_rad
    starttime = time.time()
    phot_sources = perform_photometry(data_sub, data, hdu, apertures_source, instrument, filter, type='source', gal=gal, calc_error=True, suffix=suffix, savefile=False)
    endtime = time.time()
    print("Time for source photometry:", (endtime-starttime)/60., "minutes")

    # Collects the photometry that will be used as the photometry for clusters, 
    # collected within an aperture of radius extended_rad
    starttime = time.time()
    phot_extended = perform_photometry(data_sub, data, hdu, apertures_extended, instrument, filter, type='extended', gal=gal, calc_error=True, suffix=suffix, savefile=False)
    endtime = time.time()
    print("Time for extended photometry:", (endtime-starttime)/60., "minutes")

    # If aperture corrections need to be calculated, run CorrectAp()
    if aperture_correction:
        print("Aperture corrections...")
        apcorrections = CorrectAp(phot_full, gal=gal, filter=filter, radii=ap_rads, EEF=EEF, num_stars=num_stars, zmag=zeropoint, \
                              		  min_rad=min_rad, max_rad=max_rad, extended_rad=extended_rad)
        if len(apcorrections) > 0:
            apcorr = apcorrections[0], 
            aperr = apcorrections[1]
            apcorr_ext = apcorrections[2]
            aperr_ext = apcorrections[3]
    else: 
        apcorr = 0
        aperr = 0
        apcorr_ext = 0
        aperr_ext = 0

    # Calculates magnitudes from the photometry on point sources
    # If an aperture correction is given or calculated, include that in the source photometry.
    print("Calculating magnitudes...") 

    # Non-corrected magnitude from the photometry:
    phot_sources["aperture_mag"] = -2.5 * np.log10(phot_sources['aperture_sum'])

    # Ignoring the aperture error for now, as it seems to mess up measurements
    # phot_sources["aperture_mag_err"] = np.sqrt((0.434 * -2.5 * phot_sources["aperture_sum_err"]/
    # 					     phot_sources["aperture_sum"])**2 + aperr**2)

    # Error on the calculated magnitude: 
    phot_sources["aperture_mag_err"] = 0.434 * -2.5 * phot_sources["aperture_sum_err"]/phot_sources["aperture_sum"]

    # Corrected with the zeropoint and the aperture correction, if given
    # NOTE: will use the aperture correction from CorrectAp, if that is run above
    phot_sources["aperture_mag_corr"] = phot_sources["aperture_mag"] + zeropoint + apcorr 

    # Writing sources to a .ecsv file
    phot_sources.write("photometry_"+gal+"_"+filter.lower()+"_"+instrument.lower()+"_sources"+suffix+".ecsv", overwrite=True)
    print("photometry_"+gal+"_"+filter.lower()+"_"+instrument.lower()+"_sources"+suffix+".ecsv", "saved")

    ## REPEATING FOR EXTENDED SOURCES 
    # Non-corrected magnitude from the photometry:
    phot_extended["aperture_mag"] = -2.5 * np.log10(phot_extended['aperture_sum'])

    # Error on the calculated magnitude: 
    phot_extended["aperture_mag_err"] = 0.434 * -2.5 * phot_extended["aperture_sum_err"]/phot_extended["aperture_sum"]

    # Corrected with the zeropoint and the aperture correction, if given
    # NOTE: will use the aperture correction from CorrectAp, if that is run above
    phot_extended["aperture_mag_corr"] = phot_extended["aperture_mag"] + zeropoint + apcorr_ext 

    # Writing sources to a .ecsv file
    phot_extended.write("photometry_"+gal+"_"+filter.lower()+"_"+instrument.lower()+"_extended"+suffix+".ecsv", overwrite=True)
    print("photometry_"+gal+"_"+filter.lower()+"_"+instrument.lower()+"_extended"+suffix+".ecsv", "saved")

    print("DONE!")

    if aperture_correction:
            return apcorr, aperr, apcorr_ext, aperr_ext
    else: return None   
###-----------------------------------------------------------------------------------------------------

def SubtractBKG(data, sigma=3.0):
    
    """
    Returns a background-subtracted fits image, with the background subtraction performed by photutil.
    This is an auxilliary function required by RunPhots(). 

    PARAMETERS
    ----------
    data 	[HDUImage] 	: HDU data extracted from a FITS file.
    sigma 	[float]	(3.0)	: Sigma used for astropy.stats.SigmaClip()

    RETURN
    ---------
    data_sub [array]   	: Array containing the background-subtracted data.

    """

    sigma_clip = SigmaClip(sigma=sigma)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    data_sub = data - bkg.background
    
    return data_sub
    
###-----------------------------------------------------------------------------------------------------

def DaoFindObjects(data, fwhm, pixtoarcs, sigma=5, threshold=5.0, sharplo=0.2, sharphi=1, roundlo=-1, roundhi=1, savereg=False, plot=False, cmap='gray_r', vmin=0, vmax=0.3, aperture_color='#0547f9'):
    
    """ Using DaoFind from photutils, generates a list of objects. 
	
	Also if `plots=True`, will plot the image with apertures around the point sources. This should be used for testing
	purposes only to confirm if your threshold or fwhm are suffiecient enought to detect all the point sources in 
	the image. 

    PARAMETERS
    ----------
    data		[HDUImage] 	: HDU data extracted from a FITS file. Should use the original file, not the background subtraction.
    fwhm		[float]	 	: Estimated FWHM of stars in image, in units arcseconds.
    pixtoarcs 		[float]		: Pixel to arcsecond conversion.
    sigma		[float] (5)	: Desired sigma for the sigma clipping of the data.
    threshold		[float] (5)	: Absolute image value above which sources are detected.
    sharplo		[float]	(.2)	: Lower limit for object sharpness read in by DAOStarFinder. 
    sharphi		[float]	(1) 	: Upper limit for object sharpness read in by DAOStarFinder. 
    roundlo		[float] (-1)	: Lower limit for object roundness read in by DAOStarFinder.
    roundhi		[float)	(1) 	: Upper limit for object roundness read in by DAOStarFinder. 
    savefile 		[str] (False)	: If sources should be saved to a region file, input should be the name of the region file to be saved.
    plot        [bool] (False)  : Plots the image with apertures around the detected sources found through the DAOfind algorithm. 
    cmap        [str] ('gray_s) : Color map used for the plot.
    vmin, vmax  [float] (0, 0.3): The data range that the colormap covers.
    aperture_color [string]('#0547f9')  : The color of the apertures

    RETURN
    ----------
    objects 	[QTable]	: A table of objects identified by DaoFind. 
	
    `AxesImage`             : If `plot=True`, will return the plot with apertures around the detected objects.

    """

    dat_mean, dat_med, dat_std = sigma_clipped_stats(data, sigma=sigma)
    daofind = DaoFind(fwhm=fwhm/pixtoarcs, threshold=threshold*dat_std, sharplo=sharplo, sharphi=sharphi, roundlo=roundlo, roundhi=roundhi)
    objects = daofind(data)

    for col in objects.colnames:
    	if col not in ('id', 'npix'):
    		objects[col].info.format = '%.2f'  # for consistent table output

    # The coordinates of these sources appear to be about a pixel off from their actual locations.
    # Making the correction here. 
    #print("\nAdding 1 pixel correction to x & y coordinates.\nPlease double-check accuracy of region file.")
    #objects['xcentroid'] = [x+1 for x in objects['xcentroid'].tolist()]
    #objects['ycentroid'] = [y+1 for y in objects['ycentroid'].tolist()]
    
    if savereg:
    	WriteReg(sources=[objects['xcentroid'].tolist(), objects['ycentroid'].tolist()], \
    		 radius=3, coordsys="image", outfile=savereg, label=objects["id"].tolist())
	
    # To plot apertures on top of sources to check if the code is identifying the sources properly.
    if plot:
        positions = np.transpose((objects["xcentroid"], objects["ycentroid"]))
        apertures = CircularAperture(positions, r=5)
        plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        apertures.plot(color=aperture_color)
        plt.show() 
		

    return objects
    
###-----------------------------------------------------------------------------------------------------

def CorrectAp(tab, radii, gal, filter, EEF=False, num_stars=20, return_err=True, zmag=0, min_rad=3, max_rad=20, extended_rad=10):
    
    """
    Generating the correction on the aperture photometry, including the EEF correction from some 
    maximum aperture radius to infinity. This code assumes the photometry of the source is being
    taken within a 3 pixel radius, unless otherwise specified with min_rad.	An additional correction
    from the maximum aperture radius to infinity is applied based on the EEF at that radius (input by user).

    The stars on which the aperture correction calculation is run are randomly selected by the code and 
    manually approved. The correction is taken to be the median difference in the flux between the minimum 
    and maximum aperture radii (in pixels).

    PARAMETERS
    ----------
    tab 		[Table]		: The table result from running photutils.aperture_photometry on a 
    				  	  particular HST image
    radii 		[list]		: The apertures on which the photometry was run
    EEF 		[float]		: The Encircled Energy Fraction at the maximum aperture pixel radius 
                              		  (default 20) for the instrument/filter of interest
    num_stars		[int] (20) 	: The number of ideal stars to use for the calculation.
    return_err 		[bool] (True)	: When set to True (default), returns the standard deviatio of the aperture correction
    zmag 		[float] (0)	: May input the zeropoint magnitude to adjust the photometry.
    min_rad 		[int] (3)	: The pixel radius of the minimum aperture size (for point sources)
    max_rad 		[int] (20)	: The pixel radius of the maximum aperture size
    extended_rad 	[int] (10)	: The pixel radius of extended sources (i.e. clusters)

    RETURN
    ---------
    correction 	(float): The full correction to be applied to magnitude measurements,
                      	equivalent to the median of the 3 to 20 pixel aperture correction + the EEF correction
    err 	(float): The standard deviation of the 3 to 20 pixel aperture correction
    corr_ext 	(float): The full correction to be applied to magnitude measurements,
                      	equivalent to the median of the extended aperture correction + the EEF correction
    err_ext 	(float): The standard deviation of the extended pixel aperture correction

    """

    # Setting up EEF, if not given
    # The WFC3 EEF is going to slightly underestimate the correction, because there is no 20 pix correction
    if not EEF: 
        instrument = input('No EEF given. Please define instrument (acs or wfc3): ')
        print("Will pull EEF from 20 pixel radius. If max_rad is different, please define EEF manually.")
        if instrument.lower() == 'acs': EEF = ACS_EEFs[ACS_EEFs['Filter']==filter.upper()].reset_index()['20'][0]
        else: EEF = WFC3_EEFs[WFC3_EEFs['Filter']==filter.upper()].reset_index()['20.75'][0]
        
    # Calculating magnitudes from the aperture sums
    for i in range(len(radii)):
    	tab["aperture_mags_"+str(i)] = -2.5*np.log10(tab['aperture_sum_'+str(i)]) + zmag


    # For each aperture radius in the table, read in the photometric measurements
    # phots = an array in which each column represents a single radius, and each row is a source
    phots = np.array([tab["aperture_mags_"+str(i)] for i in range(len(radii))]).T
    temp_select_inds = random.sample(range(0, len(tab)), num_stars) # indices of the randomly selected stars
    temp_select = []
    cont = True

    tester = 0

    # Continue searching through stars until the number of selected stars reaches num_stars
    while cont:# and tester < 3:
    	if len(temp_select) < num_stars:
    		for j in temp_select_inds:
    			temp = phots[j] # full radial profile of chosen random star
    			# checking for dips and flattened ends
    			#print(temp)
    			if all(k>l for k,l in zip(temp, temp[1:])): # ensures the profile always decreases
              			if all(k-l<0.15 for k,l in zip(temp[3:], temp[4:])): # limits the slope of the decrease
                				if j not in temp_select: # print the star. If good, add to the list
                  					plt.figure(figsize=(4,2))
                  					for k in temp_select:
                  					    plt.plot(radii, phots[k], color="gray", lw=2, alpha=0.2)
                  					plt.plot(radii, phots[j])
                  					plt.ylim(26,10)
                  					plt.xlabel("Aperture radius (pixels)")
                  					plt.ylabel("Magnitude")
                  					plt.title("Star No. " + str(j))
                  					plt.show()
                  					if len(temp_select) > 0: 
                  					    print("Stars selected:",temp_select)
                  					    print(num_stars-len(temp_select),"more to go.")
                  					ans = input("Keep? (yes/[n]o/quit)").lower()
                  					if "y" in ans: temp_select.append(j) # add if not already in the list
                  					elif "q" in ans: cont = False; # allows user to quit the code
    		temp_select_inds = random.sample(range(0, len(tab)), num_stars-len(temp_select))
    		#tester =+ 1
    	else: cont = False; pass;

    #temp_select.sort()
    #print(temp_select)
    # Plotting the radial profile of all stars
    for i in temp_select:
    	plt.plot(radii, phots[i]) # where i is the index of the star
    plt.ylim(26,10)
    plt.savefig(f"radial_profile_{gal}_{filter}.png")
    plt.show()
    print(f"Saving radial_profile_{gal}_{filter}.png")

    ans = input("Check all profiles and enter 'y' to continue calculation: ")
    if "y" in ans:
    	# Continuing calculation of aperture correction
    	# finding the indices of the min and max aperture radii (default 3 and 20 pixels)
    	ind_min = radii.index(min_rad)
    	ind_max = radii.index(max_rad)
    	ind_ext = radii.index(extended_rad)
    	phot_diff = [phots[i][ind_max] - phots[i][ind_min] for i in temp_select]
    	phot_diff_ext = [phots[i][ind_max] - phots[i][ind_ext] for i in temp_select]
    	correction = np.median(phot_diff) + (1-(1./EEF))
    	err = np.std(phot_diff)
    	corr_ext = np.median(phot_diff_ext) + (1-(1./EEF))
    	err_ext = np.std(phot_diff_ext)
    	if return_err: return correction, err, corr_ext, err_ext
    	else: return correction, corr_ext
    else:
    	print("Rerun function to calculate aperture correction.")
    	return None
    
###-----------------------------------------------------------------------------------------------------

def Zeropoint(hdu, filter, instrument, date=None):
    
    """
    Retrieves the zero-points for the filter of interest based on the date of the observation.

    PARAMETERS
    ----------
    hdu	[fits]	: The HDU of the HST image for which the magnitudes are obtained.
    filter 	[str]	: The name of the filter for which the zero-point is needed.
    instrument [str]: ACS or WFC3. 
    date 	[str]	: Specify the date of the observation in "YYYY-MM-DD" format, if given by HDU.

    RETURN
    ----------
    zmag 	[float] : The zero-point magnitude of the input filter.
    
    """

    # Zeropoint for ACS
    if instrument.lower() == 'acs':
        if date: date = date
        else: 
            try: date = hdu[0].header['DATE-OBS']
            except: date = hdu['PRIMARY'].header['DATE']
        q_filter = acszpt.Query(date=date, detector="WFC", filt=filter)
        filter_zpt = q_filter.fetch()
        zmag = filter_zpt['VEGAmag'][0].value

    # Zeropoint for WFC3
    else: zmag = Find(WFC3_UVIS1_zpt, 'Filter = '+filter.upper())['Vega mag'][0]

    return zmag
    
###-----------------------------------------------------------------------------------------------------

def RemoveExt(Ebv, wave, mag):
    
    """
    Removes extinction from a the spectrum of a star given the reddening, E(B-V), using the
    extinction function and the Fitzpatrick & Massa 2007 extinction model. R_v is assumed to be 3.1.

    PARAMETERS
    ----------
    Ebv 	[float]	: The assumed reddening on the star. Can be calculated from the location of the star,
               		  Using the online tool at https://ned.ipac.caltech.edu/forms/calculator.html
    wave 	[array] : The wavelengths at which the magnitudes are evalutated.
    mag 	[array] : The magnitudes of the star at the given wavelengths.

    RETURNS
    ---------
    mag_ext [array] : Extinction-corrected magnitudes of the star at each wavelength. 
    
    """


    flux = 10**(mag/-2.5)
    Av = 3.1*Ebv # Calculating the extinction using the reddening as Av = Rv * Ebv
    ext_model = fm07(wave, Av)
    flux_ext = remove(ext_model, flux)
    mag_ext = -2.5*np.log10(flux_ext)

    return mag_ext
    
###-----------------------------------------------------------------------------------------------------

def perform_photometry(data_sub, data, hdu, apertures, instrument, filter, type, gal, suffix="", calc_error=True, savefile=False):
    '''
    A helper function to calculate the aperture photometry.
	
	PARAMETERS
	----------
	data_sub  [nd.array] : Background subtracted data
	data      [nd.array] : Data extracted from the fits file. NOT BACKGROUND SUBTRACTED
	hdu 		[FITS]	: 	The fits image of the HST/JWST field
	apertures      [int] : the aperture size passed for performing the aperture photometry.
	instrument 		[str]	: 	Name of the instrument (ACS, WFC3, or NIRCAM).
	filter 		[str]	: 	Name of the hdu filter.
	type           [str] : the type of photometry being performed -- full, extended or source.
	gal  		[str]	: 	Name of the galaxy of interest.
	suffix      [str]   : suffix to be added to the name of the saved file.
	calc_error  [bool] (True) : Calculate the error for aperture photometry.
	savefile    [bool] (True) : Save the file.

    RETURNS
	-------
	photometry   [array] : The photometric information of the sources. 
	
	OTHER PRODUCTS
	--------------
    Region file for all daofind sources in the field in fk5 coordinates.
        photometry_[GALAXY]_[FILTER]_[INSTRUMENT]_full[SUFFIX].ecsv: 
    Datafile containing the full 1-30 pixel aperture photometry of all sources in the field
        photometry_[GALAXY]_[FILTER]_[INSTRUMENT]_sources[SUFFIX].ecsv: 
    Datafile containing the 3 pixel aperture photometry of all sources in the field
        photometry_[GALAXY]_[FILTER]_[INSTRUMENT]_extended[SUFFIX].ecsv: 
    Datafile containing the extended pixel aperture photometry of all sources in the field
    '''
    hst_instrument = ['acs', 'wfc3']
    if calc_error:  # If error needs to be calculated (this is mainly for extended and source photometry)
        # If hst instrument
        if instrument in hst_instrument: 
            photometry = aperture_photometry(data_sub, apertures, error=calc_total_error(data, \
                            data-data_sub, effective_gain=hdu[0].header["EXPTIME"]))
            
        else: # If jwst instrument
            photometry = aperture_photometry(data_sub, apertures, error=calc_total_error(data, \
                                            data-data_sub, effective_gain=hdu[1].header["XPOSURE"]))
                
    else: # if you dont need to calculate error
        photometry = aperture_photometry(data_sub, apertures, method='center')
            
    if savefile:
        photometry.write("photometry_"+gal+"_"+filter.lower()+"_"+instrument.lower()+"_"+type+suffix+".ecsv", overwrite=True)
        print("photometry_"+gal+"_"+filter.lower()+"_"+instrument.lower()+"_"+type+suffix+".ecsv", "saved")
         
    return photometry

###------------------------------------------------------------------------------------------------

