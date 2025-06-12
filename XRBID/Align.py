###################################################################################
##########		For aligning region files and calculating	###########
##########		astrometric corrections and positional		###########
##########		uncertainties. 					###########
##########		Last Updated: May 12, 20205			###########
##########		Added greater flexibility to CalcPU,		###########
##########		added directionality to CorrectAstrometry,	###########
##########		and removed unneeded functions.			###########
###################################################################################

import re
import numpy as np
from numpy import median, mean, std, sqrt
from math import isnan
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.io.votable import parse
from astropy.coordinates import SkyCoord
import pandas as pd
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore")
from XRBID.Sources import GetCoords
from XRBID.WriteScript import WriteReg

#print("Changes to Align.py successful.")

###-----------------------------------------------------------------------------------------------------

def AlignReg(filename, shifts=None, outfile=None, pixtodeg=None): 

	"""Aligns all of the sources of a region file by a given translation. """

	if not filename: 
		filename = input("Region file?: ")
		filename = re.split(",", filename)

	# Needs to be a list for the rest of the code to work right		
	if not isinstance(filename, list): filename = [filename]

	if not shifts: 
		shifts = input("X and Y pixel shifts?: ")
		temp = re.split("\W+", shifts)
		shifts = [float(temp[0]), float(temp[1])]

	print(shifts)

	if not isinstance(shifts, list): shifts = [shifts]
	print(shifts)

	xshift_copy = []
	yshift_copy = []

	print(shifts)
	if isinstance(shifts[0], list): 
		for i in range(len(shifts[0])): 
			xshift_copy.append(shifts[0][i])
			yshift_copy.append(shifts[1][i])
	else: 
		xshift_copy = shifts[0]
		yshift_copy = shifts[1]

	# For each region file listed...
	for k in range(len(filename)):
		reg = filename[k] 
		with open(reg) as f: 
			lines = f.readlines()

		if lines[2] == "fk5\n": 
			xshift_copy[k] = float(xshift_copy[k])*pixtodeg
			yshift_copy[k] = float(yshift_copy[k])*pixtodeg

		# Read in the lines (excluding header) and shift the coords by the calculated shift
		for i in range(len(lines)-3): 
			line = lines[i+3]

			# Splitting line into data components
			temp = re.split("# ", line)	# First need to get out the labels
			line = temp[0]
			marks = ""
			if len(temp) > 1: marks = "#" + "".join(temp[1:])
			temp = re.split(",|\(|\)| ", line)

			temp = filter(None, temp) 

			# Adding shifts to the coordinates
			try: # Sometimes shifts aren't available.
				temp[1] = str(float(temp[1]) + xshift_copy[k]) 
				temp[2] = str(float(temp[2]) + yshift_copy[k])
			except: 
				temp[1] = str(float(temp[1]) + xshift_copy)
				temp[2] = str(float(temp[2]) + yshift_copy)
			# Adding special characters back to the line
			temp[1] = "(" + temp[1]
			temp[2] = ", " + temp[2]
			temp[3] = ", " + str(temp[3]) + ") "
			lines[i+3] = "".join(temp) + marks # Reconstructs the line in .reg format
			
		# If no outfile is given, save to same filename + "_shifted"
		if outfile == None: 
			outf = "".join(re.split(".reg", reg))+"_shifted.reg"
		else: 
			if not isinstance(outfile, list): outfile = [outfile]
			outf = outfile[k]

		# Saving region file
		print("Saving " + outf)
		with open(outf, "w") as f:	
			for line in lines: 
				f.write(line)
		f.close()


###-----------------------------------------------------------------------------------------------------

def CorrectAstrometry(base_coords, cat_coords, autoresid=True, returnshifts=True, returncoords=False, savebasereg=False):
  """
  A quick method for identifying the likely sources necessary for an aperture alignment.
  This will cross-reference between the two input coordinates to find common sources, calculate the median offset between these sources,
  and (limiting the residuals to identify the most likely true common sources) calculate the standard deviation of the median offsets.

  PARAMETERS
  -----------
  base_coords 	[array] : Coordinates of sources to be used as a base to which the catalog will be aligned. Typically, these are the 
			  HST coordinates of XRBs.
  cat_coords  	[array] : Coordinates of the sources as seen in the misaligned catalog, for which the shifts will be calculated.
			  Typically, these are the Chandra coordinates of XRBs. 
  autoresid	[bool]	: If True, automatically sets the limits of 'optimal' residuals from the calculated shift to +/- 0.02. 
			  Otherwise, will prompt the user to select the best maximum and minimum residuals from a residual plot. 
			  This allows the user to determine whether one or more selected base coordinate source should be excluded 
			  from the astrometric correction for having residuals that are too far from the median. Default is True.
  returnshifts	[bool]	: If True, returns the x- and y-axis shifts from the astrometric correction. Default is True. 
  returncoords	[bool]	: If True, applies the astrometric corrections to the catalog coordinates and return the new coordinates as new_coords. 
  savebasereg 	[str]   : Saves a region file of the base coordinates of the matches sources from the base catalog

  RETURNS
  -----------
  new_coords 	[array]  		: New, shifted catalog coordinates after astrometric correction is applied (optional). 
  medshiftx, medshifty, stdx, stdy 	: The x- and y-axis shifts and standard deviations on the astrometric corrections, 
					  in units degrees. 
  
  """

  # Setting the input coordinates as SkyCoord objects, which will allow us to calculate the alignment.
  base_skycoords = SkyCoord(base_coords[0],base_coords[1],frame="fk5",unit="degree")
  cat_skycoords = SkyCoord(cat_coords[0],cat_coords[1],frame="fk5",unit="degree")

  # Finding matches between the base objects and the catalog objects
  # This will return the ids of the sources from base_skycoords that aligns most closely with each source from cat_skycoords
  # (base_ids) as well as the separation between the two.
  base_ids, separation, _ = cat_skycoords.match_to_catalog_sky(base_skycoords)

  # Plotting the histogram of the separations, to help isolate "good sources"
  if int(len(cat_coords[0])/20) < 10: bins = 10
  else: bins = int(len(cat_coords[0])/20)

  plt.figure()
  plt.hist(separation.deg, bins=bins)
  #plt.axvline(x=median(separation.deg),color="black")
  plt.title("Coordinate separation")
  plt.xlabel("Degrees")
  plt.show()

  # Request user to input minimum and maximum separation of "good sources"
  print("Median separation:", median(separation.deg))
  minsep, maxsep = input("Input min and max separation:").split(",")
  minsep = float(minsep)
  maxsep = float(maxsep)

  good_matches = (separation.deg < maxsep) & (separation.deg > minsep)

  good_cat_x = np.array(cat_coords[0])[good_matches]
  good_cat_y = np.array(cat_coords[1])[good_matches]
  good_base_x = np.array(base_coords[0])[base_ids[good_matches]]
  good_base_y = np.array(base_coords[1])[base_ids[good_matches]]

  # Calculating the full offsets between the input coordinates and the nearest items
  #offsets_x = good_cat_x - good_base_x
  #offsets_y = good_cat_y - good_base_y
  offsets_x = good_base_x - good_cat_x
  offsets_y = good_base_y - good_cat_y

  # Plotting the offsets
  fig, axes = plt.subplots(1,2, figsize=(6,4))
  fig.suptitle("Offsets (deg)\n(Zoomed in and out)")
  axes[0].scatter(offsets_x, offsets_y)
  axes[0].axvline(x=np.median(offsets_x), color="black",linestyle="--", label="median")
  axes[0].axvline(x=np.mean(offsets_x), color="black", label="mean")
  axes[0].axhline(y=np.median(offsets_y), color="black",linestyle="--", label="median")
  axes[0].axhline(y=np.mean(offsets_y), color="black", label="mean")
  axes[1].scatter(offsets_x, offsets_y)
  axes[1].axvline(x=np.median(offsets_x), color="black",linestyle="--", label="median")
  axes[1].axvline(x=np.mean(offsets_x), color="black", label="mean")
  axes[1].axhline(y=np.median(offsets_y), color="black",linestyle="--", label="median")
  axes[1].axhline(y=np.mean(offsets_y), color="black", label="mean")
  axes[1].set_xlim(median(offsets_x)/1.5,median(offsets_x)*1.5)
  axes[1].set_ylim(median(offsets_y)/1.5,median(offsets_y)*1.5)
  plt.show()

  # Saving median shifts
  medshiftx = median(offsets_x)
  medshifty = median(offsets_y)
  print("Median offsets in x and y (deg):", medshiftx, medshifty)

  # Plotting the residuals
  #resids_x = good_cat_x - medshiftx - good_base_x
  #resids_y = good_cat_y - medshifty - good_base_y
  resids_x = good_base_x - medshiftx - good_cat_x
  resids_y = good_base_y - medshifty - good_cat_y

  fig, axes = plt.subplots(1,2, figsize=(6,4))
  fig.suptitle("Residuals (deg)\n(Zoomed in and out)")
  axes[0].scatter(resids_x, resids_y)
  axes[0].axvline(x=np.median(resids_x), color="black",linestyle="--", label="median")
  axes[0].axvline(x=np.mean(resids_x), color="black", label="mean")
  axes[0].axhline(y=np.median(resids_y), color="black",linestyle="--", label="median")
  axes[0].axhline(y=np.mean(resids_y), color="black", label="mean")
  axes[1].scatter(resids_x, resids_y)
  axes[1].axvline(x=np.median(resids_x), color="black",linestyle="--", label="median")
  axes[1].axvline(x=np.mean(resids_x), color="black", label="mean")
  axes[1].axhline(y=np.median(resids_y), color="black",linestyle="--", label="median")
  axes[1].axhline(y=np.mean(resids_y), color="black", label="mean")
  axes[1].set_xlim(median(resids_x)/1.5,median(resids_x)*1.5)
  axes[1].set_ylim(median(resids_y)/1.5,median(resids_y)*1.5)
  plt.legend()
  plt.show()

  if not autoresid:
    # Setting the minimum and maximum offsets allowed, to limit the std calculation
    minresidx,maxresidx = input("Min and max x residuals:").split(",")
    minresidx = float(minresidx)
    maxresidx = float(maxresidx)
    minresidy,maxresidy = input("Min and max y residuals:").split(",")
    minresidy = float(minresidy)
    maxresidy = float(maxresidy)
  else:
    minresidx = -0.02
    minresidy = -0.02
    maxresidx = 0.02
    maxresidy = 0.02

  # printing the standard deviation of the item shift

  stdx = std(offsets_x[(resids_x < maxresidx) & (resids_x > minresidx)])
  stdy = std(offsets_y[(resids_y < maxresidy) & (resids_y > minresidy)])
  print("\nRESULTS OF ASTROMETRY CORRECTION")
  print("\nMedian offsets in x and y:\n", medshiftx, medshifty, "degrees")
  print("",medshiftx*3600, medshifty*3600, "arcsecs")
  print("\nStandard deviation in x and y offsets:\n", stdx, stdy, "degrees")
  print("",stdx*3600, stdy*3600, "arcsecs\n")

  # Returns the new coordinates of the input catalog sources
  new_coords = np.array([cat_coords[0] + medshiftx, cat_coords[1] + medshifty])
    
  if savebasereg: 
    WriteReg(sources=[good_base_x.tolist(), good_base_y.tolist()],
             color="red", outfile=savebasereg, coordsys="fk5", 
             radius=1, radunit="arcsec")
  print("Check region file for correct assessments.")

  # Determining what to return
  if returncoords and not returnshifts: # just coords
    return new_coords
  elif returncoords and returnshifts: # both shifts and coords
    return new_coords, medshiftx, medshifty, stdx, stdy
  elif returnshifts and not returncoords: # just shifts
    return medshiftx, medshifty, stdx, stdy
  else: return None # Return nothing

###-----------------------------------------------------------------------------------------------------

def CalcPU(df=False, theta=False, counts=False, theta_head="Theta", counts_head="Counts", std=[0,0], sig2search=False): 
	
	"""

	Calculates the positional uncertainty in arseconds of X-ray sources on an HST image using the 
	formulas from Kim et al. 2007 (https://arxiv.org/pdf/astro-ph/0611840.pdf), Equations 12 and 14.

	PARAMETERS
	------------
	df		[pd.DataFrame]	: (optional) DataFrame containing the X-ray data for each source.
					  This code assumes that the off-axis angle and net counts are 
					  stored under the headers 'Theta' and 'Counts', unless otherwise given. 
	theta		[list, float]	: (optional) Off-axis angles of X-ray sources from the Chandra pointing,
					  in units arcmin. This may be found as 'theta' or 'theta_mean' in the 
					  Chandra Source Catalog. 
	counts		[list, float]	: (optional) New counts of each X-ray source. 
	theta_head 	[str]		: Header under which theta is stored in the DataFrame. Default is 'Theta'. 
	counts_head	[str]		: Header under which counts in stored in the DataFrame. Default is 'Counts'.
	std		[list] ([0,0])	: The standard deviation of the X-ray source son the HST image, in units
					  arcseconds, along the x and y axis (e.g. [xstd,ystd]).
	sig2search	[str] (False)	: In the event that an observation does not have a valid theta or counts
					  value, allows the code to search for a default 2-sigma value under the 
					  input header name. For example, CSC provides a 2-sigma major and minor 
					  radii for the error ellipse of each source (err_ellipse_r0 and err_ellipse_r1). 
					  By setting sig2search equal to the header under which the major radius is saved, 
					  user can request CalcPU to pull the major radius as the default 2-sigma positional 
					  uncertainty for all sources for which the Kim et al. 2007 calculation
					  cannot be made. This prevents the code for returning artificially small 
					  radii when the positional uncertainty cannot be calculated. 
					  It's recommended to use the header under which err_ellipse_r0 is saved. 

	RETURNS
	---------
	sig1, sig2	[list]	: The 68% and 95% positional uncertainty radii of each source.

	 """
	
	# Off-axis angle should be in units arcminutes (default units for L19)
	if not theta: 
		theta = df[theta_head].values.tolist()
	if not counts: 
		counts = df[counts_head].values.tolist()

	sig2 = []
	sig1 = []

	if not sig2search: print("WARNING: No default 2-sigma given. Some positional uncertainty values may be artificially small, unless sig2search is provided by user.")

	# Calculating the Kim07 X-ray based uncertainties
	for i in range(len(theta)):
		oaa = theta[i]
		C = np.log10(counts[i]) 
        
		if isnan(oaa) or not 0 < C < 3.3:
			if sig2search:	# for invalid counts/oaa, use default error radius, if given
				#print("Invalid observation for source", i, ". Setting radius to", df[sig2search][i])
				sig2.append(df[sig2search][i])
				sig1.append(df[sig2search][i]*0.5)
			else: 
				#print("Invalid observation for source", i, ". Setting radius to 0.")
				sig2.append(0)
				sig1.append(0)
            
		else: 
			#print(i, "good. Continuing")
			# 95% confidence
			if 0 < C <= 2.1393: sig2.append(10**(0.1145*oaa - 0.4958*C + 0.1932))	
			elif 2.1393 < C <= 3.3: sig2.append(10**(0.0968*oaa - 0.2064*C - 0.4260))
                
			# 68% confidence
			if  0 < C <= 2.1227: sig1.append(10**(0.1137*oaa - 0.4600*C - 0.2398))
			elif 2.1227 < C <= 3.3000: sig1.append(10**(0.1031*oaa -0.1945*C - 0.8034))

	# Calculating the errors from the standard deviations
	sig_std = sqrt(std[0]**2 + std[1]**2)

	sig1 = [sqrt(sig**2 + sig_std**2)for sig in sig1]
	sig2 = [sqrt(sig**2 + (2*sig_std)**2) for sig in sig2]

	return sig1, sig2