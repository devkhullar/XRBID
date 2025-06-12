###################################################################################
##########	For plotting CMDs, along with mass tracks 		########### 
##########	Last updated: June 10, 2025				###########	
##########	Update Description: Added PlotSED for best-fit model	###########
##########		plotting for stellar photometric measurements	###########	
###################################################################################

import re
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as img
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from scipy.interpolate import interp1d
from astropy.io.votable import parse
from numpy import nanmedian, nanstd, nanmean, median, std, mean, log10
import pandas as pd
import os

cd = os.chdir
pwd = os.getcwd
pd.options.mode.chained_assignment = None

# Pull directory where XRBID files are saved
file_dir = os.path.dirname(os.path.abspath(__file__))
curr_dir = pwd()

from XRBID.DataFrameMod import Find, BuildFrame
from XRBID.Sources import LoadSources

default_aps = [i for i in range(1,31)] #[0.5,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,15.,20.]

# Labels associated with each line. Sources BELOW each line (but above the previous) get this label
mass_labels = ["Low", "Intermediate", "Intermediate", "High"]

# Old code refers to headers as defined below. These will need to be updated - 4/18/25
V = "V"
B = "B" 
I = "I"
U = "U"

cd(file_dir)
# Calling in tracks for future use by MakeCMD
wfc3_masses = [pd.read_csv("isoWFC3_1Msun.frame"), pd.read_csv("isoWFC3_3Msun.frame"), pd.read_csv("isoWFC3_5Msun.frame"),
	       pd.read_csv("isoWFC3_8Msun.frame"), pd.read_csv("isoWFC3_20Msun.frame")]

acs_masses = [pd.read_csv("isoACS_WFC_1Msun.frame"), pd.read_csv("isoACS_WFC_3Msun.frame"), pd.read_csv("isoACS_WFC_5Msun.frame"),
	      pd.read_csv("isoACS_WFC_8Msun.frame"), pd.read_csv("isoACS_WFC_20Msun.frame")]

isoacs = pd.read_csv("isoACS_all.frame")
isowfc3 = pd.read_csv("isoWFC3_all.frame")
# isonircam = pd.read_csv("isoNIRCAM_all.frame") # not yet active

# Calling in model for creating the cluster color-color diagrams
# As of version 1.7.0, no longer using BC03 models. Instead use CB07 models (Bruzual 2007, arXiv:astro-ph/0703052)
#BC03 = pd.read_csv("BC03_models_solar.txt")
CB07_acs = pd.read_csv("CB07_models_acs_wfc.csv")
CB07_wfc3 = pd.read_csv("CB07_models_wfc3_uvis.csv")
cd(curr_dir)

###-----------------------------------------------------------------------------------------------------

def MakeCMD(sources=False, xcolor=None, ycolor=None, xmodel=None, ymodel=None, figsize=(6,4), xlim=None, ylim=None, color="black", size=10, marker=None, label=None, save=False, savefile=None, title=None, subimg=None, annotation=None, annotation_size=None, imshow=True, fontsize=15, shift_labels=[[0,0],[0,0],[0,0],[0,0],[0,0]], set_labels=None, instrument="ACS", color_correction=[0,0], labelpoints=False, file_dir=False): 

	"""Makes a CMD from a given set of points, either from a list or an input dataframe.

	PARAMETERS: 
	sources 	[pd.dataframe, list]: 	Input may either be a pandas dataframe containing the appropriate magnitudes required for the CMD, 
				       		or a list of coordinates in the format [[xs],[ys]]. 
	xcolor 		[str or list]: 		The name of the color or magnitude to plot in the x-axis, as it is labeled in the dataframe. 
						This will be the default x-axis label. If a list is given, it is assumed to contain the 
						column names of the input dataframe to be subtracted (e.g. F555W - F814W)
	ycolor 		[str or list]: 		The name of the color or magnitude to plot in the y-axis, as it is labeled in the dataframe. 
						This will be the default y-axis label. If a list is given, it is assumed to contain the column 
						names of the input dataframe to be subtracted (e.g. F555W - F814W)
	xmodel 		[str or list]: 		The magnitude(s) of the filter(s) to be used from the stellar models for the x-axis. 
						If given as a list, it is assumed the color is xmodel[0] - xmodel[1] 
	ymodel 		[str or list]: 		The magnitude(s) of the filter(s) to be used from the stellar models for the y-axis. 
						If given as a list, it is assumed the color is ymodel[0] - ymodel[1] 
	figsize 	[tuple] (6,4): 		The desired dimensions of the figure. 
	xlim 		[tuple] (None):		The limits on the x-axis. If none are given, the limits are assumed to be the (xmin - 1, xmax + 1) 
	ylim 		[tuple] (None):		The limits on the y-axis. If none are given, the limits are assumed to be (ymax + 1, ymin - 1)
	color 		[str] ("black"): 	Marker color
	size 		[int] (10): 		Marker size
	marker 		[str] (None):		The style of the marker. Defaults to a filled point. If "o" is given, 
						the marker is set to be an open circle. 
	label 		[str] (None):		The legend label to assign to the input points from sources. 
	save 		[bool] (False): 	Sets whether to automatically same the CMD image. 
	savefile 	[str] (None): 		The name to assigned the saved CMD image. 
	title 		[str] (None): 		Title of the figure.
	subimg 		[str] (None): 		Filename of an image to include in the corner of the CMD. 
						This is to allow a subplot of the XRB plotted to be shown within the CMD. 
	annotation 	[str] (None): 		Additional annotation to add to the bottom corner of the CMD (usually XRB ID)
	annotation_size [int] (None): 		Annotation fontsize 
	imshow 		[bool] (True): 		Shows the plot.
	fontsize 	[int] (20): 		The fontsize of text other than the annoations
	shift_labels 	[list]: 		List of x and y coordinate distances by which to shift each of the model mass labels.
						Defaults to [[0,0],[0,0],[0,0],[0,0],[0,0]] (no shifts)
	set_labels 	[list] (None):		Sets the position of labels. If none is given, positions are automatically calculated.
	instrument 	[str] ("ACS"): 		Name of the instrument used, to determine which models to call. 
	color_correction [list] ([0,0]):	Corrections on the x and y position of the sources. Defaults to no correction. 
	labelpoints 	[list] (False): 	Labels to add to each point. If none are given, defaults to False and no labels added. 
	file_dir 	[str]: 			The directory within which the models may be found. By default, the 
						code attempts to find this automatically, but if it fails, it will 
						prompt the user to input the directory manually. 

	RETURNS: 
	f, ax: 		Arguments defining the figure, which can be used to add more points to the CMD after the initial plotting.
 
	"""

	# Setting the style of my plots
	#fontparams = {'font.family':'stix'}
	#labelparams = {'family':'stix', 'size':fontsize}
	
	curr_dir = pwd()

	# If no file directory is given, assume the files we need are in the same directory
	# where the module is saved
	#if not file_dir: 
	#	file_dir = os.path.dirname(os.path.abspath(__file__))

	#try: cd(file_dir)
	#except: print("Directory containg CMD models not found.\nPlease check and input the correct directory manually with file_dir.")

	# Reading in the appropriate models based on the instrument given.
	#if instrument.upper() =="WFC3":
	#	mass1 = pd.read_csv("isoWFC3_1Msun.frame")
	#	mass3 = pd.read_csv("isoWFC3_3Msun.frame")
	#	mass5 = pd.read_csv("isoWFC3_5Msun.frame")
	#	mass8 = pd.read_csv("isoWFC3_8Msun.frame")
	#	mass20 = pd.read_csv("isoWFC3_20Msun.frame")
	#elif instrument.upper() =="ACS": 
	#	mass1 = pd.read_csv("isoACS_WFC_1Msun.frame")
	#	mass3 = pd.read_csv("isoACS_WFC_3Msun.frame")
	#	mass5 = pd.read_csv("isoACS_WFC_5Msun.frame")
	#	mass8 = pd.read_csv("isoACS_WFC_8Msun.frame")
	#	mass20 = pd.read_csv("isoACS_WFC_20Msun.frame")

	if instrument.upper() =="WFC3": masses = wfc3_masses # list of DataFrames of each mass model
	else: masses = acs_masses
	mass_labels = [r"1 M$_\odot$", r"3 M$_\odot$", r"5 M$_\odot$", r"8 M$_\odot$", r"20 M$_\odot$"]

	#cd(curr_dir) 

	if savefile: save = True

	# Setting the x- and y-axes labels.
	# if xcolor and/or ycolor is a list, will need to set color[0] - color[1] as the color of the appropriate axis
	if not xcolor: xcolor=xmodel
	if not ycolor: ycolor=ymodel

	if isinstance(xcolor, list): xlabel = " - ".join(xcolor)
	else: xlabel = xcolor
	if isinstance(ycolor, list): ylabel = " - ".join(ycolor)
	else: ylabel = ycolor

	### Pulling the x and y values of the sources ###
	# if input source is a pandas dataframe, read in the appropriate colors and magnitudes (and add correction, if needed)
	if isinstance(sources, pd.DataFrame):
		if isinstance(xcolor, list): xsources = sources[xcolor[0]].values - sources[xcolor[1]].values + color_correction[0]
		else: xsources = sources[xcolor].values + color_correction[0]
		if isinstance(ycolor, list): ysources = sources[ycolor[0]].values - sources[ycolor[1]].values + color_correction[1]
		else: ysources = sources[ycolor].values + color_correction[1]
		
	elif sources: # If sources is a list or coordinates, pull the x and y values as given (with additional color correction)
		xsources = (np.array(sources[0]) + color_correction[0]).tolist()
		ysources = (np.array(sources[1]) + color_correction[1]).tolist()	
	### Will only need to call xsources or ysources from now on ###

	
	### PLOTTING MODEL MASS TRACKS ###

	f, ax = plt.subplots(figsize=figsize)

	# Setting the tick parameters
	ax.tick_params(direction="in", labelsize=15, bottom=True, \
		       top=True, left=True, right=True, length=7, width=2)
	ax.grid(alpha=0.8, linestyle="--")

	# If no xmodel or ymodel are given, default to that of the sources. 
	# This point will fail if the filters of the sources are not given a name corresponding to the filters in the model DataFrame!
	if not xmodel: xmodel=xcolor
	if not ymodel: ymodel=ycolor

	# Pulling the correct colors/magnitudes from the models and plotting
	# and setting the default mass track label position and plot limits
	xlims = []
	ylims = []
	for m, mass in enumerate(masses): # for each of the mass models, pull the color/mag given by xmodel and ymodel
		if isinstance(xmodel, list): 
			xtemp = mass[xmodel[0]].values - mass[xmodel[1]].values
			xtemp_label = max(xtemp) + 0.1 + shift_labels[m][0] # default mass label position, unless set_labels is given
			xtemp_left = max(xtemp)	# Keeping track of the leftmost x coordinate		
			if not xlim: xlims.append([min(xtemp)-1, max(xtemp)+1])
			invert_xlim = False
		else: # if x-axis is a magnitude..
			xtemp = mass[xmodel].values
			xtemp_label = min(xtemp) - 0.1 + shift_labels[m][0]
			xtemp_left = min(xtemp)
			if not xlim: xlims.append([max(xtemp)+1, min(xtemp)-1])
			invert_xlim = True
		if isinstance(ymodel, list): 
			ytemp = mass[ymodel[0]].values - mass[ymodel[1]].values
			if not ylim: ylims.append([min(ytemp)-1, max(ytemp)+1])
			invert_ylim = False
		else: # of y-axis is a magnitude...
			ytemp = mass[ymodel].values
			if not ylim: ylims.append([max(ytemp)+1, min(ytemp)-1])
			invert_ylim = True

		# Finding the best y-coordinate for the model label based on the leftmost
		ytemp_label = ytemp[xtemp.tolist().index(xtemp_left)] + invert_ylim*0.5 + shift_labels[m][1]

		# If set_labels is given, use this as the coordinate of the label
		# (overrides the label positions set above)
		if set_labels: 
			xtemp_label = set_labels[m][0]
			ytemp_label = set_labels[m][1]

		# Plotting mass track and mass label
		plt.plot(xtemp, ytemp, color="black", lw=1)
		ax.annotate(mass_labels[m], xy=(xtemp_label, ytemp_label), size=15)

	xlims = np.array(xlims)
	ylims = np.array(ylims) 

	# PLOTTING SOURCE POINTS
	if isinstance(sources, pd.DataFrame) or isinstance(sources, list): 
		if marker == "o": 	# 'o' used for open circle 
			ax.scatter(xsources, ysources, facecolor="none", edgecolor=color, s=size, label=label)
		elif marker == None: 	# default is a closed circle 
			ax.scatter(xsources, ysources, color=color, s=size, label=label)
		else: 
			ax.scatter(xsources, ysources, color=color, s=size, label=label, marker=marker)

		# PLOTTING POINT NAMES, IF GIVEN
		if labelpoints: 
			for i in range(len(labelpoints)): 
				ax.annotate(labelpoints[i], xy=[xsources[i], ysources[i]-.2], size=10, horizontalalignment="center")

	# Setting plot limits
	if not xlim:
		if invert_xlim: plt.xlim(max(xlims.T[0]), min(xlims.T[1]))
		else: plt.xlim(min(xlims.T[0]), max(xlims.T[1]))
	else: plt.xlim(xlim)
	if not ylim: 
		if invert_ylim: plt.ylim(max(ylims.T[0]), min(ylims.T[1]))
		else: plt.ylim(min(ylims.T[0]), max(ylims.T[1]))
	else: plt.ylim(ylim)

	# plotting mass track labels
	plt.xlabel(xlabel, labelpad=0, fontsize=fontsize)#, labelparams, labelpad=0)
	plt.ylabel(ylabel, labelpad=-10, fontsize=fontsize)#, labelparams, labelpad=-10)


	# If another title (such as name of the object) is given, plot
	# Adjusts size to make it fit the size of the plot
	if 0.6*figsize[0]*figsize[1] > 30: titlesize = 30
	else: titlesize = 0.6*figsize[0]*figsize[1]

	if not annotation_size: annotation_size = titlesize

	if title: ax.set_title(title, fontsize=titlesize)

	# If an annotation is given, add it to the bottom of the figure
	if annotation: ax.annotate(annotation, xy=(xlim[1]-0.05*abs(xlim[1]), ylim[0]-0.1*abs(ylim[1])), size=annotation_size, horizontalalignment="right")

	# if a subimage is given as a filename, read in. 
	if subimg and "." in subimg: 
		subimg = img.imread(subimg)

	# If an image is passed to overlay on plot, add 
	if hasattr(subimg, 'shape'): # tests if subimg was fed in
		try: 
			XY = [figsize[0], figsize[1]]
			ax2 = f.add_axes([.9 - 0.22*float(XY[1])/float(XY[0]), 0.66,  float(XY[1])/float(XY[0])*0.22, 0.22], zorder=1)
			ax2.imshow(subimg)
			ax2.axis('off')
		except: 
			im = plt.imread(subimg)
			XY = [figsize[0], figsize[1]]
			ax2 = f.add_axes([.9 - 0.22*float(XY[1])/float(XY[0]), 0.66,  float(XY[1])/float(XY[0])*0.22, 0.22], zorder=1)
			ax2.imshow(im)
			ax2.axis('off')

	# saving image, if prompted
	if savefile != None: save = True; pass;
	if save:
		if savefile == None: 
			savefile = df["ID"][0].split("X")[0] + "_" + xcolors + "_" + ycolors
		plt.savefig(savefile.split(".")[0]+".jpg", dpi=300, bbox_inches="tight")

	# Returning plot information, in case I need this later
	# need to retrieve ax if using both subimg followed by AddCMD

	cd(curr_dir)
	return f, ax
				

###-----------------------------------------------------------------------------------------------------

def AddCMD(df=None, xcolor=False, ycolor=False, color="black", size=10, marker=None, label=None, f=None, ax=None, color_correction=[0,0]): 

	"""Adds multiple sets of points to a single CMD plot. Should be used after MakeCMD. 
	If plots do not print as expected, call in f and ax from MakeCMD.
	NOTE: This code us currently under construction"""

	
	# Setting the x- and y-axes labels.
	# if xcolor and/or ycolor is a list, will need to set color[0] - color[1] as the color of the appropriate axis
	if not xcolor: xcolor=xmodel
	if not ycolor: ycolor=ymodel

	if isinstance(xcolor, list): xlabel = " - ".join(xcolor)
	else: xlabel = xcolor
	if isinstance(ycolor, list): ylabel = " - ".join(ycolor)
	else: ylabel = ycolor

	### Pulling the x and y values of the sources ###
	# if input source is a pandas dataframe, read in the appropriate colors and magnitudes (and add correction, if needed)
	if isinstance(sources, pd.DataFrame):
		if isinstance(xcolor, list): xsources = sources[xcolor[0]].values - sources[xcolor[1]].values + color_correction[0]
		else: xsources = sources[xcolor].values + color_correction[0]
		if isinstance(ycolor, list): ysources = sources[ycolor[0]].values - sources[ycolor[1]].values + color_correction[1]
		else: ysources = sources[ycolor].values + color_correction[1]
		
	else: # If sources is a list or coordinates, pull the x and y values as given (with additional color correction)
		xsources = (np.array(sources[0]) + color_correction[0]).tolist()
		ysources = (np.array(sources[1]) + color_correction[1]).tolist()	
	### Will only need to call xsources or ysources from now on ###

	try: 
		if ax: 	# ax MUST be read in if subimg is used in MakeCMD. 
			if marker == "o": 	# 'o' used for open circle 
				ax.scatter(xsources, ysources, facecolor="none", edgecolor=color, s=size, label=label)
			elif marker == None: 	# default is a closed circle 
				ax.scatter(xsources, ysources, color=color, s=size, label=label)
		else: 
			if marker == "o": 	# 'o' used for open circle 
				plt.scatter(xsources, ysources, facecolor="none", edgecolor=color, s=size, label=label)
			elif marker == None: 	# default is a closed circle 
				plt.scatter(xsources, ysources, color=color, s=size, label=label)

	except: return "Failed to add points."		
	

###-----------------------------------------------------------------------------------------------------

def CorrectMags(frame=None, phots=None, corrections=None, field=None, apertures=[3,20], headers=["V", "B", "I"], instrument="ACS", filters=["F606W", "F435W", "F814W"], distance=False, savefile=None, idheader="ID", ID_header=False, coordheads=["X", "Y"], coord_headers=False, extinction=[0,0,0,0]): 

	"""Calculating magnitudes with given aperture corrections. The input 'instrument' can be 'ACS' or 'WFC3', which defines which EEF file to read from. Filters should be read in the order [V,B,I]. If given, 'extinction' should also be in [Av,Ab,Ai,Au] order. (NOTE: note RGB) or [V,B,I,U], if U is given. Corrections should also be read in VBI order. """

	try: frame = frame.copy()
	except: pass;

	if not distance: 
		distance = float(input("Distance to galaxy (in units parsec): "))

	# ID_headers and coord_headers are depricated as parameters, but still used as variables in the code below.
	if coord_headers == False: coord_headers = coordheads
	if ID_headers == False: ID_headers = idheader

	# If U is given, add U corrections to all commands
	if len(filters) == 4: U_true = True
	else: U_true = False
	
	curr_dir = pwd()
	# Loading in the EEFs file
	cd(file_dir)
	EEFs = LoadSources(instrument + "_EEFs.txt", verbose=False)
	cd(curr_dir)

	# Reading in the 20px (or highest given) EEF from each filter 
	V_EEF = Find(EEFs, "Filter = " + filters[0])[str(apertures[-1])][0]
	B_EEF = Find(EEFs, "Filter = " + filters[1])[str(apertures[-1])][0]
	I_EEF = Find(EEFs, "Filter = " + filters[2])[str(apertures[-1])][0]
	if U_true: U_EEF = Find(EEFs, "Filter = " + filters[3])[str(apertures[-1])][0]

	#EEFracts=[Vee["20"],Bee["20"],Iee["20"],Uee["20"]]
	#V_EEF = EEFracts[0]
	#B_EEF = EEFracts[1]
	#I_EEF = EEFracts[2]
	#U_EEF = EEFracts[3]

	if not phots: 
		phots = []
		phots.append(frame[headers[0]])
		phots.append(frame[headers[1]])
		phots.append(frame[headers[2]])
		if U_true: phots.append(frame[headers[3]])

	dmod = 5.*log10(distance/10.)

	# Converting to absolute magnitudes? 
	V_M = phots[0] - dmod - extinction[0]
	B_M = phots[1] - dmod - extinction[1]
	I_M = phots[2] - dmod - extinction[2]
	if U_true: U_M = phots[3] - dmod - extinction[3]

	# Finding the proper corrections factor. If none given, find proper defaults.
	if not corrections: 
		if not field: 
			try: field = frame["Field"].values.tolist()
			except: field = input("Image Field? (f_):")
		if isinstance(field, list): 
			corrections = [[Corrections[V][int(i)-1], Corrections[B][int(i)-1], Corrections[I][int(i)-1], Corrections[U][int(i)-1]] for i in field]
			corrections = np.array(corrections)
		else: 
			try: corr = int(re.split("f", field.lower())[-1][0])
			except: corr = int(field)
			corrections = [Corrections[V][corr-1], Corrections[B][corr-1], Corrections[I][corr-1], Corrections[U][corr-1]]
			corrections = np.array(corrections)

	# Calculating the corrections
	try: 
		V_corr = np.array(V_M) - corrections.T[0] + 2.5*log10(float(V_EEF))
		B_corr = np.array(B_M) - corrections.T[1] + 2.5*log10(float(B_EEF))
		I_corr = np.array(I_M) - corrections.T[2] + 2.5*log10(float(I_EEF))
		if U_true: U_corr = np.array(U_M) - corrections.T[3] + 2.5*log10(float(U_EEF))
	except: 
		V_corr = np.array(V_M) - corrections[0] + 2.5*log10(float(V_EEF))
		B_corr = np.array(B_M) - corrections[1] + 2.5*log10(float(B_EEF))
		I_corr = np.array(I_M) - corrections[2] + 2.5*log10(float(I_EEF))
		if U_true: U_corr = np.array(U_M) - corrections[3] + 2.5*log10(float(U_EEF))

	for i in range(len(V_corr)): 
		if V_corr[i] > 100 or V_corr[i] < -100: V_corr[i] = np.nan
		if B_corr[i] > 100 or B_corr[i] < -100: B_corr[i] = np.nan
		if I_corr[i] > 100 or I_corr[i] < -100: I_corr[i] = np.nan
		if U_true: 
			if U_corr[i] > 100 or U_corr[i] < -100: U_corr[i] = np.nan

	if savefile: 
		try: Mags = BuildFrame(headers=[ID_header, X, Y, V, B, I, U, VI, BV, BI], \
		         values = [frame[ID_header], frame[coord_headers[0]], frame[coord_headers[1]], V_corr, B_corr, I_corr, U_corr, V_corr-I_corr, B_corr-V_corr, B_corr-I_corr])
		except: 
			Mags = BuildFrame(headers=[ID_header, X, Y, V, B, I, U, VI, BV, BI], \
		         values = [frame[ID_header], frame[coord_headers[0]], frame[coord_headers[1]], V_corr, B_corr, I_corr, V_corr-I_corr, B_corr-V_corr, B_corr-I_corr])

		Mags.to_csv(savefile)

	try: 
		frame[headers[0]] = V_corr
		frame[headers[1]] = B_corr
		frame[headers[2]] = I_corr
		if U_true: frame[headers[3]] = U_corr
		return frame
	except: 
		try: return V_corr, B_corr, I_corr, U_corr
		except: return V_corr, B_corr, I_corr

###-----------------------------------------------------------------------------------------------------

def CorrectMag(df=False, phots=None, correction=None, field=None, apertures=[3,20], instrument="ACS", filt="F606W", distance=False, savefile=None, idheader="ID", ID_header=False, coordheads=["X", "Y"], coord_headers=False, extinction=0): 

	"""Calculating magnitude with given aperture correction, like CorrectMags, but specifically for a single input filter (so that it doesn't require all filters to be given if only one measurement is needed). The input 'instrument' can be 'ACS' or 'WFC3', which defines which EEF file to read from. Filters should be read in the order [V,B,I]. If given, 'extinction' should also be in [Av,Ab,Ai,Au] order. (NOTE: note RGB) or [V,B,I,U], if U is given. Corrections should also be read in VBI order. 

	NOTE: ID_header and coord_headers have been depricated. They are now called idheader and coordheads, to match other parts of XRBID. """

	if df: frame = df.copy()

	# ID_headers and coord_headers are depricated as parameters, but still used as variables in the code below.
	if coord_headers == False: coord_headers = coordheads
	if ID_headers == False: ID_headers = idheader

	curr_dir = pwd()
	# Loading in the EEFs file
	cd(file_dir)
	EEFs = LoadSources(instrument + "_EEFs.txt", verbose=False)
	cd(curr_dir)

	if filt in V_filts: 
		EEF = Find(EEFs, "Filter = " + filt)[str(apertures[-1])][0]
		header = V
	elif filt in B_filts: 
		EEF = Find(EEFs, "Filter = " + filt)[str(apertures[-1])][0]
		header = B
	elif filt in I_filts: 
		EEF = Find(EEFs, "Filter = " + filt)[str(apertures[-1])][0]
		header = I
	elif filt in U_filts: 
		EEF = Find(EEFs, "Filter = " + filt)[str(apertures[-1])][0]
		header = U

	if not phots: 
		phots = []
		phots.append(frame[header])

	if not distance: distance = input("Input distance to galaxy (in parsecs): ")
	dmod = 5.*log10(distance/10.)

	# Converting to absolute magnitudes? 
	Mag = phots - dmod - extinction

	# Calculating the corrections
	try: 
		corr = np.array(Mag) - correction + 2.5*log10(float(EEF))
	except: 
		corr = np.array(Mag) - correction + 2.5*log10(float(EEF))

	#for i in range(len(corr)): 
	#	if corr[i] > 100 or corr[i] < -100: corr[i] = np.nan

	if savefile: 
		try: Mags = BuildFrame(headers=[ID_header, coord_headers[0], coord_headers[1], header], \
		         values = [frame[ID_header], frame[coord_headers[0]], frame[coord_headers[1]], corr])
		except: 
			Mags = BuildFrame(headers=[ID_header, coord_headers[0], coord_headers[1], header], \
		         values = [frame[ID_header], frame[coord_headers[0]], frame[coord_headers[1]], corr])

		Mags.to_csv(savefile)

	try: 
		frame[header] = corr.tolist()[0]
		return frame
	except: 
		return corr

###-----------------------------------------------------------------------------------------------------

def MakeCCD(clusters=False, xcolor=["F555W", "F814W"], ycolor=["F435W", "F555W"], colors=["V-I","B-V"], instrument="acs", correct_ext=False, E_BV=0.08, label_ages=True, color="black", label="", model_color="gray", model_label="", size=15, title="", xlim=(-3,3), ylim=(3,-3), Z=0.02, stellar_lib="BaSeL"): 

	"""
	Creates a color-color diagram for comparing the photometric properties of input sources 
	to the cluster color evolutionary models of Bruzual & Charlot (2003), assuming solar metallicity.
	The ACS/WFC models contain the filters F220W, F250W, F330W, F410W, F435W, F475W, F555W, F606W, 
	F625W, F775W, F814W. The WFC3/UVIS models contain F225W, F336W,  F438W, F547M, F555W, F606W, 
	F625W, F656N, F658N, and F814W. These are stored in the file CB07_models_*.csv.
	
	PARAMETERS: 
	-----------
	clusters [pd.DataFrame]	:	DataFrame containing the magnitude of each cluster in each filter
					denoted by xcolor and ycolor. 
	xcolor	[list]		:	List containing the filters used to calculate the x-axis colors.
					By default, set to ["F555W","F814W"], which equates to a V-I 
					color on the x-axis.
	ycolor	[list]		:	List containing the filters used to calculate the y-axis colors.
					By default, set to ["F435W","F555W"], which equates to a B-V 
					color on the y-axis.
	colors	[list]		: 	List containing the short-hand for the color in the x and y axes. 
					This will be used to determine which extinction factor would be 
					applied to the x and y colors and the direction/magnitude of the
					reddening arrow. 
	instrument [str] ('acs'):	HST instrument (acs or wfc3) with which observations were taken. Default is 'acs'.
	correct_ext [bool]	: 	Adjust the color of clusters to correct for extinction (reddening)
					using the Milky Way extinction law and E_BV.
	E_BV	[float]	(0.08)	:	Galactic reddening towards the galaxy or source of interest.
					Used to adjust the extinction arrow vector.
	label_ages [bool]	: 	If true, plots markers to indicate cluster ages of 10 Myr and 400 Myr. Defaults to True.
	color	[str]	(black)	:	Color of the cluster markers. 
	label	[str]		:	Legend label of the cluster points. 
	model_color [str] (gray):	Color of the model line.
	model_label [str]	:	Legend label of the model. 
	size	[int]	(15)	:	Cluster marker size. 
	title	[str]		: 	Title of the figure. 
	xlim	[tuple] (-3,3) 	:	Limits on the x-axis of the figure. 
	ylim 	[tuple] (3,-3) 	:	Limits on the y-axis of the figure.  
	Z	[float]	(0.02)	:	Stellar metallicity to compare to models. Defaults to approximately solar (Z=0.02). 
					Other options are: 0.0001, 0.0004, 0.004, 0.008, 0.05, 0.10.
	stellar_lib [str]	:	The name of the stellar libary to use, available in CB07 (from https://www.bruzual.org/). 
					Defaults to "BaSel" (Lastennet et al. 2001). Other options are "xmiless" 
					(MILES, Falcón-Barroso et al. 2011) or "stelib" (Le Borgne et al. 2003). 
	
	RETURNS: 
	----------- 

	Plots input clusters against the cluster color evolution models, including an arrow pointing in the direction of reddening. 

	Returns plt figure.

	"""

	if instrument.lower() == "acs": model = CB07_acs.copy()
	elif instrument.lower() == "wfc3": model = CB07_wfc3.copy()

	model = Find(model, ["Z = " + str(Z), "Library = " + stellar_lib])

	# Calculating reddening factors from the MW reddening law
	Rv = 3.19  # from MW extinction law
	Av = Rv * E_BV

	# From other relations: 
	Au = 1.586 * Av     # 5.06
	Ab = (1 + Rv)*E_BV  # 4.19
	Ai = 0.536 * Av     # 1.71

	E_UB = Au - Ab
	E_UV = Au - Av
	E_UI = Au - Ai
	E_VI = Av - Ai
	E_BI = Ab - Ai
	E_BV = Ab - Av

	# Finding the appropriate reddening factor based on the input x and y colors
	if colors[0] == "U-B": Ex = E_UB
	elif colors[0] == "U-V": Ex = E_UV
	elif colors[0] == "U-I": Ex = E_UI
	elif colors[0] == "V-I": Ex = E_VI
	elif colors[0] == "B-I": Ex = E_BI
	elif colors[0] == "B-V": Ex = E_BV

	if colors[1] == "U-B": Ey = E_UB
	elif colors[1] == "U-V": Ey = E_UV
	elif colors[1] == "U-I": Ey = E_UI
	elif colors[1] == "V-I": Ey = E_VI
	elif colors[1] == "B-I": Ey = E_BI
	elif colors[1] == "B-V": Ey = E_BV

	# If user wishes to apply extinction correction, set the extinction factor
	if correct_ext: 
		Ex_clust = Ex
		Ey_clust = Ey
	else: 
		Ex_clust = 0
		Ey_clust = 0

	plt.figure(figsize=(4.5,4.5))
	plt.tick_params(direction="in", width=1.4, length=7)


	# Pulling x and y colors from the model based on the input DataFrame
	# These should be colors, but this code allows user to input a magnitude, in case it's more useful for special cases
	for head in model.columns.values.tolist(): 
		if isinstance(xcolor, list):
			if xcolor[0] in head: x0 = head
			if xcolor[1] in head: x1 = head
		if isinstance(xcolor, str): 
			print("WARNING: Single magnitude detected for xcolor. It is advised to only use colors for these models!")
			if xcolor in head: x0 = head
		if isinstance(ycolor, list): 
			if ycolor[0] in head: y0 = head
			if ycolor[1] in head: y1 = head
		if isinstance(ycolor, str): 
			print("WARNING: Single magnitude detected for ycolor. It is advised to only use colors for these models!")
			if ycolor in head: y0 = head


	# Headers in the models are in V-<filter> format, so they must be subtracted backwards
	try: xmodel = model[x1]-model[x0]
	except: xmodel = model["Vmag"] - model[x0]
	try: ymodel = model[y1]-model[y0]
	except: ymodel = model["Vmag"]-model[y0]

	# Plotting the cluster evolutionary model
	plt.plot(xmodel, ymodel, color=model_color, label=model_label, alpha=0.5)

	if label_ages:
		# Plotting the models for young and globular clusters
		TempAge = Find(model, "log-age-yr = 7") # 10 Myrs

		if isinstance(xcolor, list): xage = TempAge[x1]-TempAge[x0]
		else: TempAge[x0]
		if isinstance(ycolor, list): yage = TempAge[y1]-TempAge[y0]
		else: yage = TempAge[y0]

		plt.scatter(xage,yage, marker="v", color=model_color, s=75, zorder=5)
		plt.annotate("10 Myrs", (xage, yage), zorder=999)

		TempAge = Find(model, "log-age-yr = 8.606543") # ~400 Myr

		if isinstance(xcolor, list): xage = TempAge[x1]-TempAge[x0]
		else: TempAge[x0]
		if isinstance(ycolor, list): yage = TempAge[y1]-TempAge[y0]
		else: yage = TempAge[y0]

		plt.scatter(xage, yage, marker="v", color=model_color, s=75, zorder=5)
		plt.annotate("400 Myrs", (xage, yage), zorder=999)

	# Plotting the reddening arrow
	print("Plotting reddening arrow for", colors[0], "vs.", colors[1])
	plt.arrow(x=xlim[1]*0.75,y=ylim[1]*0.75, dx=Ex, dy=Ey, head_width=.05, color="black")


	# Plotting clusters from input dataframe
	if isinstance(clusters, pd.DataFrame): 
		clusters = clusters.copy()

		if isinstance(xcolor, list): xvals = clusters[xcolor[0]] - clusters[xcolor[1]]+Ex_clust
		else: xvales = clusters[xcolor] + Ex_clust

		if isinstance(ycolor, list): yvals = clusters[ycolor[0]] - clusters[ycolor[1]]+Ey_clust
		else: yvals = clusters[ycolor] + Ey_clust

		plt.scatter(xvals, yvals, s=size, color=color, label=label)

	plt.xlim(xlim)
	plt.ylim(ylim)
	if isinstance(xcolor, list): plt.xlabel(xcolor[0] + " - " + xcolor[1],fontsize=20)
	else: plt.xlabel(xcolor,fontsize=20)
	if isinstance(ycolor, list): plt.ylabel(ycolor[0] + " - " + ycolor[1],fontsize=20)
	else:plt.ylabel(ycolor,fontsize=20)

	plt.title(title)
	return plt

###-----------------------------------------------------------------------------------------------------

def AddCCD(fig, clusters=False, xcolor=["F555W", "F814W"], ycolor=["F435W", "F555W"], colors=["V-I","B-V"], instrument="acs", correct_ext=False, E_BV=0.08, label_ages=True, color="black", label="", model_color="gray", model_label="", size=15, Z=0.02, stellar_lib="BaSeL"): 

	"""
	Adds a secondary color-color diagram to one already built with MakeCCD. User must read in the object returned by MakeCCD for this to plot properly.
	
	PARAMETERS: 
	-----------
	fig	[plt object]	:	plt returned from MakeCCD.
	clusters [pd.DataFrame]	:	DataFrame containing the magnitude of each cluster in each filter
					denoted by xcolor and ycolor. 
	xcolor	[list]		:	List containing the filters used to calculate the x-axis colors.
					By default, set to ["F555W","F814W"], which equates to a V-I 
					color on the x-axis.
	ycolor	[list]		:	List containing the filters used to calculate the y-axis colors.
					By default, set to ["F435W","F555W"], which equates to a B-V 
					color on the y-axis.
	colors	[list]		: 	List containing the short-hand for the color in the x and y axes. 
					This will be used to determine which extinction factor would be 
					applied to the x and y colors and the direction/magnitude of the
					reddening arrow. 
	correct_ext [bool]	: 	Adjust the color of clusters to correct for extinction (reddening)
					using the Milky Way extinction law and E_BV.
	E_BV	[float]	(0.08)	:	Galactic reddening towards the galaxy or source of interest.
					Used to adjust the extinction arrow vector.
	label_ages [bool]	: 	If true, plots markers to indicate cluster ages of 10 Myr and 400 Myr. Defaults to True.
	color	[str]	(black)	:	Color of the cluster markers. 
	size	[int]	(15)	:	Cluster marker size. 
	label	[str]		:	Legend label of the cluster points. 
	model_color [str] (gray):	Color of the model line.
	model_label [str]	:	Legend label of the model. 
	instrument [str] ('acs'):	HST instrument (acs or wfc3) with which observations were taken. Default is 'acs'. 
	Z	[float]	(0.02)	:	Stellar metallicity to compare to models. Defaults to approximately solar (Z=0.02). 
					Other options are [0.0001, 0.0004, 0.004, 0.008, 0.05, 0.10].
	stellar_lib [str]	:	The name of the stellar libary to use, available in CB07 (from https://www.bruzual.org/). 
					Defaults to "BaSel" (Lastennet et al. 2001). Other options are "xmiless" (MILES, Falcón-Barroso et al. 2011)
					or "stelib" (Le Borgne et al. 2003). 
	
	RETURNS: 
	----------- 

	Plots input clusters against the cluster color evolution models, including an arrow pointing in the direction of reddening. 

	Returns plt figure.

	"""

	if instrument.lower() == "acs": model = CB07_acs.copy()
	elif instrument.lower() == "wfc3": model = CB07_wfc3.copy()

	model = Find(model, ["Z = " + str(Z), "Library = " + stellar_lib])

	# Calculating reddening factors from the MW reddening law
	Rv = 3.19  # from MW extinction law
	Av = Rv * E_BV

	# From other relations: 
	Au = 1.586 * Av     # 5.06
	Ab = (1 + Rv)*E_BV  # 4.19
	Ai = 0.536 * Av     # 1.71

	E_UB = Au - Ab
	E_UV = Au - Av
	E_UI = Au - Ai
	E_VI = Av - Ai
	E_BI = Ab - Ai
	E_BV = Ab - Av

	# Finding the appropriate reddening factor based on the input x and y colors
	if colors[0] == "U-B": Ex = E_UB
	elif colors[0] == "U-V": Ex = E_UV
	elif colors[0] == "U-I": Ex = E_UI
	elif colors[0] == "V-I": Ex = E_VI
	elif colors[0] == "B-I": Ex = E_BI
	elif colors[0] == "B-V": Ex = E_BV

	if colors[1] == "U-B": Ey = E_UB
	elif colors[1] == "U-V": Ey = E_UV
	elif colors[1] == "U-I": Ey = E_UI
	elif colors[1] == "V-I": Ey = E_VI
	elif colors[1] == "B-I": Ey = E_BI
	elif colors[1] == "B-V": Ey = E_BV

	# If user wishes to apply extinction correction, set the extinction factor
	if correct_ext: 
		Ex_clust = Ex
		Ey_clust = Ey
	else: 
		Ex_clust = 0
		Ey_clust = 0


	# Pulling x and y colors from the model based on the input DataFrame
	# These should be colors, but this code allows user to input a magnitude, in case it's more useful for special cases
	for head in model.columns.values.tolist(): 
		if isinstance(xcolor, list):
			if xcolor[0] in head: x0 = head
			if xcolor[1] in head: x1 = head
		if isinstance(xcolor, str): 
			print("WARNING: Single magnitude detected for xcolor. It is advised to only use colors for these models!")
			if xcolor in head: x0 = head
		if isinstance(ycolor, list): 
			if ycolor[0] in head: y0 = head
			if ycolor[1] in head: y1 = head
		if isinstance(ycolor, str): 
			print("WARNING: Single magnitude detected for ycolor. It is advised to only use colors for these models!")
			if ycolor in head: y0 = head

	# Headers in the models are in V-<filter> format, so they must be subtracted backwards
	try: xmodel = model[x1]-model[x0]
	except: xmodel = model["Vmag"] - model[x0]
	try: ymodel = model[y1]-model[y0]
	except: ymodel = model["Vmag"]-model[y0]

	# Plotting the Solar model
	fig.plot(xmodel, ymodel, color=model_color, label=model_label, alpha=0.5)

	if label_ages:
		# Plotting the models for young and globular clusters
		TempAge = Find(model, "log-age-yr = 7") # 10 Myrs

		if isinstance(xcolor, list): xage = TempAge[x1]-TempAge[x0]
		else: TempAge[x0]
		if isinstance(ycolor, list): yage = TempAge[y1]-TempAge[y0]
		else: yage = TempAge[y0]

		plt.scatter(xage,yage, marker="v", color=model_color, s=75, zorder=5)
		plt.annotate("10 Myrs", (xage, yage), zorder=999)

		TempAge = Find(model, "log-age-yr = 8.606543") # ~400 Myr

		if isinstance(xcolor, list): xage = TempAge[x1]-TempAge[x0]
		else: TempAge[x0]
		if isinstance(ycolor, list): yage = TempAge[y1]-TempAge[y0]
		else: yage = TempAge[y0]

		plt.scatter(xage, yage, marker="v", color=model_color, s=75, zorder=5)
		plt.annotate("400 Myrs", (xage, yage), zorder=999)

	# Plotting clusters from input dataframe
	if isinstance(clusters, pd.DataFrame): 
		clusters = clusters.copy()

		if isinstance(xcolor, list): xvals = clusters[xcolor[0]] - clusters[xcolor[1]]+Ex_clust
		else: xvales = clusters[xcolor] + Ex_clust

		if isinstance(ycolor, list): yvals = clusters[ycolor[0]] - clusters[ycolor[1]]+Ey_clust
		else: yvals = clusters[ycolor] + Ey_clust

		fig.scatter(xvals, yvals, s=size, color=color, label=label)

	return fig
###-----------------------------------------------------------------------------------------------------

def FitSED(df, instrument, idheader, photheads=False, errorheads=False, fittype="reduced chi2", min_models=1, input_model=False, model_header_index=13): 

	"""
	Function for finding the best fit stellar SED from isochrone models. Reads in the photometric measurements from input source(s) across 
	multiple filters and compares their values to those given by the Padova isochrones for theoretical stellar evolutionary models. 

	PARAMETERS: 
	-----------
	df	[pd.DataFrame]	:	DataFrame containing the magnitudes and photometric errors of the source(s) of interest. 
	instrument 	[str]	:	Instrument with which the measurements were taken. Currently accepts "acs" and "wfc3", but 
					will be able to take "nircam" for JWST observations in the future. 
	idheader	[str]	:	Header under which the source ID is stored. This will be used used to indicate which best-fit model is associated 
					with which source when df contains more than one source to fit.
	photheads	[list]	:	List of headers under which the photometric measurements per filter are stored within the 'df' DataFrame. 
					The measured magnitudes should be stored under the name of the filter with which they were taken 
					(e.g. "F814W", "F814Wmag", etc.), matching the headers of the isochrone models. If 'filterheads' is left blank, 
					the code will use the filter headers found in the isochrone models as the photometry headers. In searching for 
					the appropriate model headers, the code assumes the headers in the isochrone models begin with F and that no 
					other header in the model table does. This is a reasonable assumption for both HST and JWST, but may need to be 
					revisited for other models.
	errorheads 	[list]	:	List of headers under which the photometric errors for each filter are stored (e.g. "F814W Err", "F555W Err"). 
					If left blank, the code will assign values based on the values in 'photheads'.
	fittype		[str]	:	Defines the algorithm use to determine the best-fit isochrone (currently not necessary, as only reduced chi2
					has been coded. In the future MCMC will also be included). 
					"reduced chi2" (default) selects the model for which the resulting reduced chi-squared is closest to 1. 
					"mcmc" (pending) uses an MCMC algorithm to determine the best-fit model. 
	min_models	[int]	:	Minimum number of models to save for each source. By default, only the best-fit model will be returned. 
					If min_models >= 2, the next closest fit(s) up to min_models will be returned as well 
					(e.g. if min_models = 3, the 2 next closest fits will also be given). If several models fit equally well,
					all will be included in isoMatches.
	input_model	[str]	:	If preferred, user can input the name of a file containing the preferred isochrone models from the 
					Padova website. The code assumes the document is copied and pasted from the CMD output page, with the table
					headers on the 14th line (unless the file is a CSV DataFrame); this can be modified using the 'model_header_index' 
					parameter. One should be sure to change the magnitude headers to match those of their 'df', or vice versa. 
	model_header_index [int]:	Index of the line containing the headers of the isochrone models, if a .txt file is read in as input_model. 
					By default, this is line 13. This parameter can be ignored in input_model is the name of a CSV DataFrame.  

	RETURNS: 
	---------
	isoMatches [pd.DataFrame] :	DataFrame containing the best-fit models for each source in 'df'. 
	
	"""

	# If input_model is given, read in that file for the isochrone
	# Otherwise, the instrument determines which isochrones to use
	if input_model != False:
		# Assumes first that the input is a CSV DataFrame. If not, the length of the header will = 1
		isoTemp = pd.read_csv(input_model, comment="#")
		if len(isoTemp.columns.tolist()) == 1: # the file was not a DataFrame, so treat as regular .txt file
			# Pulls the header of the model file based on model_header_index
			names = [n for n in open(input_model).readlines()[model_header_index].strip("# ").split()]
			isoTemp = pd.read_csv(input_model, comment="#", delim_whitespace=True, names=names)
		else: 
			try: isoTemp = isoTemp.drop(columns=["Unnamed: 0"])
			except: pass;
	elif "acs" in instrument.lower(): isoTemp = isoacs.copy()
	elif "wfc3" in instrument.lower(): isoTemp = isowfc3.copy()


	# Tries to find the filter headers in isoTemp, assuming they all start with "F" and no other headers do. 
	filters = [filt for filt in isoTemp.columns.tolist() if filt[0] == "F" and "ID" not in filt]
	filters.sort()
	
	# Figure out the source headers to pull the photometry from df. If not given, assume they match the model header format
	if photheads == False: photheads = [h for h in filters if h in df.columns.tolist()]
	if errorheads == False: errorheads = [f"{h} Err" for h in photheads]


	# Keeping track of the ID of each source we will model
	sourceids = df[idheader].values.tolist()
	
	# Keeping track of the photometry of each source in each filter (and errors)
	# Each row of this list represents a single source, and each column represents the magnitude for each
	sourcemags = [[df[f][i] for f in photheads] for i in range(len(df))]
	sourcemag_errs = [[df[e][i] for e in errorheads] for i in range(len(df))]

	# Each best-fit model will be added to a separate DataFrame, which will be returned to the user at the end
	isoMatches = BuildFrame(headers=isoTemp.columns.tolist())


	# IN THE FUTURE, THIS PART WILL BE FLAGGED BY fittype
	
	# The fastest way to determine the best-fit model is to do so directly in the isoTemp DataFrame
	# For each star, grab the photometry and compare to the isochrones. 
	# Then find the isochrone for which the reduced chi-square is closest to 1, the best-fit
	
	isoMatches["Reduced Chi2"] = np.nan
	isoMatches["Reduced Chi2 - 1"] = np.nan
	isoMatches[idheader] = None

	print("Finding best-fit model(s)...")
	for star in range(len(df)): 
		# As long as there is at least one good magnitude value associated with the star...
		if False in [np.isnan(sourcemags[star][f]) for f in range(len(photheads))]:
			# For each filter, find the difference of the (measurements - model)^2/(errors)^2 and take the sum
			isoTemp["Reduced Chi2"] = np.nansum([(sourcemags[star][f]-isoTemp[photheads[f]].values)**2/sourcemag_errs[star][f]**2 if isinstance(sourcemags[star][f], float) else 0 for f in range(len(photheads))], axis=0)

			# The best model is one such that reduced chi2 is close to 1. Values much higher than 1 are underfit, much lower than 1 is overfit. 
			isoTemp["Reduced Chi2 - 1"] = np.abs(isoTemp["Reduced Chi2"] - 1)
		
			redchi2s = sorted(isoTemp["Reduced Chi2 - 1"].values.tolist()) # sorted list of reduced chi2 - 1
			
			# Searching for at least min_models number of best-fit models
			temp = Find(isoTemp, f"Reduced Chi2 - 1 < {redchi2s[min_models]}")

			# Adding the source ID to the DataFrame, to be added to isoMatches
			temp[idheader] = sourceids[star]

			# Adding the best-fit model(s) to the DataFrame to return to the user
			isoMatches = pd.concat([isoMatches, temp], ignore_index=True)

		else: pass; # if there are no good values, skip this star
	print("DONE")

	return isoMatches
	
###-----------------------------------------------------------------------------------------------------

def PlotSED(df_sources, df_models, idheader, fitheader="Reduced Chi2 - 1", massheader="Mass", sourceheads=False, errorheads=False, modelheads=False, modelparams=["Mass", "logAge", "logL", "logTe", "Reduced Chi2 - 1"], showtable=True, showHR=False): 
	"""
	Takes in the photometric measurements of sources in a DataFrame and the best-fit isochrones DataFrame from FitSED and plots 
	them together onto a chart. If more than one model is given for a single source ID (given as idheader), then the model with 
	the smallest Reduced Chi2 - 1 (or whatever header is read in as fitheader) is marked as the best fit, and minimum and 
	maximum estimated mass is indicated. 
	
	PARAMETERS: 
	------------
	df_sources 	[pd.DataFrame]	:	DataFrame containing the measured photometry of the sources 
	df_models 	[pd.DataFrame]	:	DataFrame containing the best-fit SED models returned as isoMatches by FitSED.
	idheader	[str]		:	Header containing the source ID in the df_model DataFrame. This is used to 
						separate the SEDs into separate plots per source. 
	fitheader	[str]		:	Header containing the value used as the criteria for finding the best fit
						by FitSED. The default is "Reduced Chi2 - 1", as used in FitSED. 
	massheader	[str]		:	Header containing the mass of each stellar model, in the event that there are multiple models
						per source. The default is "Mass". 
	sourceheads	[list]		:	Headers containing the measured magnitudes of each source per filter, in increasing 
						wavelength order, in df_sources (e.g. F814W or F814Wmag). If left blank, PlotSED will 
						search for headers that match those in df_models (or modelheads). 
	errorheads	[list]		:	Headers containing the photometric errors of each sourcce. If none are given, assumes 
						the headers are in the format '<modelheads> Err'.
	modelheads	[list]		:	Headers containing the model magnitudes of the simulated star per filter, in increasing
						wavelength order, in df_models. If left blank, PlotSED will automatically search 
						the headers of df_models to find all possible wavelength headers (e.g. F814W or F814Wmag).
	modelparams	[list]		:	List of parameters from the df_model to pull and display. These will be printed
						as a table alongside the SED plot, sorted in order of best to worst fit. The defaults are 
						["Mass", "logAge", "logL", "logTe", "Reduced Chi2 - 1"].
	showtable	[bool]		:	If True, shows the table of model parameters given by 'modelparams', in order of best to worst fit.
	showHR		[bool]		:	If True, plots the H-R diagram of the model star(s), indicating the most likely spectral type(s). 
						(Feature coming soon!)

	RETURNS: 
	----------
	Plots the best-fit models against the measured magnitudes of each source. 

	"""

	# Finding all unique sources from df_sources, to find their corresponding models in df_models 
	sourceids = FindUnique(df_sources, header=idheader)[idheader].values.tolist()
	
	# Setting up all of the headers

	# If no model headers are given, tries to find the photometry headers in df_models, assuming they all start with "F" and no other headers do. 
	if modelheads == False: 
		modelheads = [filt for filt in df_models.columns.tolist() if filt[0] == "F" and "ID" not in filt]
		modelheads.sort()
	
	# Figure out the source headers to pull the photometry from df_sources. If not given, assume they match the modelheads format
	if sourceheads == False: sourceheads = [h for h in modelheads if h in df_sources.columns.tolist()]
	if errorheads == False: errorheads = [f"{h} Err" for h in sourceheads]

	# Setting up wavelengths of sources and models, in angstroms
	# JWST has filters containing W2, which mess up the function below. Replacing those with X
	modelwavs = [int(re.sub('\D', '', h.replace('W2','W')))*10 for h in modelheads]
	sourcewavs = [int(re.sub('\D', '', h.replace('W2','W')))*10 for h in sourceheads]

	print(modelwavs)
	print(sourcewavs)

	# Plot each source separately
	for s in sourceids: 
		TempModel = Find(df_models, f"{idheader} = {s}")	# May contain multiple models, from min_models in FitSED
		TempSource = Find(df_sources, f"{idheader} = {s}") 	# Should be a single source

		# If there is no model for the given star, let user know
		if len(TempModel) == 0: print(f"No best-fit model available for Source ID {s}.")

		# Otherwise, plot the models
		else: 
			
			# Pulls the magnitudes and wavelengths from the models
			modelmags_all = [[TempModel[h][m] for h in modelheads] for m in range(len(TempModel))]
			modelchis_all = TempModel[fitheader].values.tolist()
			modelmass_all = TempModel[massheader].values.tolist()

			# Also keep track of parameters the user wants to print from each model
			# The table will be sorted from best to worst fit.
			chisort = sorted(enumerate(modelchis_all), key=lambda i: i[1]) # Sorting the fit parameters
			modelparams_all = [[round(TempModel[p][m[0]],8) for p in modelparams] for m in chisort]
	
			# Finding the index of the best fit, the minimum mass model, and the maximum mass model.
			bestind = chisort[0][0]
			minmassind = sorted(enumerate(modelmass_all), key=lambda i: i[1])[0][0]
			maxmassind = sorted(enumerate(modelmass_all), key=lambda i: i[1])[-1][0]
			
			# Pulling the magnitudes and wavelengths from source DataFrame
			sourcemags = [TempSource[h][0] for h in sourceheads] # list of photometries (mags)
			errormags = [TempSource[h][0] for h in errorheads] # list of photometries (mags)

			# Plotting the models
			plt.figure(figsize=(4,4))

			# On the left, plot the models and observations
			#plt.subplot(1, 2, 1)
			plt.xlabel("HST Filter (Angstrom)")
			plt.ylabel("Absolute Magnitude")

			# Plotting all models with low opacity
			for mod in modelmags_all: 
				plt.plot(modelwavs, mod, color="silver", alpha=0.3)

			# Plotting the best-fit model
			plt.plot(modelwavs, modelmags_all[bestind], lw=3, label="Best-Fit")
			
			# If there is more than one model, indicate the minimum and maximum mass
			if len(TempModel) > 1: 
				plt.plot(modelwavs, modelmags_all[minmassind], lw=3, color="pink", label="Min. Mass", linestyle=":")
				plt.plot(modelwavs, modelmags_all[maxmassind], lw=2, color="lightblue", label="Max. Mass", linestyle="--")

			# Plotting observed magnitudes and their errorbars
			plt.scatter(sourcewavs, sourcemags, marker="s", edgecolor="black", facecolor="none", s=40, linestyle='None', label="Observed", zorder=999)
			plt.errorbar(sourcewavs, sourcemags, yerr=[np.abs(m) for m in errormags], c="black", linestyle='None', zorder=999)

			if showtable:
				# On the right or bottom of plot, plot a table of model parameters
				# Setting table position/size, where bbox = [xmin, ymin, width, height] of table
				bbox = [1,1-max(0.2,0.1*len(TempModel)),1+(.25*len(modelparams)),max(0.2,0.1*len(TempModel))]

				# Plotting table
				the_table = plt.table(cellText=modelparams_all, colLabels=modelparams, bbox=bbox)
				the_table.auto_set_font_size(False)
				the_table.set_fontsize(10)
			
			plt.title(f"Source ID: {s}")
			plt.legend()
			plt.show()

		if showHR: PlotHR(TempModel, figsize=(4,4))

###-----------------------------------------------------------------------------------------------------

def PlotHR(df=False, logTeheader="logTe", logLheader="logL", idheader=False, figsize=(5,5), colormap="RdYlBu_r"):
 
	"""
	Plots a rough estimate of the HR diagram with MS, giant, and supergiant regions denoted. 
	The values for each region come from a mixture of HR diagram images, Wikipedia, and van Belle et al. 2021. 
	Stars can be plotted on the diagram by reading in isoMatches (or a derivative thereof) from FitSED. 

	PARAMETERS:
	-----------
	df	[pd.DataFrame; list]	:	DataFrame containing the log effective temperature and log luminosity of each star, 
						or a list of [logTe, logL] for each star.
						If none is given, a blank H-R diagram will be plotted instead
	logTeheader	[str]		:	Header under which the log effective temperature is stored in 'df'. 
						The default is logTe (matching isoMatches).
	logLheader	[str]		:	Header under which the log stellar luminosity is stored in 'df'. 
						The default is logL (matching isoMatches).
	idheader	[str, list]	:	Header under which the source ID is stored in 'df', or a list of IDs if df is a list of values. 
	figsize		[tuple]		: 	Size of the plot to print. Default is (5,5)
	colormap	[str]		:	Matplotlib colormap used to differentiate between spectral types. Default is "RdYlBu_r".		

	"""

	# Estimated HR diagram regions (mins and maxes)
	ms_teffs = [[4.48,6],[4.0,4.48],[3.88,4.0],[3.78,3.88],[3.72,3.78],[3.57,3.72],[3.38,3.57]]
	ms_logls = [[4.48,6],[1.4,4.48],[0.7,1.4],[.18,.7],[-.22,.18],[-1.1,-.22],[-1.1,-3]]

	g_teffs = [[np.nan,np.nan],[4.,4.08],[3.84,3.98],[3.75,3.83],[3.68,3.74],[3.59,3.68],[3.36,3.58]]
	g_logls = [[np.nan,np.nan],[np.nan,np.nan],[1,2],[1,2],[1.5,2.2],[1.8,2.4],[2,2.3]]

	sg_teffs = [[4.48,6],[4.0,4.48],[3.88,4.0],[3.78,3.88],[3.72,3.78],[3.57,3.72],[3.38,3.57]]
	sg_logls = [[5,6],[5,6],[5,6],[5,6],[5,6],[5,6],[5,6]]

	startypes=["O","B","A","F","G", "K","M"]

	cmap = matplotlib.cm.get_cmap(colormap)
	colors = [cmap(i) for i in np.linspace(0,1,7)]

	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(111)

	# Plotting the colormap
	cax = ax.scatter(10,10,c=[6], cmap=colormap)
	cbar = fig.colorbar(cax, pad=0)
	cbar.ax.set_yticklabels(startypes)  # vertically oriented colorbar

	# for each star type, plot the regions within which we see main sequence, giant, and supergiant stars
	for i,star in enumerate(startypes):
		try:
			ms_patch =  Rectangle((ms_teffs[i][0], ms_logls[i][0]), ms_teffs[i][1]-ms_teffs[i][0], ms_logls[i][1]-ms_logls[i][0], color=colors[i], alpha=0.7, lw=0)
			ax.add_patch(ms_patch)
		except: pass;
		try:
			giant_patch =  Rectangle((g_teffs[i][0], g_logls[i][0]), g_teffs[i][1]-g_teffs[i][0], g_logls[i][1]-g_logls[i][0], edgecolor=colors[i], facecolor="none", hatch="\\\\", lw=2)
			ax.add_patch(giant_patch)
		except: pass;
		try:
			sg_patch =  Rectangle((sg_teffs[i][0], sg_logls[i][0]), sg_teffs[i][1]-sg_teffs[i][0], sg_logls[i][1]-sg_logls[i][0], edgecolor=colors[i], facecolor="none", hatch="||", lw=2)
			ax.add_patch(sg_patch)
		except: pass;

	# Extra patches for labels
	ms_patch =  Rectangle((10,10),1,1, color="black", alpha=0.5, lw=0, label="MS")
	giant_patch =  Rectangle((10,10),1,1, edgecolor="black", facecolor="none", hatch="\\\\", lw=2, label="Giant")
	sg_patch =  Rectangle((10,10), 1,1,edgecolor="black", facecolor="none", hatch="||", lw=2,  label="SG")
	ax.add_patch(ms_patch)
	ax.add_patch(giant_patch)
	ax.add_patch(sg_patch)
	plt.xlim(4.6,3.4)
	plt.ylim(-2,6)
	plt.xlabel(r"log T$_{eff}$ [K]", size=15)
	plt.ylabel(r"log L [L$_{\odot}$]", size=15)
	plt.legend()

	# Plot the stars in df if given
	
	if isinstance(df, pd.DataFrame): 
		plt.scatter(df[logTeheader].values.tolist(), df[logLheader].values.tolist(), marker=r"$\star$", color="black", s=150)
		
		# Changing the x and y limits to always include the stars
		plt.xlim(max(4.6, max(df[logTeheader])+0.1), min(3.4, min(df[logTeheader])-0.1))
		plt.ylim(min(-2, min(df[logLheader])-0.1), max(6, max(df[logLheader])+0.1))
		
		if isinstance(idheader, str): 
			for i in range(len(df)): plt.text(df[logTeheader][i], df[logLheader][i]+0.3, df[idheader][i], fontsize=12)

	if isinstance(df, list): 
		if not isinstance(df[0], list): df = [df]
		plt.scatter(np.array(df).T[0], np.array(df).T[1], marker=r"$\star$", color="black", s=150)

		# Changing the x and y limits to always include the stars
		plt.xlim(min(np.array(df).T[0])-0.3, max(np.array(df).T[0])+0.3)
		plt.ylim(min(np.array(df).T[1])-0.5, max(np.array(df).T[1])+0.5)

		if isinstance(idheader, list): 
			for i in range(len(df)): plt.text(df[i][0], df[i][1]+0.3, idheader[i], fontsize=12)

	plt.show()

###-----------------------------------------------------------------------------------------------------
	
# PLANNED FUNCTIONS, UPDATE TBD
#
#def FitCCD(): 
#	"""
#	Function for finding the best fit cluster model from CB07 models. Coming soon!
#	"""
#

