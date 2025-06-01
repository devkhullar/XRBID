	
##################################################################################
##########	For calculating and plotting  X-ray properties of CSC 	##########
##########	sources, such as X-ray luminosity, hardness-ratio 	##########
##########	plots, XLFs, etc. 					##########
##########	Last updated: May 12, 2025				##########
##########	NOTE: I'm still actively working on adding descriptions	##########
##########	      to the XLF fitting algorithms. 			##########	
##################################################################################

import re
import random
import numpy as np
from numpy import pi, sqrt, log10, log
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from astropy.io.votable import parse
import pandas as pd
pd.options.mode.chained_assignment = None

# from Headers import LogL
from XRBID.DataFrameMod import BuildFrame, Find


# Setting the style of my plots
fontparams = {'font.family':'Times New Roman'}
labelparams = {'family':'Times New Roman', 'size':20}

###-----------------------------------------------------------------------------------------------------

def Lum(F, dist): 

	"""
	Calculates the luminosity of a set of sources given the flux and distance to the host galaxy. 

	PARAMETERS
	----------
	F	[list]	:	Flux of X-ray source(s), in units ergs/s/cm^2. 
	dist	[float]	:	Distance to the galaxy in units pc or cm.

	RETURNS
	---------
	L	[list]	:	List of luminosities in units ergs/s
	"""

	if not dist: dist = input("Galaxy distance (in pc, Mpc, or cm)? ")
	
	# if dist is less than 100, probably given in Mpcs. Convert to pcs.	
	if dist < 100: 
		dist = dist * 10**6
	
	# need dist in cm, since F is in ergs/s/cm2. If not given in cm, convert. 
	if dist < 1e18: 
		dist = dist * 3e18


	F = np.array(F)
	L = F*4*pi*(dist)**2	# luminosity calculation
	L = L.tolist()
	return L


###-----------------------------------------------------------------------------------------------------

def FindSNRs(df=False, HR=False, Lx=False, HR_head="HS Ratio", Lx_head="LogLx", cut_criteria=[-0.75,37.5], figsize=(4,4), plotfig=True): 

	"""
	Isolates candidate supernova remnants based on X-ray luminosity and harness ratio cuts, 
	based on the limits from Hunt et al. 2021. The limiting HR and Lx can be adjusted based on 
	user preference with cut_criteria, and additional SNR catalogs can be included with add_snr.
	
	PARAMETERS
	----------

	df		[pd.DataFrame]	:	DataFrame containing the X-ray luminosity and hardness ratio of X-ray sources 
						to be plotted. If not given, please provide HR and Lx separately. 
	HR		[list]		:	If df is not provided, list of hardness ratios to plot against X-ray luminosity.
	Lx		[list]		: 	If df is not provided, list of X-ray luminositied to plot against hardness ratios. 
						Should be given in log units. 
	HR_head		[str]		:	Header under which hardness ratios are stored in input DataFrame, df. Default is "HS Ratio". 
	Lx_head		[str]		:	Header under which X-ray luminosities are stored in input DataFrame, df. 
						Default is "LogLx". 
	cut_criteria	[list]		: 	Criteria for identifying candidate supernova remnants, in format [HR limit, Lx limit]. 
						The default is [-0.75, 37.5]. The code will mark any source with HR-Lx less than or 
						equal to these values as SNR candidates.
						Sources with Lx and HR beneath these values will be marked as SNR candidates. 
	figsize		[tuple]		:	Figure dimensions. 
	plotfig		[bool]		:	If True, generates a plot that shows SNR candidates plotted against the full sample, with 
						the cut criteria indicated with dashed lines.  
	
	RETURNS
	---------
	df_snr		[DataFrame]	:	Original DataFrame with a new header, "(SNR)", where SNR candidates are flagged with 1, and 
						sources that pass the cut are flagged with 0. 
	"""

	df_snr = df.copy()
	df_snr["(SNR)"] = 0	# Setting up new header for SNR candidates

	for i in range(len(df)): 
		if (df[HR_head][i] <= cut_criteria[0]) and (df[Lx_head][i] <= cut_criteria[1]): 
			df_snr["(SNR)"][i] = 1
		else: pass;
	
	temp_snr = Find(df_snr, "(SNR) = 1")
	if plotfig:
		fig, ax = plt.subplots()
		ax.scatter(df[HR_head], df[Lx_head], color="black", s=10)
		ax.scatter(temp_snr[HR_head], temp_snr[Lx_head], edgecolor="red", facecolor="none", 
			   s=45, label="SNR Candidates\n("+str(len(temp_snr))+" sources)", marker="d")
		ax.axvline(x=cut_criteria[0], color="grey", linestyle="--", alpha=0.4)
		ax.axhline(y=cut_criteria[1], color="grey", linestyle="--", alpha=0.4)
		plt.legend()
		plt.show()
	
	return df_snr
###-----------------------------------------------------------------------------------------------------

def MakeXLF(L, L_lo=None, L_hi=None, L_header="LogL", completeness=36.2, gal=None, bins=50, binlim=[35,39], xlim=[35,40], ylim=None, scale="log", cumul=True, color="black", label=None, savehist=False, returnhist=False, fontsize=20, labelsize=None, legendsize=20, figsize=(8,6), lw=2, linestyle="-", title=None, xlabel=True, ylabel=True, xticks=True, yticks=True, showlegend=True): 
	
	"""
	Builds an X-ray luminosity function given the log-luminosity of sources and the upper and lower limits, if available. 
	The XLF may be plotted as either a cumulative or differential XLF, with luminosity in the x-axis and number of sources
	in the y-axis.
	
	PARAMETERS
	----------
	L	[list, pd.DataFrame]	: 	Luminosity of the sources, in units ergs/s or log(ergs/s), or a pd.DataFrame in which 
						the luminosities are stored.
	L_lo		[list]		:	Lower limits on the source luminosities, in units ergs/s or log(ers/s). 
	L_hi		[list]		:	Upper limits on the source luminosities, in units ergs/s or log(ers/s).
	L_header	[str]		:	Header in which the luminosities are stored, if L is given as a pd.DataFrame.
	completeness	[float]		:	X-ray completeness limit of the X-ray observations. 
	gal		[str]		: 	Name of the galaxy. 
	bins		[int, array]	: 	Number of bins between which the sources will be divided, or an array of bins as they 
						will be plotted. For an 'unbinned' plot, can also leave empty or set to 'unbinned'. Default is 50. 
	binlim		[list]		: 	The lower and upper limit on the bins, in luminosity units. Default is [35,39]. 
	xlim 		[list]		: 	The lower and upper limit of the x-axis in the units of the luminosity. Defaults is [35,40].
	ylim		[list]		: 	The lower and upper limit of the y-axis in units counts. No default is given. 
	scale		[str]		:	The scale of the luminosity. By default it is 'log' (for standard log luminosities), 
						but one may also plot the XLF linearly ('linear').
	cumul		[bool]		: 	Determines whether the XLF will be plotted as a cumulative or differential function. Defaults to True.
	color		[str]		: 	Color of the XLF line (default is black). 
	label 		[str]		: 	Label given to the XLF being plotted, as it will appear on the legend. 
	savehist	[str]		:	Name of file to save the results of the histogram to. If no name is given, no file will be saved. 
	returnhist	[bool]		:	If set to True, returns the results of the histogram, the edges of the bins, and the centers of the bins.
	fontsize	[int]		:	Size of the font of the axes and title. Default is 20. 
	labelsize	[int]		:	Size of the tick lables in the x and y axes. Default is half of the fontsize. 
	legendsize	[int]		:	Size of the font of the legend. Default is 20. 
	figsize		[tuple]		:	Size of the figure. Defaults is (8,6). 
	lw		[int]		:	Linewidth of the XLF. Default is 2. 
	linestyle	[str]		: 	Linestyle of the XLF. Default is a solid line, '-'. Other common styles are '--', '-.', ':', and '.'.
	title		[str]		:	Title of the figure. 
	xlabel		[bool]		:	Sets whether or not to show the x-axis label. Defaults to True. 
	ylabel		[bool]		:	Sets whether or not to show the y-axis label. Defaults to True.
	xticks		[bool]		:	Sets whether or not to show the x-axis ticks. Defaults to True. 
	yticks		[bool]		:	Sets whether or not to show the y-axis ticks. Defaults to True.
	showlegend	[bool]		:	Sets whether or not to show the legend. Defaults to True.

	RETURNS 
	--------
	
	If returnhist = True, returns the histogram, bin centers, and bins from plt.hist() as h, h_cent, bins.
	
	""" 
	

	labelparams = {'family':'Times New Roman', 'size':fontsize}
	if not labelsize: labelsize=fontsize/1.2

	# If input is a DataFrame, look for the luminosities (should be under LogL, but let user define header name)
	if isinstance(L, pd.DataFrame): L = L[L_header].values.tolist()

	# Convert L to array for manipulation
	L = np.array(L)

	# If scale is linear but xlim undefined, convert xlim to linear scale
	if scale != "log" and xlim[0] < 1e3: xlim = [10**xlim[0], 10**xlim[1]]

	cumul = cumul*(-1)	# cumul=0 if False and -1 if True

	# set the bins to evenly-spaced logarithmic bins, unless otherwise directed
	if isinstance(bins, int):
		numbins = bins
		bins = np.linspace(binlim[0], binlim[1], numbins)
		if scale == "linear": bins = 10**bins
	elif bins == "unbinned" or bins == None:
		bins = np.arange(binlim[0]+0.05,binlim[1]+1.15,0.1).tolist()
	
	# If upper and lower luminosity bounds are given, use to make error bars
	try: 
		h_lo = plt.hist(L_lo[~np.isnan(L_lo)], bins=bins, histtype="step", cumulative=cumul, alpha=0)
		h_hi = plt.hist(L_hi[~np.isnan(L_hi)], bins=bins, histtype="step", cumulative=cumul, alpha=0)
		plt.close()
	except: pass;

	plt.figure(figsize=figsize)

	# Plotting the XLF
	h, h_edge, patch = plt.hist(L[~np.isnan(L)], bins=bins, histtype="step", cumulative=cumul, color=color, lw=lw, label=label, linestyle=linestyle);
	h_cent = .5*(h_edge[1:]+h_edge[:-1]);	# Finding bin centers, for errorbar alignment

	# Adding errors, if applicable
	try: 
		matplotlib.rcParams.update({'errorbar.capsize': 2})
		plt.errorbar(h_cent, h, yerr=[h-h_lo[0], h_hi[0]-h], fmt='.', \
		     color='red', lw=1, alpha=.5);
	except: pass;

	# Plotting completeness line
	if completeness: 
		plt.axvline(x=completeness, color="black", linestyle="--")

	# Setting tick parameters
	plt.tick_params(which="major", length=10, direction="in", width=1.5, labelsize=labelsize, zorder=0)
	plt.tick_params(which="minor", length=5, direction="in", width=1.5, zorder=0)
	plt.tick_params(labelbottom=xticks, labelleft=yticks, zorder=0)
	plt.yscale('log');

	# Defining the x axis based on the scale (scale will change the units)
	if scale == "linear": 
		plt.xscale('log');		
		if xlabel: plt.xlabel('L [erg/s]', labelparams, fontsize=fontsize);
	else: 
		if xlabel: plt.xlabel('log L [erg/s]', labelparams, fontsize=fontsize);
	plt.xlim(xlim[0], xlim[1]);
	if ylim: plt.ylim(ylim[0], ylim[1]);
	else: plt.ylim(0.1, 500)
	if title: 
		if isinstance(title, bool): 
			plt.title(gal + " XLF", size=fontsize);
		else: plt.title(title, fontsize=fontsize);
	if ylabel: plt.ylabel('N(>L)', labelparams);
	if label and showlegend: plt.legend(fontsize=legendsize)

	plt.rcParams.update(fontparams)

	if savehist: 
		with open(savehist, "w") as f: 
			np.savetxt(f, np.column_stack([h, h_cent]))
		print(str(savehist) + " saved.")

	# If requested, return the results of the histogram, the edges of the bins, and the centers of the bins
	if returnhist: return h, h_cent, bins

###-----------------------------------------------------------------------------------------------------

def AddXLF(L, L_lo=None, L_hi=None, L_header='LogL', label=None, bins=50, scale="log", cumul=True, color=None, lw=2, linestyle="-",savehist=None, legendsize=20, showlegend=True): 

	"""
	Adds in multiple XLFs to a single plot. Must do sequencially when called, or plot will automatically print after calling. 

	
	PARAMETERS
	----------
	L	[list, pd.DataFrame]	: 	Luminosity of the sources, in units ergs/s or log(ergs/s), or a pd.DataFrame in which the luminosities are stored.
	L_lo		[list]		:	Lower limits on the source luminosities, in units ergs/s or log(ers/s). 
	L_hi		[list]		:	Upper limits on the source luminosities, in units ergs/s or log(ers/s).
	L_header	[str]		:	Header in which the luminosities are stored, if L is given as a pd.DataFrame.
	label 		[str]		: 	Label given to the XLF being plotted, as it will appear on the legend. 
	bins		[int, array]	: 	Number of bins between which the sources will be divided, or an array of bins as they 
						will be plotted. For an 'unbinned' plot, can also leave empty or set to 'unbinned'. Default is 50. 
	scale		[str]		:	The scale of the luminosity. By default it is 'log' (for standard log luminosities), 
						but one may also plot the XLF linearly ('linear').
	cumul		[bool]		: 	Determines whether the XLF will be plotted as a cumulative or differential function. Defaults to True.
	color		[str]		: 	Color of the XLF line (default is black). 
	lw		[int]		:	Linewidth of the XLF. Default is 2. 
	linestyle	[str]		: 	Linestyle of the XLF. Default is a solid line, '-'. Other common styles are '--', '-.', ':', and '.'.
	savehist	[str]		:	Name of file to save the results of the histogram to. If no name is given, no file will be saved. 
	legendsize	[int]		:	Size of the font of the legend. Default is 20. 
	showlegend	[bool]		:	Sets whether or not to show the legend. Defaults to True.
	
	"""

	if not color: color = np.random.rand(3,)

	# If DataFrame is read in, find luminosities
	if isinstance(L, pd.DataFrame): L = L[L_header].values.tolist()

	# Set luminosities to an array
	L = np.array(L)

	# Set cumul=0 or cumul=-1 based on bool
	cumul = cumul*(-1)

	# set the bins to evenly-spaced logarithmic bins, unless otherwise directed
	if isinstance(bins, int):
		numbins = bins
		bins = np.linspace(35, 39, numbins)
		if scale == "linear": bins = 10**bins
	else: pass;
	
	# If upper and lower luminosity bounds are given, use to make error bars
	try: 
		h_lo = plt.hist(L_lo[~np.isnan(L_lo)], bins=bins, histtype="step", cumulative=cumul, alpha=0)
		h_hi = plt.hist(L_hi[~np.isnan(L_hi)], bins=bins, histtype="step", cumulative=cumul, alpha=0)
	except: pass;

	# Plotting the XLF
	h, h_edge, patch = plt.hist(L[~np.isnan(L)], bins=bins, histtype="step", cumulative=cumul,lw=lw, linestyle=linestyle, color=color, label=label);
	h_cent = .5*(h_edge[1:]+h_edge[:-1]);

	# Adding errors, if applicable
	try: 
		matplotlib.rcParams.update({'errorbar.capsize': 2})
		plt.errorbar(h_cent, h, yerr=[h-h_lo[0], h_hi[0]-h], fmt='.', \
		     color='red', lw=1, alpha=.5);
	except: pass;

	if label and showlegend: plt.legend(fontsize=legendsize)

	if savehist: 
		with open(savehist, "w") as f: 
			np.savetxt(f, np.column_stack([h, h_edge]))


###-----------------------------------------------------------------------------------------------------

def GoodBins(L_bins, counts, completeness=36.2, recenter=0, sigma=None): 

	"""
	Returns the good bins from the histogram binning, based on the lower completeness limit [completeness]. 
	Removes all luminosity bins below this limit. Also removes all bins with zero counts, and recenters the L_bins based 
	on the given bin widths [recenter].
	
	This function is called by fitXLF(). It returns	L_bins, counts, and sigma of bins above the completeness limit.
	""" 

	limit = completeness 
	# plt.hist() will return luminosity bins including the left edge of the first bin, followed by 
	# the right edge of all the other bins, making len(L) = len(counts) + 1.
	# If given, must first remove this bin. 

	if len(L_bins) != len(counts):
		L_bins = L_bins[1:] 

	# Removing all bins below the limit
	counts = counts[L_bins > limit]
	try: sigma[L_bins > limit]
	except: pass;
	L_bins = L_bins[L_bins > limit]
	
	# Removing all unpopulated bins and recentering L_bins by the bin width
	L_bins = L_bins[counts > 0] - recenter	
	try: sigma = sigma[counts > 0]
	except: pass;
	counts = counts[counts > 0]

	return L_bins, counts, sigma

###-----------------------------------------------------------------------------------------------------

def linPL(L, index, K, alpha=False, L0_in=False, cumul=True, convert_power=False):

	"""
	Linear version of the power law, for fitting data to a basic power law function. The general form is some log(N) = -index * log(L) + K.
	The form of index and K depend on what kind of N is input: if N = cumulative = N(>L), index = gamma = (alpha - 1), K = -gamma*log(L0); 
	if N = differential = dN/dL or dN/dlogL, index = alpha, K = -alpha*log(L0).
	
	This function is called by fitXLF() to calculate N (number of sources) from the input values. 
	"""

	# general form is some log(N) = -index * log(L) + K
	# index represents the slope of the power-law
	# K is some normalizing factor, usually related to L0
	# The form of index and K depend on what kind of N is input
	# if N = N(>L), index = gamma = (alpha - 1), K = -gamma*log(L0)
	# if N = dN/dL or dN/dlogL, index = alpha, K = -alpha*log(L0)

	# make sure the input is in log-scale
	if max(L) > 50: L = log10(L)

	# if cumulative and alpha input, must convert index to gamma (index - 1)
	# if neither cumulative nor alpha input, must convert to alpha (index + 1)
	# else (True and False), index can stay the same
	index = index - (cumul + alpha) + 1

	# need integer to = +1 if both are false
	# need integer to = -1 if both are true
	# need integer to = 0 if they are different
	# cumul == alpha : equals 0 only if different (otherwise equals 1)
	# cumul*alpha    : equals one only if both are true
	

	# if entered L0 instead of K, then set K equal to index*L0
	if L0_in: K = K*index

	# depending on whether using differential or cumulative, 
	# index can equal alpha or gamma
	if convert_power: return 10**(-index*L + K)
	else: return -index*L + K

	# NOTE: The L here is always log10(L)		


###-----------------------------------------------------------------------------------------------------

def fitPL(L, N, index=None, L0=None, init=None, sigma=None, alpha=False, cumul=True, returnalpha=True, returnfit=False, returnerrors=True):
	
	"""
	Function for fitting the given data as a power law, given a starting break luminosity. All data should be given as logs for a 
	proper fit. The uncertainty on N is taken as a Poisson uncertainty (0.434/sqrt(N)) if none is given. May also return the value of alpha, 
	assuming the index of the power law is gamma = -(alpha - 1). The default format of the power law function is N. 
	
	This function is called by fitXLF() to find the best fit to the power-law function given a sample of 
	sources with known X-ray luminosities. 
	Returns index, L0, and fit (optional) or errors (default/optional, [index, L0]). 

	"""

	L = L.copy()
	N = N.copy()

	# making sure all input material is in the proper log-space
	if max(N) > 4: N = log10(N)
	if max(L) > 50: L = log10(L)

	# In the event of bad N values, remove
	L = L[np.abs(N) != np.inf]
	try: sigma = sigma[np.abs(N) != np.inf]
	except: pass;
	N = N[np.abs(N) != np.inf]

	if not cumul: N = N - L

	# Initial guess
	if not init:
		init = [1,1]
	if index: init[0] = index
	if L0: # calculating initial 'b' from input L0
	    if L0 > 50: L0 = log10(L0)
	    init[1] = init[0] * L0      

	# Calculating the uncertainties
	try: 
		if not sigma: 
		    sigma = 0.434/sqrt(10.0**N)
	except: pass; 

	# curve_fit returns two arrays: fits and covariance matrix
	fit = curve_fit(linPL, L, N, p0=init, sigma=sigma, maxfev=50000)

	# Retrieving/calculating fits
	# first value = index
	# second value = K = normalization
	index = fit[0][0]
	K = fit[0][1]
	L0 = K / index

	# Calculating errors on fits	
	fit_errs = sqrt(np.diag(fit[1]))
	L0_err = sqrt((fit_errs[0]/index)**2 + (fit_errs[1]/K)**2)

	# Histogram should be either cumulative or differential. 
	# Default, cumul=True. If returnalpha set, calculate alpha from gamma.
	# index = gamma = alpha - 1 -> alpha = gamma + 1
	# See: https://iopscience.iop.org/article/10.1086/300577/fulltext/
	# if both cumulative and returnalpha, convert index from gamma (cumul) to alpha
	index = index + (cumul + returnalpha) - 1

	if returnfit: 
		return index, K, L0, fit
	elif returnerrors: 
		return index, K, L0, [fit_errs[0], fit_errs[1], L0_err]
	else: return index, K, L0

	# NOTE: returns both the normalization as given by the fit, and the calculated L0

###-----------------------------------------------------------------------------------------------------

def linBPL(logL, logLb, index1, index2, K1, K2, alpha=True, L0_in=False, convert_power=False, cumul=True):
	
	"""
	Returns the resulting N values for a broken power law, given some break luminosity and indices.

	This function is called by fitXLF() to calculate N in a broken power-law. 
	"""

	logL = logL.copy()

	# Converts the indices to gammas before calling power-law
	index1 = index1 - alpha
	index2 = index2 - alpha
	
	# Splitting the function at the break luminosity
	# Fit each part separately and combine later

	loL = logL[logL <= logLb]
	hiL = logL[logL >= max(loL)]	# start where the last one left off

	# if logLb does not exist in the arrays, this may cause an issue with the plotting. 
	# Can possibly fix if I interpolate
	# REVISIT THIS LATER (add before the hiL line)
	# if max(loL) < logLb: 	

	loN = (linPL(loL, index1, K1, L0_in=L0_in, cumul=cumul)).tolist()
	hiN = (linPL(hiL, index2, K2, L0_in=L0_in, cumul=cumul)).tolist()

	for i in hiN[1::]: 
		loN.append(i)

	if convert_power: return 10**(np.array(loN))
	else: return np.array(loN)

###-----------------------------------------------------------------------------------------------------

def fitBPL(L, N, Lb=None, index1=None, index2=None, L0=None, init=None, sigma=None, cumul=True, returnalphas=True, returnfit=False, returnerrors=True):

	"""
	Fits function to a broken powerlaw, calculated as a combination of single power-laws split by given break luminosity. 
	If returnfit=True, returns the power-law fits for parts below and above the break luminosity. 
	Else, returns [index1, index2], [L01, L02], Lb (estimated), and errors (default/optional, [[index1, index2],[L01, L02]]).

	This function is called by fitXLF() to calculate the parameters of the best-fit broken power-law given a sample of 
	sources with known X-ray luminosities.
	"""
    
	L = L.copy()
	N = N.copy()

	# Needs to be in logspace before continuing. 
	if max(N) > 4: N = log10(N)
	if max(L) > 50: L = log10(L)

	# In the event of bad N values, remove
	L = L[np.abs(N) != np.inf]
	N = N[np.abs(N) != np.inf]

	# Calculating the uncertainties
	try: 
		if not sigma: sigma = 0.434/sqrt(10.0**N)
	except: pass; 


	if not Lb:
		# If no Lb is given, use a chi-squared analysis to fit the best Lb
		Lbs = np.linspace(36,39.9,40).tolist()
		chis = []
		for Lb in Lbs:	
			loL = L[L <= Lb]
			hiL = L[L >= Lb]
			loN = N[L <= Lb]
			hiN = N[L >= Lb]
			losig = sigma[L <= Lb]
			hisig = sigma[L >= Lb]

			try:
				lofit = fitPL(loL, loN, sigma=losig, cumul=cumul, returnalpha=returnalphas)
				hifit = fitPL(hiL, hiN, sigma=hisig, cumul=cumul, returnalpha=returnalphas)
				# for calculating and returning the break luminosity for this fit
				try: 
					alpha1 = lofit[0][0]
					alpha2 = hifit[0][0]
					K1 = lofit[0][1]
					K2 = hifit[0][1]
				except: 
					alpha1 = lofit[0]
					alpha2 = hifit[0]
					K1 = lofit[1]
					K2 = hifit[1]

				# Calculating the chi-squared 
				temp = np.array(linBPL(L, Lb, alpha1, alpha2, K1, K2, alpha=True, L0_in=False, convert_power=True))
				temp_N = 10**(np.array(N))
				chis.append(sum((temp_N - temp)**2 / temp_N))
			except: 
				chis.append(np.inf)

		best = np.where(chis == min(chis))[0][0]
		Lb = Lbs[best]

	if Lb > 50: Lb = log10(Lb)

	# Finding the fit for each part of the broken power-law
	# Edits should be made here, if I want to fix the broken power-law plots
	loL = L[L <= Lb]
	hiL = L[L >= Lb]
	loN = N[L <= Lb]
	hiN = N[L >= Lb]
	losig = sigma[L <= Lb]
	hisig = sigma[L >= Lb]


	# If no source has the break luminosity, it can cause problems with the fit
	# The following smooths the fit by forcing the fit to include the break luminosity
	# and setting the counts equal to the preceeding counts
	#if max(loL) != Lb: 
	#	i = np.where(loL == max(loL))[0][0]
	#	hiL = np.insert(hiL, 0, Lb)
	#	loL = np.append(loL, Lb)
	#	hiN = np.insert(hiN, 0, loN[i])
	#	loN = np.append(loN, loN[i])
	#	hisig = np.insert(hisig, 0, losig[i])
	#	losig = np.append(losig, losig[i])
		

	lofit = fitPL(loL, loN, init=init, sigma=losig, cumul=cumul, returnalpha=returnalphas, returnfit=returnfit, returnerrors=returnerrors)
	hifit = fitPL(hiL, hiN, init=init, sigma=hisig, cumul=cumul, returnalpha=returnalphas, returnfit=returnfit, returnerrors=returnerrors)

	# for calculating and returning the break luminosity for this fit
	try: 
		alpha1 = lofit[0][0]
		alpha2 = hifit[0][0]
		K1 = lofit[0][1] 
		K2 = hifit[0][1] 
		L01 = lofit[0][2]
		L02 = hifit[0][2]
	except: 
		alpha1 = lofit[0]
		alpha2 = hifit[0]
		K1 = lofit[1] 
		K2 = hifit[1] 
		L01 = lofit[2]
		L02 = hifit[2]


	if returnfit: 
		return lofit, hifit, Lb
	elif returnerrors: 
		return [alpha1, alpha2], [K1, K2], [L01, L02], Lb, [[lofit[-1][0], hifit[-1][0]], [lofit[-1][1], hifit[-1][1]], [lofit[-1][2], hifit[-1][2]]]
	else: return [alpha1, alpha2], [K1, K2], [L01, L02], Lb

###-----------------------------------------------------------------------------------------------------

def Schechter(L, index, L0, K, alpha=True, cumul=True, convert_power=False): 

	"""Linear form of the Schechter function, defined as log(N) = logK - index*logL + index*logL0 - L/(L0*ln10). 
	Returns the log (base 10) of the expected counts per bin.

	This function is called by fitXLF() to calculate N of a given Schechter function. """

	# Definition of Schechter function: 
	# log(N) = logK - index*logL + index*logL0 - L/(L0*ln10)
	# Will need to test if the variables are given as logs. If so, need to convert to linear.

	if cumul: index = index - alpha

	if max(L) > 50: L = log10(L)
	if L0 > 50: L0 = log10(L0)
		

	if convert_power: return 10**(-index*L + index*L0 - 10**(L)/10**(L0)/log(10) + K)
	else: return -index*L + index*L0 - 10**(L)/10**(L0)/log(10) + K

###-----------------------------------------------------------------------------------------------------

def fitSchechter(L, N, index=None, L0=None, K=None, init=None, sigma=None, returnalpha=True, cumul=True, returnfit=False, returnerrors=True):

	"""
	Fits to a Schechter function. Returns fitted index (gamma or alpha, where gamma = -(alpha - 1)), 
	L0, N0 (normalization), and optional fit or errors (default/optional, [alpha, L0, N0]).

	This function is called by fitXLF() to calculate the best fit parameters of a Schechter function given a sample of 
	sources with known X-ray luminosities.  
	"""    

	L = L.copy()
	N = N.copy()

	# making sure all input material is in the proper log-space
	if max(N) > 4: N = log10(N)
	if max(L) > 50: L = log10(L)
	try: 
		if L0 > 50: L0 = log10(L0)
	except: pass;
	 
	L = L[np.abs(N) != np.inf]
	N = N[np.abs(N) != np.inf]

	#if histtype=="differential": N = N - L

	# Initial guess
	if not init:
		init = [1,1,1]
		if index: init[0] = index
		if L0: init[1] = L0
		if K: init[2] = K
	try: 
		if not sigma: 
			sigma = 0.434/sqrt(10.0**N)
	except: pass; 


	# Running fit. Curve_fit returns two arrays: the fits and the covariance matrix
	fit = curve_fit(Schechter, L, N, p0=init, sigma=sigma, maxfev=50000)
	index = fit[0][0]
	L0 = fit[0][1]
	K = fit[0][2]
	if returnfit: return index, L0, K, fit
	elif returnerrors: 
		fit_errs = sqrt(np.diag(fit[1]))
		return index, L0, K, [fit_errs[0], fit_errs[1], fit_errs[2]]
	else: return index, L0, K



###-----------------------------------------------------------------------------------------------------

def fitXLF(L, N=None, Lb=None, index=None, L0=None, K=None, init_Schechter=None, sigma=None, returnalpha=True, cumul=True, returnfit=True, bins=50, title=None, figsize=(8,6), xlim=(35,40), ylim=None, verbose=False, fontsize=20, labelsize=None, completeness=36.2, xlabel=True, ylabel=True, xticks=True, yticks=True, fitlw=2.5, lw=2, scale="log", L_error=None, showPL=True, showBPL=True, showSchechter=True, PLcolor="#85C0F9", BPLcolor="slateblue", Schechtercolor="goldenrod", showlegend=True, legendsize=20):

	"""
	Plots XLF of given sources and calculates the best fits to power-law, broken power-law, and Schechter functions. 

	PARAMETERS
	----------
	L     [list, DataFrame] :	DataFrame containing the luminosities of each source, in log (base 10) units, or list containing the binned luminosities
					of sources of interest
	N		[list]	:	If parameter 'L' is not a DataFrame, then N is the results of a histogram on the luminosities of sources of interest.
	Lb		[float]	:	Inital guess at the break luminosity of the broken power-law (see Lehmer et al. 2019 for more info). 
	init_Schechter	[list]	:	Initial guess on the Schechter function fit, defined as log(N) = logK - index*logL + index*logL0 - L/(L0*ln10). 
					Input should be given as [index, L0 (normalizing luminosity), and K (general normalization factor)].
	sigma		[float]	:	Sigma of bins, given as the inverse of the bin errors. 
	returnalpha	[bool]	:	Default is True. 
	cumul		[bool]	:	If True, plots and fits the cumulative X-ray luminosity function. If False, used a 
					differential form of each XLF instead. Default is True. 
	returnfit	[bool]	:	If True, returns a table containing the parameters of the best fit power-law, broken
					power-law, and Schechter function lines based on the formulas from Lehmer et al. 2019. 
					Default is True. 
	bins		[int]	:	Default is 50. For unbinned XLFs, can input 'unbinned' or None instead. 
	title		[str]	:	Title of the plot. 
	figsize		[tuple]	:	Size of the figure. Default is (8,6). 
	xlim		[tuple]	:	Limits on the x-axis (luminosity). Defaults is (35,40). 
	ylim		[tuple]	:	Limits on the y-axis (N or dN/dL). If none is given, will attempt to calculate a reasonable limit. 
	verbose		[bool]	:	If true, will print outputs describing each step. 
	fontsize	[int]	:	Size of font on figure. Default is 20. 
	labelsize	[int]	:	Size of the tick labels on the XLF. If none is given, size will depend on the fontsize parameter. 
	completeness	[float]	:	X-ray completeness limit (usually 90% or 95%) in log (base 10) units. 
	xlabel		[bool]	:	Turns the x-axis label on (True) or off (False). Default is True.  	
	ylabel		[bool]	:	Turns the y-axis label on (True) or off (False). Default is True. 
	xticks		[bool]	:	Turns the x-axis ticks on (True) or off (False). Default is True. 
	yticks		[bool]	:	Turns the y-axis ticks on (True) or off (False). Default is True. 
	fitlw		[float]	:	Linewidth of the best fit line. Default is 2.5. 
	lw		[float]	:	Linewidth of the X-ray luminosity function. Default is 2.
	scale		[str]	:	Scale on the y-axis (N). Default is 'log', but can also be given as 'linear'. 
	L_error		[float]	:	Error on the binned luminosities, given as a fraction of the luminosity. 
	showPL		[bool]	:	If True, plots the best fit power-law on the XLF figure. Default is True.
	showBPL		[bool]	:	If True, plots the best fit broken power-law on the XLF figure. Default is True. 
	showSchechter	[bool]	: 	If True, plots the best fit Schechter function on the XLF figure. Default is True. 
	PLcolor		[str]	: 	Color of the power-law best fit line. Default is '#85C0F9' (purple) 
	BPLcolor	[str]	: 	Color of the broken power-law best fit line. Default is 'slateblue'. 
	Schechtercolor	[str]	:	Color of the Schechter function best fit line. Default is 'goldenrod'.
	showlegend	[bool]	:	If True, plots the legend on the XLF figure. 
	legendsize	[int]	:	Size of the legend font. Default is 20. 

	"""

	#L = L.copy()
	

	# The Lehmer data gives luminosities with 3 significant figures, with a 
	# min luminosity difference of 0.1. If we don't want the data binned
	# feed in set bin edges representing the possible data.
	if bins == "unbinned" or bins == None: 
		bins = np.arange(35.05,40.15,0.1).tolist()

	# If N is not given, histogram has not been constructed yet. 
	# Run MakeXLF from beginning to obtain N
	if N == None:
		 N_bins, L_bins, _ = MakeXLF(L, bins=bins, returnhist=True, title=title, figsize=figsize, xlim=xlim, ylim=ylim, fontsize=fontsize, labelsize=labelsize, completeness=completeness, xlabel=xlabel, ylabel=ylabel, xticks=xticks, yticks=yticks, lw=lw, scale=scale, cumul=cumul)
	else: 
		MakeXLF(L, bins=bins, title=title, figsize=figsize, xlim=xlim, ylim=ylim, fontsize=fontsize, labelsize=labelsize, completeness=completeness, xlabel=xlabel, ylabel=ylabel, xticks=xticks, yticks=yticks, lw=lw, scale=scale, cumul=cumul)
		try: 
			L_bins = L.copy()
			N_bins = N.copy()
		except: pass;

	# L_error may be given as a fraction of the L
	if L_error: 
		sigma = L_bins * L_error

	# Ensuring only luminosities above the limiting luminosity are fit.
	# completeness = 36.2 for Lehmer M83 sources

	L_plot = np.linspace(min(L_bins), max(L_bins), 100)

	try: sigma = sigma[L_bins >= completeness]
	except: pass; 
	N_bins = N_bins[L_bins >= completeness]
	L_bins = L_bins[L_bins >= completeness]

	# Removing bad bins
	goodL, goodN, goodsig = GoodBins(L_bins, N_bins, sigma=sigma, completeness=completeness) 


	# Power Law Fit
	if verbose: print("Calculating Power Law fit.") 
	PLFit = fitPL(goodL, goodN, sigma=goodsig, cumul=cumul)
	if verbose: print("Plotting Power Law fit.")
	if showPL: plt.plot(L_plot, linPL(L_plot, PLFit[0], PLFit[1], alpha=True, L0_in=False, convert_power=True, cumul=cumul), lw=fitlw, color=PLcolor, linestyle="--", label="Power Law")

	#Broken Power Law Fit
	if verbose: print("Calculating Broken Power Law fit.")
	BPLFit = fitBPL(goodL, goodN, Lb, sigma=goodsig, cumul=cumul)
	Lb = BPLFit[3]

	if verbose: print("Plotting Broken Power Law fit.")
	if showBPL: plt.plot(L_plot, linBPL(L_plot, Lb, BPLFit[0][0], BPLFit[0][1], BPLFit[1][0], BPLFit[1][1], alpha=True, L0_in=False, convert_power=True, cumul=cumul), lw=fitlw, linestyle="-.", color=BPLcolor, label="Broken PL")
	
	# Schechter Fit
	if verbose: print("Calculating Schechter fit.")  
	SchFit = fitSchechter(goodL, goodN, init=[1.4, 38., 1.], sigma=goodsig, cumul=cumul)	

	if verbose: print("Plotting Schechter fit.")
	if showSchechter: plt.plot(L_plot, Schechter(L_plot, SchFit[0], SchFit[1], SchFit[2], alpha=True, convert_power=True, cumul=cumul), lw=fitlw, color=Schechtercolor, label="Schechter")

	# NOTE: Instead of using the normalization returned by the fit, use the 
	#       Lb * alpha for K.
	#       If this doesn't look right, go back to using the returned normalization

	if showlegend: plt.legend(fontsize=legendsize)
	if returnfit: 
		funcfits = ["Power-Law", "Broken PL", "Schechter"]
		alpha1 = [r"%1.2f $\pm$ %1.2f" %(PLFit[0], PLFit[-1][0]), \
			  r"%1.2f $\pm$ %1.2f" %(BPLFit[0][0], BPLFit[-1][0][0]), \
			  r"%1.2f $\pm$ %1.2f" %(SchFit[0], SchFit[-1][0])]
		alpha2 = ["-", r"%1.2f $\pm$ %1.2f" %(BPLFit[0][1], BPLFit[-1][0][1]), "-"]

		K1 = [r"%1.2f $\pm$ %1.2f" %(PLFit[2]*PLFit[0], PLFit[-1][1]), \
		      r"%1.2f $\pm$ %1.2f" %(BPLFit[2][0]*BPLFit[0][0], BPLFit[-1][1][0]), \
		      r"%1.2f $\pm$ %1.2f" %(SchFit[2], SchFit[-1][2])]
		K2 = ["-", r"%1.2f $\pm$ %1.2f" %(BPLFit[2][1]*BPLFit[0][1], BPLFit[-1][1][1]), "-"]
		Lb = ["-", Lb, "-"]
		L1 = [r"%1.2f $\pm$ %1.2f" %(PLFit[2], PLFit[-1][2]), \
		      r"%1.2f $\pm$ %1.2f" %(BPLFit[2][0], BPLFit[-1][2][0]), \
		      r"%1.2f $\pm$ %1.2f" % (SchFit[1], SchFit[-1][1])]
		L2 = ["-", r"%1.2f $\pm$ %1.2f" %(BPLFit[2][1], BPLFit[-1][2][1]), "-"]

		XRB_df = BuildFrame(headers=["Fit", r"$\alpha_{1}$", r"$\alpha_{2}$", \
				            r"L$_{1}$", r"L$_{2}$", r"L$_{B}$", r"K$_{1}$", r"K$_{2}$"], \
				   values=[funcfits, alpha1, alpha2, L1, L2, Lb, K1, K2])
		return XRB_df


###-----------------------------------------------------------------------------------------------------

def CalcXRBs(Lb=2.16e38, Lnorm=1e38, Lc=6.31e40, L90=1.58e36, w=1, enclosed=None, alpha1=None, alpha2=None, K=None): 

	"""
	Calculates the expected number of XRBs (of either high or low mass) by integrating the XLF based on the 
	models from Lehmer et al. 2019 (hereafter L19). The parameter 'enclosed' should be given as either the enclosed 
	stellar mass or star formation rate of the region of interest, depending on the intended output. 
	'w' is the L19 normalization factor for the specific galaxy of interest, obtained from Table 5. 
	'K' is the normalization from L19 Table 4. 'alpha1' may refer to the first slope for the LMXB BPL case, 
	or equivalent to gamma in the HMXB PL case in Table 4. 
	NOTE: I did not include a doubly-broken PL here as it is irrelevant for my case. May include in later updates (v1.6.1+). 
	"""

	if not K: K = input("Normalization constant: ")
	if not enclosed: enclosed = input("Enclosed stellar mass or star formation rate?: ")
	if not alpha1: alpha1 = input("Slope: ")

	if alpha2: 
		return w*enclosed*N_BPL_total(L90, Lb, Lc, Lnorm, alpha1, alpha2, K)
	else: 
		return w*enclosed*quad(N_PL, L90, Lc, args=(alpha1, K, Lnorm))[0]
	
	
###-----------------------------------------------------------------------------------------------------

### The following functions are used in the XRB calculation above. They are the PL and BPL formulations of dN/dL needed for integration. ###

def N_PL(L, alpha1, K, Lnorm): 
    return (K/(Lnorm))*(L/(Lnorm))**(-alpha1)

# For the broken power law
def N_BPL1(L, alpha1, K, Lnorm):
    return (K/(Lnorm))*(L/(Lnorm))**(-alpha1)

def N_BPL2(L, Lb, alpha1, alpha2, K, Lnorm):
    return (K/(Lnorm))*(Lb/(Lnorm))**(alpha2-alpha1)*(L/(Lnorm))**(-alpha2)

def N_BPL_total(L90, Lb, Lc, Lnorm, alpha1, alpha2, K): 
    return quad(N_BPL1, L90, Lb, args=(alpha1, K, Lnorm))[0] + quad(N_BPL2, Lb, Lc, args=(Lb, alpha1, alpha2, K, Lnorm))[0]
