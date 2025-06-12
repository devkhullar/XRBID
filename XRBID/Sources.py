###################################################################################
##########	For reading in, identifying, cross-referencing,		###########
##########	and classifying sources, particularly CSC-derived	########### 
##########	X-ray sources or stars ID'd w/ DaoStarFinder.		###########
##########	Last update: June 9, 2025 				########### 
##########	Update desc: Updated GetDaoPhots to allow the return	########### 
##########		     of photometric errors.		 	########### 
###################################################################################

import re
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from astropy.io.votable import parse, parse_single_table
import astropy.io.votable
import glob
import pandas as pd
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore")
imext = [0., 13500.]
import os

from XRBID.DataFrameMod import BuildFrame, Find, Convert_to_Number

###-----------------------------------------------------------------------------------------------------

def LoadSources(infile=None, verbose=True):

	"""
	Creates and returns a DataFrame using the specified text file.
	
	PARAMETERS
	----------
	infile		[str] 		: Input file name
	verbose		[bool] (True)	: If True, prints updates on the process
	
	RETURNS
	---------
	Returns a DataFrame created from the input file
	
	"""

	if infile == None: infile = raw_input("File containing data for sources: ") 
	if len(infile.split(".")) < 2: infile = infile + ".txt";

	if verbose: print("Reading in sources from " + infile + "...") 

	try: 
		# ID will sometimes be given as a number but should be read in as string
		try: return Convert_to_Number(pd.read_csv(infile, sep=",", converters={"ID": str}, dtype=None).drop("Unnamed: 0", axis=1))
		# If ID not in the file, just return this
		except: 
			return Convert_to_Number(pd.read_csv(infile, sep=",", dtype=None).drop("Unnamed: 0", axis=1))
			
	except: 
		try: return Convert_to_Number(pd.read_csv(infile, sep=",", dtype=None))
		except: 
			return Convert_to_Number(pd.read_csv(infile, dtype=None))

###-----------------------------------------------------------------------------------------------------

def NewSources(infile=None, headers=None, rename=False, outfile=False):

	"""
	Creates a new DataFrame from a specified VOTable. If headers are known, can be read in to start. 
	If rename = True, list all headers and offer user to rename in order.
	
	PARAMETERS
	----------
	infile		[str] 		: Name of the VOTable file to read from
	headers		[list]		: Optional; list of header names
	rename		[bool] (False)	: If True, allows user to rename the headers
	outfile		[str] (False)	: Name of save file. 
	
	RETURNS
	---------
	sources		[pd.DataFrame]	: DataFrame created from the input file

	""" 

	# user must give VOTable file to read from
	if infile == None: infile = input("VOTable file: ") 
	if len(infile.split(".")) < 2: infile = infile + ".vot"

	print("Reading in table from " + infile)

	votable = parse(infile)   # reads in table 
	table = votable.get_first_table().to_table(use_names_over_ids=True)

	values = []		# Keeps track of the values in the votable

	# Keeping track of the headers from votable
	temp_headers = [f.name for f in parse_single_table(infile).fields]

	# Populating values and temp_headers
	for i in temp_headers: 
		values.append(table[i])


	# If no headers are speficied, read in all in the votable file
	if not headers and rename:
		print("Enter new header names, or hit 'Enter' to keep name.\n")
		frame_heads = []
		for i in temp_headers:
			temp_head = input(i+": ")
			if not temp_head: 
				temp_head = i
			frame_heads.append(temp_head)
	elif headers: frame_heads = headers
	else: frame_heads = temp_headers

	sources = BuildFrame(headers=frame_heads, values=values)

	# For some reason, reading directly from a CSC .vot does not allow
	# Find() to be used on the DataFrame unless read back in by LoadSources.
	# Saving to a file, then reading back in with LoadSources. 
	if outfile: 
		sources.to_csv(outfile)
		sources = LoadSources(outfile)
	else: 
		sources.to_csv("temp.txt")
		sources = LoadSources("temp.txt", verbose=False)
		os.remove("temp.txt")

	print("DONE")
	return sources

###-----------------------------------------------------------------------------------------------------

def GetCoords(infile=None, IDs=None, savecoords=None, checkcoords=False, verbose=True):

	"""
	Gets the image (X/Y) coordinates of the sources from a given region, text file, or DataFrame file. 
	The IDs argument can be used to only return certain coordinates, such as in the case of finding DaoFind 
	source coordinates from a region file of specific DaoFind source IDs. 
	NOTE: to enable this to work in non-Daofind cases, will need to edit code some more.
	
	PARAMETERS
	----------
	infile		[str] 		: Name of the VOTable file to read from
	headers		[list]		: Optional; list of header names
	rename		[bool] (False)	: If True, allows user to rename the headers
	
	RETURNS
	---------
	sources		[pd.DataFrame]	: DataFrame created from the input file

	"""

	if verbose: print("Retrieving coordinates from " + infile)

	if ".frame" in infile: # DataFrame files can be read in using LoadSources
		temp = LoadSources(infile=infile, verbose=False)
		try: 	# Let image coordinates be the default
			x_coords = temp["X"].values
			y_coords = temp["Y"].values
		except: # If image coordinates not available, use RA and Dec.
			x_coords = temp["RA"].values
			y_coords = temp["Dec"].values
	else: 
		try: 	# If the file is a .txt file (or some other), this should work fine
			coords = np.genfromtxt(infile)
			if checkcoords: 
				userin = "n"
				print("Input check on " + infile + ": ")
				i = -1
			else: 
				userin = "y"
				i = 0
			
			while userin != "y": 
				i += 1
				# Trying to find the coordinates in the file. 
				# This should cover all kind of file formats
				try: 
					userin = raw_input("Is (" + str(coords[0][i]) + ", " + str(coords[0][i+1]) + ") a valid coordinate? ").lower()[0]
				except: print("No coordinates found. Check file."); break;
				
			x_coords = coords.T[i]
			y_coords = coords.T[i+1]
		except:	# If the file is a region file (.reg), the following should apply 
			with open(infile) as f: 
			    lines = f.readlines()[3:]
			    
			x_coords = []
			y_coords = []

			for i in range(len(lines)):
				line = lines[i]
				try: 
					l0 = float(line.strip().split()[0].split('(')[-1].split(',')[0])
					l1 = float(line.strip().split()[1].split(",")[0])
				except: 
					l = line.strip().split()[0].split('(')[-1].split(',')
					l0 = float(l[0])
					l1 = float(l[1])
				x_coords.append(l0)
				y_coords.append(l1)
	
	if IDs != None: 
		# Create a mask of all False. Then replace the corresponding ID with True to mask out all coordinates that do not correspond with the sources in IDs.
		mask = [False]*len(x_coords)
		for i in IDs: 
			j = int(i) - 1
			mask[j] = True
		# Applying the mask
		x_coords = np.array(x_coords)[mask]
		y_coords = np.array(y_coords)[mask]
		x_coords = x_coords.tolist()
		y_coords = y_coords.tolist()

	if savecoords: 
		print("Saving " + savecoords)
		with open(savecoords, "w") as f: 
			np.savetxt(f, np.column_stack([x_coords, y_coords]))
	return x_coords, y_coords

###-----------------------------------------------------------------------------------------------------

def GetIDs(infile=None, verbose=True):

	"""
	Gets the IDs (or printed text) of the sources from a given region or text file.
	
	PARAMETERS
	----------
	infile	[str]		: Name of input file
	verbose	[bool] (True)	: If True, prints file info

	RETURNS
	---------
	ids	[list]		: List of source IDs
	
	"""

	if verbose: print("Retrieving IDs from " + infile)

	# If the file is a region file (.reg), the following should apply 
	with open(infile) as f: 
	    lines = f.readlines()[3:]
	    
	ids = []

	for i in range(len(lines)):
		line = lines[i]
		try: 
			ids.append(re.split("}", re.split("text={", line)[-1])[0])
		except: 
			ids.append("None")

	return ids

###-----------------------------------------------------------------------------------------------------

def SourceList(savefile, df=None, columns=['ID']):

	""" 
	Creates a text file containing some input data stacked into columns. 
	By default, if a DataFrame is read in and no columns are specified, 
	the code searches for a header in the DataFrame called 'ID' and prints
	those values to a file. Otherwise, SourceList() can be used to print
	the values of any header within DataFrame given in a list called 
	columns, or columns can be used as a list of values to print to an
	output file (if not DataFrame is given).
	(Modified for simplification Nov 26, 2024)
	
	PARAMETERS
	----------
	savefile	[str]		: Name of output file
	df		[pd.DataFrame] 	: DataFrame containing information to save. Optional only if columns is given.
	columns		[list] (['ID'])	: Optional; contains either the names of the headers to pull from the input
					  DataFrame, or a list of values to save to the output file. 
	
	RETURNS
	---------
	Saves a file under the name savefile

	""" 

	if df: 
		with open(savefile, 'w') as f:
			tempstack = []
			for i in columns: 
				tempstack.append(df[i].values)
			np.savetxt(f, np.column_stack(tempstack), fmt="%s")
	else: np.savetxt(f, np.column_stack(columns), fmt="%s")

	print(savefile + " saved!")

###-----------------------------------------------------------------------------------------------------

def DaoClean(daosources=None, sources=None, sourceid="ID", coordsys="img", coordheads=False, radheader="Radius", wiggle=0, outfile=False): 

	"""
	Cleaning the DaoFind sources to exclude any candidate that falls outside of the radius of the 
	X-ray sources, taking into account given wiggle room (in pixels). Sources should be read in as
	a dataframe with image coordinates (X, Y) and 2-sig radius saved under header Radius. The best
	practice is to save both the image and fk5 coordinates from the DaoFind region files to a single
	DataFrame per field/filter, to allow better flexibility. 
	
	PARAMETERS
	----------
	daosources	[df.DataFrame]	: DataFrame containing the coordinates of sources identified by DaoFind.
	sources		[df.DataFrame]	: DataFrame containing the X-ray sources (or others of interest), their
					  coordinates, and their 2sig radii. 
	sourceid	[str] ('ID')	: Name of the header under which the ID of each source is stored.
	coordsys	[str] ('img')	: Coordinate system defining the units of the coordinates and the 2sigma radius. 
					  Options are 'img' (pix, default) or 'fk5' (coordinates in degs, radii in arcsecs). 
	coordheads	[list] 		: Name of the headers under which the coordinates are saved. These should be the same 
					  between daosources and sources. If coordheaders is not given, assumes ['X','Y'] if
					  the unit is 'img' and ['RA', 'Dec'] if the unit is 'fk5'. 
	radheader	[str] ('Radius'): Name of header under which the 2sig radius is saved. 
	wiggle		[float] (0)	: Additional pixels/degrees to add to the search radius for a less stringent search.
	outfile		[str]		: Name of file to save resulting DataFrame to (optional). 
	
	RETURNS
	---------
	GoodSources	[df.DataFrame]	: DataFrame equivalent to 'daosources' with only sources that fall within
					  the radii of the sources in the DataFrame 'sources.'

	"""


	try: 
		daosources = daosources.copy()
		sources = sources.copy()
	except: pass;

	# Retrieving headers from the DataFrames
	daoheads = daosources.columns.values.tolist()
	headlist = daosources.columns.values.tolist()
	headlist.append(sourceid)

	arcsec2deg = 0.000277778

	# If the coordinate system is fk5, assumes the radius is in arcseconds and converts to degrees
	if coordsys == 'fk5': 
		print("WARNING: Due to the way HST images are stretched and plotted, you may find using the fk5 coordinate system gives odd results. It is recommended to use image (pixel) coordinates. If fk5 are used, please check results manually.")
		sources[radheader] = sources[radheader]*arcsec2deg
		if not coordheads: # checks if the coordinate header is given. If not, set to [RA, Dec]
			coordheads = ['RA', 'Dec']
	elif coordsys == 'img': 
		if not coordheads: coordheads = ['x', 'y']
	else: print('Coordinate system not recognized.')

	# Cleaning daofind sources to only include those around our sample sources
	daocleaned = np.empty((0,len(headlist)))

	print("Cleaning DAOFind sources. This will take a few minutes. Please wait.. ")
	for i in range(len(sources)): 
		# Properties of each Lehmer source
		temprad = sources[radheader][i]
		xtemp = sources[coordheads[0]][i]
		ytemp = sources[coordheads[1]][i]
		tempid = sources[sourceid][i]

		# Search area around each source
		#print(temprad, xtemp, ytemp)
		tempxmax = xtemp+temprad+wiggle
		tempxmin = xtemp-temprad-wiggle
		tempymax = ytemp+temprad+wiggle
		tempymin = ytemp-temprad-wiggle

		# Finding the coordinates of daofind sources within the square area of each source
		# Then refine search to look within the search radius of
		# (This method is quicker than comparing all daosources to the source radius)
		tempdao = Find(daosources, [coordheads[0] + " >= " + str(tempxmin), coordheads[0] + " <= " + str(tempxmax), \
				            coordheads[1] + " >= " + str(tempymin), coordheads[1] + " <= " + str(tempymax)])

		for j in range(len(tempdao)): 
			if sqrt((tempdao[coordheads[0]][j] - xtemp)**2 + (tempdao[coordheads[1]][j] - ytemp)**2) <= temprad+wiggle:
				tempstack = [tempdao[k][j] for k in daoheads]
				tempstack.append(tempid)
				daocleaned = np.vstack((daocleaned, tempstack))

	print("DONE WITH CLEANING. CREATING DATAFRAME...")

	daotemp = [daocleaned.T[k] for k in range(len(daocleaned.T))]
	GoodSources = BuildFrame(headers=headlist, values=daotemp)

	if outfile: GoodSources.to_csv(outfile)

	return GoodSources

###-----------------------------------------------------------------------------------------------------

def Crossref(df=None, regions=False, catalogs=False, coords=False, sourceid="ID", search_radius=3, coordsys="img", coordheads=False, verbose=True, shorten_df=False, outfile="crossref_results.txt"): 

	"""

	UPDATE NEEDED: Keep the other columns in the dataframe.

	From input DataFrame and/or region files (in image coordinate format), finds overlaps within a given 
	search radius of the DataFrame sources and prints all ID names to a file as a DataFrame. 
	If the coordinates are given as ['RA', 'Dec'] instead of ['X','Y'], must change coordsys from "img" to "fk5" 
	and convert search_radius from pixels to degrees. Can feed in the name of the catalogs used to output 
	as DataFrame headers. Otherwise, the region name will be used.

	NOTE: There is an error in this where if the first region file doesn't have a counterpart in the first 
	entry of the overlap file, the first entry may be split into multiple entries. Check file.

	PARAMETERS
	-----------
	df		[pd.DataFrame]	: DataFrame containing the coordinates of the sources for which the counterparts 
					  will be found in the given region files or catalogs. 
	regions 	[list]		: List of filenames for regions to cross-reference sources from. This should be
					  in the same coordinate system as the units in df. 
	catalogs	[list]		: Name of the catalogs associated with the input region files. This will be used to
					  define the ID header for sources in each region file. If none is given, then the  
					  region file name is used as the respective source ID header.
	coords 		[list]		: List of coordinates to cross-reference; can be given instead of regions. 
	sourceid	[str] ('ID')	: Name of header containing the ID of each source in df. 
	search_radius	[list] (3)	: Search radius (in appropriate units for the coordinate system) around each source in df. 
					  Can be read in as a single value or a list of values (for unique radii).
	coordsys	[str] ('img')	: Coordinate system of the region files. NOTE: there may be issues reading in 'fk5'. 
				   	  'img' (pixel) coordinates are recommended. 
	coordheads	[list]		: Name of header under which coordinates are stored. Will assume ['X','Y'] or ['x','y'] if coordsys='img'
					  or ['RA','Dec'] if coordsys is 'fk5'. 
	verbose 	[bool] (True)	: Set to False to avoid string outputs. 
	shorten_df	[bool] (False)	: If True, shortens the output DataFrame to only include the original ID of each source in df, 
					  the coordinates, and the counterpart IDs of each of the other catalogs. Otherwise, will maintain
					  the original headers of the input DataFrame df.
	outfile		[str]		: Name of output file to save matches to. By default, saves to a file called 'crossref_results.txt'

	RETURNS
	---------
	Matches		[pd.DataFrame]	: DataFrame containing the original ID of each source, its coordinates, and the ID of all 
					  corresponding matches in each of the input region files or coordinates. 
	
	"""

	sources = df.copy()

	xlist = []
	ylist = []
	idlist = []

	# headerlist keeps track of the headers in the input DataFrame
	# if shorten_df = False, will use this to reapply addition headers to the Matches DataFrame
	dfheaderlist = sources.columns.tolist()
	
	# Removing headers that will be duplicated later
	dfheaderlist.remove(sourceid)

	if not isinstance(search_radius, list): search_radius = [search_radius]*len(sources)

	masterlist = [] # list of all matched sources

	if regions:
		if not isinstance(regions, list): regions = [regions]
		for i in regions: 
			idlist.append(GetIDs(i, verbose=False))
			xtemp, ytemp = GetCoords(i, verbose=False)
			xlist.append(xtemp)
			ylist.append(ytemp)
	elif coords: 
		# if given coords, they should be read in as a list of [xcoords, ycoords]. 
		xlist = coords[0]
		ylist = coords[1]
		if not isinstance(xlist, list): 
			xlist = [xlist]
			ylist = [ylist]

	blockend = 0 # keeps track of the index of the current 'block' of counterparts associated with a single base source 

	# Figuring out the coordinate headers if not given
	if not isinstance(coordheads, list): 
		coordheads = [False, False]
		if coordsys == "fk5": 
			if "RA" in dfheaderlist: coordheads[0] = "RA" 
			elif "ra" in dfheaderlist: coordheads[0] = "ra" 
			if "Dec" in dfheaderlist: coordheads[1] = "Dec" 
			elif "dec" in dfheaderlist: coordheads[1] = "dec"
		elif coordsys == "img":
			if "X" in dfheaderlist: coordheads[0] = "X"
			elif "x" in dfheaderlist: coordheads[0] = "x"
			if "Y" in dfheaderlist: coordheads[1] = "Y" 
			elif "y" in dfheaderlist: coordheads[1] = "y" 
		if not coordheads[0] and not coordheads[1]: # if coordinates not found, prompt user to input 
			coordheads = input("Coordinate headers not found. Please input headers separated by comma (xhead,yhead): ")
			coordheads = [i.strip() for i in coordheads.split(",")]
	
	# Removes headers from list, to avoid duplications later
	dfheaderlist.remove(coordheads[0])
	dfheaderlist.remove(coordheads[1])

	if verbose: print("Finding cross-references between sources. This will take a few minutes. Please wait.. ")
	for i in range(len(sources)): # for each source in the DataFrame
		# Properties of each source
		# Pulling the coordinates of each source
		xtemp = sources[coordheads[0]][i]
		ytemp = sources[coordheads[1]][i]

		tempid = sources[sourceid][i]
		tempn = 0  
		# tempn keeps track of the number of overlap sources identified in the current list for the current base source (used as index)

		# Search area around each source
		tempxmax = xtemp+search_radius[i]
		tempxmin = xtemp-search_radius[i]
		tempymax = ytemp+search_radius[i]
		tempymin = ytemp-search_radius[i]

		# Adding each new source to the list, starting as a list of "None" values
		# If no counterparts are found, the source will appear with "None" for counterparts.
		tempids = [None]*(len(idlist) + 1)
		tempids[0] = tempid 			# adding original source ID to the front
		tempids = [xtemp, ytemp] + tempids 	# adding original coordinates to the front
		if not shorten_df: tempids = tempids + [sources[head][i] for head in dfheaderlist] # adding additional header values at end, if requested
		masterlist.append(tempids)		# saving to the full source list

		# Searching each list of sources from each region file to identify overlaps
		for j in range(len(idlist)): # Number of lists (region files) to search through (e.g. for each catalog, search...)
			for k in range(len(xlist[j])): # Number of sources to search through for the current list/region file (e.g. for each source in a specific catalog...)
				# When overlap is found, see if masterlist has room to add it. 
				# If not, add a new row to make room.
				if tempxmax > xlist[j][k] > tempxmin and tempymax > ylist[j][k] > tempymin and \
				sqrt((xlist[j][k]-xtemp)**2 + (ylist[j][k]-ytemp)**2) <= search_radius[i]: # If the catalog source falls within the range of the base source
					try: 
						# With blockend showing how many total items were found prior to the search on this source, 
						# and tempn showing how many counterparts were identified for the current source, 
						# blockend+tempn should identify the index of the current source
						# The following will cycle through all indices from blockend to blockend+tempn 
						# to see where the last open space is
						for n in range(tempn+1):
							if masterlist[blockend+n][j+3] == None: 
								masterlist[blockend+n][j+3] = idlist[j][k]
								break; # After the last open space, break the chain.
							else: pass;
					except: 
						# Exception will be raised once we reach the end of the current list without finding a free space. 
						# Add a new line, if that's the case.
						tempids = [None]*(len(idlist) + 1) # keeps track of the ids associated with the identified source
						tempids[0] = tempid	# adding current source id to front of list
						tempids = [xtemp, ytemp] + tempids	# adding current coordinates to list
						if not shorten_df: 
							tempids = tempids + [sources[head][i] for head in dfheaderlist] # adding additional header values at end
						tempids[j+3] = idlist[j][k] # Add the source to the list of matches
						masterlist.append(tempids)

					tempn = tempn + 1 # adds a count to the identified sources for this file.

		blockend = len(masterlist) # the index of the end of the previous "block" of sources already detected.

	if verbose: print("DONE WITH CLEANING. CREATING DATAFRAME...")

	# If catalogs not given, use the name of the region files as the headers of the DataFrame
	if not catalogs:
		catalogs = []
		try: 
			for r in regions: catalogs.append(r.split(".reg")[0]) 
		except: 
			for i in len(range(xlist)): catalogs.append("ID "+str(i))
	else: catalogs = [cat+" ID" for cat in catalogs]

	# Adding catalogs to the headers to be read into the DataFrame
	headlist = [coordheads[0], coordheads[1], sourceid]
	for i in catalogs: headlist.append(i)	# adding the name of the catalogs to the header list
	if not shorten_df: headlist = headlist + dfheaderlist	# if requesting, readding other original headers to end of list

	vallist = []
	# Converting the masterlist into an array to be read in as DataFrame values
	temp_array = np.array(masterlist).T
	for i in range(len(temp_array)): vallist.append(temp_array[i].tolist())

	Matches = BuildFrame(headers=headlist, values=vallist)
	Matches.to_csv(outfile)

	return Matches


###-----------------------------------------------------------------------------------------------------

def GetDaoPhots(df, photfiles, idheads, filters, magheader="aperture_mag", dmod=0, return_err=True, errorheader="aperture_mag_err"): 

	"""
	Retrieving the photometry for each daosource by their IDs as listed in a given DataFrame. 
	Usually, this is run after running DaoClean() on X-ray sources, followed by Crossref()
	to identify all daosource IDs across all filters for each point source from DaoClean().

	PARAMETERS
	----------
	df		[pd.DataFrame]	: DataFrame containing the source ID of each point source in the filters of interest.
	photfiles	[list]		: List of the files containing the photometry to pull. This assumes the file is saved
					  as it is given by photutils aperture photometry, with the ID of each point source listed
					  under the header 'id'. 
	idheads		[list]		: List of the headers under which each ID is found, in the order the associated photometric
					  file is given in photfiles.
	filters		[list]		: List of the name of the filters for each file in photfiles.
	magheader	[str]		: Name of the header under which the photometry of each point source is stored in each photfile. 
					  By default, sets the header to 'aperture_mag', but if you performed and saved an aperture correction
					  (through AutoPhots.RunPhots or manually), you may wish to set this to 'aperture_mag_corr'.
	dmod		[float]		: Distance modulus (equal to 5*np.log10(distance)-5) used to convert from apparent to absolute 
					  magnitudes. Defaults to 0 to assume magnitudes are already converted, or to return photometry in
					  apparent mags.
	return_err	[bool]		: If True, returns the photometric error from the photometry file and adds to the returned DataFrame.
	errorheader	[str]		: Header under which the photometric errors are stored. If 'return_errs = True' and no errorheader is 
					  given, assumes the header is 'aperture_mag_err'.

	RETURNS
	---------
	df_phots	[pd.DataFrame]	: Returns df with the magnitudes and colors pulled from photfiles appended as additional headers.
	
	"""

	df_phots = df.copy()
	
	# Starting by adding each fitler to the DataFrame as a header under which the photometry will be pulled
	# Then searching the associated photometric file for the measurements of the identified point source
	for i,f in enumerate(filters): 
		# Defining new header for current filter
		df_phots[f] = [np.nan]*len(df_phots) 	# defaults to np.nan if no ID is found. 

		# If errors are requested, add it to the DataFrame
		if return_err: df_phots[f"{f} Err"] = [np.nan]*len(df_phots)

		print("Searching", photfiles[i])

		# Reading in photometry for the current filter
		tempphots = pd.read_csv(photfiles[i], delimiter=" ", comment="#")
		
		# For each ID in source DataFrame, search for the corresponding photometry in temphots
		for j,tempid in enumerate(df_phots[idheads[i]]): 
			tempph = Find(tempphots, "id = " + str(tempid))
			try: df_phots[f][j] = tempph[magheader][0] - dmod	# pulling the photometry from the appropriate header
			except: pass;	# if there is no ID given for this particular line in DataFrame, keep photometric value as np.nan

			if return_err: 
				try: df_phots[f"{f} Err"][j] = tempph[errorheader][0]
				except: pass;
		
	return df_phots

###-----------------------------------------------------------------------------------------------------

def GalComponents(sources, rad=[0], locs=["Disk", "Outskirt"], theta=0, center=None, savereg=False, regname="sources"): 

	"""
	Breaks up the sources within a given galaxy into the the regional components of the galaxy as given by the 'locs' argument. 
	The radii should be given in pixel units. It may be read in as a list of radii, in order of innermost regions outward. 
	The argument 'locs' should give the corresponding location names and should be one element larger than the size of the radius 
	list (preferrably with outskirt being the last element, corresponding to all sources outside of the defined regions of interest). 
	If a given element in the rad list is a list of multiple radii, assume an ellipse at an angle theta and run the ellipse 
	function InEllipse. GalComponents returns the same DataFrame with an added header, Location, which details which component 
	the source appears within.
	"""
	
	from XRBID.WriteScript import WriteReg

	sources = sources.copy()
	locations = ["Outskirt"]*len(sources)
	
	# Finding the center of the galaxy using either given coordinates or the nucleus of the galaxy.
	if center: 
		xcenter = center[0]
		ycenter = center[1]
	else: 
		temp = Find(sources, "Class = Nucleus")
		try: 
			xcenter = temp["X"][0]
			ycenter = temp["Y"][0]
		except: 
			xcenter = temp["x (mosaic)"][0]
			ycenter = temp["y (mosaic)"][0]

	## Start with the outermost region and work inwords
	for i in range(len(sources)): 
		try: 
			xtemp = sources["X"][i]
			ytemp = sources["Y"][i]
		except: 
			xtemp = sources["x (mosaic)"][i]
			ytemp = sources["y (mosaic)"][i]

		#starting with the outermost region and works way inword
		for j in range(1, len(rad)+1): 
			r = rad[-j] # first element is the last in the radius list
			if not isinstance(r, list): # if only one radius given, assume circle
				if sqrt((xcenter - xtemp)**2 + (ycenter - ytemp)**2) <= r:
					# ignoring the last element of the locs list, save the location
					# of the next outermost region
					locations[i] = locs[-j-1]
				else: pass;
			else: # if multiple radii are given, assume ellipse 
				if InEllipse(xtemp, ytemp, [xcenter, ycenter], r[0], r[1], theta): 
					locations[i] = locs[-j-1]
				else: pass;

	sources["Location"] = locations

	if savereg: 
		for i in locs: 

			print(i)
			WriteReg(Find(sources, "Location = " + i), outfile=regname+"_"+i.lower()+".reg", color="red", width=2, radius=50)		

	print("Locations of sources successfully identified")
	return sources

###-----------------------------------------------------------------------------------------------------

def CheckBounds(df, imext=imext, remove=False, search=["x", "y"], resetbounds=False, ID="ID"):

	""" 
	Checking whether sources in the given DataFrame is inside the bounds in the image. 
	Returns a DataFrame with the boundary conditions (whether the source is in or out) marked. 

	""" 

	# For now, assume Bounds is given. If it isn't, set remove to False and only return sources within the bounds.

	frame = df.copy()

	try: 
		if resetbounds: frame[Bounds] = "Out" # resets bounds
	except: remove = True

	# converting image extent (imext) to an array
	# if the length of the array is 1, use imext[0] as min and imext[1] as max for both x and y
	# otherwise, the length should 2, for x-extent and y-extent
	imext = np.array(imext)
	if len(imext) == 1: 
		xmin = ymin = imext[0]
		xmax = ymax = imext[1]
	else: 
		xmin = imext[0][0]
		xmax = imext[0][1]
		ymin = imext[1][0]
		ymax = imext[1][1]
	
	# finding all of the sources within bounds
	temp = Find(frame, [search[0] + " >= " + str(xmin),  search[0] + " <= " + str(xmax), search[1] + " >= " + \
		    str(ymin), search[1] + " <= " + str(ymax)])

	if remove: 
		frame = temp
		try: 
			for i in range(len(frame)): 
				frame[Bounds][i] = "In"
		except: pass; 
	else:	# if not removing all out of bounds, then find all 
		try: inID = temp[ID].values.tolist()
		except: ID = raw_input("ID not found. Enter ID Header: ") 

		inID = temp[ID].values.tolist()  # list of in-bound sources
		frame = frame.set_index(frame[ID].values) # Allows frame to be searchable by ID name

		# setting all in-bound sources to "In"
		for i in inID: 
			frame[Bounds][i] = "In" 

	# resetting the indices of the frame to source number
	frame = frame.set_index(np.arange(len(frame)))
	print("DONE")
	return frame

###-----------------------------------------------------------------------------------------------------

def InEllipse(x,y,center,rad1,rad2,theta): 

	"""Checks whether a source falls within a given ellipse and returns a bool."""

	# Calculating the position of the source relative to the center of the galaxy
	# If pos =< 1, the source is within the ellipse
	theta = theta * 0.0174533
	pos = (((x-center[0])*np.cos(theta)+(y-center[1])*np.sin(theta))**2)/rad1**2 + \
	      (((x-center[0])*np.sin(theta)-(y-center[1])*np.cos(theta))**2)/rad2**2

	# Return True with the source is within the ellipse
	return pos <= 1

###-----------------------------------------------------------------------------------------------------

def Mosaic(df=None, sourceimgs=None, findimgs=None, rows=6, columns=5, filename="mosaic_temp", toplabels=None, bottomlabels=None, top_coords=[225,60], bottom_coords=[225,420], remove_env=True, fontsize=20):

	"""
	Plots a mosaic of the given sources. Either a DataFrame of the sources or a list of the images can be given. Alternatively, 
	can read in a search term [e.g. *.png] to find the appropriate files. 
	NOTE: I don't think the search works currently, probably because of the last step of the search process (4/25/22).

	"""

	# If sourceimgs not given, search for the files with fingimgs	
	if findimgs: 
		sourceimgs = []
		for file in glob.glob(findimgs): 
			sourceimgs.append(file)
		sourceimgs.sort()
	
		# If a DataFrame is given, only use the images of sources within that DataFrame
		tempimgs = []
		for i in sourceimgs: 
			temp = i.split(findimgs.split("*")[0])[1].split(".")[0].split("_")[0] # Finds the ID of the image
			# This assumes the name of the file puts the ID right after *
			if len(Find(df, "ID = " + str(temp))) > 0: tempimgs.append(i)
		sourceimgs = tempimgs

	# Remove the environmental snapshots from the sourceimg list.
	for i in sourceimgs: 
		if remove_env and "env" in i: sourceimgs.remove(i)

	# Calculates the number of mosaics needed to contain all images in the given format
	addone = len(sourceimgs) % (rows*columns) > 0
	totmos = round(len(sourceimgs)/(rows*columns)) + addone

	# If labels isn't a list, assume it's a header.
	if not isinstance(toplabels, list): 
		try: toplabels = df[toplabels].values.tolist()
		except: toplabels=None


	if not isinstance(bottomlabels, list): 
		try: bottomlabels = df[bottomlabels].values.tolist()
		except: bottomlabels=None

	i = 0
	for l in range(1,totmos+1):
		f, ax = plt.subplots(rows,columns, figsize=(columns*3-1,rows*3-1))
		for j in range(0,rows):
			
			for k in range(0,columns):
				try:
					ax[j,k].axis("off")
					ax[j,k].set_aspect("equal")
					ax[j,k].imshow(plt.imread(sourceimgs[i]))
					if toplabels: 
						txttop = ax[j,k].text(top_coords[0], top_coords[1], toplabels[i], color='white', ha='center', weight="extra bold", size=fontsize)
						txttop.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])
					if bottomlabels: 
						txtbot = ax[j,k].text(bottom_coords[0], bottom_coords[1], bottomlabels[i], color='white', ha='center', weight="extra bold", size=fontsize)
						txtbot.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])
				except: pass;
				i += 1
		plt.subplots_adjust(wspace=0, hspace=0.02)
		plt.savefig(filename+"_"+str(l)+".png", dpi=300, bbox_inches="tight")
		plt.show()
	
	print (filename+"_*.png saved!")

