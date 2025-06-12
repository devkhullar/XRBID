###################################################################################
##########		For searching and modifying DataFrames		########### 
##########		Last update: Mar. 4, 2025 			########### 
##########		Update desc: Removed the use of Headers.py 	########### 
##########		   to define a search header in Find().		########### 
##########		   Code will instead assume the user input	########### 
##########		   the proper header name. 			########### 
###################################################################################

import re
import numpy as np
import operator
from astropy.io import ascii
from astropy.io.votable import parse
import pandas as pd
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore")

imext = [0., 13500.] # extent of 

# Operators need to be set up so that criteria can be searched appropriately
# Library of operations (strings) and their functional defintions
oper = {"=": operator.eq, "==": operator.eq, \
	"!": operator.ne, "!=": operator.ne, "=!": operator.ne, \
	">": operator.gt, ">=": operator.ge, "=>": operator.ge, \
	"<": operator.lt, "<=": operator.le, "=<": operator.le}	

# operations that may define a search, used to split header from the criteria and 
# to find correct functional defintions in "oper"
ops = "=<|=>|=!|<=|>=|!=|=|<|>|!" 

# Possible values associated with headers (from Long et al. 2014 https://arxiv.org/pdf/1404.3218.pdf)
# classes include my own classifications as well. 

### NOTE: Searching for N/A does not work! ###

classes = ["AGN", "SS", "SNR", "XRB", "GAL", "Star", "AGN/GAL", "AGN/XRB", \
	   "None", "HMXB", "IMXB", "LMXB", "Cluster", "Quasar", "Unknown", "N/A"]
lowerclass = [x.lower() for x in classes]
spects = ["P", "T", "D", "P+D", "P/T", "P/D"]
varis = ["V", "B", "C", "N", "A", "AN", "V*", "BN"]


###-----------------------------------------------------------------------------------------------------

def Find(df, criteria, verbose=False): 

	"""
	Allows DataFrames to be easily searchable without the need for weird DataFrame nesting.

	PARMETERS
	---------
	df	  [pd.DataFrame]  : DataFrame containing sources/data to search
	criteria  [list]	  : List of search criteria, each written as a string
				    in the format of '[header] [operator] [criteria]'.
				    If no header or operator is given, the code attempts
				    to assume the header based on the input.
	verbose	  [bool] (False)  : If True, prints a warning when the search fails
	
	RETURNS
	--------
	df 	  [pd.DataFrame]  : DataFrame containing only sources that fit all search criteria

	"""

	df = df.copy()

	# allowing the inclusion of several criteria
	# if only one criterion given not as a list, turn into a list. Avoids errors. 
	if not isinstance(criteria, list): criteria = [criteria] 

	# For each criterion given in the list... 
	for crit in criteria: 

		### DECODING THE SEARCH CRITERIA ###	
		
		raw_search = re.split(ops, crit) # keeping track of the raw input, for if case matters
		crit = crit.lower() 		 # working first in lower-case space
		temp = re.split(ops, crit)	 # splitting the search by operators, but set to lowercase
		
		# if there are extra spaces included in the search, they should be removed. 
		if temp[0][-1] == " ": 
			temp[0] = temp[0][:-1]
			raw_search[0] = raw_search[0][:-1]
		if temp[-1][0] == " " : 
			temp[-1] = temp[-1][1:]
			raw_search[-1] = raw_search[-1][1:]

		# Parentheses indicate Long candidate classifications. Keep track of these! 
		if "(" in temp[-1]: 	# AKA if candidate...
			temp[-1] = re.sub(r"\(|\)", "", temp[-1])	# remove () from temp for now
			raw_search[-1] = re.sub(r"\(|\)", "", raw_search[-1])
			cand = True
		else: cand = False

		### At this point, temp is a list split at the operator, if available 	###
		### (operator removed). If the length of temp is larger than 1, 	###
		### then an operator was given, [0] should be the search header, and 	###
		### [-1] should be the search criteria. If the length == 1, then the 	###
		### search may be for specified Class, Spectral Type, variability, 	###
		### extent, overlap, or bounds. First try for former case. 		###
		### Then override with conditions of latter, if applicable. 		###


		# Setting the header based on the user input
		search_head = raw_search[0]
		if search_head[-1] == " ": search_head = search_head[:-1]
		search = raw_search[-1]
		
		# NOTE: search is now the criteria we're searching for under the criteria.	###
		### It is still string! The following code will override the search and 	###
		### search_head definitions if applicable. This is for searches on the basis 	###
		### of bounds, classification, spectral type, extent, or overlap.               ###

		# First search for any variations of Bounds, Overlap, and Extended criteria
		# All booleans here should be set as strings first, to allow proper searching.
		if "bounds" in temp[0]:	
			# Searching if sources is within bounds. Bounds is a header and should always be in [0]
			search_head = "Bounds"
			# Out/In can be in [3] or [0] (both can be [-1])
			if "out" in temp[-1] or "false" in temp[-1]: search = "Out" 
			else: search = "In"	# In is the default, unless Out is specified
		elif "overlap" in temp[0]: 	# Searching for overlap with Long (or other) catalog
			search_head = "Overlap"
			if "no " in temp[-1] or "false" in temp[-1]: search = "No"; pass;
			else: search = "Yes"; pass;	# default is yes, unless no or not is specified
		elif "variability" in temp[0]: 
			search_head = "Variability Flag" 
			if "no " in temp[-1] or "false" in temp[-1]: search = "False" 
			else: search = "True"; pass;

		# Searches without operators return length == 1. These correspond to searches
		#	of class, spectral type, variability, extent, or bounds. Check accordingly.
		if len(temp) == 1: 
			temp_split = re.split(" ", temp[0])[-1]  # Necessary, for reasons I've forgotten

			# Long classifications can have alternative formats. Standardize. 
			if temp_split == "xrb/agn": 
				temp_split = "agn/xrb" 
			elif temp_split == "gal/agn": 
				temp_split = "agn/gal"
			elif temp_split == "t/p":
				temp_split = "p/t"
			elif temp_split == "d/p": 
				temp_split = "p/d" 
			elif temp_split == "d+p": 
				temp_split = "p+d"    # only variation that isn't automatically a candidate

			# Now checking for header type, if temp_split is contained in object classes, 
			# spectral classes, or variable classes.
			if temp_split in [x.lower() for x in classes]: 
				# setting all classes to lowercase for comparison
				temp2 = [x.lower() for x in classes]	
				search = classes[temp2.index(temp_split)]
				search_head = "Class"
				try: df[search_head]
				except: search_head = "Long Class"
			elif temp_split in [x.lower() for x in spects]: 
				temp2 = [x.lower() for x in spects]
				search = spects[temp2.index(temp_split)]
				search_head = "Spectra"
			elif temp_split in [x.lower() for x in varis]: 
				temp2 = [x.lower() for x in varis]
				search = varis[temp2.index(temp_split)]
				search_head = "Variable"
			pass;

		# Setting the operator for the search 
		# Tells the function how to implement the search
		op = ""
		for i in ops.split("|"): 
			if i in crit: op = i; break; # searching for the correct input operator
		if not op: op = "="	# default operation is equal to

		if "/" in search and search != "N/A": cand = True	# multiple identifications imply candidacy
		if search == "True": search = True; pass;	# These need to be boolean for the search to work
		elif search == "False": search = False; pass;

		if "none" not in crit: 
			# avoid double-negatives in the logic below
			if "no " in crit and search != False and search_head.lower() != "dao no": op = "!=" 
		# In case search above doesn't catch the search of classes. 
		if search in lowerclass: search = classes[lowerclass.index(search)]

		datatype = df[search_head].values.dtype

		# Try to convert the search to a number, if applicable. 
		try: search = float(search)
		except: pass;

		### At this point, search should be set to the appropriate search criteria, ###
		### search_head should be the corrected header, and whether or not the the  ###
		### search is for a class candidate and the given operation are noted.      ###
		### Should also now know if user is doing a 'negative' search 		    ###
		### (False or ! given criteria) 					    ###


		### IMPLEMENTING SEARCH ###
		#print(search, search_head)
		searching = ""

		# Checking if search is NaN. If so, change to string 
		if not isinstance(search,str): 
			try:
				if np.isnan(search): search = "nan"		
			except: pass;

		if search == "nan": 	# if search criteria is NaN, need a special search function
			if "!" in op or "false" in search: 
				df = df[~df[search_head].isnull()].reset_index().drop("index", axis=1) # excluding NaNs
			else: df = df[df[search_head].isnull()].reset_index().drop("index", axis=1) # searching for NaNs
		elif "all" in crit: 	# if searching for all possible instances, including candidates
			if "/" in search and search != "N/A": temp = ["("+search+")"]; pass;
			else: temp = []				
			for i in re.split(r"\W+", search):  # adding all possible variations to a temp list
				temp.append(i)
				temp.append("("+i+")")
			for i in temp: 		# adding all variations from temp to the searching function
				try: searching = operator.or_(searching, oper[op](df[search_head], i))
				except: searching = oper[op](df[search_head], i)
		elif "no " in crit and "none" not in crit and search_head.lower() != "dao no": 	
			# if searching for all possible instances to exclude
			if "/" in search and search != "N/A": temp = ["("+search+")"]; pass;
			else: temp = []				
			for i in re.split(r"\W+", search):  # adding all possible variations to a temp list
				temp.append(i)
				temp.append("("+i+")")
			for i in temp: 		# adding all variations from temp to the searching function
				try: searching = operator.and_(searching, oper[op](df[search_head], i))
				except: searching = oper[op](df[search_head], i)		
		else: 				# all other searches can be implimented the same way
			try: 
				if datatype != "object": 
					# ID will sometimes be a number, 
					# but should be treated as string. Else, convert to float.
					search = float(search)	
			except: pass;	# if failed, search is a string.

			# Check if candidate and reapply parentheses
			if cand: search = "("+search+")"; pass;	# reapplying candidate parentheses
			searching = oper[op](df[search_head], search)

		# Catching all possible case variations, so that search is fairly case insensitive
		try: 	# operator used will depend on op
			if op == "!=": 
				searching = operator.and_(searching, oper[op](df[search_head], search.lower()))
				searching = operator.and_(searching, oper[op](df[search_head], search.upper()))
				searching = operator.and_(searching, oper[op](df[search_head], search.title()))
				searching = operator.and_(searching, oper[op](df[search_head], raw_search))
				if search == "None" or search == "none":  # Could be string or actual None
					searching = operator.and_(searching, oper[op](df[search_head], None))
			else: 
				searching = operator.or_(searching, oper[op](df[search_head], search.lower()))
				searching = operator.or_(searching, oper[op](df[search_head], search.upper()))
				searching = operator.or_(searching, oper[op](df[search_head], search.title()))
				searching = operator.or_(searching, oper[op](df[search_head], raw_search))
				if search == "None" or search == "none":  # Could be string or actual None
					searching = operator.or_(searching, oper[op](df[search_head], None))
		except: pass;	# above will fail if search is a float

		try: df = df[searching].reset_index().drop("index", axis=1)
		except: 
			if verbose: print("Search failed. Objects matching criterion not found.") 
	
	return df

###-----------------------------------------------------------------------------------------------------

def FindUnique(df, criteria=None, header='ID'):

	"""
	Finds unique sources from the given DataFrame. Sources from CSCView may be duplicated if multiple 
	observations are available. If search criteria are given, will return only unique matching sources.
	
	PARMETERS
	---------
	df	  [pd.DataFrame]  : DataFrame containing sources/data to search
	criteria  [list]	  : List of search criteria, each written as a string
				    in the format of '[header] [operator] [criteria]'
	header	  [str] ('ID')    : Header under which only unique values will be returned
	
	RETURNS
	--------
	df 	  [pd.DataFrame]  : DataFrame containing only sources that fit all search criteria

	"""

	# NOTE: This needs to be modified to take some kind of mean or median of values 
	#       that differ between observations, such as HR

	df = df.copy()
	remove = []
	for i in range(1, len(df)): 
		try: 
			if df[header][i] == df[header][i-1]: remove.append(i)
		except: 
			if df["CSC ID"][i] == df["CSC ID"][i-1]: remove.append(i)
	df = df.drop(df.index[remove])
	if criteria == None: return df.reset_index().drop("index", axis=1)
	else: return Find(df, criteria)

###-----------------------------------------------------------------------------------------------------

def FindAll(df,ids,header="ID"): 

	"""
	Find all IDs in a given list
	
	PARMETERS
	---------
	df	  [pd.DataFrame]  : DataFrame containing sources/data to search
	ids	  [list]	  : List of source IDs to search for
	header	  [str] ('ID')	  : Header under which IDs are stored 
	
	RETURNS
	--------
	df 	  [pd.DataFrame]  : DataFrame containing only sources that match the ID list

	"""

	searching = operator.eq(df[header], ids[0])
	for i in range(1, len(ids)): 
		searching = operator.or_(searching, operator.eq(df[header], ids[i]))
	return df[searching]
###-----------------------------------------------------------------------------------------------------

def GetVals(df, header, tolist=False): 

	"""
	Returns the values from the header within a given DataFrame as a list that can be plotted or 
	otherwise manipulated.
	
	PARMETERS
	---------
	df	  [pd.DataFrame]  : DataFrame containing sources/data to search
	header	  [str]		  : Header from which to obtain the values
	tolist	  [bool] (False)  : If true, returns the values as a list
	
	RETURNS
	--------
	List of values under the input header

	"""
	
	df = df.copy()

	if tolist: return df[header].values.tolist()
	else: return df[header].values

###-----------------------------------------------------------------------------------------------------

def BuildFrame(headers=None, values=None, infile=None, sources=None, size=None, headers_index=None, savefile=None, ascii_file=False):

	"""
	Creates a new DataFrame from given headers and sources. May use anticipated size instead, if no sources
	are given. If an infile name is given but headers are not, this function assumes the first line in the file 
	are the headers and will align values accordingly. Otherwise, headers_index represents which column in 
	the datafile aligns to which index. 
	
	PARMETERS
	---------
	headers	  	[list] 	  	: List of headers under which values are stored. Optional only if infile is given.
	values	  	[list]	 	: Optional; List of values (or list of values) add under each header
	infile  	[str]		: Optional; name of file to generate DataFrame from
	sources	  	[pd.DataFrame]  : Optional; DataFrame that can be used to build the new DataFrame. 
				     	  If sources contains a defined header, set the value(s) for that header to 
					  that of sources
	size	  	[int]		: Optional; if no values are known, fill DataFrame with [size] x "None" 
					  under each header
	headers_index	[list]		: Optional; if the DataFrame is being built from a text file, defines the 
					  column index of each header as they appear in that text file. 
					  If this is not set properly, the wrong values
					  may be read in from the data file for each of the desired headers.
	savefile	[str]		: Optional; name of the file to save the DataFrame to
	ascii_file	[bool] (False)	: Set to true if the data file people read in as infile is an ASCII file; 
					  This is useful when reading in a data file downloaded from a paper (AAS style)
	
	RETURNS
	--------
	df 	  [pd.DataFrame]  :  DataFrame containing the specified headers and (optional) values
	
	""" 
	
	# If infile is given, build from file. Whether or not headers are given 
	# will determine the assumed format of the data file.
	if infile: 
		if headers:
			if headers_index == None:
				headers_index = [i for i in range(len(headers))]
			elif len(headers_index) != len(headers): 
				print("WARNING: incorrect number of header indices given.")
				headers_index = [i for i in range(len(headers))]

			if ascii_file: 
				# If file is an ascii file, read in this way...
				try: temp = ascii.read(infile)
				except: temp = ascii.read(infile, delimiter="\t")
				values = [temp[i] for i in temp.columns]
			else: 
				# Read in values from file
				temp = np.genfromtxt(infile, dtype=str)
				temp = np.array(temp).T.tolist()
				values = [temp[i] for i in headers_index]

		else: 
			# in the event no headers are given, assume first line is headers 
			# (as is the case when copying over the master file)
			if ascii_file: 
				temp = ascii.read(infile)
				headers = [i for i in temp.columns]

				# Convert coordinates if given in ApJ format
				try:			
					tempra = [str(temp["RAh"][i]) + ":" + str(temp["RAm"][i]) + ":" + str(temp["RAs"][i]) for i in range(len(temp))]
					tempdec = [str(temp["DEd"][i]) + ":" + str(temp["DEm"][i]) + ":" + str(temp["DEs"][i]) for i in range(len(temp))]
					tempra, tempdec = HMS2deg(tempra, tempdec)
					headers.remove("RAh")
					headers.remove("RAm")
					headers.remove("RAs")
					headers.remove("DEd")
					headers.remove("DEm")
					headers.remove("DEs")
				except: pass;
	
				values = [temp[i] for i in headers]
				try: 
					values.append(tempra)
					values.append(tempdec)
					headers.append("RA")
					headers.append("Dec")
				except: pass;

			else: 
				with open(infile) as f: 
				    lines = f.readlines()[:]
				    
				temp0 = [line.strip("\n") for line in lines]
				temp1 = [line.split("\t") for line in temp0]
				headers = temp1[0]
				values = np.array(temp1[1:]).T.tolist()
				
				# Convert proper values to floats instead of strings
				for i in range(len(values)): 
					try: 
						for j in range(len(values[i])): 
							if values[j][i][-1] == " ": 
								values[j][i] = values[j][i][:-1]
							if values[i][j] == "nan" or values[i][j] == "NaN": 
								values[i][j] = np.nan
							try: values[i][j] = float(values[i][j])
							except: pass;
					except: pass;


	df = pd.DataFrame(columns=headers)

	if isinstance(sources, pd.DataFrame): 
		for i in headers: 
			try: df[i] = sources[i]; pass;
			except: pass;
	elif values: 
		for i in range(len(headers)): 
			df[headers[i]] = values[i]
		pass;
	elif size: 
		df[headers[0]] = ["None"]*size

	# Trying again to convert proper values to float, if it failed the first time: 
	for j in df.columns.values: 
		for i in range(len(df)): 
			df[j][i] = pd.to_numeric(df[j][i], errors="ignore")

	if savefile: df.to_csv(savefile)

	return df

###-----------------------------------------------------------------------------------------------------

def RemoveHeader(df=None, headers=None, savefile=None):

	"""
	Removes input headers and values to the input DataFrame and returns as a new DataFrame.

	
	PARMETERS
	---------
	df	  	[pd.DataFrame]  : DataFrame to modify
	headers	  	[list] 	  	: List of headers to remove from the DataFrame
	savefile	[str]		: Optional; name of the file to save the DataFrame to
	
	RETURNS
	--------
	NewFrame	[pd.DataFrame]  : New DataFrame with headers removed

	""" 

	df = df.copy()

	# Setting headers as a list, if not already
	if not isinstance(headers, list): headers = [headers]

	# Remove all headers given in the list
	allheads = list(df.columns)
	for i in headers: 
		allheads.remove(i)
	
	allvals = []
	for i in allheads: 
		allvals.append(df[i].values.tolist())

	NewFrame = BuildFrame(headers=allheads, values=allvals)

	if savefile: NewFrame.to_csv(savefile)
	
	return NewFrame

###-----------------------------------------------------------------------------------------------------

def Remove(df, remove=None, header='ID', criteria=None): 

	"""
	Removes all IDs in a given list. Good for filtering out specific objects if IDs are known.
	
	PARMETERS
	---------
	df	  	[pd.DataFrame]  : DataFrame to modify
	remove		[list]		: Optional; List of values to removed from the DataFrame
	header	  	[str] ('ID') 	: Optional; Header under which to search for the removable values
	criteria 	[list]	 	: Optional; can define removal criteria instead of defining remove and header above
	
	RETURNS
	--------
	Returns a new DataFrame with the specified entries removed

	"""

	# Criteria = None with removing just a list of IDs. 
	# If instead trying to remove sources that meet all criteria, 
	# run search first and then remove.

	try: 
		# remove should be a list of IDs
		if not isinstance(remove, list): remove = [remove]

	except: 
		if criteria: remove = Find(df, criteria)[header].values.tolist()

	criteria = [header + " != " +  str(i) for i in remove]

	return Find(df, criteria)

###-----------------------------------------------------------------------------------------------------

def RemoveElse(df, keep=None, header='ID', criteria=None): 

	"""
	Removes all IDs NOT in a given list. Good for picking out specific objects.
	
	PARMETERS
	---------
	df	  	[pd.DataFrame]  : DataFrame to modify
	keep		[list]		: Optional; Values to keep
	header	  	[str] ('ID') 	: Optional; Header under which to search for the removable values
	criteria 	[list]	 	: Optional; can define criteria instead of defining headers and values to keep
	
	RETURNS
	--------
	Returns a new DataFrame containing only the specified entries

	"""

	df = df.copy()

	try: 
		# remove should be a list of IDs
		if isinstance(keep, pd.DataFrame): keep = keep[header].values.tolist()
		elif not isinstance(keep, list): keep = [keep]

	except: 
		if criteria: keep = Find(df, criteria)[header].values.tolist()

	criteria = [header + " != " +  str(i) for i in keep]
	df_temp = Find(df, criteria)
	df_temp = [str(i) for i in df_temp[header]]

	return Remove(df, df_temp, header=header)	
		
###-----------------------------------------------------------------------------------------------------

def Convert_to_Number(df, headers=None): 

	""" 
	Converting the values in a header to numeric, if possible 

	PARMETERS
	---------
	df	  [pd.DataFrame]  : DataFrame to modify
	headers	  [list]	  : Optional; Specifying headers to convert to numbers. 
				    If not specified, code will attempt to convert all values but
			            will only convert those that are actually numbers.
	
	RETURNS
	--------
	df	  [pd.DataFrame]  : DataFrame in which applicable values are converted from str to a number

	""" 

	df = df.copy()

	if not headers: headers = df.columns.values.tolist()
	if not isinstance(headers, list): headers = [headers] 

	for h in headers: 
		for i in range(len(df)): 
			try: 
				test = float(df[h][i])# = pd.to_numeric(df[j][i])
				df[h][i] = pd.to_numeric(df[h][i], errors="coerce")
			except: pass;
	return df

###-----------------------------------------------------------------------------------------------------

def Remove_Spaces(df, headers=None): 

	""" Removing spaces accidentally placed after DataFrame values. """ 

	df = df.copy()

	if not headers: headers = df.columns.values.tolist()
	if not isinstance(headers, list): headers = [headers] 

	for h in headers: 
		if h == "CSC ID" or h == "Notes" or h == "Note": pass;
		else: 
			for i in range(len(df)): 
				try: 
					df[h][i] = df[h][i].strip(" ")
				except: pass;
	return df

###-----------------------------------------------------------------------------------------------------

def Reset_Index(df): 

	""" Resets the index of a DataFrame, renumbering from 0 to len(df) """

	return df.reset_index().drop("index", axis=1)

###-----------------------------------------------------------------------------------------------------

def HMS2deg(ra='', dec=''):
	
    """
	Converts from hours, minutes, and seconds (and days for Dec) into degrees. 
	Good for building DataFrames from ascii tables from ApJ papers.


	PARMETERS
	---------
	ra	  [str]  : RA in H:M:S format, or with spaces instead of ':'
	dec	  [str]	 : Dec in D:M:S format, or with spaces instead of ':', with preceeding + or - 
	
	RETURNS
	--------
	RA, DEC (or whichever is input)

	"""

    RA, DEC, rs, ds = [],[], 1, 1
    if dec:
        for j in dec: 
            if ":" in j: D, M, S = [float(i) for i in j.split(":")]
            else: D, M, S = [float(i) for i in j.split(" ")]
            if str(D)[0] == '-': ds, D = -1, abs(D)
            deg = D + (M/60) + (S/3600)
            DEC.append(deg*ds)

    if ra:
        for j in ra:
            if ":" in j: H, M, S = [float(i) for i in j.split(":")]
            else: H, M, S = [float(i) for i in j.split(" ")]
            if str(H)[0] == '-': rs, H = -1, abs(H)
            deg = (H*15) + (M/4) + (S/240)
            RA.append(deg*rs)
  
    if ra and dec:
        return (RA, DEC)
    else:
        return RA or DEC
