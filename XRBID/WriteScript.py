###########################################################################################
##########	For writing scripts, such as bash scripts and region files	########### 
##########	                Last Update: Jun 03, 2025               	########### 
##########	Streamlined some I/O code, made minor changes to WriteDS9.	########### 
##########	Major fix leftover from removing tiling from WriteDS9.		########### 
###########################################################################################

import math
import datetime
import re
import numpy as np
from astropy.io.votable import parse
import pandas as pd
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore")

from XRBID.DataFrameMod import FindUnique
from XRBID.Sources import GetCoords
from XRBID.DataFrameMod import BuildFrame

X = "x" 
Y = "y"
RA = "RA" 
Dec = "Dec" 
ID = "ID"

###-----------------------------------------------------------------------------------------------------

def WriteDS9(df=None, galaxy="galaxy", colorfiles=None, regions=None, scales="zscale", imgnames=None, imgsize=[305,305], outfile=None, unique_scale=False, coords=None, ids=None, idheader="ID", filetype="jpeg", basefilter=["red"], filterorder=["red", "green", "blue"], zoom=8, env_zoom=2, coordsys=None, path_to_ds9=False): 

	"""
	Generates a bash script that will write a program for taking thumbnails/snapshots in DS9 of sources in a given DataFrame. 
	Will take in one or more region files to open over the given galaxy file, and different scales to use to adjust the RGB colors of the image. 
	Scales should be input with the format ["scale name", redscale, greenscale, bluescale], or, if unique_scale is True, the name of the file 
	with the unique scalings should be used. If zscale is being used, only ["zscale"] is needed (**** SETTING TO ZSCALE IS CURRENTLY BROKEN. 
	INSTEAD, ALWAYS USE A UNIQUE SCALES***). Parameter basefilter can be used to set which filter the region file should be aligned to, if not 
	the default green (Ex. red used as the base filter in M81). The very first basefilter should be the main filter used for imaging, even if 
	the region files are aligned to different filters. Filters should be called in the order they're intended to be used (default [red, green, blue]). 
	If a different order is used, the order should be declared using the 'filterorder' parameter. 

	PLANNED UPDATE (post v1.7.1): 
	WriteDS9 will be updated such that the user can read in a list of zoom factors, rather than set just the 'zoom' and 'zoom_env' 
	parameters. This update will render the 'env_zoom' parameters obsolete. WriteDS9 will check whether 'zoom' is a list and, if so,
	will add to the image names to label them `_zoom{zoom factor}` before the user-defined suffix (if given). This will allow more 
	than two zoom factors to be given, allowing greater flexibility for the user. 


	PARAMETERS: 
	-----------
	df		[pd.DataFrame]	:	DataFrame containing the sources to image.
	galaxy		[str]		:	Name of the galaxy containing the images. This is used to name certain files. 
	colorfiles	[list]		:	List of the FITS files used to create color images, in RGB order. Currently requires
						all 3 channels to be filled. If an alternate order is used, it should be defined by the 'filterorder' parameter.
	regions		[list]		: 	List of region files to include in DS9 images. 
	scales		[list, str]	: 	Either a list of scaling parameters for each filter (e.g. ["scale name", redscale, greenscale, bluescale])
						or, if unique_scale is set to True, the name of the file with the unique scalings output by WriteScalings. 
	imgnames	[list, str]	:	Either a list containing the unique names of each image to be saved, or the prefix to append to each 
						image name. If a single string is given, the images will be differentiated by image number.
	imgsize		[list]		:	Integer dimensions of saved images. The DS9 window will be rescaled to this size. 
	outfile		[str]		:	Name of the bash script to save. 
	unique_scale	[bool]		:	If True, looks for unique scale parameters for each source. These should be read in as a txt file in 'scales' parameter. 
	coords		[list]		:	Instead of reading in a DataFrame, can instead read in a list of coordinates. Assumes the coodinate system is 'image'
						unless otherwise given in 'coordsys'. 
	ids		[list]		:	List of source IDs. Only needed if only coordinates are provided (to build new DataFrame). 
	idheader	[str]		:	Header of the ID column in the DataFrame. Defaults to "ID". 
	filetype	[str]		: 	File format of the images to be saved. Defaults to "jpeg" to save memory, but "png" may also be used. 
	basefilter	[str]		:	The filter/channel whose coordinates are to be used when opening region files. This is useful when there are slight 
						coordinate differences between the filters and/or the region files were created using a specific base filter. 
						Default is "red". 
	filterorder	[list]		:	Order in which the filters were input. The default is ["red","green","blue"]. 
	zoom		[int]		:	DS9 zoom setting, for the closest zoom. Default is 8. 
	env_zoom	[int]		:	DS9 zoom setting, for the farthest zoom. This allows DS9 to take an image of the 
						environment around each source. Default is 2. (Will be depricated after v1.7.2).
	coordsys	[str]		:	Coordinate system of the source coordinates (image or fk5 preferred). 
	tile		[bool]		:	Determines whether to save a tiled image, in which each of the color channels are tiled side-by-side instead of opening a 
						full color image. Default is False. 
	path_to_ds9 [string]    :   Path to the directory which stores ds9. Only to be used if your ds9 is not configured properly for the bash script to work.
						
	"""
	
	frame = df.copy()

	if ".txt" in scales: unique_scale = True

	good = False

	# If only coordinates are given, build DataFrame from those.
	try: 
		if not frame: 
			if coordsys == "fk5" or coordsys == "wcs": 
				headers = ["ID", "RA", "Dec"]
			else: headers = ["ID", "x", "y"]
			if not ids: ids = np.arange(coords[0]).tolist()
			frame = BuildFrame(headers=headers, values=[ids, coords[0], coords[1]])
	except: pass;

	frame = frame.copy()
	#display(frame) 

	if colorfiles == None: # Name of the files for RGB frames needed
		while not good: 
			colorfiles = input("Names of .fits files for galaxy color map (red, green, blue; separate by commas): ")
			colorfiles = [c.strip() for c in colorfiles.split(",")] # breaks up filters and removes trailing/leading whitespace
			if len(colorfiles) == 3: good = True
			else: print("Not enough color files provided."); colorfiles = None; 
		
	if regions == None: # In case region files were forgotten, ask user for input. 
		regions = input("Region files? (separate by commas, or Enter if none):") 
		regions = [r.strip() for r in regions.split(",")]
	if not isinstance(regions, list): regions = [regions]  # region files should be lists

	# basefilter should be indicated for every region
	# If only one basefilter is given, assume all regions are aligned to that filter.
	if not isinstance(basefilter, list): 
		basefilter = [basefilter]
	if len(basefilter) == 1 and regions: basefilter = [basefilter[0]]*len(regions)
	base = filterorder[0]

	if outfile == None: # Name of output script file
		outfile = input("Output (script) file name?: ")
		if len(outfile) == 0: outfile = galaxy+".sh" # if no name is given, just use the galaxy name
		if len(outfile.split(".")) < 2: outfile = outfile+".sh" # if .sh is not included in the name, add it


	### If using unique scales for each source ###
	if unique_scale == True: 
		# Currently, this looks at the unique scalings file and finds all sources in there in the DataFrame. 
		# It might be more useful to look for the scaling for each DataFrame source read in. 
		# To do that, may want to convert the scalings to a DataFrame and perform as search on it.

		if ".txt" in scales: scalings = np.genfromtxt(scales, dtype=str).tolist()
		else: scalings = np.genfromtxt(input("Unique scalings file?: "), dtype=None).tolist()

		#try: 
		print(f"Writing {outfile}...")
		with open(outfile, 'w') as f: 
			f.write(f"#! /bin/bash\necho \"Creating images of {galaxy} sources. Please wait...\"\n")
			# code for opening color image of galaxies
			f.write(f"ds9 -height {imgsize[0]} -width {imgsize[1]} -colorbar no \\\n-rgb -{filterorder[0]} -zscale {colorfiles[0]}")	# green needs to be opened first, for alignment reasons
			f.write(f" \\\n-{filterorder[1]} -zscale {colorfiles[1]}")
			f.write(f" \\\n-{filterorder[2]} -zscale {colorfiles[2]} \\\n") 
			# code for opening region files (if applicable)
			try: 
				for i in range(len(regions)): 
					f.write(f"-{basefilter[i]} -region load {regions[i]} ")
			except: pass;
			# For each of the scalings, look to see if the source is in the DataFrame
			#  Making sure the scalings is in the right form
			if not isinstance(scalings[0], list): scalings = [scalings]
			# If so, save image. If not, pass.
			for k in range(len(scalings)):
				j = scalings[k]
				#try: 
				if isinstance(imgnames, list): tempname = str(imgnames[k])
				else: tempname = f"{imgnames}_{j[0]}" # if imgname is given, use it
				#except: tempname = j[0]

				# Search dataframe for the scaling ID
				temp = FindUnique(frame, f"{idheader} = {j[0]}")
					
				# If ID is found, print pan script to .sh file
				if len(temp) > 0:
					f.write(f"\\\n-zoom to {zoom} ")
					if coordsys == "img" or coordsys == "image":
						try: f.write(f"\\\n\\\n-{base} -pan to {temp["x"].values[0]} {temp["y"].values[0]} image " )
						except: f.write(f"\\\n\\\n-{base} -pan to {temp["X"].values[0]} {temp["Y"].values[0]} image " )
					elif coordsys == "galaxy" or coordsys == "fk5":
						f.write(f"\\\n-{base} -pan to {temp["RA"].values[0]} {temp["Dec"].values[0]} fk5 ")
					else:
						try: f.write(f"\\\n\\\n-{base} -pan to {temp["x"].values[0]} {temp["y"].values[0]} image " )
						except: f.write(f"\\\n-{base} -pan to {temp["RA"].values[0]} {temp["Dec"].values[0]} fk5 ")
					if j[1] == "-zscale" or j[1] == "zscale":
						f.write("\\\n-red -zscale -green -zscale -blue -zscale ")
					else: f.write(f"\\\n-red -scale limits 0 {j[1]} -green -scale limits 0 {j[2]} -blue -scale limits 0 {j[3]} ")
					f.write(f"\\\n-saveimage {filetype} {tempname}.{filetype} ")
					f.write(f"\\\n-zoom to {env_zoom} ")
					f.write(f"\\\n-saveimage {filetype} {tempname}_env.{filetype} ")
				else: pass;
			f.write("\\\n-exit\necho \"Done.\"")               
		f.close()
		print("DONE")
		#except: print("Error creating file.")


	### If not using unique scales for each source (default) ###
	else: 
		if scales == None:
			scales = []
			temp = "none" 
			while len(temp) > 0:
				temp = input("Color scales? (enter one at a time. Example: \"zscale\" or [\"red\", 10, 5, 2.5]. Press Enter when finished.)")
				if len(temp)>0: scales.append([x for x in re.split(",", temp)])
			
		if not isinstance(scales, list): scales = [scales] # Scales should be a list
		print(f"Writing {outfile}...")
		with open(outfile, 'w') as f: 
			f.write(f"#! /bin/bash\necho \"Creating images of {galaxy} sources. Please wait...\"\n")
			# code for opening color image of galaxies
			f.write(f"ds9 -height {imgsize[0]} -width {imgsize[1]} -colorbar no \\\n-rgb -{filterorder[0]} -zscale {colorfiles[0]}")
			f.write(f" \\\n-{filterorder[1]} -zscale {colorfiles[1]}")
			f.write(f" \\\n-{filterorder[2]} -zscale {colorfiles[2]} \\\n") 
			# code for opening region files
			try:
				for i in range(len(regions)): 
					f.write(f"-{basefilter[i]} -region load {regions[i]} ")
			except: pass;
			for i in range(len(frame)): 
				f.write(f"\\\n-zoom to {zoom} ")
				if coordsys == "img" or coordsys == "image":
					try: f.write(f"\\\n-{base} -pan to {frame["x"][i]} {frame["y"][i]} image ")
					except: f.write(f"\\\n-{base} -pan to {frame["X"][i]} {frame["Y"][i]} image ")
				elif coordsys == "galaxy" or coordsys == "fk5":
					f.write(f"\\\n-{base} -pan to {frame["RA"][i]} {frame["Dec"][i]} fk5 ")
				else:
					try: f.write(f"\\\n-{base} -pan to {frame["x"][i]} {frame["y"][i]} image " )
					except: f.write(f"\\\n-{base} -pan to {frame["RA"][i]} {frame["Dec"][i]} fk5 ")
				for j in scales: 
					try: j = j.strip(",")
					except: j[0] = j[0].strip(",")
					if not isinstance(j, list): j = [j] # Convert to list

					# If a list of image names is given, use this. 
					# Else, if imgnames is a prefix, apply source numbers to image file name.
					# NOTE: SOMETHING IS GOING WRONG HERE WHEN UNIQUE SCALES AREN'T USED. WILL NEED TO LOOK INTO THIS LATER!
					if isinstance(imgnames, list): imgtemp = imgnames[i]
					else: imgtemp = imgnames+"%03i"%(i)

					if len(j) == 1:
						f.write("\\\n\\\n-red -zscale -green -zscale -blue -zscale ")
						pass;
					elif len(j) > 3: 
						f.write(f"\\\n\\\n-red -scale limits 0 {j[1]} -green -scale limits 0 {j[2]} -blue -scale limits 0 {j[3]} ")
						# In the case where multiple, non-unique scales are given, append current scale to imagenames
						if len(scales) > 1: imgtemp = [n + "_" + j[0] for n in imgtemp]
						else: pass;
						pass;
					f.write(f"\\\n-saveimage jpeg {imgtemp}.jpg ")
			f.write("\\\n-exit\necho \"Done.\"")               
		f.close()
		print("DONE") 

###-----------------------------------------------------------------------------------------------------

def WriteReg(sources, outfile, coordsys=False, coordheads=False, coordnames=False, idheader=False, color="#FFC107", radius=False, radunit=False, label=False, width=1, fontsize=10, bold=False, dash=False, addshift=[0,0], savecoords=None, marker=False, fill=False): 

    """
    Writes a DS9 region file for all sources given a DataFrame containing their coordinates or a list of the source coordinates. 
    
    PARAMETERS
    ----------
    sources     [DataFrame, list]:	Sources for which to plot the regions. Can be provided as either a 
                                        DataFrame containing the the galactic or image coordinates, or a list of 
                                        coordinates in [[xcoords], [ycoords]] format.
    outfile     [str]		:	Name of the file to save the regions to.
    coordsys    [str]		:	Defines the coordinate system to use in DS9. Options are 'image' or 'fk5'.
					If no coordsys is given, will attempt to infer from other inputs.
    coordheads  [list]		:	Name of the headers containing the coordinates of each source, 
                                        read in as a list in [xcoordname, ycoordname] format. This is only needed if
					sources is a DataFrame. If coordheads is not defined, will attempt to infer
					the correct coordinate headers from the DataFrame or other inputs.
    coordnames  [list]		: 	Depricated parameter, now called coordheads (as of v1.6.0).
    idheader    [str]           :   	Name of the header containing the ID of each source. By default, checks 
                                        whether the DataFrame contains a header called 'ID'. If not, it's assumed
                                        no IDs are given for each source. 
    idname 	[str] 		: 	Depricated parameter, now called idheader (as of v1.6.0)
    color 	[str] 		: 	Color of the regions to plot. Default is '#FFC107' (yellow-orange).
    radius 	[int] 		: 	Radius or size of the region to plot, given as a list of unique values or a 
					single value to use on all sources. If no radius is given, all radii are set to 
					10 pixels or 1 arcsecond, depending on radunits or the coordinate system.
    radunit 	[str] 		: 	Unit of the region radius. If no unit is given, algorithm makes a guess at the
					best unit to use based on the coordinate system. 
    label 	[list] 		: 	Labels to apply to each source. This is overwritten if idheader is given.
    width 	[int] 		: 	Width of the region outline if circular regions are used.
    fontsize 	[int]		:	Size of the label text. 
    bold 	[bool] 		: 	If True, sets text to boldface. Default is False.
    dash 	[bool] 		:	If True, sets region circles to be outlined with dashed lines. Default is False.
    addshift 	[list] 		: 	Adds a shift to the source coordinates. Shifts must be given in the same units as the coordinates!
    savecoords 	[str] 		: 	Saves the source coordinates to a text file with the input filename. 
    marker 	[str] 		: 	Defines a marker to use instead of circular units. For DS9, options include circle, box, diamond, 
					cross, x, arrow, or boxcircle.
    fill 	[bool] 		: 	If True, fills the region with the color determined by the 'color' parameter. 
					Only circular regions can be filled. 
    """

    # Coordnames has been renamed coordheads to standardize parameter names across XRBID, 
    # This code is added so that old calls of WriteReg still work without error. 
    if coordnames: coordheads=coordnames

    # xcoord and ycoord keep track of the coordinate headers if source is a DataFrame
    xcoord = False
    ycoord = False
    
    # Pulling the name of the coordinate headers for source DataFrame, if given
    if coordheads: 
        xcoord = coordheads[0]
        ycoord = coordheads[1]

    # Pulling source coordinates from sources, depending on the type of input
    if isinstance(sources, list): # if sources is a list, assume they are a list of coordinates
        if len(np.asarray(sources)) == 2: # if in the format [x_coords, y_coords]....
            x_coords = sources[0]
            y_coords = sources[1]
        else: # if not, assume given as [[x1,y1], [x2,y2]...]
            x_coords = np.array(sources).T[0]
            y_coords = np.array(sources).T[1]
    elif isinstance(sources, str) and len(re.split(r"\.", sources)) > 1: # if sources is a filename, use GetCoords to retrieve coordinates
        x_coords, y_coords = GetCoords(infile=sources)
    elif isinstance(sources, pd.DataFrame): # If df read in, try to decide on coordinate system, if not given
        if not coordheads:
            # The header name can be inferred from the coordinate system
            # Assumes image coordinates have headers [X,Y] or [x,y] and 
            # fk5 coordinates have headers [RA,Dec] or [ra,dec]
            if not coordsys or coordsys.lower() == "fk5": 
                if "RA" in sources.columns.tolist():
                    xcoord = "RA"
                elif "ra" in sources.columns.tolist(): 
                    xcoord = "ra" 
                else: pass;
                if "Dec" in sources.columns.tolist(): 
                    ycoord = "Dec" 
                elif "dec" in sources.columns.tolist(): 
                    ycoord = "dec"
                else: pass;
            if not coordsys or "im" in coordsys.lower(): 
                if "X" in sources.columns.tolist(): 
                    xcoord = "X" 
                elif "x" in sources.columns.tolist(): 
                    xcoord = "x" 
                else: pass; 
                if "Y" in sources.columns.tolist(): 
                    ycoord = "Y" 
                elif "y" in sources.columns.tolist():
                    ycoord = "y" 
                else: pass;                
        # Whether or not coordheads is given, we should have a valid xcoord and ycoord by now
        # If not, this code will pass an error, and the user will be asked to provide the header names. 
        try: 
            x_coords = sources[xcoord].values.tolist()
            y_coords = sources[ycoord].values.tolist()
        except: 
            coordheads = input("Coordinate headers not found. Please enter headers (separated by comma, no space):")
            xcoord,ycoord = coordheads.split(",")
            # If this still fails, then user has erred and program will end in error. 
            x_coords = sources[xcoord].values.tolist()
            y_coords = sources[ycoord].values.tolist()

    # At this point, WriteReg should now have the x and y coordinates of each region to plot. 
    # We also need to make sure coordsys, radius, and radunits are known, if not given. 

    # Sorting out coordsys options
    if coordsys: 
        coordsys = coordsys.lower()
        if "im" in coordsys: coordsys = "image" # allows user to input img instead of image and still get same results
    if not coordsys: 
        # If any values are beyond the acceptable range of fk5, this must be in image coordinates
        if max(x_coords) > 360 or max(np.abs(y_coords)) > 90 or xcoord in ["X","x"]: coordsys = "image"
        else: coordsys="fk5"

    # Setting up the units to add to the radii
    if (radunit and 'arcs' in radunit) or (not radunit and coordsys=="fk5"): radmark='\"' # add arcsec
    elif (radunit and 'arcm' in radunit): radmark='\'' # add arcmin
    elif (radunit and 'deg' in radunit): radmark='d'   # add degree marker
    else: radmark=''  # if pixels or marker is given, no unit marker needed
        
    # Setting default radius, if not given
    if isinstance(radius, list): radius = [str(r)+radmark for r in radius] # converting radii to strings with unit markers added
    elif not radius: # defaults to 3 pixels or 0.5 arcsec, depending on coordinate system and marker type
        if coordsys == "image" or marker: radius = ['10']*len(x_coords) # pixels
        else: radius = ['1'+radmark]*len(x_coords) # will add unit arcsecond soon
    elif not isinstance(radius, list): radius = [str(radius)+radmark]*len(x_coords) # making radius a list of strings, if single value given 

    # If only one width is given, use it as the default in the header
    # otherwise, unique widths will be added to each region, and the header width will be set to the first in list
    if not isinstance(width, list): 
        uniquewidth = False
        defaultwidth=width
    else:
        uniquewidth = True
        defaultwidth = width[0]
        
    # Now have radii as a list of strings with the unit markers included, 
    # coordinates of each region to plot, and the coordinate system 
    # Can start to put together the strings to write to file.
    
    #### PARAMETERS FOR PLOTTING #####        
    # Setting text to bold or normal based on user input
    if bold: bold = " bold"
    else: bold = " normal" 
    # Setting up the header of the region file based on parameters read in
    f_head = "# Region file format: DS9 version 4.1\nglobal color=" +str(color)+" dashlist=8 3 width="+str(defaultwidth)+\
            " font=\"helvetica "+str(fontsize)+bold+" roman\" select=1 highlite=1 dash="+str(int(dash))+\
            " fill="+str(int(fill))+" fixed=0 edit=1 move=0 delete=1 include=1 source=1\n"+coordsys+"\n" 
    # NOTE, each source can theoretically have a different width, but if not, we'll want to set the width in the header of the .reg

    # SETTING UP REGION OF EACH SOURCE
    # reg determines the type of region (circle or point), preceeding the coordinates of each point
    # reg_props determines the properties (label, color, etc) of each point following the coordindates of each point
    if marker: # If it is a marker, radius is the markersize and added to reg_props after the parenthesis        
        reg = "point("  
        reg_props = [") # point="+marker+" "+r for r in radius] 
    else: # If the region is a circle, the radius is added within the parenthesis
        reg = "circle("
        reg_props = [", "+r+") # " for r in radius]

    # If list of labels or idheader given, add them to the end of reg_props
    if idheader: label = sources[idheader].values.tolist()
    if label: reg_props = [r+" text={"+str(label[i])+"}" for i,r in enumerate(reg_props)]
    # If each source has a unique width, add them to reg_props
    if uniquewidth: reg_props = [r+" width="+str(width[i]) for i,r in enumerate(reg_props)]

    # Finally, adding endline character to end of reg_props to place each region on a new line in the .reg file
    reg_props = [r+"\n" for r in reg_props] 
    
    # Putting together the full region line using the coordinates of each source
    f_reg = [reg+str(x_coords[i]+addshift[0])+", "+str(y_coords[i]+addshift[1])+r for i,r in enumerate(reg_props)]

    #### WRITING THE REGION FILE ####
    print("Saving", outfile)
    with open(outfile, 'w') as f:
        f.write(f_head)
        for r in f_reg: # for each source in the list, print the 
            f.write(r)
    print(outfile,"saved!")

    if savecoords: 
        with open(savecoords, "w") as f: 
            np.savetxt(f, np.column_stack([x_coords, y_coords]))
        print(savecoords, "saved!")

###---------------------------------------------

def CombineReg(regions, outfile): 
	"""
	Combines multiple region files, assuming all files are written in the same coordinate system. 
	NEXT UPDATE: add parameters from the header of each region file to the reg_param after each line. 

	PARAMETERS
	-----------
	regions [list] : List of region file names to combine. 
	Only region files with the same coordinate system as the first file will be added to the file. 
	outfile [str] : Name of the combined region file. 
	"""

	# First, read in the region file and find the coordinate system. 
	# If the coordinate system of the region file matches the first file, add to the combined file.  
	# Otherwise, reject the region file and do not add to the combined region file. 

	# Reading in the first region file and pulling the coordinate system and header information
	# The new region file will use the header information of the first region file
	basereg = regions[0]
    
	f = open(basereg, "r")
	basetext = f.read()
	basetext = basetext.split("\n") # splitting the text into lines
	basecoord = basetext[2] # base coordinate system. 
	f.close()

	newtext = basetext
    
	for reg in regions[1::]: 
		f = open(reg, "r")
		text = f.read()		# Text from the other region files (besides the base file)
		text = text.split("\n") 

		# If the coordinate system is the same, add it to the text. Otherwise, skip. 
		if text[2] == basecoord: 
			# If this region file is good to add to the new combined region file, 
			# Take the properties of the regions in the file and add them to the end
			# of each region in the file, to ensure these properties are preserved
			font = text[1].split("font=")[1].split("\"")[1]
			dash = text[1].split("dash=")[1][0]
			width = text[1].split("width=")[1][0]
			color = text[1].split("color=")[1].split(" ")[0]
			for i,line in enumerate(text[3:-1]):
				text[i+3] = line + " font=\""+font+"\" dash="+str(dash)+" width="+str(width)+" color="+color
			# Adding the text from current region file to the full text saved to newtext 
			newtext = newtext + text[3:-1]
		else: print(reg, "is in wrong coordinate system. Regions will not be added to",outfile)
		f.close()

    
	#### WRITING THE REGION FILE ####
	print("Saving", outfile)
	with open(outfile, 'w') as f:
		for line in newtext: # for each source in the list, print the 
			f.write(line+"\n")
	print(outfile,"saved!")

###---------------------------------------------

def WriteScalings(sources=None, outfile="scalings.txt", scalings=None, default_scalings=["zscale","zscale","zscale"], regions=None, savescalings="autoscalings.txt", coordheads=False, coords_header=['X','Y'], idheader="ID"): 
	"""Script for automatically writing a default unique scaling for each source in the input DataFrame based on the image coordinates 
	and input square regions. It is assumed that the regions get brighter closer to the center of the image and that regions are read in 
	going from the outer regions moving inward. The regions are read in as rectangles in the form [[xmin, xmax], [ymin, ymax]] in image 
	coordinates. Regions will be given as [red, green, blue]. If the source is found outside of all of the regions given, assumed to be 
	in zscale. Input scalings and regions will automatically be saved in a file with the default name 'autoscalings.txt'. This file can 
	be read in place of scalings and regions in situations where the same regions can be for the galaxy in all cases. This same file can
	then be called in as the scalings parameter, in place of a list of scalings. 

	The best steps are the following: 

	(1) Manually create an autoscaling file with a list of [redscale, greenscale, bluescale, xmin, xmax, ymin, ymax]; 
	(2) Run WriteScalings using the new autoscaling file name name as the 'scalings' argument to create a file with a 
	    list of unique location-based scalings for each source; 
	(3) Run WriteDS9 using unique_scale=True and the new unique scaling file as the 'scales' argument. Use the resulting 
	    .sh file to get scaled images of each source. 

	PARAMETERS:
	----------
	sources		[DataFrame]	:	DataFrame containing the sources to image. 
	outfile		[str]		:	Name of the file to save the source scalings to. Defaults to "scalings.txt".
	scalings	[list, str]	:	Name of the file containing the manually-determined scale parameters of nested 
						rectangular regions, or a list containing these scalings in the format
						[[red, green, blue], ...]. If the scalings are read in 
						as a list, will save them to a file named by the parameter 'savescalings'. 
	default_scalings [list]		:	Default scalings for any source falling outside of the nested rectangular regions
						in scalings, in RGB order. Defaults to ['zscale','zscale','zscale'].
	regions		[list]		:	If scalings is given as a list of scale parameters rather than a file containing the scalings, 
						then the nested regions can be defined here in the format [[[xmin, xmax], [ymin, ymax]], ...]
	savescalings	[str]		:	If scalings is a list of scale parameters instead of a file, then save the scalings to a file
						using this filename. 
	coordheads	[list]		:	Headers under which the coordinates are stored in the sources DataFrame.
	coords_header	[list]		:	Depricated parameter name, now called coordheads (as of v1.6.0).
	idheader	[str]		:	Header under which source IDs are stored. These will be used to save the appropriate scale 
						parameter of each source, which will then be called by WriteDS9. 
					
	"""

	if coordheads: coords_header = coordheads

	sourcex = sources[coords_header[0]].values.tolist()
	sourcey = sources[coords_header[1]].values.tolist()
	
	if ".txt" in scalings: 
		temp = np.genfromtxt(scalings, dtype=str)
		scalings = []
		regions = []
		for i in temp: 
			scalings.append([i[0], i[1], i[2]])
			regions.append([[i[3], i[4]],[i[5], i[6]]])	
	else: 
		print ("Auto-saving scalings as " + savescalings)
		with open(savescalings, "w") as f: 
			f.write("# Scalings and region coordinates readable by WriteScalings\n")
			for i in range(len(scalings)): 
				temp = ""
				for j in range(3): temp = temp + str(scalings[i][j]) + " "
				for j in range(2): temp = temp + str(regions[i][j][0]) + " " + str(regions[i][j][1]) + " "
				f.write(temp + "\n")
		 
	with open(outfile, "w") as f: 
		f.write("### Last update: " + str(datetime.date.today()))
		f.write("\n### Keeps track of the best unique color scaling for each source")
		f.write("\n# [0] ID, [1] Red scale, [2] Green scale, [3] Blue scale\n")
		# Looking at each source independently
		for s in range(len(sources)):
			# Default scale is zscale. If the source falls within a given region, change accordingly 
			red = default_scalings[0]
			green = default_scalings[1]
			blue = default_scalings[2]
			# Each region (reg) will be a list filled with two lists, [xmin, xmax] and [ymin, ymax]
			for i in range(len(regions)):
				reg = regions[i]
				scale = scalings[i]
				xmin = reg[0][0]
				xmax = reg[0][1]
				ymin = reg[1][0]
				ymax = reg[1][1]
				if float(xmin) < sourcex[s] < float(xmax) and float(ymin) < sourcey[s] < float(ymax):
					red = str(scale[0])
					green = str(scale[1])
					blue = str(scale[2])
				# Each of these scalings will be overwritten if the source appears in a later (i.e. inner) region
			temp = str(sources[idheader][s]) + " "  + str(red) + " " + str(green) + " " + str(blue)
			f.write(temp)
			f.write("\n")
	f.close()
	print (outfile + " saved\nDONE.")
				


###-----------------------------------------------------------------------------------------------------

def WriteFig(images, outfile=None, dimensions=(8,5), dir=""): #, imgnames=None, outfile=None): 

	"""For writing LaTex code for figure containing thumbnails. Dimensions are given as rows x columns

	images		[list]	:	List of image names to add to the figure. 
	outfile		[str]	:	Name of the file to save the LaTex code to. 
	dimensions	[tuple]	:	Size of the figure, given as (rows, columns).
	dir		[str]	:	Name of the directory containing the images (including the trailing backslash)
	"""

	images = images.copy()

	imgpertable = dimensions[0]*dimensions[1]  	# Number of images per table
	numtables = int(math.ceil(float(len(images))/float(imgpertable))) 	# Number of tables needed
	tables = [""]*numtables 	# a list for holding all of the tables generated
	img = 0 	# keeping track of which image is being added to the table

	try: 
		for i in images: images[i] = dir+images[i]
	except: pass;

	print("Writing " + outfile + "...\n")

	for i in range(numtables):	# This needs to be repeated for every table
		tables[i] = tables[i] + "\\begin{table}[ht]\n\t\\centering\n\t\\begin{tabular}{c"
		for j in range(dimensions[1]): tables[i] = tables[i] + "c"
		tables[i] = tables[i] + "}\t\t"
		try: 
			for j in range(dimensions[0]): 
				tables[i] = tables[i] + "\n\t\t"
				for k in range(dimensions[1]): 
					tables[i] = tables[i] + " & \\includegraphics[width=" + str(0.75/dimensions[1])+"\\textwidth]{"+images[img]+"}"
					img = img + 1
				tables[i] = tables[i] + " \\\\"
		except: tables[i] = tables[i] + "\\\\"
		tables[i] = tables[i] + "\n\t\\end{tabular}\n\t\\caption{}\n\t\\label{}\n\\end{table}\n"
		
	with open(outfile, 'w') as f: 
		for table in tables: 
			f.write(table + "\n")

	f.close()
	print("Done")

###-----------------------------------------------------------------------------------------------------

def WriteTable(df, outfile=None, headers=None, dimensions=None): 

	"""
	For writing LaTex code for tables containing data properties, etc.

	df		[DataFrame]	:	DataFrame containing the sources in table. 
	outfile		[str]		:	Name of file to save LaTex code to. 
	headers		[list]		:	List of headers to add to the table. 
	dimensions	[tuple]		:	Number of rows/columns to include in a single table. This allows the table to be broken up into smaller 
						tables that continue across multiple pages. 
	"""

	frame = df.copy()

	# If no dimensions are given, use full df.
	if dimensions == None: dimensions = [frame.shape[0], frame.shape[1]]

	# If no headers are given, use all headers in the DataFrame
	if headers == None: headers = list(frame.columns.values) 
	else: dimensions[1] = len(headers)	# else, use only the given headers.

	numtables = int(math.ceil(float(len(frame))/float(dimensions[0])))
	tables = [""]*numtables
	source = 0
	# If headers are specified, only include those headers. 
	
	print("Writing " + outfile + "...\n")

	for i in range(numtables):	# This needs to be repeated for every table
		#tables[i] = tables[i] + "\\pagebreak\n\\begin{sidewaysfigure}\n\t\\begin{tabular}{c|" 
		tables[i] = tables[i] + "\n\\begin{center}\n\\afterpage{\n\t\\begin{longtable}[ht!]{c|"
		for j in range(dimensions[1]-1): tables[i] = tables[i] + "c"  # adding centered columns 
		tables[i] = tables[i] + "}\n\t\\hline\n\t"
		for head in headers: tables[i] = tables[i] + str(head) + " & "
		tables[i] = tables[i] + "\\\\ "
		try: 
			for j in range(dimensions[0]): 
				tables[i] = tables[i] + "\n\t"
				for head in headers: 
					if head != headers[-1]:
						try: tables[i] = tables[i] + str(round(frame[head][source], 3)) + " & "
						except: tables[i] = tables[i] + str(frame[head][source]) + " & "
					else: 
						try: tables[i] = tables[i] + str(round(frame[head][source], 3))
						except: tables[i] = tables[i] + str(frame[head][source])
				source = source + 1
				tables[i] = tables[i] + " \\\\ "
		except: tables[i] = tables[i] + "\\\\ "
		#tables[i] = tables[i] + "\n\t\\end{tabular}\n\\caption{}\n\\label{}\n\end{sidewaysfigure}\n"
		tables[i] = tables[i] + "\n\t\\end{longtable}\n\\clearpage\n}\n\\end{center}"
		
	with open(outfile, 'w') as f: 
		for table in tables: 
			f.write(table + "\n")

	f.close()
	print("Done")	



