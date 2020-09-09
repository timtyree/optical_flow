
# lograt.py
# Tim Tyree
# 8.31.2020
import numpy as np
#TODO: convert .tiff to .mov with lossless compression using ffmpeg?
#TODO: filter a grayscale .mov to/from an ndarray
#TODO: filter a grayscale ndarray to lograt value ndarray

def map_lograt(ndarr, x0=None):
	'''ndarr is a nonnegative scalar field in d-space dimensional space (and/or 1 dimensional time).  
	Scalar fields are 1-channels.
	x0 will be mapped to zero.'''
	if x0 is None:
		x0 = ndarr.flatten().quantile(0.1) #what I take to be the implicit default background value
	return np.log(ndarr/x0)