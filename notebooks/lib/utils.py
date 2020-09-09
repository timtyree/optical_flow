import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pims, tifffile
from skimage.filters import scharr_h, scharr_v, gaussian
import random as rand
from .optical_flow import OpticalFlowClient
from cv2 import drawMarker

from .utils_fluorescence import *
from .utils_outward_flow import *
from .lograt import *
#serial parallelization tools
from joblib import Parallel, delayed
import multiprocessing
# TODO: put this in any relevant functions --> num_cores = multiprocessing.cpu_count()

#import tkinter for simple gui
from tkinter import filedialog, Tk, Canvas, PhotoImage 
from PIL import ImageTk,Image 

#automate the boring stuff
import time, os, sys, re
beep = lambda x: os.system("echo -n '\a';sleep 0.2;" * x)
if not 'here_dir' in globals():
	here_dir = os.getcwd()
# print('notebook is at: ' + here_dir)

def search_for_file (currdir = os.getcwd()):
	'''#make functions for save file name, input cell frames, and input cell trajectories'''
	#TODO: eventually make this ^take cell trajectories or cell positions
	root = Tk()
	tempdir = filedialog.askopenfilename(parent=root, 
										 initialdir=currdir, 
										 title="Please select a file")#, 
										 # filetypes = (("all files","*.*")))
	root.destroy()
	if len(tempdir) > 0:
		print ("File: %s" % tempdir)
	return tempdir

def find_files(filename, search_path):
	'''recursively search everywhere inside of search_path and return all files matching filename.'''
	result = []
	for root, dir, files in os.walk(search_path):
		if filename in files:
			result.append(os.path.join(root, filename))
	return result

def find_file(**kwargs):
	'''recursively search everywhere inside of search_path for filename.  Returns the first found.  This could be optimized with a greedy algorithm.'''
	return find_files(**kwargs)[0]

#setup user interface for file selection
def search_for_frame_path (currdir = os.getcwd()):
	root = Tk()
	tempdir = filedialog.askopenfilename(parent=root, 
										 initialdir=currdir, 
		  
										 title="Please select a file.")
	root.destroy()
	if len(tempdir) > 0:
		print ("Frames: %s" % tempdir)
	return tempdir

def tiffstack_to_avi(input_file_name, save_dir= None, fps=8):
	'''saves tiffstack found in the local 'path' to save_dir to a similarly named .avi file if save_dir=None.
	fps = frames per second.'''
	path = input_file_name
	if save_dir == None:
		save_dir= path[:path.find(r'.tif')]+'.avi'
	start = time.time()
	boo, cap = cv.imreadmulti(path)
	width = int(cap[0].shape[0])
	height = int(cap[0].shape[1])
	try:
		chnl_no = int(cap[0].shape[2])
	except:
		chnl_no = 0#or 1, not sure
		print('Error: depth shape not given, returning 0')
	# uncompressed YUV 4:2:0 chroma subsampled
	fourcc = cv.VideoWriter_fourcc('I','4','2','0')
	writer = cv.VideoWriter()
	retval = writer.open(save_dir, fourcc, fps, (width, height), 1)
	assert(writer.isOpened())#assert the writer is properly initialized
	for i  in range(len(cap)):
		#TODO: make this step faster by using something like the (missing) cv.GrabFrame command 
		frame = cap[i]
		writer.write(frame)
	writer.release()
	end = time.time()
	print('{} seconds elapsed reading tiffstack and writing video to avi.'.format(np.around(end-start)))
	return True

#########
# Example Usage
#########
# cluster_dir = search_for_file('cluster_122019_pos1_middle_cubic_spline.csv')
# print(cluster_dir)


def load_input_grayscale_data_second_order(file_name, **kwargs):
	'''file_name is the .png file name of a single frame of preprocessed intensities from DIC microscopy.

	load input file_name and its previous file_name_previous into our pythonic vm-memory
	suppose file_name is an absolute path file legible to plt.imread(file_name)
	
	Example Usage: 
	current, previous, previous_previous = load_input_grayscale_data_second_order(file_name)
	'''
	lst = file_name.split('.')
	assert(len(lst)>2)
	frm = int(eval(lst[-2]))
	file_name_previous = '.'.join(lst[:-2])
	file_name_previous = '.'.join([file_name_previous,f'{frm-1:d}',lst[-1]])
	file_name_previous_previous = '.'.join(['.'.join(lst[:-2]),f'{frm-2:d}',lst[-1]])
	
	# #current time 
	# frm0 = 0; dt = 1.
	# tme = ( frm - frm0 ) * dt 

	def import_preprocessed_frame(input_file_name, **kwargs):
	    img = plt.imread(input_file_name)
	    norm_image = cv2.normalize(img, None, alpha=0., beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	    img_gray = norm_image[...,0]
	    return img_gray

	#load frames
	current = import_preprocessed_frame(input_file_name=file_name, **kwargs)       
	try: 
		previous = import_preprocessed_frame(input_file_name=file_name_previous, **kwargs)       
	except Exception as e:
		print (e)
		previous = None
	try: 
		previous_previous = import_preprocessed_frame(input_file_name=file_name_previous_previous, **kwargs)       
	except Exception as e:
		print (e)
		previous_previous = None
	return current, previous, previous_previous




def load_input_grayscale(file_name):
	'''
	load input file_name and its previous file_name_previous into our pythonic vm-memory
	suppose file_name is an absolute path file legible to plt.imread(file_name)
	
	Example Usage: 
	previous, current = load_input_grayscale(file_name)
	'''
	lst = file_name.split('.')
	assert(len(lst)>2)
	frm = int(eval(lst[-2]))

	# #current time 
	# frm0 = 0; dt = 1.
	# tme = ( frm - frm0 ) * dt 

	#load frame
	file_name_previous = '.'.join(lst[:-2])
	file_name_previous = '.'.join([file_name_previous,f'{frm-1:d}',lst[-1]])
	previous_previous = plt.imread(file_name_previous)#, cmap='gray')
	try: 
		previous = plt.imread(file_name_previous)#, cmap='gray')
		current = plt.imread(file_name)#, cmap='gray')
	except Exception as e:
		print (e)
		previous = None
		current = None
	return current, previous, previous_previous
