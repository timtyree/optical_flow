import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pims, tifffile
from skimage.filters import scharr_h, scharr_v, gaussian
import random as rand
from .optical_flow import OpticalFlowClient
from .mark_images import *
from cv2 import drawMarker
#serial parallelization tools
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

#import tkinter for simple gui
from tkinter import filedialog, Tk, Canvas, PhotoImage 
from PIL import ImageTk,Image 

#automate the boring stuff
import time, os, sys, re
beep = lambda x: os.system("echo -n '\a';sleep 0.2;" * x)
if not 'here_dir' in globals():
	here_dir = os.getcwd()
# print('here we at ' + here_dir)


###############
# Get the data
###############
def search_for_frame_path (currdir = os.getcwd()):
	'''setup user interface for file selection'''
	root = Tk()
	tempdir = filedialog.askopenfilename(parent=root, 
										 initialdir=currdir, 
		  
										 title="Please select a file.")
	root.destroy()
	if len(tempdir) > 0:
		print ("Frames: %s" % tempdir)
	return tempdir

def get_timeseries_filename(r1,r2):
	return f'results_r1_{int(r1)}_r2_{int(r2)}_navg_1_08042020 c2P1.csv'
# data_dir = "/Users/timothytyree/Desktop/Research/Rappel/Dicty. Dispersal/Richa's High Time Resolution Experiment/Data/"
# os.chdir(data_dir)
# #channel A is the center of the cluster
# df_A = pd.read_csv(get_timeseries_filename(0,10))
# #channel B is the perifery of the cluster
# df_B = pd.read_csv(get_timeseries_filename(10,40))
# # channel C is the disconnected neighboring cells
# df_C = pd.read_csv(get_timeseries_filename(75,100))
# # channel D is the disconnected neighboring cells further still
# df_D = pd.read_csv(get_timeseries_filename(100,200))
# t_values = df_A.t.values.copy()
# plt.plot(df_A.t,df_A.fret, label='channel A')
# plt.plot(df_B.t,df_B.fret, label='channel B')
# plt.plot(df_C.t,df_C.fret, label='channel C')
# plt.plot(df_D.t,df_D.fret, label='channel D')
# plt.xlim([60,80])
# plt.legend()

from PIL import Image 

def shift_right_by(arr,x_shiftby):
	'''#shift array to the right'''
	H,L = arr.shape
	mn = np.min(arr)
	out = 0*arr #+ mn
	# for j in range(H-1):
	for i in range(L-1):
		k = i-x_shiftby
		out[:,k%L]  = np.array(arr)[:,(k-int(x_shiftby))%L]
	# out[j%H,k%L]  = np.array(arr)[j%H,(k-int(x_shiftby))%L]
	return out
# def shift_right_by(arr,x_shiftby):
#     H,L = arr.shape
#     mn = np.min(arr)
#     out = 0*arr#*0+mn#np.zeros_like(arr).copy()+mn
#     for j in range(H-1):
#         for i in range(L-1):
#             k = i-x_shiftby
#             if k>-1:
#                 out[j%H,k%L]  = np.array(arr)[j,k]
# #                 arr[j,k]
#             else:
#                 out[j,i] = mn
#     return out

def mytransform_array(input_array, x_shiftby, rotateby):
	'''transform the array by shifting and rotating
	Example Usage:
	arry = mytransform_array(input_array, x_shiftby=3, rotateby=-15)
	'''
	return rotateby_then_shiftby(input_array, x_shiftby, rotateby)

def shiftby_then_rotateby(input_array, x_shiftby, rotateby):
	'''transform the array by shifting and rotating
	Example Usage:
	arry = mytransform_array(input_array, x_shiftby=3, rotateby=-15)
	'''
	arr = input_array.copy()
	out = shift_right_by(arr,x_shiftby)
	img = Image.fromarray(out)
	img = img.rotate(rotateby)

	# set new pixels to minimum value
	arry = np.array(img)
	mn = np.min(input_array) #minimum value before rotating
	boo = np.isclose(arry,0)
	arry[boo] = mn
	return arry

def rotateby_then_shiftby(input_array, x_shiftby, rotateby):
	'''transform the array by shifting and rotating
	Example Usage:
	arry = mytransform_array(input_array, x_shiftby=3, rotateby=-15)
	'''
	arr = input_array.copy()
	img = Image.fromarray(arr)
	img = img.rotate(rotateby)

	# set new pixels to minimum value
	arry = np.array(img)
	mn = np.min(input_array) #minimum value before rotating
	boo = np.isclose(arry,0)
	arry[boo] = mn
	out = shift_right_by(arry,x_shiftby)
	return out

def get_green_channel_from(input_array, baseline = 1000, x_shiftby = 2, rotateby = 15):
	'''shift and rotate the fluorescence data
	Example Usage:
	green_channel = get_green_channel_from(input_array, baseline = 1000)
	'''
	arry = mytransform_array(input_array, x_shiftby=x_shiftby, rotateby=-rotateby)
	img = Image.fromarray(arry)
	#norm values, choosing a baseline that is independent of frame number
	green_channel = np.log(np.array(img)/baseline) 
	return green_channel




# #####################################################################
# # Measure the fluor. intens. timeseries from scratch,
# #####################################################################
# # make a histogram of intensity values and test/justify the signal measure.
# class FluorescentClient():
# 	pass





#####################################################################
# TODO: calculate flow for all frames and save as a .tif
#####################################################################
# TODO: step 2: use ^that routine to calculate flow for all frames and save as a .tif
#put all of step 1.b together into a function that takes two frames and returns the output texture
# def calculate_output_texture(prv, nxt, raw_frame):
#     #calculate DIS optical flow
#     dis = cv.cv2.DISOpticalFlow_create(cv.cv2.DISOpticalFlow_PRESET_MEDIUM)
#     img_flow = dis.calc(prv,nxt, flow=None)
#     flow_x = img_flow[...,0]
#     flow_right = flow_x.copy()
#     flow_right[flow_right<=0] = 0
#     flow_left = -flow_x.copy()
#     flow_left[flow_left<=0] = 0
	
#     #calculate cell area
#     cell_area = gaussian(nxt+prv, sigma=3)
#     threshold = 0.1
#     cell_area[cell_area>threshold]  = 1
#     cell_area[cell_area<=threshold] = 0
	
#     #select flow only where there are cells
#     flow_left_cells  = (cell_area*flow_left)
#     flow_right_cells = (cell_area*flow_right)

#     #compress data type (optional, but reccomended)
#     flow_left_cells   = flow_left_cells.astype('float16')
#     flow_right_cells  = flow_right_cells.astype('float16')
#     cell_area_cells   = cell_area.astype('float16')
#     raw_frame         = raw_frame.astype('float16')
#     output_texture    = np.stack([flow_left_cells, flow_right_cells, cell_area_cells, raw_frame], axis=-1)
#     return output_texture

# #(optional) reload raw DIC frames
# @pims.pipeline
# def gray(image):
#     return np.uint8(image[:, :, 0])  # Take just the red channel
# frames = pims.TiffStack_libtiff(data_file_name)



##########################
# (start of) Example Usage
##########################
# fret_dir = "/Users/timothytyree/Desktop/Research/Rappel/Dicty. Dispersal/Richa's High Time Resolution Experiment/Data/08042020 c2P1_20X_C1.tif"
# cluster_dir = "/Users/timothytyree/Desktop/Research/Rappel/Dicty. Dispersal/Richa's High Time Resolution Experiment/Data/08042020 c2P1/cluster_trajectory_08042020 c2P1.csv"
# frames= pims.TiffStack_libtiff(fret_dir)
# df = pd.read_csv(cluster_dir)

# dt    = float(1/6)    #time between two frames (minutes)
# lamda = float(2/3) #microns per pixel
# #avoid using clients, but they can hold environment variables without dask workers stepping on eachother.
# # fc  = FluorescentClient(**kwargs)

# of  = OpticalFlowClient(dt=dt)
# # of.lamda = lamda
# #is ^this needed?

# #matrix of distances from the cluster
# xpos,ypos = df.head(1).values.T[2:]
# position=(xpos,ypos)
# r_hat_mat, r_c_mat = of.get_r_hat_mat(position)
# print (position)

##########################
# Deprecated)
##########################
# def shift_right_by(arr,x_shiftby):
#     H,L = arr.shape
#     mn = np.min(arr)
#     out = 0*arr#*0+mn#np.zeros_like(arr).copy()+mn
#     for j in range(H-1):
#         for i in range(L-1):
#             k = i-x_shiftby
#             if k>-1:
#                 out[j%H,k%L]  = np.array(arr)[j,k]
# #                 arr[j,k]
#             else:
#                 out[j,i] = mn
#     return out

# def shift(xs, n):
#     if n >= 0:
#         return np.r_[np.full(n, np.nan), xs[:-n]]
#     else:
#         return np.r_[xs[-n:], np.full(-n, np.nan)]



