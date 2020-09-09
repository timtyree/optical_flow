# utils_outward_flow.py
import cv2 as cv
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pims, tifffile
from skimage.filters import scharr_h, scharr_v, gaussian
import random as rand
from .optical_flow import OpticalFlowClient
from .mark_images import *
from .utils import *
from matplotlib import patches
# from cv2 import drawMarker
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

#TODO: skimage.morphology.remove_small_objects(ar=retval[3],min_size=64)
#     if use_cell_area_filter:
def erosion_dilate_erode(img, rad = 1):
	kernel   = np.ones((rad,rad),np.uint8)
	erosion  = cv2.erode (img,kernel,iterations = 1)
	dilation = cv2.dilate(erosion,kernel,iterations = 2)
	erosion  = cv2.erode (dilation,kernel,iterations = 1)
	return erosion

def mydot(flow, rhat, use_angthresh = False, angthresh = 0):
	'''flow_in, flow_out = dot(self, flow, rhat)
	#define flow as out/in if it is within 45degrees of directly out/in
	# <--> v_r/vtot = CI < np.sqrt(0.5)
	'''
	r_hat_mat = rhat
	#dot dense optical flow direction with the outward r_hat
	#TODO: add a cross product option for curl flow about the center marked with a "yellow x"
	flow_out_x  = np.multiply(flow[...,0],r_hat_mat[...,0])
	flow_out_y  = -np.multiply(flow[...,1],r_hat_mat[...,1])
	#TODO:try undoing both yaxis flips. consider saving a previous version first.
	flow_out = flow_out_x + flow_out_y


	# flow_out[flow_out<1] = 0

	mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
	boo = (mag>0)
	# CI = 0.*flow_out
	CI = np.divide(flow_out_x+flow_out_y, mag, where=boo)
	#     angthresh = self.angthresh
	#remove small flows under threshold
	# calculate chemotactic index texture CI where defined
	# find boolean index of inward flows (dot product is negative) and set inward flow to zero
	flow_in = flow_out.copy()
	if use_angthresh:
		flow_in[CI>-angthresh] = 0.
	flow_in       = -flow_in
	if use_angthresh:
		flow_out[CI<angthresh] = 0.
	return flow_in, flow_out

def calc_flow(previous, current, current_flow=None, minthresh=0):
	dis = cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_MEDIUM)
	flow = dis.calc(previous,current, current_flow)
	mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])

	#remove small flows under threshold
	flow[...,0][mag<minthresh] = 0
	flow[...,1][mag<minthresh] = 0
	return flow

def calc_sharp_optical_flow(img,img_before, position, dt, frm = -1, 
	area_channel = None, img_before_before = None, of=None):
	'''calculate radial inward/outward flow.
	Example Usage: 
	flow_radial = calc_sharp_optical_flow(img,img_before, position, dt, frm = -1, area_channel = None, img_before_before = None, of=None):

	'''
	if of is None:
		of = OpticalFlowClient(dt=dt)
	#NB: requires edges and df as arguments
	#TODO: make sure that flow_list is appended to only here in the module
	#     start = time.time()
	# get radial coordinates
	#TODO: fix get_r_hat_mat so there's no arbitrary max radius of support :(
	rhat,rmat = of.get_r_hat_mat(position)

	# if img_before_before is given, compute the flow from the next previous frame
	if img_before_before is not None:
		dis = cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_MEDIUM) # the most precise builtin setting as of now...
		current_flow = dis.calc(img_before_before,img_before, flow=None)
	else: 
		current_flow = None

	# get flow field in radial coordinates
	flow      = of.calc_flow(img,img_before, current_flow=current_flow)
	flow_out, flow_in = mydot(flow, rhat)

	# #blurred binary cell area mask
	# fltr     = edges[frm]/2+edges[frm+1]/2
	# fltr     = gaussian(fltr, sigma=sigma)
	# fltr[fltr<20] = 0
	# fltr[fltr>=20] = 1
	# area_channel = fltr#pims.frame.Frame(fltr.astype('uint16'))
	if area_channel is None:
		area_channel = img/np.max(img)

	#filter off-cell flow and return 
	output_texture = np.stack([(area_channel*flow_in).astype('float32'), 
							   (area_channel*flow_out).astype('float32'),
							  rmat.astype('uint32'),
							  area_channel.astype('uint16')], axis=2)
	#include original frame_no metadata
	flow_radial       = pims.frame.Frame(output_texture, frame_no = frm)
	return flow_radial

##################################################
def get_image(frames, frm):
	img = frames[frm]
	return prepare_image(img)
	
def prepare_image(img):
	norm_image = cv2.normalize(img, None, alpha=0., beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	return cv2.cvtColor(norm_image,cv2.COLOR_GRAY2BGR)


def highlight_flow_bwr(image, flow_in, vmin, vmax, figsize = (10,10), mydpi = 512/10):
	fig, ax = plt.subplots(1,1,figsize=figsize)
	ax.imshow(image)
	ax.imshow(flow_in, cmap='bwr', vmin = vmin, vmax = vmax, alpha = 0.5)
	ax.axis('off')
	return fig

def get_lograt_out(flow_out, area_channel, baseline = .1):
	'''
	it looks like vmin = -4, vmax = +4 might be reasonable 
	for lograt_out replacing flow_out.
	Example Usage:
	lograt_out = get_lograt_out(flow_out, baseline = .1)
	'''
	#ignoring flow slower than .1
	arr  = np.array(flow_out)
	boo  = area_channel==1
	boo &= arr > 0
	arr[boo] = np.log(arr[boo]/baseline)
	arr[arr<0]=0
	lograt_out = arr.copy()
	# plt.hist(np.array(arr[boo]).flatten(),bins=30)
	# plt.show()

	#for positive pixels, scale the outward flow to a logratio w.r.t. some basline
	#histogram of inward/outward intensities
	arr  = np.array(flow_out)
	boo  = area_channel==1
	boo &= arr < 0
	arr[boo] = np.log(-arr[boo]/baseline)
	arr[arr<0]=0
	lograt_in = arr.copy()
	# plt.hist(np.array(arr[boo]).flatten(),bins=30)
	# plt.show()

	lograt_out = lograt_out-lograt_in
	return lograt_out

def average_image_list(img_list):
	img = np.stack(img_list[:], axis=0)
	img = np.mean(img, axis=0)
	return img

def compute_sharp_optical_flow(current, previous, previous_previous, dt, 
	of=None, **kwargs):
	'''Example Usage
	flow = compute_sharp_optical_flow(current, previous, previous_previous, dt, of=None, **kwargs)
	'''
	# if of is None:
	# 	of = OpticalFlowClient(dt=dt)
	# 	kwargs['of'] = of
	dis = cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_MEDIUM)
	# if img_before_before is given, compute the flow from the next previous frame
	if previous_previous is not None:
		current_flow = dis.calc(previous_previous,previous, flow=None)
	else: 
		current_flow = None
	# flow=None is not entirely unreasonable, though unstable... flow = dis.calc(previous, current, flow=None)
	flow = dis.calc(previous, current, flow=current_flow)
	#I tried the average of current and previous flow.  no clear improvement.
	# flow = flow/2 + current_flow/2
	#no apparrent difference.  see if compute_sharp_optical_flow is in use 
	return flow	
# try:
# 	flow      = of.calc_flow(current,previous, current_flow=current_flow)
# except Exception as e:
# 	print (e)
# 	print(current.shape)
# 	print(previous.shape)
# 	print(current_flow.shape)

def compute_outward_from_flow(flow, position, frm, area_channel, of = None, **kwargs):
	'''
	Example Usage:
	flow_radial = compute_outward_from_flow(flow, position, frm, area_channel, of = None, **kwargs)
	'''
	#NB: requires edges and df as arguments
	#TODO: make sure that flow_list is appended to only here in the module
	#     start = time.time()
	# get radial coordinates
	#TODO: fix get_r_hat_mat so there's no arbitrary max radius of support :(
	if of is None:
		of = OpticalFlowClient(dt=kwargs['dt'])
		kwargs['of'] = of
	rhat,rmat = of.get_r_hat_mat(position)
	# get flow field in radial coordinates
	flow_out, flow_in = mydot(flow, rhat)

	# #blurred binary cell area mask
	# fltr     = edges[frm]/2+edges[frm+1]/2
	# fltr     = gaussian(fltr, sigma=sigma)
	# fltr[fltr<20] = 0
	# fltr[fltr>=20] = 1
	# area_channel = fltr#pims.frame.Frame(fltr.astype('uint16'))
	# if area_channel is None:
	# 	#normalize area_channel
	# 	#try img = current = 512x512 matrix indicating the locations of cells
	# 	imgp = img - np.min(img)
	# 	area_channel = imgp/np.max(imgp)

	# print(flow_in.shape)
	# print(area_channel.shape)
	# print(rmat.shape)
	#filter off-cell flow and return 
	output_texture = np.stack([(area_channel*flow_in).astype('float64'), 
							   (area_channel*flow_out).astype('float64'),
							  rmat.astype('float64'),
							  area_channel.astype('uint16')], axis=2)
	#include original frame_no metadata
	flow_radial       = pims.frame.Frame(output_texture, frame_no = frm)
	return flow_radial

def get_mean_ci(i,img_list,r1=100,r2=300):
	'''calculate the mean chemotactic index averaged 
	over cell area in the annulus centered at the centroid. 
	r1 and r2 is the inner/outer diameter of the annulus in microns.
	i is the index of the "before" frame in img_list.  
	DIS dense optical flow is computed from img_list.
	Example Usage:
	ci_avg = get_mean_ci(i,img_list,r1=100,r2=300)
	'''
	#calculate dense optical flow
	# prv = img_list[i][...,0].astype('uint8')
	# nxt = img_list[i+1][...,0].astype('uint8')		
	prv = np.array(img_list[i-1])[...,0].astype('uint8')
	nxt = np.array(img_list[i])[...,0].astype('uint8')
	# TypeError: Expected Ptr<cv::UMat> for argument 'I0'
	#heretim
	flow = calc_flow(previous=prv, current=next, current_flow=None, minthresh=0)
	# flow = of.calc_flow(prv,nxt)
	position = tuple(df.loc[i,['x','y']].values)
	#r_c_mat is the distance from the centroid in microns
	r_hat_mat, r_c_mat = of.get_r_hat_mat(position)
	boo = (r2>r_c_mat) & (r_c_mat>r1)
	#get radial component
	flow_out_x  = np.multiply(flow[...,0],r_hat_mat[...,0])
	flow_out_y  = -np.multiply(flow[...,1],r_hat_mat[...,1])
	flow_out = flow_out_x+flow_out_y

	#get total component
	length = np.sqrt(np.multiply(flow[...,0],flow[...,0])
			+np.multiply(flow[...,1],flow[...,1]))

	#calculate chemotactic index everywhere
	ci = np.divide(flow_out,length)
	ci[np.isnan(ci)]=0
	
	#threshold by speed
	speed_thresh = 0/lamda*dt
	boo_speed = length>=speed_thresh
	
	#import cell area phase field channel
	area = radial_results[i][...,-1]
	#multiply chemotactic index by cell area channel
	result = np.multiply(ci,area)
	#average over result[boo]
	ci_avg = np.median(result[(result!=0) & boo & boo_speed])

#     ci_avg = np.mean(result[(result!=0) & boo & boo_speed])
#     ci_avg = result[(result!=0) & boo & boo_speed].mean()
#     ci_total   = np.array(result)[boo].sum()
	#TODO: repeat with normalizing by nonzero elements of ci instead of area
#     ci_num_nonzero = (ci!=0)[boo].sum()
#     area_total = np.array(area)[boo].sum()
	#return mean ci
#     ci_mean = ci_total/area_total
#     ci_mean = ci_total/ci_num_nonzero
	return ci_avg

class Worker(object):
	"""A pythonic class that holds recycleable objects/clients/tools for the Dask worker"""
	def __init__(self, lamda, dt):
		of  = OpticalFlowClient(dt=dt)
		of.lamda = lamda
		self.of = of
		dis = cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_MEDIUM)    
		self.dis = dis
		self.asserting = True
		
	def optical_flow_client(self):
		return self.of
	def dis_client(self):
		return self.dis

def my_input_file_name_to_highlight(input_file_name, **kwargs):
	'''if a blue figure is returnd, check that vmin == - vmax'''
	#initialize for a given frame
	dt = kwargs['dt']
	worker = Worker(lamda=kwargs['lamda'], dt=dt)
	of  = worker.optical_flow_client()
	dis = worker.dis_client()
	asserting = worker.asserting

	#get frame number, frm, for a given file_name
	lst = input_file_name.split('.')
	assert(len(lst)>2)
	frm = int(eval(lst[-2]))
	#only consider frames with a history, so that we may calculate optical flow
	if frm>1:
		out = None
	else:
		return False
	# import the aggregate's trajectory
	os.chdir(kwargs['data_dir'])
	df = pd.read_csv(kwargs['cluster_dir'])	
	position = df.loc[df.frame==frm][['x','y']].values.T

	#raw dic frames
	frames = pims.TiffStack_libtiff(kwargs['dic_dir'])
	
	#     file_name = f"{tmp_folder}/preprocessed_snapshot.{int(frm)}.png"
	if asserting:
		assert ( os.path.exists(input_file_name) ) 

	#import from a folder of preprocessed frames stored as .png's
	current, previous, previous_previous = load_input_grayscale_data_second_order(input_file_name)
	#convert to grayscale uint8 (as required by dis.calc *sad face* )
	current = current[...,0].astype('uint8')
	previous = previous[...,0].astype('uint8')
	previous_previous = previous_previous[...,0].astype('uint8')
	area_channel = current.copy()

	#compute the flow
	current_flow = dis.calc(previous_previous, previous, flow = None)
	new_flow = dis.calc(previous, current, flow = current_flow)

	#  compute the flow_out and the flow_in
	flow_radial = calc_sharp_optical_flow(img=current,img_before=previous, 
										  position = position, dt=dt, frm = frm, 
										  area_channel = current, 
										  img_before_before = previous_previous, of=of)
	flow_out = flow_radial[...,1]
	flow_in = flow_radial[...,0]

	image = get_image(frames, frm)
	fig = highlight_flow_bwr(image, flow_in, kwargs['vmin'], kwargs['vmax'], figsize = (10,10), mydpi = 512/10)


	#save fig as .png in tmp2
	if input_file_name.find('tmp/preprocessed_snapshot')==-1:
		return False
	save_marked_fig_dir = input_file_name.replace('tmp/preprocessed_snapshot','tmp2/highlighted_snapshot')
	assert(save_marked_fig_dir != input_file_name)
	#if ^this assert fails, check that input_file_name is an absolute directory
	#     save_figure(fig, save_marked_fig_dir=save_marked_fig_dir)
	img_fn = save_marked_fig_dir
	fig.tight_layout()
	fig.savefig(img_fn, dpi =  512/10)
	plt.close()
	return img_fn

def load_data(file_name):
	current, previous, previous_previous = load_input_grayscale_data_second_order(file_name)
	return current, previous, previous_previous

def compute_flow(current, previous, previous_previous, **kwargs):
	flow = compute_sharp_optical_flow(current, previous, previous_previous, of=None, **kwargs)
	return flow

def compute_flow_out(flow, position, **kwargs):
	flow_radial = compute_outward_from_flow(flow, position, **kwargs)
	return flow_radial


def input_to_highlighted_png(input_file_name, **kwargs):
	'''if a blue figure is returnd, check that vmin == - vmax'''
	#get frame number, frm, for a given file_name
	# from cv2 import drawMarker
	lst = input_file_name.split('.')
	assert(len(lst)>2)
	frm = int(eval(lst[-2]))
	#only consider frames with a history, so that we may calculate optical flow
	if frm>1:
		out = None
	else:
		return False
	
	os.chdir(kwargs['data_dir'])
	df = pd.read_csv(kwargs['cluster_dir'])
	#raw dic frames
	frames = pims.TiffStack_libtiff(kwargs['dic_dir'])

	# compute outward flow
	os.chdir(kwargs['tmp_dir'])
	current, previous, previous_previous = load_data(input_file_name)
	current = current[...,1].astype('uint8')
	previous = previous[...,1].astype('uint8')
	previous_previous = previous_previous[...,1].astype('uint8')

	flow = compute_flow(current, previous, previous_previous, **kwargs)
	position = df.loc[df.frame==frm][['x','y']].values.T
	area_channel = current.copy()

	flow_radial = compute_flow_out(flow, position, area_channel = area_channel, frm=frm, **kwargs)

	image = get_image(frames, frm)
	# flow_out = flow_radial[...,1]
	flow_in = flow_radial[...,0]
	lograt_ = get_lograt_out(-1.*flow_in, area_channel, baseline = 0.01)
	fig = highlight_flow_bwr(image, lograt_, vmin = kwargs['vmin'], vmax = kwargs['vmax'], figsize = (10,10), mydpi = 512/10)

	#save fig as .png in tmp2
	if input_file_name.find('tmp/preprocessed_snapshot')==-1:
		return False
	save_marked_fig_dir = input_file_name.replace('tmp/preprocessed_snapshot','tmp2/highlighted_snapshot')
	assert(save_marked_fig_dir != input_file_name)
	#if ^this assert fails, check that input_file_name is an absolute directory
	#     save_figure(fig, save_marked_fig_dir=save_marked_fig_dir)
	img_fn = save_marked_fig_dir
	# fig.tight_layout()
	plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
	fig.savefig(img_fn, dpi = 512/10)
	plt.close()
	return img_fn

def plot_figure_and_save(img, save_marked_fig_dir, **kwargs):
	'''Example Usage: mark a figure
	retval = plot_figure_and_save(img, save_marked_fig_dir, **kwargs)
	'''
	lst = save_marked_fig_dir.split('.')
	assert(len(lst)>2)
	frm = int(eval(lst[-2]))

	tme = kwargs['dt']*(frm-kwargs['time_origin_frameno'])
	time_stamp = f"t = {+int(tme)}:00 min"
	#     time_stamp = f't = {+int(tme):3d} minutes'
	fig, ax = plt.subplots(1)
	fontsize = 19
	save=True
	
#     rect = patches.Rectangle((400,25),50/lamda,5, edgecolor='w', facecolor="w")
	rect = patches.Rectangle((420,25),50/kwargs['lamda'],5, edgecolor='w', facecolor="w")
	ax.imshow(img)
	tme = kwargs['dt']*(frm-kwargs['time_origin_frameno'])
	ax.annotate(text=time_stamp,
			   xy=(.2,.9),xytext=(.2,.9),
				textcoords='figure fraction',
			   fontsize=fontsize,
			   color='w',
			   fontweight='bold')
#     ax.annotate(text=f'red = in, blue = out',
#                xy=(.2,.04),
#                 textcoords='figure fraction',
#                 xytext=(.2,.04),
#                fontsize=fontsize,
#                color='w',
#                fontweight='bold')
	plt.tight_layout()
	#            **kwargs)
	# Displays an image


	ax.add_patch(rect)
	ax.axis('off')
	plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
	if not save:
		plt.show()
	else:
		plt.savefig(save_marked_fig_dir, dpi =  512/5)
		plt.close()
	return True
# #save fig as .png in tmp2
# save_marked_fig_dir = input_file_name.replace('tmp/preprocessed_snapshot','tmp2/highlighted_snapshot')
# assert(save_marked_fig_dir != input_file_name)
# #if ^this assert fails, check that input_file_name is an absolute directory
# #     save_figure(fig, save_marked_fig_dir=save_marked_fig_dir)
# img_fn = save_marked_fig_dir
# fig.tight_layout()
# fig.savefig(img_fn, dpi =  512/10)


def mark_from_highlighted_snapshot_png(input_file_name, **kwargs):
	'''mark_from_highlighted_snapshot_png(input_file_name)'''
	img = plt.imread(input_file_name)
	lst = input_file_name.split('.')
	assert(len(lst)>2)
	frm = int(eval(lst[-2]))
	save_marked_fig_dir = input_file_name.replace('tmp2/highlighted_snapshot','tmp2d2/marked_snapshot')
	out = mark_image(img, frm, save_marked_fig_dir, **kwargs)
	retval = plot_figure_and_save(out, save_marked_fig_dir, **kwargs)
	plt.close()
	return retval
	# plt.show()
	
	# plt.imshow(img)
def mark_image(img, frm, save_marked_fig_dir, **kwargs):
	'''add the yellow x to the figure'''
	#time_stamp is not the message
	#mark img with a scale bar
	lst = save_marked_fig_dir.split('.')
	assert(len(lst)>2)
	frm = int(eval(lst[-2]))
	os.chdir(kwargs['data_dir'])
	df = pd.read_csv(kwargs['cluster_dir'])

	#simple scale bar
	l = int(50/lamda)
	img = cv.rectangle(img,(400,40),(400+l,30),(255,255,255),-1)

	#mark img with a yellow marker
	position = df.loc[df.frame==frm][['x','y']].values.T
	drawMarker(img, position = (int(position[0]),int(position[1])), 
			   color = (1,1,0,1), markerType = cv.MARKER_TILTED_CROSS , 
			   markerSize = 15, thickness = 2, line_type = cv.LINE_AA)
	return img

def measure_outward_flow_in_range(flow, area_channel, x0,y0,r1=100,r2=300):
	'''calculate the mean  index averaged 
	over cell area in the annulus centered at the centroid. 
	r1 and r2 is the inner/outer diameter of the annulus in microns.
	i is the index of the "before" frame in img_list.  
	DIS dense optical flow is computed from img_list.
	Example Usage:
	ci_avg = get_mean_ci(i,img_list,r1=100,r2=300)
	'''
	#r_c_mat is the distance from the centroid in microns
	of = OpticalFlowClient(dt=dt)
	r_hat_mat, r_c_mat = of.get_r_hat_mat(position)
	boo = (r_c_mat>=r1 ) & (r_c_mat<=r2  )

	#get total component
	length = np.sqrt(np.multiply(flow[...,0],flow[...,0])+np.multiply(flow[...,1],flow[...,1]))

	#get radial component
	flow_out_x  = np.multiply(flow[...,0],r_hat_mat[...,0])
	flow_out_y  = -np.multiply(flow[...,1],r_hat_mat[...,1])
	flow_out    = flow_out_x + flow_out_y 
	#not this: np.sqrt  (  flow_out_x  ** 2  +  flow_out_y  **2  )              
				
	#calculate chemotactic index everywhere
	ci = np.divide(flow_out,length)
	ci[np.isnan(ci)]=0
				 
	#average over results with arr[booku]
	booku   = (area_channel!=0) & boo
	avg_ci  = float(np.median(ci[booku]))
	avg_ln  = float(np.median(length[booku]))    
	#calculate mean outward flow
	booku   &= flow_out>=0            
	avg_out = float(np.median(flow_out[booku]))
	#calculate mean inward flow
	booku   = (area_channel!=0) & boo
	booku   &= flow_out<=0            
	avg_in  = float(np.median(-flow_out[booku]))
	return avg_out, avg_in, avg_ci, avg_ln

def input_to_measures ( input_file_name, **kwargs):
	#get frame number, frm, for a given file_name
	lst = input_file_name.split('.')
	assert(len(lst)>2)
	frm = int(eval(lst[-2]))
	#only consider frames with a history, so that we may calculate optical flow
	if frm<=1:
		return None
	#import images    
	current, previous, previous_previous = load_input_grayscale_data_second_order(input_file_name)
	# import the aggregate's trajectory
	os.chdir(kwargs['data_dir'])
	df = pd.read_csv(kwargs['cluster_dir'])	
	position = df.loc[df.frame==frm][['x','y']].values.T
	# compute the measures of flow
	flow = compute_sharp_optical_flow(current, previous, previous_previous, of=None, **kwargs)
	avg_out, avg_in, avg_ci, avg_ln = measure_outward_flow_in_range(flow, area_channel, x0=position[1],y0=position[0],r1=100,r2=300)
	return {frm : (avg_out, avg_in, avg_ci, avg_ln)}
# col_names = ['avg_out', 'avg_in', 'avg_ci', 'avg_ln']


# def mark_and_save_highlight(image, flow_in, vmin, vmax, save=save, save_dir = None, save_fn = None, figsize = (10,10), mydpi = 512/10):
#     fig = highlight_flow_bwr(image, flow_in, vmin, vmax, figsize = figsize, mydpi = mydpi)
#     if not save:
#         plt.show()
#     else:
#         os.chdir(save_dir)
#         plt.tight_layout()
#         plt.savefig(save_fn, dpi = mydpi)

# def mydot(flow, rhat,angthresh = 0):
#     '''flow_in, flow_out = dot(self, flow, rhat)
#     #define flow as out/in if it is within 45degrees of directly out/in
#     # <--> v_r/vtot = CI < np.sqrt(0.5)
#     '''
#     mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])

#     r_hat_mat = rhat
#     # # calculate r_hat
#     # df = self.df
#     # position = self.get_centroid(i, df)
#     # r_hat_mat, r_c_mat   = self.get_r_hat_mat(position = position)

#     #dot dense optical flow direction with the outward r_hat
#     #TODO: add a cross product option for curl flow about the center marked with a "yellow x"
#     flow_out_x  = np.multiply(flow[...,0],r_hat_mat[...,0])
#     flow_out_y  = -np.multiply(flow[...,1],r_hat_mat[...,1])
#     #TODO:try undoing both yaxis flips. consider saving a previous version first.
#     flow_out = flow_out_x + flow_out_y


# #     angthresh = self.angthresh
#     #remove small flows under threshold
#     # flow_out[flow_out<1] = 0

#     #calculate chemotactic index texture CI where defined
#     boo = (mag>0)
#     # CI = 0.*flow_out
#     CI = np.divide(flow_out_x+flow_out_y, mag, where=boo)

#     #find boolean index of inward flows (dot product is negative) and set inward flow to zero
#     flow_in = flow_out.copy()
#     flow_in[CI>-angthresh] = 0.
#     flow_in       = -flow_in

#     flow_out[CI<angthresh] = 0.
#     return flow_in, flow_out

# def calc_flow(previous, current, current_flow=None, minthresh=0):
#     dis = cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_MEDIUM)
#     flow = dis.calc(previous,current, current_flow)
#     mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])

#     #remove small flows under threshold
#     flow[...,0][mag<minthresh] = 0
#     flow[...,1][mag<minthresh] = 0
#     return flow

# def calc_sharp_optical_flow(img,img_before,dt, position):
#     '''calculate radial inward/outward flow.'''

#     of = OpticalFlowClient(dt=dt)
#     #NB: requires edges and df as arguments
#     #TODO: make sure that flow_list is appended to only here in the module
# #     start = time.time()
#     # get radial coordinates
#     rhat,rmat = of.get_r_hat_mat(position)

#     # get flow field in radial coordinates
#     flow      = of.calc_flow(img,img_before)
#     flow_out, flow_in = mydot(flow, rhat)

#     # #blurred binary cell area mask
#     # fltr     = edges[frm]/2+edges[frm+1]/2
#     # fltr     = gaussian(fltr, sigma=sigma)
#     # fltr[fltr<20] = 0
#     # fltr[fltr>=20] = 1
#     # area_channel = fltr#pims.frame.Frame(fltr.astype('uint16'))
#     area_channel = img/np.max(img)

#     #filter off-cell flow and return 
#     output_texture = np.stack([(area_channel*flow_in).astype('float32'), 
#                                (area_channel*flow_out).astype('float32'),
#                               rmat.astype('uint32'),
#                               area_channel.astype('uint16')], axis=2)
#     #include original frame_no metadata
#     flow_radial       = pims.frame.Frame(output_texture, frame_no = frm)
#     return flow_radial