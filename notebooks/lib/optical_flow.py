# /*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#  * Python       :   Optical Flow for Cell Motion independent of Cell Tracking
#  *
#  * PROGRAMMER   :   Timothy Tyree
#  * DATE         :   Fri 22 Nov 2019
#  * PLACE        :   Rappel Lab @ UCSD, CA
#  *@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#  */

#TODO: import only relevant functions
import pandas as pd
import numpy as np
import pims
from skimage.filters import scharr_h, scharr_v, gaussian
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
import tifffile
from cv2 import drawMarker, calcOpticalFlowFarneback
import cv2 as cv
from tkinter import filedialog, Tk
from scipy.interpolate import CubicSpline
# import trackpy as tp
# import skimage as sk

#automate the boring stuff
import time, os, sys
beep = lambda x: os.system("echo -n '\a';sleep 0.2;" * x)
if not 'nb_dir' in globals():
	nb_dir = os.getcwd()
print('notebook is at: ' + nb_dir)

#TODO: append mean flow to .tiff file as they're calculated instead of calculating them
# all for saving all at once.
#  rgb float64 textures for each frame:
#  r channel: flow x
#  g channel: flow y
#  b channel: r_hat

#TODO: get flow_prev to be treated properly between calls to get_flow_field_moving_average
#TODO: make raw_save_file_name robust to absolute/local directory usage
#TODO: glance at all inputs, checking for sensibility
#TODO: add DIS_OF function and allow keywords to parse kwargs for DIS_OF
#TODO: remove any mentions of a needletip. this is deprecated.

#TODO: use draw_flow to visualize raw flow (input frames with quiver plot overlay)
#TODO: test non-512x512 inputs are handled correctly
#TODO: test cross_flow functionality

#TODO: allow inputs like '.mpg' video inputs
#TODO: allow video stream inputs
#Hint: elif v_data_type == 'video':
#     cap = cv2.VideoCapture("vtest.avi")
#     ret, frame1 = cap.read()

#TODO: add a .input_file_name function using my current method
#TODO: elabourate all docstrings
#TODO: make more test cases!
#TODO: organize functions by subject.
#TODO: collect similar functions into subclasses
#TODO: give user option to specify fields to use for df instead of 'frame', 'x', and 'y'
#TODO: put parameter parsing + defaults into all helper functions so all moving parts are accessable from the top layer
#TODO: add getter and setter functions for parameters
#TODO: hide functions the user doesn't want to look at
#TODO: change preprocess to handle uint8 inputs if it is given
#TODO: add test case for highlight_flow)

#TODO: check that I can delete the following commented code.
# OF_PARAMS_DEFAULT = {'pyr_scale': 0.5, 'levels':3 , 'winsize': 10, 'iterations': 3, 'poly_n': 5, 'poly_sigma':1.2, 'flags':0}
# OF_PARAMS = OF_PARAMS_DEFAULT
# kwargs = {'pyr_scale': 0.3, 'levels':3 }
# keys_given = set(kwargs.keys()).intersection(OF_PARAMS_DEFAULT.keys())
# for key in keys_given:
#     OF_PARAMS[key] = kwargs[key]
# print(OF_PARAMS)
# test = pims.TiffStack(frame_dir)
# pims.TiffStack?
# frame_dir
# pims.TiffStack?
# pims.

class OpticalFlowClient(object):
	'''
		initialize optical flow client and specify any nondefault parameters
		self.dot and self.cross methods separate flow about some origin specified by
		self.position.
		PARAMS_DEFAULT = {'navg':8, 'width':512, 'height':512, 'rthresh': 0, 'lamda':1.33, 'dt':1/3, 'method':'dis'}

Example Usage:
cf = OpticalFlowClient.interpolate_trajectory(df)
plt.scatter(x=cf.x, y=cf.y, c=cf.frame, cmap='Blues')
plt.scatter(x=df.x, y=df.y, c='r')

	'''
	def __init__(self,  **kwargs):
		'''set client parameters'''
		kwargs = (kwargs if kwargs is not None else {})
		PARAMS_DEFAULT = {'navg':8, 'width':512, 'height':512, 'rthresh': 0, 'lamda':1.33, 'dt':1/3, 'method':'dis'}#'method':'farneback'}
		PARAMS = PARAMS_DEFAULT
		keys_given = set(kwargs.keys()).intersection(PARAMS_DEFAULT.keys())
		for key in keys_given:
			PARAMS[key] = kwargs[key]
		self.save_params= save_params = {'append':'True', 'imagej':False, 'contiguous':True}#, 'truncate':True,  'metadata':None}#, 'photometric':'RGB', } #'writeonce':True}

		self.navg      = PARAMS['navg']
		self.r_thresh  = PARAMS['rthresh']
		self.hue_scale = 10*self.navg
		self.nb_dir    = nb_dir
		self.lamda     = PARAMS['lamda']
		self.dt        = PARAMS['dt']
		self.method    = PARAMS['method']
		self.width     = PARAMS['width']
		self.fps       = 8#24
		self.height    = PARAMS['height']
		# self.angthresh = np.sqrt(0.5)#equal parts perpendicular and radial
		self.angthresh = 0#all in/out flow considered
		self.minthresh = 1#set minimum threshold flow magnitude for consideration
		#set optical flow parameters for Farneback optical flow
		OF_PARAMS_DEFAULT = {'pyr_scale': 0.5, 'levels':3 , 'winsize': 15, 'iterations': 3, 'poly_n': 5, 'poly_sigma':1.2, 'flags':0}
		OF_PARAMS = OF_PARAMS_DEFAULT
		keys_given = set(kwargs.keys()).intersection(OF_PARAMS_DEFAULT.keys())
		for key in keys_given:
			OF_PARAMS[key] = kwargs[key]
		self.OF_PARAMS = OF_PARAMS
		self.dis = cv.cv2.DISOpticalFlow_create(cv.cv2.DISOpticalFlow_PRESET_MEDIUM)
		self.flows = []
		self.hue_scale     = 10*self.navg
		self.hue_scale_red = 10*self.navg-10
		self.hue_scale_blue= 10*self.navg+10
		return None

	def set_df(self, df):
		self.df = df
		return self

	# /*========================================================================
	#  * Code for optical flow
	#  *========================================================================
	#  */
	def write(self, save_file_name, texture):
		'''appends texture to save_file_name as a .tiff'''
		rgba = texture
		tifffile.imsave(save_file_name, rgba, rgba.shape, **self.save_params)
		return self

	def append_texture(self, save_file_name, texture):
		'''appends texture to save_file_name as a .tiff'''
		rgba = texture
		tifffile.imsave(save_file_name, rgba, rgba.shape, **self.save_params)
		return self

	def write(self, save_file_name, flow, r_hat_mat):
		'''rg channels are flow.
		ba channels are r_hat_mat
		writes output to save_file_name as a tiff'''
		rgba = np.concatenate([flow,r_hat_mat], axis=2)
		tifffile.imsave(save_file_name, rgba, rgba.shape, **self.save_params)
		return self

	def write_list(self, save_file_name, flow_lst, r_hat_mat_lst):
		'''TODO: streamline this using nb sketches.'''
	#append rbg texture of raw flow data to file
		# kwrd = save_file_name[:save_file_name.find('_')]
		# raw_save_file_name = save_file_name.replace(kwrd,'raw')
		try:
			os.remove(save_file_name)
			# os.remove(raw_save_file_name)
		except:
			pass
		#TODO: try parallelizing this loop for ~8x speedup
		#is flow pickleable?
		for flow in flow_lst:
			self.write(save_file_name, flow_lst, r_hat_mat_lst)
		return self



	def run_optical_flow_main(self, df,file_name, save_file_name = None , background_file_name = None,
							  first_frame = None, last_frame =  None, average_then_dot = False,
							  assert_override=True, **kwargs):
		'''input file_name of dic frames (as a .tif or .tiff stack), a background_file_name of comparable type,
		and a trajectory DataFrame to highlight cell motion inwards/outwards from df. kwargs passed to
		cv2.calcOpticalFlowFarneback. save to save_file_name.

		Parameters:

		file_name = (directory for optical flow calculatons)
			^the (absolute) directory of the .tiff file of microscopy data (currently grayscale)
		background_file_name = (directory for background overlay)
			^the (absolute) directory of the .tiff file of microscopy data (currently grayscale)
		df is a pandas.DataFrame containing the fields
			df.frame = frame number is 1 to final frame (consistent with file_name)
			df.x     = x coordinate of marker in pixels
			df.y     = y coordinate of marker in pixels
			**kwargs = dict of keyword arguments passed to optical flow subroutines'''

		navg      = self.navg
		hue_scale = self.hue_scale
		r_thresh  = self.r_thresh




		@pims.pipeline
		def gray(image):
			return np.uint8(image[:, :, 1])  # Take just the green channel

		#set defaults to positional arguments if not specified
		if save_file_name == None:
			save_file_name = 'out_'+file_name
		if background_file_name == None:
			background_file_name = file_name

		#import DIC frames
		#TODO(later): IF pims.TiffStack_libtiff is actually too slow, try fast import method and use the slow one if the fast one fails
		#TODO: search for file and only use parse only if file is not found.
		#	TODO: make parse take absolute directories for autocompletion
		#     file_name = parse(file_name)
		#     background_file_name = parse(background_file_name)
		#	NB: pims.TiffStack(file_name) is faster (for finding median and max in preprocessing) but fails at importing .tiff's I personally exported in this nb.
		#     boo, frames = cv.imreadmulti(file_name)
		frames = pims.TiffStack_libtiff(file_name)

		#import background frames
		#     boo, background_frames = cv.imreadmulti(background_file_name)
		background_frames = pims.TiffStack_libtiff(background_file_name)

		print('frames imported.')
		#select particular frames if specified
		if (first_frame != None and last_frame != None):
			frames = frames[first_frame:last_frame]
			background_frames = background_frames[first_frame:last_frame]
		elif (first_frame != None):
			frames = frames[first_frame:]
			background_frames = background_frames[first_frame:]
		elif (last_frame != None):
			frames = frames[:last_frame]
			background_frames = background_frames[:last_frame]

		# assert lengths agree
		#TODO: add assert_override option to kwargs
		#TODO: turn this into proper exception handling/a proper switch statement of loading routines
		print('frame number = {}, trajectory length = {}'.format(len(frames),df.frame.size))
		if not assert_override:
			assert(df.frame.size == len(frames))
			assert(df.frame.size == len(background_frames))
		print('frame number agrees.')

		#perform Scharr filter in preprocessing
		imgs = self.preprocess(frames)

		print('calculating dense optical flow.  this may take a few minutes...')
		#highlight and save optical flows as bulky tiffstacks
		#TODO(later): let save_params be specified in kwargs
		save_params = {'append':'True', 'imagej':False, 'contiguous':True}#, 'truncate':True,  'metadata':None}#, 'photometric':'RGB', } #'writeonce':True}
		start = time.time()
		try:
			os.remove(save_file_name)
		except:
			pass
		frame_max = len(frames)-navg
		lst_out = []
		lst_keys= ['frame', 'mfi_inner_avg', 'mfo_inner_avg', 'mfi_outer_avg', 'mfo_outer_avg']
		#TODO: add option to not save flow


		#append rbg texture of raw flow data to file
		kwrd = save_file_name[:save_file_name.find('_')]
		raw_save_file_name = save_file_name.replace(kwrd,'raw')
		try:
			os.remove(raw_save_file_name)
		except:
			pass

		#TODO: try parallelizing this loop for ~8x speedup
		for j in range(frame_max):
			background_frame = background_frames[j+navg]

			# calculate r_hat
			#TODO: remove redundant calculations of r_hat and r_c_hat
			position = self.get_centroid(j, df)
			r_hat_mat, r_c_mat   = self.get_r_hat_mat(position = position)

			#calculate mean inward/outward flow
			if average_then_dot:#gave gibberish when I first tried it
				flow =  self.get_flow_field_moving_average(j, imgs, flow_prev= None, **kwargs)
				mfi, mfo, r_c_mat = self.dot_flow(j, flow, df, **kwargs)
				rgba = np.concatenate([flow,r_hat_mat], axis=2)
				tifffile.imsave(raw_save_file_name, rgba, rgba.shape, **save_params)
			else:
				mf, mfi, mfo, r_c_mat = self.get_mean_flow(j, imgs = imgs, df = df, **self.OF_PARAMS)
				rgba = np.concatenate([mf,r_hat_mat], axis=2)
				tifffile.imsave(raw_save_file_name, rgba, rgba.shape, **save_params)

			#define inner averaging annulus in microns
			lamda = self.lamda
			r1, r2 = (50/lamda,100/lamda)
			mag, ang = cv.cartToPolar(rgba[...,0], rgba[...,1])
			mfi_inner_avg = self.annulus_avg(mfi, mag, r1, r2, r_c_mat)
			mfo_inner_avg = self.annulus_avg(mfo, mag, r1, r2, r_c_mat)

			#define outer averaging annulus in microns
			r1, r2 = (150/lamda,200/lamda)
			mfi_outer_avg = self.annulus_avg(mfi, mag, r1, r2, r_c_mat)
			mfo_outer_avg = self.annulus_avg(mfo, mag, r1, r2, r_c_mat)
			out = [j, mfi_inner_avg, mfo_inner_avg, mfi_inner_avg, mfo_outer_avg]
			lst_out.append(out)

			img = self.highlight_flow(j, background_frame, df, mfi, mfo, r_c_mat).astype('uint8')
			#         img = highlight_flow(j, background_img, imgs = imgs, df = df).astype('uint8')
			#     img.metadata = None
			tifffile.imsave(save_file_name, img, img.shape, **save_params)


		#TODO: verify something dumb like a transpost isn't needed before casting to dataframe     lst_out = pd.DataFrame(dict(zip(lst_keys, lst_out)))
		end = time.time()
		print(str(np.around(end - start,1)) + ' seconds elapsed highlighting dense optical flow. ')
		#(TODO: make the following optional through the kwargs. )
		# output to avi ( / quicktime )
		self.tiffstack_to_avi(save_file_name)
		return lst_out

	def annulus_sum(self,txt,rtxt,r1,r2):
		'''returns the sum of texture txt in the annulus r1:r2, with rtxt as the texture fo radial distances from the origin.
		Sum of all channels contained in axis=2 of txt is returned.
		'''
		return txt[(rtxt>r1) & (r2>rtxt)].sum()

	def search_for_frame_path (self, currdir = os.getcwd()):
		'''#make functions for save file name, input cell frames, and input cell trajectories'''
		#TODO: eventually make this ^take cell trajectories or cell positions
		root = Tk()
		tempdir = filedialog.askopenfilename(parent=root,
											 initialdir=currdir,
											 title="Please select .tiff of cell frames",
											 filetypes = (("tiff files","*.tiff"),("tif files","*.tif"),("all files","*.*")))
		root.destroy()
		if len(tempdir) > 0:
			print ("Frames: %s" % tempdir)
		return tempdir

	def search_for_traj_path (self, currdir = os.getcwd()):
		root = Tk()
		tempdir = filedialog.askopenfilename(parent=root,
											 initialdir=currdir,
											 title='Please select .csv of cell trajectories',
											 filetypes = (("csv files","*.csv"),("excel files","*.xlsx"),("all files","*.*")))
		root.destroy()
		if len(tempdir) > 0:
			print ("Trajectories: %s" % tempdir)
		return tempdir

	def mypreprocess(self, frames, thresh=90):
		return preprocess(self, frames, thresh=90)

	def preprocess(self, frames, thresh=90):
		'''frames is an iterable of numpy.array objects.
		uses scharr filter to get cell edges from DIC channel,
		scaling down to uint8 dtypes.
		returns a stack of image list of frames.
	    preprocessing should be done with this output
	    to avoid normalizing intensity with respect
	    to variable empirical parameters.'''
		md1 = np.median(frames[0])
		mx1 = np.max(frames[0])#nearly 65535 for uint16
		md2 = np.median(frames[-1])
		mx2 = np.max(frames[-1])
		md  = md1/2+md2/2
		mx  = mx1/2+mx2/2
		def processInput(img):
			fno  = img.frame_no
			img = 255/(mx-md)*(img-md)
			edges = np.sqrt(scharr_h(img)**2+scharr_v(img)**2)
			return pims.Frame(np.uint8(edges), frame_no=fno+1)
		start = time.time()
		inputs = frames
		edges = Parallel(n_jobs=num_cores, verbose=.1)(delayed(processInput)(img) for img in inputs)
		end = time.time()
		print('{} seconds elapsed taking scharr filtration.'.format(np.around(end-start,1)))
		#denoise the background of the scharr transform with a lower bound threshold edge intensity
		imgs = edges.copy()
		for img in imgs:
			img[img<thresh] = 0
			img[img>thresh] = 255
		return imgs


	def set_frames(self, frames):
		self.frames = frames
		return self

	def set_flow_frames(self, frames):
		self.flow_frames = frames
		return self

	def get_frames(self, frames):
		return self.frames

	def get_flow_frames(self, frames):
		return self.flow_frames

	def calc_flow(self, previous, current, current_flow=None):
		#TODO: if self.dis = None: make it, else use self.dis
		flow = self.dis.calc(previous,current, current_flow)
		mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])

		#remove small flows under threshold
		minthresh = self.minthresh
		flow[...,0][mag<minthresh] = 0
		flow[...,1][mag<minthresh] = 0
		return flow

	def dot(self, flow, rhat):
		'''flow_in, flow_out = dot(self, flow, rhat)'''
		mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])

		r_hat_mat = rhat
		# # calculate r_hat
		# df = self.df
		# position = self.get_centroid(i, df)
		# r_hat_mat, r_c_mat   = self.get_r_hat_mat(position = position)

		#dot dense optical flow direction with the outward r_hat
		#TODO: add a cross product option for curl flow about the center marked with a "yellow x"
		flow_out_x  = np.multiply(flow[...,0],r_hat_mat[...,0])
		flow_out_y  = -np.multiply(flow[...,1],r_hat_mat[...,1])
		#TODO:try undoing both yaxis flips. consider saving a previous version first.
		flow_out = flow_out_x + flow_out_y

		#define flow as out/in if it is within 45degrees of directly out/in
		# <--> v_r/vtot = CI < np.sqrt(0.5)
		angthresh = self.angthresh
		#remove small flows under threshold
		# flow_out[flow_out<1] = 0

		#calculate chemotactic index texture CI where defined
		boo = (mag>0)
		# CI = 0.*flow_out
		CI = np.divide(flow_out_x+flow_out_y, mag, where=boo)

		#find boolean index of inward flows (dot product is negative) and set inward flow to zero
		flow_in = flow_out.copy()
		flow_in[CI>-angthresh] = 0.
		flow_in       = -flow_in
		self.flow_in  = flow_in

		flow_out[CI<angthresh] = 0.
		self.flow_out = flow_out

		return flow_in, flow_out

	def cross(self, flow, rhat):
		'''flow_cw, flow_cc = cross(self, flow, rhat)
		flow_tw, flow_ws = cross(self, flow, rhat)'''
		r_hat_mat = rhat
		# # calculate r_hat
		# df = self.df
		# position = self.get_centroid(i, self.df)
		# r_hat_mat, r_c_mat   = self.get_r_hat_mat(position = position)

		#dot dense optical flow direction with the outward r_hat
		#TODO: add a cross product option for curl flow about the center marked with a "yellow x"
		flow_perp_x  = (np.multiply(flow[...,0],r_hat_mat[...,1]))
		flow_perp_y  = -(-np.multiply(flow[...,1],r_hat_mat[...,0]))
		#TODO: consider undoing both yaxis flips (all three here)
		# flow_out    = np.sqrt(np.multiply(flow_out_y, flow_out_y) + np.multiply(flow_out_x, flow_out_x))

		#define flow as out/in if it is within 45degrees of directly out/in
		# <--> v_r/vtot = CI < np.sqrt(0.5)
		angthresh = self.angthresh


		#find out where there's zero flow
		mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
		boo = (mag>0)
		#calculate chemotactic index texture CI where defined

		#not chemotactic index anymore
		CI = np.divide(flow_perp_x+flow_perp_y, mag, where=boo)
		self.CI = CI
		flow_cw = flow_perp_x + flow_perp_y

		#remove small flows under threshold
		flow_cw[CI>-angthresh] = 0.
		flow_cc = flow_perp_x + flow_perp_y
		flow_cc[CI<angthresh] = 0.

		#find boolean index of inward flows (dot product is negative) and set inward flow to zero
		# flow_in[(flow_out_x+flow_out_y)>0] = 0.
		flow_cw[CI<angthresh] = 0.
		flow_cw = -flow_cw
		self.flows.append(flow_cw)
		self.flow_cw = flow_cw
		self.flow_cc = flow_cc
		self.flow_tw = flow_cw#turnwise
		self.flow_ws = flow_cc#wittershins
		return flow_cw, flow_cc

	def times(self, txt, txt2):
		return np.multiply(txt, txt2)

	def parse(self, target):
		'''parse files in the current directory and return the first file that contains the string, target.'''
		dir_name = None
		for string in os.listdir():
			if target in string:
				dir_name = string
				break
		if not dir_name: raise Exception('File for target string, {}, was not found.'.format(target))
		return dir_name

	def get_r_hat_mat(self, position):
		'''get_r_hat_mat([x_coord,y_coord])--> r_hat_mat, r_c_mat
		returns texture width xy channels populated by the unit vector pointing radially away from position.'''
		#make a texture of the r_hat outward from the cluster
		width = self.width
		height= self.height
		xc,yc = tuple(position)
		y_mat = np.array([x for x in range(width) for y in range(height)]).reshape(width,height)
		x_mat = np.array([y for x in range(width) for y in range(height)]).reshape(width,height)
		dx_mat = x_mat - xc
		dy_mat = y_mat - yc
		r_c_mat  = np.sqrt(dx_mat**2+dy_mat**2)
		boo      = (r_c_mat>=1)#the origin doesn't like this
		r_hat_mat_x = np.divide(dx_mat,r_c_mat, where=boo)
		r_hat_mat_y = np.divide(-dy_mat,r_c_mat, where=boo)#flip y axis
		r_hat_mat   = np.stack([r_hat_mat_x,r_hat_mat_y], axis=2)
		self.r_hat_mat  = r_hat_mat
		self.r_c_mat  = r_c_mat
		return r_hat_mat, r_c_mat

	def average_texture_list(self, imgs):
		'''returns the mean image for a list of images stored in imgs'''
		#average the entries of imgs
		mf = np.stack(imgs[:])
		mf = np.mean(mf, axis=0)
		return mf

	def imread(file_name, channel=0):
	    '''load grayscale image. channel = 0 takes just the red channel.'''
	    return cv.imread(file_name)[:,:,channel]

	def load_tiff_stack(self, file_dir):
		'''returns stack of numpy arrays with frame number metadata stored in `frame_no`'''
		return pims.TiffStack_libtiff(file_dir)

	def get_mean_flow(self, frm, imgs, df, **kwargs):
		'''initial and final frames of apparent outward/inward flow.  takes one int argument frm as frame number.
		flow is dotted for each frame and then averaged over navg frames.  For slow moving origins, it would be better
		to average flows over navg frames and then take the dot (or cross) product.'''
		#define optical flow parameters.  set defaults if not specified.

		OF_PARAMS_DEFAULT = {'pyr_scale': 0.5, 'levels':3 , 'winsize': 15, 'iterations': 3, 'poly_n': 5, 'poly_sigma':1.2, 'flags':0}
		OF_PARAMS = OF_PARAMS_DEFAULT
		keys_given = set(kwargs.keys()).intersection(OF_PARAMS_DEFAULT.keys())
		for key in keys_given:
			OF_PARAMS[key] = kwargs[key]

		navg      = self.navg
		r_thresh  = self.r_thresh
		angthresh = self.angthresh

		f_init = frm
		f_final= f_init+navg

		#TODO: calculate mean position of df for marking with a yellow x to marginally decrease runtime.
		#TODO: test for when the variance or speed of the yellow x trajectory is too large,
		#TODO: when ^that is too large, switch to multiple evaluations of r_hat_mat inside the for loop
		#     d = df.query('frame=={}'.format(int(np.around(np.mean([f_init, f_final])))))[['x', 'y']]
		flow_out_lst = []
		flow_in_lst  = []
		flow_lst     = []
		if self.method == 'dis':
			self.dis = cv.cv2.DISOpticalFlow_create(cv.cv2.DISOpticalFlow_PRESET_MEDIUM)

		for i in range(f_init,f_final):
			#get the optical flow
			prv = imgs[i]
			nxt = imgs[i+1]

			if self.method == 'farneback':
				flow = cv.calcOpticalFlowFarneback(prv,nxt, None, **OF_PARAMS)
			elif self.method == 'dis':
				flow = self.dis.calc(prv,nxt, None)
			else:
				Exception('invalid method entered')

			mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])

			#remove small flows under threshold
			minthresh = self.minthresh
			flow[...,0][mag<minthresh] = 0
			flow[...,1][mag<minthresh] = 0

			#append flow before dotting
			flow_lst.append(flow)

			# calculate r_hat
			position = self.get_centroid(i, df)
			r_hat_mat, r_c_mat   = self.get_r_hat_mat(position = position)

			#dot dense optical flow direction with the outward r_hat
			#TODO: add a cross product option for curl flow about the center marked with a "yellow x"
			flow_out_x  = np.multiply(flow[...,0],r_hat_mat[...,0])
			flow_out_y  = -np.multiply(flow[...,1],r_hat_mat[...,1])#consider undoing both yaxis flips
			# flow_out    = np.sqrt(np.multiply(flow_out_y, flow_out_y) + np.multiply(flow_out_x, flow_out_x))
			flow_out = flow_out_x + flow_out_y

			#define flow as out/in if it is within 45degrees of directly out/in
			# <--> v_r/vtot = CI < np.sqrt(0.5)
			# angthresh = 0#all in/out flow considered
			# angthresh = np.sqrt(0.5)#45 degree bins
			#remove small flows under threshold
			# flow_out[flow_out<1] = 0

			#calculate chemotactic index texture CI where defined
			boo = (mag>0)
			# CI = 0.*flow_out
			CI = np.divide(flow_out_x+flow_out_y, mag, where=boo)

			#find boolean index of inward flows (dot product is negative) and set inward flow to zero
			flow_in = flow_out.copy()
			flow_in[CI>-angthresh] = 0.
			# flow_in[(flow_out_x+flow_out_y)>0] = 0.
			flow_in_lst.append(-flow_in)
			flow_out[CI<angthresh] = 0.
			# flow_out[(flow_out_x+flow_out_y)<0] = 0.
			flow_out_lst.append(flow_out)

		#average the entries of flow_out
		mf = np.stack(flow_lst[:])
		mf = np.mean(mf, axis=0)
		mfo = np.stack(flow_out_lst[:])
		mfo = np.mean(mfo, axis=0)
		mfi = np.stack(flow_in_lst[:])
		mfi = np.mean(mfi, axis=0)
		return mf, mfi, mfo, r_c_mat


	def get_centroid(self, frm, df):
		x = df.query('frame == {}'.format(frm)).x.values[0]
		y = df.query('frame == {}'.format(frm)).y.values[0]
		return (x,y)

	def annulus_avg(self, texture, mag_flow, r1, r2, r_c_mat):
		'''average the np array, texture, between radii r1 and r2 from radial texture r_c_mat.'''
		#TODO: do exception handling to make sure texture.shape is (width,height) only (i.e. only 1 channel)
		boo_mat = (r_c_mat>r1) & (r_c_mat<r2)
		a = texture[boo_mat].sum()
		b = mag_flow[boo_mat].sum()
		return a/b


	def highlight(self, position, flow_in, flow_out, background_frame, r_c_mat=None, **kwargs):
		'''visualize overlay of outward optical flow over the background channel.'''
		#TODO: use more sophisticated/correct kwargs parsing as done elsewhere
		#TODO: add r_thresh set method to argument of highlight
		mfo = flow_out
		mfi = flow_in
		# hue_scale = self.hue_scale
		hue_scale_red = self.hue_scale_red
		hue_scale_blue = self.hue_scale_blue
		r_thresh  = self.r_thresh

		filt = np.zeros([self.width,self.height,3])
		img  = pims.to_rgb(background_frame)

		# navg      = self.navg
		# img = pims.to_rgb(frames[int(np.around(np.mean([f_init, f_final])))])

		if r_c_mat == None:
			r_filter = 1
		elif r_c_mat.shape == (self.width, self.height):
			r_filter = (r_c_mat>r_thresh)
		else:
			 r_filter = 1

		#TODO: use colormap such as # im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_COOL)
		#subtract nonblue for outward flow
		filt[:,:,0] = hue_scale_blue*mfo*r_filter
		filt[:,:,1] = hue_scale_blue*mfo*r_filter

		#subtract nonred for inward flow
		filt[:,:,1] = hue_scale_red*mfi*r_filter
		filt[:,:,2] = hue_scale_red*mfi*r_filter
		# filt[:,:,2] = filt[:,:,2]/np.max(filt[:,:,2])+1
		flowing = np.add(img, -filt)

		#clip to 0 to 255
		# if np.isnan(np.sum(out_vec)):
		# 	out_vec = out_vec[~np.isnan(out_vec)] # just remove nan elements from out_vec
		flowing[flowing<0] = 0
		flowing[flowing>255] = 255

		#highlight needletip with a yellow x
		#TODO: make marker parameters accessable through **kwargs
		return drawMarker(flowing, position = position,
						  color = (255,255,0), markerType = 1 , markerSize = 15, thickness = 2)

	def highlight_flow(self, frm, background_frame, df, mfi, mfo, r_c_mat, **kwargs):
		'''visualize overlay of outward optical flow over the background channel.'''
		#TODO: use more sophisticated/correct kwargs parsing as done elsewhere


		# hue_scale = self.hue_scale
		hue_scale_red = self.hue_scale_red
		hue_scale_blue = self.hue_scale_blue
		r_thresh  = self.r_thresh

		filt = np.zeros([self.width,self.height,3])
		img  = pims.to_rgb(background_frame)

		# navg      = self.navg
		# img = pims.to_rgb(frames[int(np.around(np.mean([f_init, f_final])))])

		#TODO: use colormap such as # im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_COOL)
		#subtract nonblue for outward flow
		filt[:,:,0] = hue_scale_blue*mfo*(r_c_mat>r_thresh)
		filt[:,:,1] = hue_scale_blue*mfo*(r_c_mat>r_thresh)

		#subtract nonred for inward flow
		filt[:,:,1] = hue_scale_red*mfi*(r_c_mat>r_thresh)
		filt[:,:,2] = hue_scale_red*mfi*(r_c_mat>r_thresh)
		# filt[:,:,2] = filt[:,:,2]/np.max(filt[:,:,2])+1
		flowing = np.add(img, -filt)

		#clip to 0 to 255
		# if np.isnan(np.sum(out_vec)):
		# 	out_vec = out_vec[~np.isnan(out_vec)] # just remove nan elements from out_vec
		flowing[flowing<0] = 0
		flowing[flowing>255] = 255

		#TODO: position = get_centroid(frm) should be able to use subpixel accuracy
		position = (int(self.get_centroid(frm, df=df)[0]),int(self.get_centroid(frm, df=df)[1]))

		#highlight needletip with a yellow x
		#TODO: make marker parameters accessable through **kwargs
		return drawMarker(flowing, position = position,
						  color = (255,255,0), markerType = 1 , markerSize = 15, thickness = 2)

	def tiffstack_to_avi(self, path, save_dir= None):
		'''saves tiffstack in local 'path' to similarly named avi.
		TODO: formalize this io a bit. '''

		if save_dir == None:
			save_dir= path[:path.find(r'.')]+'.avi'
		start = time.time()
		boo, cap = cv.imreadmulti(path)
		fps = self.fps
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
		print('{} seconds elapsed reading and writing video to avi.'.format(np.around(end-start)))
		return self

	def get_flow_field_moving_average(self, frm, imgs, flow_prev= None, **kwargs):
		'''apparent flow averaged from frm to frm + navg. Fails for fast moving origins.'''
		#define optical flow parameters.  set defaults if not specified.

		OF_PARAMS_DEFAULT = {'pyr_scale': 0.5, 'levels':3 , 'winsize': 15, 'iterations': 3, 'poly_n': 5, 'poly_sigma':1.2, 'flags':0}
		OF_PARAMS = OF_PARAMS_DEFAULT
		keys_given = set(kwargs.keys()).intersection(OF_PARAMS_DEFAULT.keys())
		for key in keys_given:
			OF_PARAMS[key] = kwargs[key]

		navg      = self.navg
		r_thresh  = self.r_thresh

		using_flow_prev = (flow_prev != None)
		f_init = frm
		f_final= f_init+navg

		#precalculate mean position of df to marginally decrease runtime.
		#TODO: test for when the variance or speed of the yellow x trajectory is too large,
		#TODO: when ^that is too large, switch to multiple evaluations of r_hat_mat inside the for loop
		#scratchwork     d = df.query('frame=={}'.format(int(np.around(np.mean([f_init, f_final])))))[['x', 'y']]
		flow_lst = []
		if self.method == 'dis':
			self.dis = cv.cv2.DISOpticalFlow_create(cv.cv2.DISOpticalFlow_PRESET_MEDIUM)

		for i in range(f_init,f_final):
			#get the optical flow
			prv = imgs[i]
			nxt = imgs[i+1]

			if self.method == 'farneback':
				flow = cv.calcOpticalFlowFarneback(prv,nxt, flow_prev, **OF_PARAMS)
			elif self.method == 'dis':
				flow = self.dis.calc(prv,nxt, flow_prev)
			else:
				Exception('invalid method entered')
			#mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])

			flow_lst.append(flow)
			if not using_flow_prev:
				flow_prev = flow

			mf = np.stack(flow_lst[:])
			mf = np.mean(mf, axis=0)
			return mf

	#TODO: def dot_flow(self, flow, r_hat_mat, **kwargs):
	def dot_flow(self, frm, flow, df, **kwargs):
		'''flow is dotted with the unit vector extending radially away from the position of df in frame frm.
		For sufficiently slow moving origins, we average flows over navg frames and then take the dot (or cross) product.'''

		navg      = self.navg
		hue_scale = self.hue_scale
		r_thresh  = self.r_thresh

		# calculate r_hat
		i = int(np.around(np.mean([frm, frm+navg])))
		position = self.get_centroid(i, df)
		r_hat_mat, r_c_mat   = self.get_r_hat_mat(position = position)

		# if np.isnan(np.sum(r_hat_mat)):
		# 	r_hat_mat = r_hat_mat[~np.isnan(r_hat_mat)] # just remove nan elements from out_vec

		#dot dense optical flow direction with the outward r_hat
		#TODO: add a cross product option for curl flow about the yellow x
		flow_out_x  = np.multiply(flow[...,0],r_hat_mat[...,0])
		flow_out_y  = -np.multiply(flow[...,1],r_hat_mat[...,1])
		flow_out    = np.sqrt(np.multiply(flow_out_y, flow_out_y) + np.multiply(flow_out_x, flow_out_x))

		#remove small flows under threshold
		flow_out[flow_out<1] = 0

		#find boolean index of inward flows (dot product is negative) and set inward flow to zero
		flow_in = flow_out.copy()
		flow_in[(flow_out_x+flow_out_y)>0] = 0.
		flow_out[(flow_out_x+flow_out_y)<0] = 0.
		return flow_in, flow_out, r_c_mat

	def draw_flow(self, img, flow, step=16):
		h, w = img.shape[:2]
		y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
		fx, fy = flow[y,x].T
		lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
		lines = np.int32(lines + 0.5)
		vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
		cv.polylines(vis, lines, 0, (0, 255, 0))
		for (x1, y1), (_x2, _y2) in lines:
			cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
		return vis

	def interpolate_trajectory(self, df, f1=None, f2=None):
	    """returns a pandas.DataFrame with exactly one position for every frame.
	    Positions are calculated by cubic spline interpolation.
	    df is a pandas.DataFrame of pixel positions for sparse frames.
	    f1 is the first frame number to interpolate, and is the first value in df.frame by default.
	    f2 is the last frame number to interpolate, and is the last value in df.frame by default.
	    Nota Bene: it is always good to plot your interpolations before relying on them.
Example Usage:
	    cf = OpticalFlowClient.interpolate_trajectory(df)
	    plt.scatter(x=cf.x, y=cf.y, c=cf.frame, cmap='Blues')
	    plt.scatter(x=df.x, y=df.y, c='r')
	    """
	    knots= df.frame.values
	    if f1==None:
	        f1   = knots.min();
	    if f2==None:
	        f2   = knots.max()
	    x_values   = df.x
	    y_values   = df.y
	    xSpl = CubicSpline(knots,x_values); ySpl = CubicSpline(knots,y_values)
	    f    = np.arange(f1,f2)
	    x    = xSpl(f)
	    y    = ySpl(f)
	    cf = pd.DataFrame({'frame':f,'x':x,'y':y})
	    return cf

	def time_average_then_spatial_sum(self, texture_lst, r1,r2, navg=10, columns=None, units=None, lamda=None, dt=None, rmat_col = 2):
	    '''returns a list of dictionaries with a field for each channel in texture_lst. texture_lst is a list of textures.
	    dictionary values contain the sum over the annulus from r1 to r2 of the navg-frame moving average of texture_lst.
	    columns == None is equivalent to columns = ['in', 'out', 'r', 'area']
	    units == None is equivalent to units   = [lamda/dt, lamda/dt, lamda, lamda**2]
	    dt == None is equivalent to dt=self.dt is the time between two frames.
	    lamda == None is equivalent to lamda = self.lamda is the distance between two pixels.
	  	rmat_col is the axis int that refers to rmat.
	    '''
	    if lamda==None:
	    	lamda = self.lamda
	    if dt == None:
	    	dt = self.dt

	    #idiot proof the calculations here.
	    assert(r1<=r2)
	    channel_no = texture_lst[0].shape[-1]
	    if columns==None:
	        columns = ['in', 'out', 'r', 'area']
	    assert(len(columns)==channel_no)
	    if units==None:
	        units   = [lamda/dt, lamda/dt, lamda, lamda**2]
	    assert(len(units)==channel_no)
	    #time average then take sum over annulus
	    start = time.time()
	    dict_out_lst = []
	    inputs = np.arange(navg,len(texture_lst))
	    for f in inputs:
	        avg_txt = self.average_texture_list(texture_lst[f-navg:f])
	        rmat = avg_txt[..., rmat_col]
	        values = []
	        for val_no in np.arange(channel_no):
	            values.append(self.annulus_sum(avg_txt[...,val_no], rmat, r1, r2)*units[val_no])
	        dict_out_lst.append(dict(zip(columns, values)))
	    end = time.time()
	    print(f"{np.around(end-start)} seconds elapsed time averaging and then summing over annulus from {int(r1//1)} to {int(r2//1)} pixels.")
	    return dict_out_lst

	def reset_flow_lst(self):
		#TODO: make function to reset flow_lst
		self.flow_lst = []
		return self

	def create_radial_flow_texture_list(self, df, edges, trialnum, sigma=3):
	    '''calculate radial and perpendicular flow.
	     sigma = 2 #for 4X data works.
	     sigma = 3 #for 10X data works..'''
	    # self = OpticalFlowClient(dt=dt)
	    file_name_radial = 'flow_radial_'+trialnum+'.tiff'
	    try:
	        os.remove(file_name_radial)
	    except:
	        pass

	    #NB: requires edges and df as arguments
	    #TODO: make sure that flow_list is appended to only here in the module
	    start = time.time()
	    self.flow_lst = []
	    flow_lst = self.flow_lst
	    f_values = df.frame.values
	    for frm in f_values[:-1]:
	        # get radial coordinates
	        position  = self.get_centroid(frm,df)
	        rhat,rmat = self.get_r_hat_mat(position)

	        # get flow field in radial coordinates
	        #TODO: check that I'm not setting the turnwise channel to zero accidently.  It looks black everywhere while ws does not.
	        #     flow_tw, flow_ws  = self.cross(flow, rhat)
	        flow      = self.calc_flow(edges[frm+1],edges[frm])
	        flow_out, flow_in = self.dot(flow,rhat)

	        #raw fret channel (should already be uint16)
	        #     fret_channel = fretframes[frm].astype('uint16')

	        #binary area channel
	        fltr     = edges[frm]/2+edges[frm+1]/2
	        fltr     = gaussian(fltr, sigma=sigma)
	        fltr[fltr<20] = 0
	        fltr[fltr>=20] = 1
	        area_channel = fltr#pims.frame.Frame(fltr.astype('uint16'))

	        #filter off-cell flow
	        output_texture = np.stack([(area_channel*flow_in).astype('float32'),
	                                   (area_channel*flow_out).astype('float32'),
	                                  rmat.astype('uint8'),
	                                  area_channel.astype('uint8')], axis=2)
	        #include original frame_no metadata
	        #TODO: fix ^this in self.write, it's not working
	        #     flow_radial       = pims.frame.Frame(flow_radial, frame_no = frm)

	        #stack to rgba frames and write to files
	        self.append_texture(file_name_radial  ,output_texture)
	    end = time.time()
	    print("{} seconds elapsed calculating/exporting radial flow textures".format(np.around(end-start,1)))
	    return file_name_radial




	def run_tests(self, **kwargs):
		'''test cases for the optical client.'''
		# Example 1:
		#calculate texture of radial unit vectors, r_hat_mat, and radial vectors, r_c_mat
		print('running test cases...')
		dt = 1 #1 minute between each frame
		of = OpticalFlowClient(dt=dt, width = 512, height = 512)
		assert(of != None)
		x_coord, y_coord = [100, 150]
		r_hat_mat, r_c_mat = of.get_r_hat_mat([x_coord,y_coord])
		assert(r_hat_mat.shape == (512,512,2))
		assert(r_c_mat.shape == (512,512))

		# Example 2:
		#load and mark two frames with the apparent inward/outward flow from a pixel position
		#TODO: make shape test cases for Example 2:
		frame1       = of.imread('data/test_frm_450.png')
		frame2       = of.imread('data/test_frm_451.png')
		assert(frame1!=None)
		edge1, edge2 = of.preprocess([frame1, frame2])
		assert(edge1!=None)
		flow         = of.calc_flow(edge1, edge2)
		assert(flow!=None)
		flow_in, flow_out  = of.dot(flow, r_hat_mat)
		assert(flow_in!=None)
		img = of.highlight(position=(x_coord, y_coord), flow_in=flow_in, flow_out=flow_out, background_frame=frame_2)
		assert(img!=None)
		print('all test cases passed!')
		return True
