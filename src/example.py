import numpy as np, os, time
import cv2 as cv
from tkinter import filedialog, Tk
from highlight_func import highlight
#########################
# Compute Optical Flow
#########################
def compute_optical_flow_simple(prv,nxt,clf=None,**kwargs):
    """
    Example Usage:
flow=compute_optical_flow_simple(prv,nxt)
flow_x=flow[...,0]
flow_y=flow[...,1]
mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    """
    if clf is None:
        clf=cv.cv2.DISOpticalFlow_create(cv.cv2.DISOpticalFlow_PRESET_MEDIUM)
    flow=clf.calc(prv,nxt, None)
    return flow

###########################
# Compute Radial Component
###########################
def get_r_hat_mat(width,height, x_center, y_center):
    '''returns matrices of r_hat outward from the center at x_center, y_center as r_hat_mat.
    also returns the radial distance in r_c_mat.
    Example Usage:
r_hat_mat, r_c_mat=get_r_hat_mat(width,height, x_center, y_center)
    '''
    #make a texture of the r_hat outward from the cluster
    xc=x_center
    yc=y_center
    y_mat = np.array([x for x in range(width) for y in range(height)]).reshape(width,height)
    x_mat = np.array([y for x in range(width) for y in range(height)]).reshape(width,height)
    dx_mat = x_mat - xc
    dy_mat = y_mat - yc
    r_c_mat  = np.sqrt(dx_mat**2+dy_mat**2)
    boo      = (r_c_mat>=1)#the origin doesn't like this
    r_hat_mat_x = np.divide(dx_mat,r_c_mat, where=boo)
    r_hat_mat_y = np.divide(-dy_mat,r_c_mat, where=boo)#flip y axis
    r_hat_mat   = np.stack([r_hat_mat_x,r_hat_mat_y], axis=2)
    return r_hat_mat, r_c_mat

def dot(flow, r_hat_mat,angthresh = 0,**kwargs):
    '''
    angthresh = 0 #all in/out flow considered
    angthresh = np.sqrt(0.5) #45 degree bins

    Example Usage:
flow_in, flow_out = dot(flow, r_hat_mat)'''
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    #dot dense optical flow direction with the outward r_hat
    flow_out_x  = np.multiply(flow[...,0],r_hat_mat[...,0])
    flow_out_y  = -np.multiply(flow[...,1],r_hat_mat[...,1])
    flow_out = flow_out_x + flow_out_y

    #calculate chemotactic index texture CI where defined
    boo = (mag>0)
    CI = np.divide(flow_out_x+flow_out_y, mag, where=boo)

    #find boolean index of inward flows (dot product is negative) and set inward flow to zero
    flow_in = flow_out.copy()
    flow_in[CI>-angthresh] = 0.
    flow_in       = -flow_in

    flow_out[CI<angthresh] = 0.

    return flow_in, flow_out


#####################################################
# Image Preprocessing for our DIC microscopic images
#####################################################
from joblib import Parallel, delayed
import pims
from skimage.filters import scharr_h, scharr_v, gaussian

def preprocess(frames, thresh=90):
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

#####################################################
# input support of .tiff files from imageJ
#####################################################
#setup user interface for file selection
def search_for_file_path (currdir):
    root = Tk()
    tempdir = filedialog.askopenfilename(parent=root,initialdir=currdir, title="Please select desired file.")
    root.destroy()
    if len(tempdir) > 0: print ("File Name: %s" % tempdir)
    return tempdir

#####################################################
# viewer / output support to .avi files
#####################################################
def tiffstack_to_avi(path, fps=8, save_dir= None):
	'''saves tiffstack in local 'path' to similarly named avi.
	TODO: simplify this io a bit. '''

	if save_dir == None:
		save_dir= path[:path.find(r'.')]+'.avi'
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
	print('{} seconds elapsed reading and writing video to avi.'.format(np.around(end-start)))
	return save_dir

#####################################################
# example routine
#####################################################
if __name__=='__main__':
    fps=40
    save_dir='out.mp4'

    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    print(f"This example is for computing inward/outward flow of cells from our DIC setup.")
    # x_center=input(f"please enter the x coordinate of the fixed center, in pixels:")
    # y_center=input(f"please enter the y coordinate of the fixed center, in pixels:")
    x_center=300
    y_center=300
    position=(int(x_center),int(y_center))
    print(f"Fixed center is at {position}")
    print("Please select the desired .tiff file")
    file_name=search_for_file_path ( currdir = os.getcwd() )
    # file_name="/Users/timothytyree/Desktop/Old Desktop/Research/Rappel/Dicty. Dispersal/Python/Cell Positions to Cell Trajectories/frames031318.tiff"
    #preprocessing
    frames = pims.TiffStack_libtiff(file_name)
    print(f'.tiff file loaded:\nnumber of frames: {len(frames)}')
    imgs = preprocess(frames)
    # print(f"succesffully preprocessed {len(imgs)} frames")
    clf=cv.cv2.DISOpticalFlow_create(cv.cv2.DISOpticalFlow_PRESET_MEDIUM)

    #load data and initialize the writer
    #save_dir= file_name[:file_name.find(r'.')]+'.avi'
    start = time.time()

    # boo, cap = cv.imreadmulti(file_name)
    # width = int(cap[0].shape[0])
    # height = int(cap[0].shape[1])
    width = int(imgs[0].shape[0])
    height = int(imgs[0].shape[1])
    r_hat_mat, r_c_mat=get_r_hat_mat(width, height, int(x_center), int(y_center))

    # #preprocessing
    # frames = cap#pims.TiffStack_libtiff(file_name)
    # print(f'.tiff file loaded:\nnumber of frames: {len(frames)}')
    # imgs = preprocess(frames)
    # print(f"succesffully preprocessed {len(imgs)} frames")

    # #note: r_hat_mat encodes the fixed position inputed
    # try:
    #     chnl_no = int(cap[0].shape[2])
    # except:
    #     chnl_no = 0#or 1, not sure
    #     raise('Error: depth shape not given')
    # # uncompressed YUV 4:2:0 chroma subsampled
    # fourcc = cv.VideoWriter_fourcc('I','4','2','0')
    #for .mp4
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv.VideoWriter()
    retval = writer.open(save_dir, fourcc, fps, (width, height), 1)
    assert(writer.isOpened())#assert the writer is properly initialized
    frame_prev=imgs[0]
    #stream frames highlighted with inward/outward flow to .avi
    #TODO: make this step much faster by using something like the (missing) cv.GrabFrame command
    for i  in range(1,len(imgs)):
        frame_next = imgs[i]
        background_frame=frames[i]#cap[i]
        #compute in/out flow
        flow=compute_optical_flow_simple(prv=frame_prev,nxt=frame_next,clf=clf)
        flow_in, flow_out = dot(flow, r_hat_mat)

        #constrain flow to the location of cells
        flow_in*=frame_next
        flow_out*=frame_next

        img_ = highlight(position=position, flow_in=flow_in, flow_out=flow_out, background_frame=background_frame)

        #fix the colors
        # img_out=(img_).astype(np.uint8)[...,[2,0,1]]#y looks reasonable
        # img_out=(img_).astype(np.uint8)[...,[1,0,2]]
        # img_out=(img_).astype(np.uint8)[...,[0,2,1]]
        # img_out=(img_).astype(np.uint8)[...,[1,2,0]]#rb looks reasonable
        img_out=(img_).astype(np.uint8)[...,[2,1,0]]  #that's the right one
        #img_out = cv.cvtColor(img_out, cv.COLOR_RGB2YUV)
        # img_out = cv.cvtColor(img_out, cv.COLOR_RGB2YUV)
        # img_out=(background_frame).astype(np.uint8)
        #record
        writer.write(img_out)
        frame_prev=frame_next
    writer.release()
    end = time.time()
    print('{} seconds elapsed reading and writing video to avi.'.format(end-start))
    print(f"save_dir: {os.path.abspath(save_dir)}")

    #TODO(better): use colormap such as # im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_COOL)
