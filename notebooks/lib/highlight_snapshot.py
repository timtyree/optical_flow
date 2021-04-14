from lib.mark_images import make_input_image
from lib.optical_flow import OpticalFlowClient
import os, pims

def highlight_snapshot(background_frame, flow_in, flow_out, save_dir, position, lamda, dt, 
	time_stamp=None, message = None, hue_scale_blue= 512/8, hue_scale_red= 512/8, fontsize=22, y_time_stamp = 20, uml=50, **kwargs):
	''' Highlights input_file_name without the black blobs with blob optical flow.'''
	of = OpticalFlowClient(dt=dt)
	of.hue_scale_blue = hue_scale_blue
	of.hue_scale_red  = hue_scale_red
	out = of.highlight(tuple(position), flow_in, flow_out, background_frame=background_frame)
	make_input_image(out.astype('uint8'), save_dir, message=message, 
		time_stamp=time_stamp, y_start = 505, lamda=lamda, uml=uml, width=4, 
		fontsize=fontsize, y_time_stamp = y_time_stamp)
	return save_dir

# #######################################################################################
# # Example Usage to generate Fig. 5B, the 4X trial of the DTT experiment               #
# #######################################################################################
# data_folder = '/Users/timothytyree/Desktop/Research/Rappel/Dicty. Dispersal/experiment_with_DTT/4X/122019-2/'
# output_folder = "/Users/timothytyree/Desktop/Research/Rappel/Dicty. Dispersal/Python/fig/"
# cluster_fn = "cluster_122019-2_pos12_top_right_cubic_spline.csv"
# dic_fn = 'Capture 1 - Position 12_XY1576876376_Z0_T000_C0.tif'
# radial_fn = "flow_radial_122019-2_pos12_top_right.tiff"
# trialnum = "122019-2_pos12"

# save_fn = 'fig_5b.png'
# t = -7
# scl_red = 3#1.5
# scl_blue = 3.
# fontsize = 20
# navg = 6#3#6#1

# dt = 1.0 #minutes between frames
# lamda = 3.7313  #micron/pixel
# time_origin_frameno = 51
# shift_to_apparent_t0 = 1
# onset = time_origin_frameno
# save_dir = os.path.join(output_folder, save_fn)
# frm_shift = -1#-1#shift_to_apparent_t0
# frm = int(t/dt+onset) #+ frm_shift
# time_stamp = f't = {int(t//1):+}:00 min'
# def plot_snapshot_routine(message = None):
#     #import DIC snapshot as background_image
#     os.chdir(data_folder)
#     frames = pims.TiffStack_libtiff(dic_fn)

#     #assert the background_image has 1 channel (i.e. is grayscale)
#     background_frame = frames[frm]
#     assert(2==len(background_frame.shape))

#     #get the optical flow results
#     flow   = pims.TiffStack_libtiff(radial_fn)
#     of = OpticalFlowClient(dt=dt)
#     txt_avg = of.average_texture_list(flow[int(frm-navg+frm_shift):int(frm+frm_shift)])
#     # txt_avg = flow[frm].copy()
#     flow_in = txt_avg[...,0]  *scl_red#* (navg-3)
#     flow_out = txt_avg[...,1] *scl_blue#* (navg-3)

#     #import position
#     df     = pd.read_csv(cluster_fn)
#     position = (int(df.loc[frm-frm_shift].x),int(df.loc[frm-frm_shift].y))

#     #make the averaged texture not overlap!  This prevents the horrid black blobs
#     tmp_out = flow_out.copy()
#     tmp_in = flow_in.copy()
#     boo = flow_out>0
#     tmp_out[boo] -= flow_in[boo]
#     tmp_out = np.maximum(tmp_out,0.)
#     boo = flow_in>0
#     tmp_in[boo] -= flow_out[boo]
#     tmp_in = np.maximum(tmp_in,0.)
#     flow_in = tmp_in.copy()
#     flow_out = tmp_out.copy()

#     #plot and save the figure
#     kwargs = {
#         "background_frame":background_frame,"flow_in":flow_in,"flow_out":flow_out,
#         "save_dir":save_dir, "position":position,"lamda":lamda,"dt":dt, "time_stamp":time_stamp,"fontsize":fontsize, "message":message}
#     highlight_snapshot(**kwargs)
# plot_snapshot_routine()

# def plot_snapshot_routine(data_folder, dic_fn, radial_fn, frm, dt, message = None):
#     #import DIC snapshot as background_image
#     os.chdir(data_folder)
#     frames = pims.TiffStack_libtiff(dic_fn)

#     #assert the background_image has 1 channel (i.e. is grayscale)
#     background_frame = frames[frm]
#     assert(2==len(background_frame.shape))

#     #get the optical flow results
#     flow   = pims.TiffStack_libtiff(radial_fn)
#     of = OpticalFlowClient(dt=dt)
#     txt_avg = of.average_texture_list(flow[int(frm-navg+frm_shift):int(frm+frm_shift)])
#     # txt_avg = flow[frm].copy()
#     flow_in = txt_avg[...,0]  *scl_red#* (navg-3)
#     flow_out = txt_avg[...,1] *scl_blue#* (navg-3)

#     #import position
#     df     = pd.read_csv(cluster_fn)
#     position = (int(df.loc[frm-frm_shift].x),int(df.loc[frm-frm_shift].y))

#     #make the averaged texture not overlap!  This prevents the horrid black blobs
#     tmp_out = flow_out.copy()
#     tmp_in = flow_in.copy()
#     boo = flow_out>0
#     tmp_out[boo] -= flow_in[boo]
#     tmp_out = np.maximum(tmp_out,0.)
#     boo = flow_in>0
#     tmp_in[boo] -= flow_out[boo]
#     tmp_in = np.maximum(tmp_in,0.)
#     flow_in = tmp_in.copy()
#     flow_out = tmp_out.copy()

#     #plot and save the figure
#     kwargs = {
#         "background_frame":background_frame,"flow_in":flow_in,"flow_out":flow_out,
#         "save_dir":save_dir, "position":position,"lamda":lamda,"dt":dt, "time_stamp":time_stamp,"fontsize":fontsize, "message":message}
#     highlight_snapshot(**kwargs)
# ###############
# # Example Usage    
# ###############
# # plot_snapshot_routine()
