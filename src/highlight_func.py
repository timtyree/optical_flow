from cv2 import drawMarker
import cv2 as cv, numpy as np, pims

def highlight(position, flow_in, flow_out, background_frame, r_c_mat=None,
    hue_scale=80, hue_scale_red=70, hue_scale_blue=90,
    r_thresh  = 0,
    **kwargs):
    '''visualize overlay of outward optical flow over the background channel.
    for masking the highlight to not consider r_thresh pixels from the center,
    the center must also be provided by r_c_mat as returned by get_r_hat_mat

    Example Usage:
img = highlight(position=(x_coord, y_coord), flow_in=flow_in, flow_out=flow_out, background_frame=frame_2)
    '''
    #TODO: use more sophisticated/correct kwargs parsing as done elsewhere
    #TODO: add r_thresh set method to argument of highlight
    mfo = flow_out
    mfi = flow_in
    width=background_frame.shape[0]
    height=background_frame.shape[1]
    filt = np.zeros([width,height,3]) #3 channels for rgb
    img  = pims.to_rgb(background_frame)

    if r_c_mat == None:
        r_filter = 1
    elif r_c_mat.shape == (width, height):
        r_filter = (r_c_mat>r_thresh)
    else:
        r_filter = 1

    #TODO(better): use colormap such as # im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_COOL)
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
