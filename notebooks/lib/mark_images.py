import pandas as pid
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
import pims
import time

# def __init__(self):  
    #TODO: get scales from log file or perhaps .tiff metadata

    # return self

# def make_input_image(self, img, message = None, time_stamp = None, save_file_name = None):    
def make_input_image(img, save_file_name, message = None, time_stamp = None, x_start = 440, y_start = 70, lamda = 1.33, uml=50, width = 6):
    """mark the input image, img, at position needletip with a 50µm white scale bar.  
    time_stamp and message are passed as content to plt.text.
    marked image saved to save_file_name.
    DONE: checked that the output still has 512x512 pixels.
    x_start and y_start are for the scale bar position and have units of pixels.
    time stamp might be '+{} min '.format(str(int(np.around(tmax,0))))', for instance.
    lamda is lengthscale in microns per pixel.
    uml is the length of the scale bar in microns
    width is the width of the scalebar"""

    #parse inputs
    # lamda = 1.33 #microns per pixel for 10X magnification
    ycoord = y_start#pixel
    xstart = x_start#pixel
    # dt    =  1   #minutes per frame
    # self.lamda = 1.33 #microns per pixel
    # self.dt    =  1   #minutes per frame
    # self.frm   = -1
    # lamda = self.lamda#1.33

    #send grayscale/1_channel images to rgb
    if len(img.shape)==2:
        img = pims.to_rgb(img.copy())

    mydpi=128#64
    mult = 1.32642487

    #calculate scale bar length
    s = 3
    # uml = 50#um
    pxl = uml/lamda#pxl
    
    #plot image and mark
    fig = plt.figure(num=None, frameon=False, figsize=(mult*img.shape[0]/mydpi, mult*img.shape[1]/mydpi), dpi=mydpi, facecolor='w', edgecolor='k')
    #alternative initialization      #fig = plt.figure(num=None, figsize=(8, 8), dpi=64, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.axis('off')

    ##    add scale lines
    l1 = lines.Line2D([xstart, xstart+pxl], [ycoord, ycoord], color='white', linewidth=width, solid_capstyle='butt')
    fig.lines.extend([l1])
    
    ##  add text boxes
    fontsize = 18
    if message is not None:
        ax.text(x=15, y=50, c='white', s=message, weight='bold', fontsize=fontsize, horizontalalignment='left', verticalalignment='center')
    if time_stamp is not None:
        ax.text(x=15, y=20, c='white', s=time_stamp, weight='bold', fontsize=fontsize, horizontalalignment='left', verticalalignment='center')
    
    plt.savefig(save_file_name, bbox_inches='tight', pad_inches=0.)
    return True

def make_input_image_list(time_stamp_lst,file_name_pic_lst):
    '''passes arguments to make_input_image, returns True.'''
    start = time.time()
    for time_stamp, file_name_pic in zip(time_stamp_lst, file_name_pic_lst):
        save_file_name = file_name_pic.replace('.png', '_formatted.png')
        img = pims.image_reader.imread(file_name_pic)
        img = pims.frame.Frame(img)
        make_input_image(img, save_file_name, message=None, time_stamp=time_stamp, y_start = 490);
    end = time.time()
    print('{} seconds elapsed marking images'.format(np.around(end-start,1)))
    return True
    # def mark_input_image(self, img, position = Nonecolor = None, time_stamp = None, save_file_name = None):)
    """mark the input image, img, at position needletip with an x, paint cell 
    trajectories in df and cluster trajectories in dfc.  Save file name is name.
    TODO: check that the output still has 512 pixels.  mydpi changed.
    cluster_df is the dataframe argument for dfc. needle position is in df."""
    #optional mark image
    #     # drawMarker(img, position = (needletip[0],needletip[1]), color = (255,255,0), markerType = cv.MARKER_CROSS , markerSize = 50, thickness = 2, line_type = cv.LINE_AA)
    #     # plt.imshow(img)#, 'gray')#, resample=False)
    #     return self

    # def select_image(self, image_file_name):



    #     return image