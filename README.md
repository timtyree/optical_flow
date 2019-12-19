<h4>
* Python       :   Optical Flow for Cell Motion independent of Cell Tracking<br>
*<br>
* PROGRAMMER   :   Timothy Tyree<br>
* DATE         :   Fri 22 Nov 2019 <br>
* PLACE        :   Rappel Lab @ UCSD, CA<br>
</h4>

<p>This is a module that is an ongoing project made for calculating and analyzing dense optical flow in microscopy video data that exhibits cell motion towards/away from a trajectory or around a trajectory.  It can clearly be generalized to other applications.  It primarily uses a dense inverse search (DIS) optical flow algorithm (see calc_flow).  </p>

<p>This module supports spatial and/or temporal averging functionality for flow textures/videos, calculates inward/outward and/or clockwise/counterclockwise flow from a user defined pixel positions or from a user defined trajectory stored in a sorted pandas DataFrame object (see dot_flow and cross_flow).  That DataFrame has fields 'frame' indicating frame number, 'x' indicating x pixel coordinate, and 'y' indicating y pixel coordinate.</p>

<p>This module also supports functionality for averaging textures over a user defined annulus (see annulus_sum and annulus_avg) to produce time series data that can then be stored to '.csv'.</p>

<p>This module supports video marking functionality for marking the user defined trajectory with a "x" and coloring inward/outward flow red/blue.</p>

<p>All functionality and frequently reused parameters are held in the context of an OpticalFlowClient object.</p>

<p>Microscopy video data formats currently  supported are '.tiff'/'.tif' stacks (TODO: generalize inputs to a .mpg or .avi or .mp4 or .mov formats).  Video output functionality is currently supported with '.tiff' stacks compatable with ImageJ, '.avi', and '.mov'.  My inputs were differential interference contract (DIC) microscopy videos that had edges detected by scharr transform (see preprocess).</p>

<p>I made this module for my physics PhD research on chemotaxis, and although it serves my purposes, it is reasonably well-documented and riddled with "#TODO: " comments describing simple and clear incremental improvements.  Test cases should also be developed. If you're interested in helping make this module more useful to more people, I strongly encourage you to fork me on GitHub at TimtheTyrant.  Have a nice day!</p>

## Examples

<p>Example 1: 
#calculate texture of radial unit vectors, r_hat_mat, and radial vectors, r_c_mat<br>
from optical_flow import *<br>
dt = 1 #1 minute between each frame<br>
of = OpticalFlowClient(dt=dt, width = 512, height = 512)<br>
x_coord, y_coord   = tuple(100, 150)<br>
r_hat_mat, r_c_mat = of.get_r_hat_mat([x_coord,y_coord])<br></p>

<p>Example 2:
#load and mark two frames with the apparent inward/outward flow from a pixel position<br>
frame1       = of.imread('data/test_frm_450.png')<br>
frame2       = of.imread('data/test_frm_451.png')<br>
edge1, edge2 = of.preprocess([frame1, frame2])<br>
flow         = of.calc_flow(edge1, edge2)<br>
flow_in, flow_out  = of.dot(flow, r_hat_mat)<br>
img = of.highlight(position=(x_coord, y_coord), flow_in=flow_in, flow_out=flow_out, <br>background_frame2)</p>
