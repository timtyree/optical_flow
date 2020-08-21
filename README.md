<h1>
Optical Flow for Cell Motion independent of Single Cell Tracking
</h1>
<h3>
The Rappel Lab at the University of California, San Diego<br>
This repository was created/is maintained by Timothy Tyree<br>
The manuscript associated with this repository may be found [manuscript in review]<br>
  </h3>

<p>This repository contains a module of python functions. It is an ongoing project, and you are welcome to collaborate! The purpose of this repository is for computing and analyzing dense optical flow from raw microscopy video data.  The preprocessing is tuned for a differential interference contrast microscopy experiemental setup, and the analysis is geared towards video data that exhibits cell motion towards/away from a trajectory that can be taken to be stationary, for simplicity.  I encourage you to apply these analytical methods to other applications.  It primarily uses a dense inverse search (DIS) optical flow algorithm (see optical_flow.OpticalFlowClient.calc_flow).</p>

<p>This module supports spatial and/or temporal averging functionality for flow textures/videos, calculates inward/outward and/or clockwise/counterclockwise flow from a user defined pixel positions or from a user defined trajectory stored in a sorted pandas DataFrame object (see dot_flow and cross_flow).  That DataFrame has fields 'frame' indicating frame number, 'x' indicating x pixel coordinate, and 'y' indicating y pixel coordinate.</p>

<p>This module supports functionality for averaging textures over a user defined annulus (see annulus_sum and annulus_avg) to produce time series data that can then be stored to '.csv'.</p>

<p>This module supports video marking functionality for marking the user defined trajectory with a "x" and coloring inward/outward flow red/blue.</p>

<p>All functionality and frequently reused parameters are held in the context of an OpticalFlowClient object.  It would be better if the functionality was distributed into modules.</p>

<p>Microscopy video data formats currently  supported are '.tiff'/'.tif' stacks (TODO: generalize inputs to a .mpg or .avi or .mp4 or .mov formats).  Video output functionality is currently supported with '.tiff' stacks compatable with ImageJ, '.avi', and '.mov'.  My inputs were differential interference contract (DIC) microscopy videos that had edges detected by Scharr transform (see optical_flow.OpticalFlowClient.preprocess).</p>

<p>I made this module for my physics PhD research on chemotaxis, and although it serves my purposes, it is reasonably well-documented and riddled with "#TODO: " comments describing simple and clear incremental improvements.  Test cases should also be developed. If you're interested in helping make this module more useful to more people, I encourage you to fork this repository.</p>

## Example Usage

### Example 1: 
```
#calculate texture of radial unit vectors, r_hat_mat, and radial vectors, r_c_mat<br>
from optical_flow import *<br>
dt = 1 #1 minute between each frame<br>
of = OpticalFlowClient(dt=dt, width = 512, height = 512)<br>
x_coord, y_coord   = tuple(100, 150)<br>
r_hat_mat, r_c_mat = of.get_r_hat_mat([x_coord,y_coord])<br></p>
```

### Example 2:
```
#load and mark two frames with the apparent inward/outward flow from a pixel position<br>
frame1       = of.imread('data/test_frm_450.png')<br>
frame2       = of.imread('data/test_frm_451.png')<br>
edge1, edge2 = of.preprocess([frame1, frame2])<br>
flow         = of.calc_flow(edge1, edge2)<br>
flow_in, flow_out  = of.dot(flow, r_hat_mat)<br>
img = of.highlight(position=(x_coord, y_coord), flow_in=flow_in, flow_out=flow_out, <br>background_frame2)
```
