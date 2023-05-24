
''' core

Summary:
    This file contains

Example:
    Usage of

Todo:
    *
'''



''' imports '''

# filesystem navigation, system, regex
import os, sys, glob, re


# module
#from . import module_folder


''' core functions '''

def add_files(db, base_path, props):

    ''' Add File Reference to Database

        Given directory (str) and database instance (list), recursively search directory for files
        of given file extension (props); append properties (dict) to node

    Args:
        db (list): database instance as list of file nodes (dict)
        base_path (str): full directory path to search
        props (dict): file properties to store in file node

    Returns:
        (none): file node added to database instance
    '''

    # recursive directory search for all desired pl images
    file_paths =  glob.glob(base_path + '/**/*.' + props['file_ext'] , recursive = True)

    #print(file_paths)

    # iterate files
    for path in file_paths:

        # extract file names from matched file paths
        file_name = path.split('/')[-1:][0]

        # get directory paths, remove file name
        file_path = path[:-len(file_name)-1]

        # define new database node
        node = {**props, 'file_name': file_name, 'file_path': file_path}

        # store node in database
        db.append(node)


''' module image segmentation functions '''

import numpy as np
import cv2

# simple lens distortion correction
def apply_lens(img, k1 = -7.0e-6, f = 8.):

    # grid: image to distort
    # k1: lens curvature param
    # f: lens focal length

    s = img.shape

    distCoeff = np.zeros((4,1),np.float32)

    # barrel distortion
    #k1 = -7.0e-6 # negative to remove barrel distortion
    #k2 = 0.0
    #p1 = 0.0
    #p2 = 0.0

    distCoeff[0,0] = k1
    #distCoeff[1,0] = k2
    #distCoeff[2,0] = p1
    #distCoeff[3,0] = p2

    # set focal length
    #f = 8.

    # assume unit matrix for camera
    cam = np.eye(3,dtype=np.float32)

    cam[0,2] = s[1]/2.  # define center x
    cam[1,2] = s[0]/2. # define center y
    cam[0,0] = f        # define focal length x
    cam[1,1] = f        # define focal length y

    # here the undistortion will be computed
    return cv2.undistort(img.astype(np.float32), cam, distCoeff)



import skimage
import scipy

# otsu filter for threshold mask, binary closing to clean noise
def mask_clean(img, _r = 10, clean = True):

    # get intensity threshold
    _val = skimage.filters.threshold_otsu(img)

    # init binary mask
    mask = np.ones(img.shape)

    # get index pixels by threshold
    idx = np.where(img < _val)

    # zero pixels below threshold
    mask[idx] = 0.

    if clean:
        #'''
        # create disk structure element, set radius
        #_r = 10
        ele = skimage.morphology.disk(_r)

        # perform binary closing and opening on mask (clean small features)
        mask = skimage.morphology.binary_closing(mask, footprint=ele,)
        mask = skimage.morphology.binary_opening(mask, footprint=ele,)
        #'''

    # return clean mask
    return mask


''' isolate central mask feature '''

def mask_single(mask):

    # use measure to identify closed features in binary mask
    lmask = skimage.measure.label(mask, background = 0)

    ### area average floored, account for individual pixel errors
    # get image dimensions
    dd = [lmask.shape[0], lmask.shape[1]]

    # get central third
    c = lmask[ int(dd[0]/3):int(2*dd[0]/3), int(dd[1]/3):int(2*dd[1]/3) ]

    # get mean class label of centre, rounded integer of mean label excluding background (zero)
    cm = round( c[c != 0].mean() )

    # get index of pixels to zero (all non-module)
    idx = np.where(lmask != cm)

    # get final binary mask of module area pixels
    lmask[idx] = 0.

    # noramlise to binary 0/1 (account for feature label value)
    lmask = lmask/lmask.max()

    # return mask with single central feature
    return lmask


''' detect corner features, convex hull '''

def mask_corner_hull(mask, _filter = True):

    # blur and prepare mask image (type float for cv2)
    _mask = scipy.ndimage.gaussian_filter(mask, sigma = 5.).astype(np.float32)

    # set min dist between corners
    d = min([mask.shape[0], mask.shape[1]])// 10

    # upper limit number features (corners), goodness
    N = 20; g = 0.001
    #N = 20; g = 0.1

    # corner feature detection (Shi-Tomashi algorithm),
    corners = cv2.goodFeaturesToTrack(_mask, N, g, d)


    # prep corner feature list
    crns = corners.reshape(-1,2)

    # compute convex hull of contour
    #hull = scipy.spatial.ConvexHull(crns, incremental=True)
    hull = scipy.spatial.ConvexHull(crns)

    # get ordered convex hull points
    cnt = np.array(hull.points[hull.vertices,...], dtype = np.int16)

    if _filter:

        #'''
        ## find only corners (euclidian dist to image corners)
        # define corners by image range
        cr = np.stack([
            [0,0],
            [0,mask.shape[0]],
            [mask.shape[1],0],
            [mask.shape[1],mask.shape[0]]
        ])

        cns = []

        # iterate over each corner
        for c in cr[:]:

            # compute abs eucl dist of corner features from point
            dist = np.linalg.norm(cnt - c, axis = 1)

            # get closest point
            i = np.where(dist == np.min(dist))[0][0]

            # save corner location
            cns.append(cnt[i,...])

        # stack corners
        cns = np.stack(cns)
        #'''

        return cns

    else:
        return cnt



def mask_corner_hull_image(mask, _filter = True, _edge = False):

    # compute convex hull of image mask
    _hull = skimage.morphology.convex_hull_image(mask).astype(np.int8)

    # compute gradient to obtain edge contour mask
    _img = scipy.ndimage.morphological_gradient( _hull, size=(3))

    # get points of edge contour
    _cpts = np.stack( np.where(_img == 1), axis = 1)

    print(_cpts)


    # compute convex hull of contour points
    hull = scipy.spatial.ConvexHull(_cpts)

    # get ordered convex hull points, swap axes
    cnt = np.array(hull.points[hull.vertices,...], dtype = np.int16)[:,::-1]

    #print(cnt)

    if _filter:

        #'''
        ## find only corners (euclidian dist to image corners)
        # define corners by image range
        cr = np.stack([
            [0,0],
            [0,mask.shape[0]],
            [mask.shape[1],0],
            [mask.shape[1],mask.shape[0]]
        ])

        cns = []

        # iterate over each corner
        for c in cr[:]:

            # compute abs eucl dist of corner features from point
            dist = np.linalg.norm(cnt - c, axis = 1)

            # get closest point
            i = np.where(dist == np.min(dist))[0][0]

            # save corner location
            cns.append(cnt[i,...])

        # stack corners
        cns = np.stack(cns)
        #'''

        return cns

    else:
        if _edge:
            return cnt, _cpts[:,::-1]
        else:
            return cnt


import hdbscan

def lines_corners(pts, _edg):

    # input edge contour points and convec hull points
    # compute edge lines, intersection points (corners)
    # return corner points


    ''' compute points for edge lines '''

    # sum distance all points to line (adjacent point pair)
    ds = []

    # iterate over all points pairs
    for i in range(pts.shape[0]-1)[:]:
        #print(i, i+1)

        # get adjacent points
        p1 = pts[i,:].astype(np.float32)
        p2 = pts[i+1,:].astype(np.float32)

        # compute distance of each point from line (two adjacent points)
        #d = np.abs( np.cross(p2-p1,pts-p1)/np.linalg.norm(p2-p1) )

        # compute distance of all edge point (downsample)
        d = np.abs( np.cross(p2-p1,_edg[::10]-p1)/np.linalg.norm(p2-p1) )
        #print(d.sum())

        # remove point pair (zeros), apply weighting by distance (higher weight to points close to line)
        _d = d[np.where(d != 0)]**0.8

        # store sum total distance
        ds.append(np.sum(_d)/len(_d))


    # get index max distance (a corner)
    i = np.where(ds == np.max(ds))[0][0]

    # re-order to start at corner
    ds = np.roll(ds, -i)

    # roll points to match
    _pts = np.roll(pts, -i, axis = 0)


    ''' compute point pair line distance derivatives '''

    # savitsky golay filter for smooth and derivatives
    _w = 5 # window
    _o = 3 # poly order

    # compute filter fit, derivatives
    sds = scipy.signal.savgol_filter(ds, _w, _o, deriv=0)
    dds = scipy.signal.savgol_filter(ds, _w, _o, deriv=1)
    ddds = scipy.signal.savgol_filter(ds, _w, _o, deriv=2)

    # compute corner likelyhood weighting (sum abs 1,2 order derivatives)
    rr = np.abs(dds)+np.abs(ddds)


    ''' compute edge point clusters - hdbscan '''

    # stack index, distance, derivative
    X = np.stack([
        list(range(len(ds))),

        # midpoint coordinates of point pair
        (_pts[:-1,1]+(_pts[:-1,1] - _pts[1:,1]))/_pts[:,1].max(),
        (_pts[:-1,0]+(_pts[:-1,0] - _pts[1:,0]))/_pts[:,0].max(),

        # norm distance, derivative measure
        #ds,
        len(ds)*(ds-np.min(ds))/np.max(ds),
        rr/np.max(rr),

        #len(ds)*np.abs(dds)/np.abs(dds).max(),
        #len(ds)*ds/np.max(ds),
    ], axis = 1)

    # compute 4 edge clusters with k-means
    #kmeans = KMeans(n_clusters = 4, random_state=0).fit(X)
    #lbls = kmeans.labels_

    # compute edge clusters with hdbscan
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(X)

    # obtain cluster point pair labels, certainty
    lbls = clusterer.labels_
    probs = clusterer.probabilities_

    print(np.unique(lbls))

    ## filter points
    # cluster label probability above threshold certainty
    idx = np.where(probs > .90)[0]

    # get only points where derivative is below std dev
    #idx = np.where(rr < np.std(rr)*1.0)[0]


    ''' find minimum pairs '''

    js = []

    # iterate over sides (labels excluding -1)
    for l in [ _ for _ in np.unique(lbls) if _ != -1 ]:

        # get label index
        k = list(np.where(lbls == l)[0])

        # filter outliers
        k = [_ for _ in k if _ in idx]

        # compute min pair index
        j = [ _ for _ in k if ds[_] == np.min(ds[k]) ][0]
        print(j)

        js.append(j)

    # sort order pairs (sequential adjacent edges)
    _js = list(np.array(js)[np.argsort(js)])


    ''' compute edge lines from optimal point pairs '''

    def line_intersection(line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        #if div == 0:
        #   raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y


    ''' compute corner points from line intersection '''

    crns = []

    # iterate over all points pairs, append first at end for closure)
    ii = list(np.array(_js)+1)
    ii += [ii[0]]
    for i in range(len(ii)-1):
        #print(i, i+1)

        # get adjacent points
        p1 = _pts[ii[i],:].astype(np.float32)
        p2 = _pts[ii[i]+1,:].astype(np.float32)

        # get second adjacent point pair
        p3 = _pts[ii[i+1],:].astype(np.float32)
        p4 = _pts[ii[i+1]+1,:].astype(np.float32)

        crns.append( line_intersection((p1, p2), (p3, p4)) )
        print( crns[-1] )

    # stack into array
    crns = np.stack(crns)

    # sort and order corners for transform
    _crns = crns[np.argsort(crns, axis = 0)[:,1]][[0,2,1,3]]


    ## return computed corner points
    return _crns


''' deskew, crop, segment '''


def deskew_crop(img, cns):

    # input raw image to deskew, corners from mask
    ## deskew image and crop to mask area from corners

    ''' get scale of module for persepctive transform '''

    # length top-left edges, aspect ratio
    h = np.linalg.norm(cns[0] - cns[1], axis = 0)
    w = np.linalg.norm(cns[0] - cns[2], axis = 0)
    asp1 = (w/h)

    # length bottom-right edges, aspect ratio
    h2 = np.linalg.norm(cns[2] - cns[3], axis = 0)
    w2 = np.linalg.norm(cns[1] - cns[3], axis = 0)
    asp2 = (w2/h2)

    # use average aspect ratio
    asp = np.mean([asp1,asp2])
    print(asp1, asp2, asp)

    ''' scale of transform, upsample '''

    # expand image points matrix
    wcr = np.stack([
        [0,0],
        [0,img.shape[1]],
        [img.shape[1]*asp,0],
        [img.shape[1]*asp,img.shape[1]]
    ])

    # compute perspective transform from detected corners to square scaled canvas with consistent aspect ratio
    M = cv2.getPerspectiveTransform(
        cns.astype(np.float32),
        wcr.astype(np.float32),)

    ''' image de-skew '''

    # prepare size tuple (type for cv2)
    shp = tuple( (wcr.copy()[-1,...]).astype(np.int16))
    print(shp)

    # perform perspective transform on raw image
    wrp = cv2.warpPerspective(img, M, shp )

    return wrp





''' compute segmentation grid '''

def edge_contrast(img, _size = 10):

    ''' prepare edge contrast image for segmentation '''

    ### assumes input module image is cropped and free of significant rotation or skew

    #gwrp = ndimage.morphological_gradient( gaussian_filter(wrp, sigma = 3.), size=(10))
    _wrp = scipy.ndimage.morphological_laplace( scipy.ndimage.gaussian_filter(img, sigma = 3.), size=_size)


    # get intensity threshold
    _val = skimage.filters.threshold_otsu(_wrp)

    # init binary mask
    fimg = np.ones(_wrp.shape)

    # get index pixels by threshold
    idx = np.where(_wrp < _val)

    # zero pixels below threshold
    fimg[idx] = 0.


    return fimg


def line_mask(fimg):

    ''' hough transform to find primary cell edge lines '''

    ## effective filter for edge line by total length

    # initialise blank image mask
    lmask = np.zeros(fimg.shape)

    # Line finding using the Probabilistic Hough Transform
    lines = skimage.transform.probabilistic_hough_line(fimg,
                                     #theta = angles,
                                     threshold = 10,
                                     #line_length = 900,
                                     line_length = int(0.8*min(fimg.shape)),
                                     line_gap = int(0.02*min(fimg.shape)))
    #print(len(lines))

    # iterate over lines, unpack points
    for p0, p1 in lines:

        # get line pixels (with AA)
        _ = skimage.draw.line_aa(p0[0], p0[1], p1[0], p1[1],)

        # draw line to mask
        lmask[_[1],_[0]] += _[2]


    return lmask


def edge_autocorr(lmask, k):

    axs = np.mean(lmask, axis = k)

    #ax[0].plot(_,)

    # autocorrelation of image axis sum
    ac = np.correlate(axs, axs, mode = 'full')

    # highpass filter to remove dc component
    sos = scipy.signal.butter(3, .2, 'hp', output='sos')
    _ac = np.abs( scipy.signal.sosfilt(sos, ac))

    # gaussian filter to clean signal, smooth peaks
    _ac = scipy.ndimage.gaussian_filter1d(_ac, 5.)

    return _ac


def period_map(_ac, sh = 20):


    ''' compute peak period map with window function '''

    # window width (+- pixels)
    #sh = 20

    # get one full autocorrelation of image axis
    tt = _ac[_ac.shape[0]//2 - sh:]
    #tt = g__[:]


    _map = []
    # iterate over periods N (pixels)
    for i in list(range(tt.shape[0]//4))[1:]:

        _t = []
        # include start pixel shift
        #for j in list(range(tt.shape[0]//30))[1:]:
        for j in list(range(sh*2))[:]:

            # get mean of every Nth sample
            _ = tt[j:][::i]

            _ = np.sum(_)/_.shape[0]
            #_ = np.mean(_/_.shape[0])/tt.mean()
            _t.append( _ )

        # peaks at period corresponding to pixels for given pixel shift
        _map.append(np.array(_t))

    # compose 2d map of period by window shift
    _map = np.stack(_map, axis = 1)


    return _map


def period_phase(_map, sh):

    ''' window average period '''

    # get window average
    per = np.max(_map, axis = 0)# / np.sum(_map, axis = 0)

    # compute cumulative sum of average period axis
    lll = np.cumsum(per) / np.arange(per.shape[0])

    # normalise to cumnsum to fix baseline bias
    _per = per / lll

    # get max peak (pixel period)
    m = np.where(_per == _per.max())[0][0]
    print(m)


    ''' window phase shift '''

    # get window average
    pha = np.mean(_map, axis = 1)

    # get peak position, adjust to zero
    _m = np.where(pha == pha.max())[0][0] - sh
    print(_m)


    return _per, m, pha, _m



def cell_spacing(lmask, sh = 20):

    ''' autocorrelation of image axis sum for cell line spacing '''

    ## effective filter for edge line by orientation (only image axis aligned edges)

    period = np.ones(2)
    shift = np.zeros(2)

    # iterate over image axis
    for k in [1,0]:


        _ac = edge_autocorr(lmask, k)

        # get edge autocorrelation map
        _map = period_map(_ac, sh)

        # compute period and phase shift
        _per, m, pha, _m = period_phase(_map, sh)

        # save period
        period[k] = m

        # save shift
        shift[k] = _m


    # return computed period and shift for axes
    return period, shift


''' prepare grid '''

def gen_grid(shape, period, shift, pad = 5, edge = False):

    # s: image shape (tuple r x c ints)
    # n: number lines (odd int)
    # w: cell width (int)
    # h: cell height (int)


    # compute n cell in row x column
    N = np.round(np.array(shape)/np.array(period)).astype(np.int16)
    print(N)

    w = int(period[0])
    h = int(period[1])

    # horizontal,vertical line spacing as function image shape, apply centre shift, discard limits
    #h_ = (np.arange(0, shape[0], h)-shift[0]).astype(np.int16)[1:]
    #w_ = (np.arange(0, shape[1], w)-shift[1]).astype(np.int16)[1:]
    h_ = (np.arange(0, shape[0], h)).astype(np.int16)[1:]
    w_ = (np.arange(0, shape[1], w)-shift[1]).astype(np.int16)[1:]
    print(h_,w_)

    # initialise grid mask
    grid = np.zeros(shape).astype(np.int16)

    # iterate, draw lines on mask with padding (thickness)
    for _h in h_:
        grid[int(_h-pad-shift[0]):int(_h+pad-shift[0]),0:int(shape[1])] = 1
    for _w in w_:
        grid[0:int(shape[0]),int(_w-pad-shift[1]):int(_w+pad-shift[1])] = 1

    # pad edges
    if edge is not None:
        grid[:,:edge] = 1
        grid[:,-edge:] = 1
        grid[:edge,:] = 1
        grid[-edge:,:] = 1

    # return grid mask
    return grid


def extract_cells(wrp, grid, _filter = False):

    ''' extract and store cell images '''

    # use measure to identify closed features in binary mask
    lbld = skimage.measure.label(grid, background = 1)

    n = np.unique(lbld)

    cells = []

    for l in range(lbld.max()+1)[1:]:

        # get cell pixels by label id
        idx = np.where(lbld == l)

        # crop cell image by min/max bounds
        _cell = wrp[ idx[0].min():idx[0].max()+1,
                    idx[1].min():idx[1].max()+1 ]

        cells.append(_cell)

    # filter edge trim
    if _filter:

        print(len(cells))

        # get max dimensions of cells
        _shape = np.max(np.stack(list(set([ _cell.shape for _cell in cells ]))), axis = 0)

        # filter for only full cells (exclude if either dimension less than 90% max)
        cells = [ _ for _ in cells if -np.min(np.array(_.shape) - _shape) < min(_shape)*0.9 ]

        print(len(cells))

    # return cell images
    return cells



''' cell dataset prep, augmentation '''

def cells_pad(cells, _shape = None, _buffer = None ):
    ''' Input set cells, Output uniform shape with padding

    '''

    if _shape is None:
        # get max shape all cells in set
        _shape = np.max(np.stack(list(set([ _cell.shape for _cell in cells ]))), axis = 0)
        print('shape ',_shape)

    if _buffer is None:
        # set buffer from fraction shape
        _buffer = (np.array(_shape)*0.1).astype(np.int32)
        print('buffer ',_buffer)

    __ = []

    # iterate cells
    for _cell in cells:

        # compute padding
        _pad = _shape - np.array(_cell.shape) + _buffer
        #print('pad ',_pad)

        #print(_cell.shape)
        # pad cell out to max shape plus buffer
        #_ = np.pad(_cell, ((_pad[0],_pad[0]),(_pad[1],_pad[1])), 'constant' )
        #_ = np.pad(_cell, ((_pad[0]//2,_pad[0]//2),(_pad[1]//2,_pad[1]//2)), 'constant' )
        _ = np.pad(_cell, [ (_pad[l]//2,_pad[l]//2) for l in range(len(_cell.shape)) ] , 'constant' )
        #_ = np.pad(_cell, ((0,_pad[0]),(0,_pad[1])), 'constant' )
        #print(_cell.shape)

        # fix uneven padding
        if not list(_cell.shape) == list(_shape + _buffer):

            _diff = _shape - np.array(_.shape) + _buffer
            #_ = np.pad(_, ((0,_diff[0]),(0,_diff[1])), 'constant' )
            _ = np.pad(_, [ (0,_diff[l]) for l in range(len(_cell.shape)) ], 'constant' )

        __.append(_)

    # return padded cells
    return __


def cells_pp(cells, _sigma = 1.0):
    ''' smooth, normalise cell images

        gaussian cmooth filter prior
        normalise intensity to bound 0,1 using full set

     '''

    # gaussian smoothing
    _cells = [ scipy.ndimage.gaussian_filter(_, _sigma) for _ in cells ]

    # max of smoothed set images
    _m = min([ np.min(_) for _ in _cells ])
    _M = max([ np.max(_) for _ in _cells ]) - _m
    print(_m, _M)

    # normalise and downsample cell images (8 bit)
    #_cells = [ (((_-_m)/_M)*2**8).astype(np.int16) for _ in _cells ]
    _cells = [ (((_-_m)/_M)).astype(np.float32) for _ in _cells ]

    return _cells


def cell_ds(cells, _sigma = 1.0, _factor = 10):
    ''' downsample cell images '''
    # get image size
    #_shape = np.array(cells[0].shape)

    # gaussian smoothing
    #_cells = [ scipy.ndimage.gaussian_filter(_, _sigma) for _ in cells ]

    # max of smoothed set images
    #_n = np.max(_cells)

    # normalise and downsample cell images (8 bit)
    #_cells = [ scipy.ndimage.zoom((_/_n)*2**8, _factor**-1).astype(np.int16) for _ in _cells ]
    _cells = [ scipy.ndimage.zoom(_, _factor**-1) for _ in cells ]

    return _cells




''' module simulation with pyspice '''


import PySpice.Logging.Logging as Logging

from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import *

logger = Logging.setup_logging()


def gen_cell(name, q = 10@u_A, rs = 1@u_mOhm, rp = 10@u_kOhm, j01 = 1e-8, ni2 = 2, j02 = 1e-12, ni1 = 1):
    ''' generate solar cell subcircuit '''

    # init cell subcircuit
    #cell = SubCircuit(name, 'X{}_in'.format(name), 'X{}_out'.format(name))
    cell = SubCircuit(name, 't_out', 't_in')

    # define cell elements
    cell.model('d1', 'D', IS = j01, N = ni1)
    cell.model('d2', 'D', IS = j02, N = ni2)

    # init cell elements
    cell.I(1, 't_load', 't_in', q)
    cell.R(2, 't_load', 't_out', rs)
    cell.R(3, 't_in', 't_load', rp)
    cell.Diode(4, 't_in', 't_load', model = 'd1')
    cell.Diode(5, 't_in', 't_load', model = 'd2')

    # return cell subcircuit
    return cell


def simulate(circuit):
    ''' current-voltage sweep simulation on module circuit '''

    # init simulation
    simulator = circuit.simulator(temperature = 25, nominal_temperature = 25)

    # set dc sweep
    analysis = simulator.dc( Vinput = slice(0,50,0.01) )

    # extract current, voltage
    I = np.array(analysis.Vinput)
    V = np.array(analysis.sweep)

    # compute performance params
    P = I*V
    #MPP = max(P)
    #VMP = V[P.argmax()]
    #IMP = I[P.argmax()]
    #VOC = max(V[I>=0])
    #ISC = I[V.argmin()]
    #FF = MPP/(VOC * ISC)

    # return current, voltage
    return I,V


def gen_module(rows = 3, cols = 3, qmap = None, jmap = None):
    ''' build module circuit '''

    if qmap is None:
        # define intensity map, [0,1] current factor
        qmap = np.ones( (rows,cols) )


    if jmap is None:
        # define j0 map, scale factor
        jmap = np.ones( (rows,cols) )

    # init module circuit
    module = Circuit('module')

    # store list cells, cell subcircuits
    _cells = []
    cells = []

    # iterate rows in module
    for row in range(rows):

        # get list list
        _cols = list(range(cols))

        # invert on even rows
        if (row+1)%2 == 0:
        #    #print('invert')
            _cols = _cols[::-1]

        # iterate columns
        for col in _cols:

            # define cell name
            _cell = 'cell_{}_{}'.format(row, col)

            # save ordered cells list
            _cells.append(_cell)

            # generate new cell, current from intensity map
            cell = gen_cell(name = _cell, q = qmap[row, col] * 10@u_A, j01 = jmap[row, col] * 1e-13)

            # store cell in list
            cells.append(cell)

            # add cell to module circuit
            module.subcircuit(cell)

    # define terminals (vlt)
    module.V('input', module.gnd, 1, 0)

    # connect input cell to terminal (pos)
    #circuit.X('{}_in'.format(_cells[0]), _cells[0], 1, 2)

    # series connect cells
    for i in range(len(_cells))[:]:

        module.X(_cells[i], _cells[i], i+1, i+2)

        #module.X('{}_in'.format(_cells[i]), _cells[i], i+1, i+2)
        #module.X('{}_out'.format(_cells[i]), _cells[i], i+2, i+3)

        #if i < len(_cells)-1:
        #    module.X('{}_out'.format(_cells[i]), _cells[i], i+2, i+3)
        #else:
        #    module.X('{}_out'.format(_cells[i]), _cells[i], i+2, 0)



    # module series resistor
    module.R('meas', len(_cells)+1, 0, 1@u_mOhm)

    #print(module.str())

    # return module circuit
    return module





def stack_cell_imgs(_cells):

    ''' stack cells, compute average images, stats '''

    # get max,min cell dimension
    _max = np.max(np.stack(list(set([ _.shape for _ in _cells ]))), 0)#.max()
    _min = np.min(np.stack(list(set([ _.shape for _ in _cells ]))), 0)#.min()
    print(_max, _min, _max-_min)
    _dx = _max-_min
    print(_dx)

    # stack cells, pad to max dims, trim to min dims
    stack = np.stack(cells_pad(_cells, _shape = _max, _buffer = (0,0)))[:,_dx[0]//2:-_dx[0]//2,_dx[1]//2:-_dx[1]//2]
    print(stack.shape)


    # normalise stack (range zero to one, outliers beyond)
    _stack = (stack.copy() - np.median(np.min(stack,0)) )
    _stack = _stack/np.median(np.max(_stack,0))

    ## compute stack average, smoothed max
    #_cell = (np.mean(stack,0) - np.min(stack))/np.max(stack)
    #_avg = np.mean(_stack,0)
    #_avg = np.median(_stack,0)
    #_best = scipy.ndimage.gaussian_filter(np.max(_stack,0), sigma = 1.5)

    #print('\n avg cell mu {:.3f} sig {:.3f}'.format(np.mean(_avg), np.std(_avg)))

    # select reference
    _ref = scipy.ndimage.gaussian_filter(np.median(_stack,0), sigma = .5)
    #_ref = np.mean(_stack,0)


    ''' compute absolute stats '''

    _min = np.median(np.min(stack,0))
    _max = np.median(np.max(stack,0))
    _med = np.median(np.median(stack,0))
    _men = np.median(np.mean(stack,0))

    print('min {:.3f} \t max {:.3f} \tmedian {:.3f} \tmean {:.3f}'.format(_min, _max, _med, _men))

    _ptp = np.median(np.ptp(stack,0))
    _std = np.median(np.std(stack,0))

    print('ptp {:.3f} \t std {:.3f}'.format(_ptp, _std,))

    # set absolute scaling factor
    _fact = _ptp/_med
    #_fact = _std/_med
    print('scale factor {:.3f}'.format( _fact))

    # compile abs stats
    abs_stats = {'min': _min,
                 'max': _max,
                 'med': _med,
                 'men': _men,
                 'ptp': _ptp,
                 'std': _std,
                }

    # return cell image stack, reference image, abs scale factor, stats
    return _stack, _ref, _fact, abs_stats


def compute_residual(_stack, _ref, _fact):

    ''' compute each cell residual to average, high/low variance '''

    res = []

    # iterate over cells in stack
    for k in range(_stack.shape[0]):
        #print(k)

        # select individual cell
        _cell = _stack[k,...]

        # smooth gaussian blur
        _cell = scipy.ndimage.gaussian_filter(_cell, sigma = .5)

        # compute residual image less avg-to-max residual
        _res = (_cell - _ref) #- (_best-_avg)
        #_res = (_cell - np.mean(_stack,0))

        #print('cell mu {:.3f} sig {:.3f}'.format(np.mean(_cell), np.std(_cell)))

        # store sum of residual
        res.append(_res)

    # stack cell residual images
    res = np.stack(res)

    ''' compute analysis, states on residual '''

    # sum residual, scale to reference, apply absolute scale factor
    _res = 10*_fact*np.sum(res,(1,2))/np.sum(_ref)

    print('residual mean {:.3f}, std dev {:.3f}, min {:.3f}'.format(np.mean(_res), np.std(_res), np.min(_res)))

    # shift residual to zero median, zero all positive values
    _std = np.stack([ res[i] - np.median(res[i]) for i in range(res.shape[0]) ])
    _std[np.where(_std>0)] = 0.
    _std = np.std(_std**2, (1,2))*10

    print('variance mean {:.3f}, std dev {:.3f}, min {:.3f}'.format(np.mean(_std), np.std(_std), np.min(_std)))

    # shift residual to zero median, zero all negative values
    _hot = np.stack([ res[i] - np.median(res[i]) for i in range(res.shape[0]) ])
    _hot[np.where(_hot<0)] = 0.
    _hot = np.std(_hot**2, (1,2))*10

    print('variance mean {:.3f}, std dev {:.3f}, min {:.3f}'.format(np.mean(_hot), np.std(_hot), np.min(_hot)))


    # return residual stack, residuals (degrade), low variance (cracks), high variance (hotspots)
    return res, _res, _std, _hot


#def classify_cells(res, _res, _std, _hot):
def classify_cells(_res, _std, _hot):

    ''' identify degraded, defective cells, hotspots by thresholds '''

    # dead and degarded cells
    _dead = [ i for i in range(len(_res)) if _res[i] < -2. ]
    _degr = [ i for i in range(len(_res)) if _res[i] < -.5 and i not in _dead ]

    # defective and damaged cells
    _damg = [ i for i in range(len(_std)) if _std[i] > .5 ]
    _defc = [ i for i in range(len(_std)) if _std[i] > .3 and i not in _damg ]

    # hot cell and hot-spot cells
    _hots = [ i for i in range(len(_hot)) if _hot[i] > .5 ]
    _hotc = [ i for i in range(len(_res)) if _res[i] > 1. and i not in _hots ]


    print('dead: {}, damg: {}, defc: {}, degr: {}, hots: {}, hotc: {}'.format(
        len(_dead), len(_damg), len(_defc), len(_degr), len(_hots), len(_hotc)) )

    #print('std residual {:.3f}'.format(np.std(res)))

    # return cell index each classfied defect
    return _dead, _damg, _defc, _degr, _hots, _hotc


