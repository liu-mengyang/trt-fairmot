import cv2


"""
Compute parameters for letterbox

Parameters
----------
raw_shape : list
    The shape of raw image formatted as [height, width]
target_shape : list
    The shape of target image formatted as [height, width]

Returns
-------
ratio : float
    The letterbox ratio.
dw : int
    Padded width.
new_shape : tuple
    The size for resizing formatted as [width, height].
dh : int
    Padded height.

"""
def letterbox_parameter_computing(raw_shape, target_shape):
    ratio = min(float(target_shape[0] / raw_shape[0]), float(target_shape[1] / raw_shape[1]))
    new_shape = (round(raw_shape[1] * ratio), round(raw_shape[0] * ratio))
    dh = (target_shape[0] - new_shape[1]) / 2
    dw = (target_shape[1] - new_shape[0]) / 2
    return ratio, new_shape, dw, dh

"""
Resize a rectangular image to a padded rectangular with the same aspect ratio.

Parameters
----------
img : ndarray
    The input image which needs to be resized, the shape of img is [height, width].
height : int
    The height of the target image.
width : int
    The width of the target image.
color : tuple
    The color of padding by RGB color code.

Returns
-------
img : ndarray
    The output image which has been padded.
ratio : float
    The ratio of resizeing according to the minimize one between with height and width.
dw : int
    The width of horizental padding.
dh : int
    The height of vertical padding.

"""
def letterbox(img, height=608, width=1088, color=(127.5, 127.5, 127.5)):  
    [h0, w0] = img.shape[:2]  # shape = [height, width]
    ratio, new_shape, dw, dh = letterbox_parameter_computing(img.shape[:2], [height, width])
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh 

"""
Inverse map the coordinates from letterboxed bbx to raw bbx.

Parameters
----------
bbx : tuple
    Letterboxed bbx formatted as (l, t, r, b)
ratio : float
    The ratio of resizeing according to the minimize one between with height and width.
dw : int
    The width of horizental padding.
dh : int
    The height of vertical padding.

Returns
-------
bbx : tuple
    Invered map bbx formatted as (l, t, r, b)

"""
def letterbox_bbx_inversemap(bbx, ratio, dw, dh):
    l = round((bbx[0] - dw) / ratio)
    r = round((bbx[2] - dw) / ratio)
    t = round((bbx[1] - dh) / ratio)
    b = round((bbx[3] - dh) / ratio)
    bbx = (l, t, r, b)
    return bbx

def letterbox_bbx_map(bbx, ratio, dw, dh):
    l = bbx[0] * ratio + dw
    r = bbx[2] * ratio + dw
    t = bbx[1] * ratio + dh
    b = bbx[3] * ratio + dh
    bbx = (l, t, r, b)
    return bbx