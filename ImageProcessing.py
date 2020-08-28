# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 13:11:06 2020

@author: Aditya
"""

from matplotlib import pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from scipy import signal
import cv2
import logging
log = logging.getLogger(__name__)
import PIL
from matplotlib import pyplot as plt
import scipy.stats as stats
import glob
from PIL import Image
from cv2.ximgproc import guidedFilter

def get_background(img, is_01_normalized=True):
    return ~get_foreground(img, is_01_normalized)

def get_foreground(img, is_01_normalized=True):
    return center_crop_and_get_foreground_mask(
        img, crop=False, is_01_normalized=is_01_normalized)[1]

def get_center_circle_coords(im, is_01_normalized: bool):
    A = np.dstack([
        signal.cspline2d(im[:,:,ch] * (255 if is_01_normalized else 1), 200.0)
        for ch in range(im.shape[-1])])
    min_r = int(min(im.shape[0], im.shape[1]) / 4)
    max_r = int(max(im.shape[0], im.shape[1]) / 4*3)
    try:
        circles = cv2.HoughCircles(
            (norm01(A).max(-1)*255).astype('uint8'), cv2.HOUGH_GRADIENT, .8,
            min(A.shape[:2]), param1=20, param2=50, minRadius=min_r, maxRadius=max_r)[0]
    except:
        log.warn('center_crop_and_get_foreground_mask failed to get background - trying again with looser parameters')
        A2 = get_foreground_slow(im)
        circles = cv2.HoughCircles(
            (A2*255).astype('uint8'), cv2.HOUGH_GRADIENT, .8,
            min(A.shape[:2]), param1=20, param2=10, minRadius=min_r, maxRadius=max_r)[0]
    x, y, r = circles[circles[:, -1].argmax()].round().astype('int')
    return x,y,r

def get_foreground_mask_from_center_circle_coords(shape, x,y,r):
    mask = np.zeros(shape, dtype='uint8')
    cv2.circle(mask, (x, y), r, 255, cv2.FILLED)
    mask = mask.astype(bool)
    return mask

def center_crop_and_get_foreground_mask(im, crop=True, is_01_normalized=True, center_circle_coords=None, label_img=None):
    if center_circle_coords is not None:
        x,y,r = center_circle_coords
    else:
        h, w, _ = im.shape
        x, y, r = get_center_circle_coords(im, is_01_normalized)
    mask = get_foreground_mask_from_center_circle_coords(im.shape[:2], x,y,r)
    if crop:
        crop_slice = np.s_[max(0, y-r):min(h,y+r),max(0,x-r):min(w,x+r)]
        rv = [im[crop_slice], mask[crop_slice]]
        if label_img is not None:
            rv.append(label_img[crop_slice])
    else:  # don't crop.  just get the mask.
        rv = [im, mask]
        if label_img is not None:
            rv.append(label_img[crop_slice])
    return rv

def get_background_slow(img):

    img = img/img.max()
    background = (img < 20/255)
    background = ndi.morphology.binary_closing(
        background, np.ones((5, 5, 1)))
    background |= np.pad(np.zeros(
        (background.shape[0]-6, background.shape[1]-6, 3), dtype='bool'),
        [(3, 3), (3, 3), (0, 0)], 'constant', constant_values=1)
    return background.sum(2) == 3

def get_foreground_slow(img):
    return ~get_background_slow(img)

def zero_mean(img, fg):
    z = img[fg]
    return (img - z.mean()) + 0.5

def norm01(img, background=None):
    if background is not None:
        tmp = img[~background]
        min_, max_ = tmp.min(), tmp.max()
    else:
        min_, max_ = img.min(), img.max()
    rv = (img - min_) / (max_ - min_)
    if background is not None:
        rv[background] = img[background]
    return rv

def solvet(I, A, use_gf=True, fsize=(5,5)):
    z = 1-ndi.minimum_filter((I/A).min(-1), fsize)
    if use_gf:
        z = gf(I, z)
    rv = z.reshape(*I.shape[:2], 1)
    return rv

def solvetmax(I, A):
    z = 1-ndi.maximum_filter((I/A).max(-1), (5, 5))
    return gf(I, z).reshape(*I.shape[:2], 1)

def solveJ(I, A, t):
    epsilon = max(np.min(t)/2, 1e-8)
    return (I-A)/np.maximum(t, epsilon) + A

def gf(guide, src, r=100, eps=1e-8):
    return guidedFilter(guide.astype('float32'), src.astype('float32'), r, eps).astype('float64')

def ta(img, ignore_ch=None, **kws):
    if ignore_ch is not None:
        I = img.copy()
        I[:,:,ignore_ch] = 0
    else:
        I = img
    return solvet(1-img, 1, **kws)

def td(img, ignore_ch=None, **kws):
    if ignore_ch is not None:
        I = img.copy()
        I[:,:,ignore_ch] = 0
    else:
        I = img
    return 1-solvet(1-img, 1, **kws)

def tb(img, ignore_ch=None, **kws):
    if ignore_ch is not None:
        I = img.copy()
        I[:,:,ignore_ch] = 1
    else:
        I = img
    return solvet(I, 1, **kws)

def tc(img, ignore_ch=None, **kws):
    if ignore_ch is not None:
        I = img.copy()
        I[:,:,ignore_ch] = 1
    else:
        I = img
    return 1-solvet(I, 1, **kws)

def A(img):
    return solveJ(img, 0, ta(img))
def B(img):
    return solveJ(img, 0, tb(img))
def C(img):
    return solveJ(img, 0, tc(img))
def D(img):
    return solveJ(img, 0, td(img))
def W(img):
    return solveJ(img, 1, ta(img))
def X(img):
    return solveJ(img, 1, tb(img))
def Y(img):
    return solveJ(img, 1, tc(img))
def Z(img):
    return solveJ(img, 1, td(img))
def B_ret(img):
    """specific to retinal fundus images, where blue channel is too sparse"""
    return solveJ(img, 0, tb(img, ignore_ch=2))
def C_ret(img):
    """specific to retinal fundus images, where blue channel is too sparse"""
    return solveJ(img, 0, tc(img, ignore_ch=2))
def X_ret(img):
    """specific to retinal fundus images, where blue channel is too sparse"""
    return solveJ(img, 1, tb(img, ignore_ch=2))
def Y_ret(img):
    """specific to retinal fundus images, where blue channel is too sparse"""
    return solveJ(img, 1, tc(img, ignore_ch=2))


def brighten_darken(img, method_name: str, focus_region=None,
                    fundus_image: bool=True):
    
    func_names = method_name.split('+')
    if fundus_image:
        _methods = dict(zip('ABCDWXYZ', [A,B_ret,C_ret,D,W,X_ret,Y_ret,Z]))
    else:
        _methods = dict(zip('ABCDWXYZ', [A,B,C,D,W,X,Y,Z]))
    I2 = np.zeros_like(img)
    for func_name in func_names:
        tmp = _methods[func_name.lstrip('s')](img)
        if func_name.startswith('s'):
            tmp = sharpen(tmp, ~focus_region)
        I2 += tmp
    I2 /= len(func_names)
    return I2

def check_and_fix_nan(A, replacement_img):
    nanmask = np.isnan(A)
    if nanmask.any():
        log.warn("sharpen: guided filter blurring operation or laplace filter returned nans. your input image has extreme values")
        A[nanmask] = replacement_img[nanmask]
    return A


def sharpen(img, bg=None, t='laplace', blur_radius=30, blur_guided_eps=1e-8,
            use_guidedfilter='if_large_img'):
   
    if bg is None:
        bg = np.zeros(img.shape[:2], dtype='bool')
    else:
        img = img.copy()
        img[bg] = 0
        A = cv2.ximgproc.guidedFilter(
        #  radiance.astype('float32'),
        img.astype('float32'),
        img.astype('float32'),
        blur_radius, blur_guided_eps)

    if t == 'laplace':
        t = 1-norm01(sharpen(ndi.morphological_laplace(
            img, (2,2,1), mode='wrap'), bg, 0.15), bg)
        
    if len(np.shape(t)) + 1 == len(img.shape):
        t_refined = np.expand_dims(t, -1).astype('float')
    else:
        t_refined = t
    if np.shape(t):
        t_refined[bg] = 1  # ignore background, but fix division by zero
    J = (
        img.astype('float')-A) / np.maximum(1e-8, np.maximum(t_refined, np.min(t_refined)/2)) + A
    #  assert np.isnan(J).sum() == 0
    if bg is not None:
        J[bg] = 0
    if use_guidedfilter == 'if_large_img':
        # note: at some point, find a better threshold?  This works.
        use_guidedfilter = min(J.shape[0], J.shape[1]) >= 1500
    if not use_guidedfilter:
        J = check_and_fix_nan(J, img)
        return J

    r2 = cv2.ximgproc.guidedFilter(
        img.astype('float32'),
        J.astype('float32'),
        2, 1e-8)
    r2 = check_and_fix_nan(r2, img)
    if bg is not None:
        r2[bg] = 0
    return r2


with PIL.Image.open('/content/drive/My Drive/right.jpg') as img:
  img.load()
  img2=img
  img = np.array(img) / 255

I = img.copy()
I, fg = center_crop_and_get_foreground_mask(I)

enhanced_img = brighten_darken(I, 'A+sB', focus_region=fg)
enhanced_img2 = sharpen(enhanced_img, bg=~fg)

fig = plt.figure()
fig.set_size_inches(img2.size[0]/100,img2.size[1]/100)
ax = plt.Axes(fig, [0., 0., 1., 1.])
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
ax.set_axis_off()
fig.add_axes(ax)
plt.set_cmap('hot')
ax.imshow(enhanced_img2, aspect='equal')
plt.savefig('/content/drive/My Drive/image.jpg', dpi=100,pad_inches=0, bbox_inches = 'tight')
