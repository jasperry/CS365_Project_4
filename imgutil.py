'''
Created on Apr 4, 2010

@author: bseastwo
'''
import cv, cv2
import numpy
from scipy.ndimage import filters


def imageInfo(image, title="image"):
    '''
    Print image information.
    '''
    print "{0}: {1} of {2}, {3} {4} {5}".format(
        title, image.shape, image.dtype,  image.min(), image.max(), image.mean())
    
def imageShow(image, title="image", norm=True, wait=1):
    '''
    Display an image in a resizable cv window.  If the image is a numpy
    array, it will be converted to a cv image before display with an
    optional normalization of the image data to the range [0 ... 255].
    Numpy arrays with 3 channels get an RGB to BGR color swap.
    
    This function returns the value from cv.WaitKey(wait); this ensures
    the image is displayed and provides the option of capturing a keystroke
    if one is pressed in the allotted time.  If the wait parameter is
    None, cv.WaitKey() is not called.
    '''
    
    if type(image) == numpy.ndarray:
        # normalize the image data
        if norm:
            image = normalize(image)
        
        # color swap RGB to BGR
        if image.ndim == 3 and image.shape[2] == 3:
            image = image[..., ::-1]
    else:
        # we actually need to go back to numpy for this
        image = cv2array(image)
            
    cv2.namedWindow(title, cv.CV_WINDOW_NORMAL)
    cv2.imshow(title, image.astype(numpy.uint8))
    
    return cv2.waitKey(wait) if wait is not None else None

def normalize(image, range=(0,255), dtype=numpy.uint8):
    '''
    Linearly remap values in input data into range (0-255, by default).  
    Returns the dtype result of the normalization (numpy.uint8 by default).
    '''
    # find input and output range of data
    if isinstance(range, (int, float, long)):
        minOut, maxOut = 0., float(range)
    else:
        minOut, maxOut = float(range[0]), float(range[1])
    minIn, maxIn = image.min(), image.max()
    
    if maxIn - minIn < 1e-8:
        maxIn += 1e-8
    ratio = (maxOut - minOut) / (maxIn - minIn)
    
    # remap data
    output = (image - minIn) * ratio + minOut
    
    return output.astype(dtype)
    
def ncc(img1, img2):
    '''
    Computes the normalized cross correlation for a pair of images.
    NCC is computed as follows: 
    \mu = \frac{1}{N} \sum_{x=1}^N I(x)
    ncc(I_1, I_2) = \frac{(I_1 - \mu_1)(I_2 - \mu_2)}{\sqrt{\sum (I_1 - \mu_1)^2 \sum (I_2 - \mu_2)^2}}
    
    where all sums are over the image plane, and the two images I_1 and I_2 
    have the same number of elements, N.
    
    If the supplied images have a different number of elements, returns -1.
    '''
    if (img1.size != img2.size):
        return -1
    
    I1 = img1 - img1.mean()
    I2 = img2 - img2.mean()
    
    correlation = (I1 * I2).sum()
    normalizer = numpy.sqrt((I1**2).sum() * (I2**2).sum())
    
    return correlation / normalizer
    
def equalize(image, alpha=1.0):
    '''
    Apply histogram equalization to an image.  Returns the uint8 result of
    the equalization.
    '''
    # build histogram and cumulative distribution function
    hist = numpy.histogram(image, 256, (0, 255))
    cdist = numpy.cumsum(hist[0])
    cdist = (255.0 / image.size) * cdist
    
    # apply distribution function to image
    output = alpha * cdist[image] + (1-alpha) * image
    return numpy.uint8(output)

def gaussian(sigma, order=0, radius=0, norm=True):
    '''
    Computes the values of a 1D Gaussian function with standard deviation
    sigma.  The number of values returned is 2*radius + 1.  If radius is 0, 
    an appropriate radius is chosen to include at least 98% of the Gaussian.  
    If norm is True, the Gaussian function values sum to 1.
    
    returns a (2*radius+1)-element numpy array
    '''
    sigma = float(sigma)
    
    # choose an appropriate radius if one is not given; 98% of Gaussian is
    # within 5 sigma of the center.
    if radius == 0:
        radius = numpy.floor(sigma * 5.0/2) 
        
    # compute Gaussian values
    xrange = numpy.arange(-radius, radius + 1)
    denom = 1 / (2 * (sigma ** 2))
    data = numpy.exp(-denom * (xrange ** 2))
    
    # derivatives of Gaussians are products of polynomials (Hermite polynomials)
    # from Front-End Vision, pg. 54
    if order == 1:
        data = -data * (xrange / (sigma ** 2))
    elif order == 2:
        data =  data * ((xrange ** 2 - sigma ** 2) / (sigma ** 4))
    elif order == 3:
        data = -data * ((xrange ** 3 - 3 * xrange * sigma ** 2) / (sigma ** 6))
    elif order == 4:
        data =  data * ((xrange ** 4 - 6 * xrange ** 2 * sigma ** 2 + 3 * sigma ** 4) / (sigma ** 8))
    
    # normalize
    if norm:
        scale = 1 / (sigma * numpy.sqrt(2 * numpy.pi))
        data = scale * data
        
    return (data, xrange)

def montage(imgs, cols=4, space=5, norm=True):
    '''
    Builds an image montage from a h x w x d stack of images.
    '''
    # determine the size of the montage
    if type(imgs) == numpy.ndarray:
        # handle an ndarray being passed in
        d = imgs.shape[-1]
        imgList = numpy.split(imgs, d, imgs.ndim-1)
    else:
        # handle a tuple or list being passed in
        d = len(imgs)
        imgList = imgs
    h,w = imgList[0].shape[0:2]
    dtype = numpy.uint8 if norm else imgList[0].dtype
    rows = numpy.ceil(d / cols)
    
    # reserve space for the montage
    output = numpy.zeros(((h+space) * rows + space, 
                          (w+space) * cols + space) + imgList[0].shape[2:], dtype=dtype)

    # add images to montage
    for idx in range(d):
        y = (idx / cols) * (h + space) + space
        x = (idx % cols) * (w + space) + space
        
        # normalize images individually
        if norm:
            cell = normalize(imgList[idx])
        else:
            cell = imgList[idx]
        # paste into montage
        output[y:y+h, x:x+w, ...] = cell
    
    return output

def labelPoints(image, pointsX, pointsY):
    '''
    Creates and labels an opencv image with a set of red points at the specified
    locations.  If the input image is a numpy array, this function converts it
    to opencv and swaps color channels.  If the image is an opencv image, a copy
    of it is made for labeling.

    Returns the labeled opencv image, which can be displayed with imageShow().
    '''
    if type(image) == numpy.ndarray:
        # color swap RGB to BGR, convert to cv
        if image.ndim == 3 and image.shape[2] == 3:
            cvimg = array2cv(image[..., numpy.r_[2, 1, 0]])
        else:
            cvimg = array2cv(image)
    else:
        # copy cv image to create one we can draw on
        cvimg = cv.CreateImage(cv.GetSize(image), image.depth, image.nChannels)
        cv.Copy(image, cvimg)
    
    # scale points with the image size
    radius = int(numpy.ceil(cvimg.height / 200.0))
        
    for pX, pY in zip(pointsX, pointsY):
        cv.Circle(cvimg, (pX, pY), radius, (0, 0, 255), -1)
    
    return cvimg


# OpenCV / numpy data conversion functions.  
# These may be obsoleted by OpenCV 2.2.
# From source at http://opencv.willowgarage.com/wiki/PythonInterface

def cv2array(im):
    depth2dtype = {
        cv.IPL_DEPTH_8U: 'uint8',
        cv.IPL_DEPTH_8S: 'int8',
        cv.IPL_DEPTH_16U: 'uint16',
        cv.IPL_DEPTH_16S: 'int16',
        cv.IPL_DEPTH_32S: 'int32',
        cv.IPL_DEPTH_32F: 'float32',
        cv.IPL_DEPTH_64F: 'float64',
    }
  
    a = numpy.fromstring(
         im.tostring(),
         dtype=depth2dtype[im.depth],
         count=im.width*im.height*im.nChannels)
    a.shape = (im.height,im.width,im.nChannels)
    return a
    
def array2cv(a):
    dtype2depth = {
        'uint8':   cv.IPL_DEPTH_8U,
        'int8':    cv.IPL_DEPTH_8S,
        'uint16':  cv.IPL_DEPTH_16U,
        'int16':   cv.IPL_DEPTH_16S,
        'int32':   cv.IPL_DEPTH_32S,
        'float32': cv.IPL_DEPTH_32F,
        'float64': cv.IPL_DEPTH_64F,
    }
    try:
        nChannels = a.shape[2]
    except:
        nChannels = 1
    cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]), 
                                 dtype2depth[str(a.dtype)],
                                 nChannels)
    cv.SetData(cv_im, a.tostring(), 
               a.dtype.itemsize*nChannels*a.shape[1])
    return cv_im

def downsample(imgIn):
    '''
    Downsample an image to half its size after smoothing with a binomial
    kernel (after Burt & Adelson).
    '''
    filter = 1.0 / 16 * numpy.array([1, 4, 6, 4, 1])
    lowpass = filters.correlate1d(imgIn, filter, 0)
    lowpass = filters.correlate1d(lowpass, filter, 1)
    
    sample = lowpass[::2, ::2, ...]
    return sample

def upsample(imgIn):
    '''
    Upsample an image to twice its size and interpolate missing pixel
    values by smoothing.
    '''
    h,w = imgIn.shape[0:2]
    sample = numpy.zeros((h*2, w*2) + imgIn.shape[2:], imgIn.dtype)
    sample[::2, ::2, ...] = imgIn
    
    filter = 1.0 / 8 * numpy.array([1, 4, 6, 4, 1])
    lowpass = filters.correlate1d(sample, filter, 0, mode="constant")
    lowpass = filters.correlate1d(lowpass, filter, 1, mode="constant")
    return lowpass


if __name__ == "__main__":
    print "testing ncc"
    
    size = 1024
    mag = 256
    I1 = mag * numpy.random.rand(size, size)
    I1n = I1 + 0.15 * mag * numpy.random.randn(size, size)
    I2 = mag * numpy.random.rand(size, size)
    
    print "ncc(I1, I1) =", ncc(I1, I1)
    print "ncc(I1, I2) =", ncc(I1, I2)
    print "ncc(I2, inv(I2)) =", ncc(I2, mag - I2)
    print "ncc(I1, I1 + N(0, .15) =", ncc(I1, I1n)
    print "ncc(I1, I1n + 4) =", ncc(I1, I1n + mag/2)
    print "ncc(I1, I1n * 4) =", ncc(I1, I1n * mag/2)
