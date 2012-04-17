'''
Created on Feb 20, 2010

@author: beastwoo
'''

import ctypes, ctypes.util
import numpy
import os.path
import threading
import time
from scipy.ndimage import filters

from dc1394 import *
import imgutil

# the firewire video library, which provides a thin wrapper to libdc1394
videolib = None

# The C signature of the image allocation function.  This allocates space
# for a new image and returns a pointer to the allocated buffer.
# int allocator(int shape[3], int bytesPerPixel)
ALLOCATOR = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.POINTER(ctypes.c_int), ctypes.c_int)

def loadVideoLibrary():
    '''
    Loads the firewire video library if it has not yet been loaded.
    '''
    global videolib
    if videolib is not None:
        return
    
    # find the C fwvideo library
    # the position is specified relative to the current path.
    filepath = os.path.split(__file__)[0]
    libName = ctypes.util.find_library(os.path.join(filepath, "libfwvideo"))
    libFound = os.path.isfile(str(libName))
    
    # find_library does not work well on Linux
    if not libFound:
        libName = os.path.join(filepath, "../../bin/libfwvideo")
        libFound = os.path.isfile(str(libName))
    if not libFound:
        libName = os.path.join(filepath, "../../bin/libfwvideo.so")
        libFound = os.path.isfile(str(libName))
    if not libFound:
        libName = os.path.join(filepath, "../../bin/libfwvideo.dylib")
        libFound = os.path.isfile(str(libName))
    
    if not libFound:
        print "*** fwvideo not found; is it compiled?. ***"
    else:
        videolib = ctypes.cdll.LoadLibrary(libName)
        print "fwvideo library: ", libName, videolib
        
        # configure the fwvideo library signatures
        videolib.FV_openVideoDevice.restype = ctypes.c_void_p
        videolib.FV_setVideoMode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        videolib.FV_acquireFrame.argtypes = [ctypes.c_void_p, ALLOCATOR]
        

class FirewireVideo:
    '''
    Class that acquires images from a firewire camera.  
    
    Each instance of this class maintains its own independent frame set, 
    which is a list of image buffers.  In other words, if multiple cameras
    are running at once, and each camera is acquiring a series of images,
    the image buffers will not get intermingled.
    
    Manufacturers handle camera features differently.  Feature settings
    are controlled through a set of registers on the camera, and these
    register values can be manipulated using libdc1394.  However, register
    values can be interpreted by different cameras in different ways.
    Further, some manufacturers support an absolute value mode, where 
    features are set by values specified in real units.
    
    In this class, each controllable camera feature is represented by a 
    Feature object.  Each Feature has a register and absolute field, and
    each of these fields has an available flag, value, minimum, and 
    maximum.  Current feature values and bounds are retrieved from the
    camera using updateFeatures().  Feature values are set using one of
    the feature setting methods, e.g. setExposureAbsolute() or
    setFrameRegister().
    '''
    def __init__(self, 
                 deviceNumber = 0,
                 isoSpeed = DC1394_ISO_SPEED_400, 
                 buffers = 10):
        '''
        Open a video device with a set of camera parameters.  User can set the 
        firewire device number, the video mode (image size and color), and the 
        number of circular buffers.
        '''
        # load the firewire video library the first time a FirewireVideo
        # object is instantiated.
        loadVideoLibrary()
        
        # camera and image fields
        self.fcd = None
        self.rgbframe = None
        self.frameSet = []
        # color coding for output frames (self.rgbframe)
        self.colorCoding = None
        
        # camera feature fields
        self.shutter = Feature(DC1394_FEATURE_SHUTTER)
        self.gain = Feature(DC1394_FEATURE_GAIN)
        self.brightness = Feature(DC1394_FEATURE_BRIGHTNESS)
        self.gamma = Feature(DC1394_FEATURE_GAMMA)
        self.exposure = Feature(DC1394_FEATURE_EXPOSURE)
        self.framerate = Feature(DC1394_FEATURE_FRAME_RATE)
        self.pan = Feature(DC1394_FEATURE_PAN)
        self.tilt = Feature(DC1394_FEATURE_TILT)
        self.saturation = Feature(DC1394_FEATURE_SATURATION)
        self.hue = Feature(DC1394_FEATURE_HUE)
        self.whiteBalance = Feature(DC1394_FEATURE_WHITE_BALANCE)
                
        # keep track of frame times for recent frames
        self.timeCount = 50
        self.timeIndex = 0
        self.timeFrames = time.time() * numpy.ones((self.timeCount,))
        
        # variables to make acquisition thread-safe
        self.cameraLock = threading.Lock()
        
        if videolib is not None:
            try:
                self.fcd = videolib.FV_openVideoDevice(deviceNumber, isoSpeed, buffers)
                self.device = deviceNumber
                
                # note: it's extremely important to recast the pointer as a
                # ctypes pointer, because otherwise it is treated as an integer
                # on subsequent API calls, and on 64-bit systems, part of the
                # address gets truncated, which is very bad.
                self.fcd = ctypes.cast(self.fcd, ctypes.c_void_p)
                print "Camera pointer (python): 0x{0:x} ({1})".format(self.fcd.value, type(self.fcd))
                
            except Exception as e:
                print e
                print "Failed to connect to firewire video device."
                self.fcd = None
    
    def setVideoMode(self, videoMode, framerate=0,
                     colorCoding=DC1394_COLOR_CODING_YUV422,
                     width=0, height=0, posX=0, posY=0, rawProcess=DEMOSAIC_RGB):
        '''
        Sets the video mode for the camera.  If a standard video mode is used,
        the color coding and image ROI information are ignored.  If the video
        mode is Format7, the colorCoding controls the color code the camera uses
        during acquisition.

        For most color codings, the frames are converted to RGB8 color format in
        software, regardless of colorCoding and the rawProcess flag.  Only when
        colorCoding is RAW8 or RAW16 and rawProcess is not DEMOSAIC_RGB does 
        this class return raw image data.
        '''
        videoText = VIDEO_MODES_INV[videoMode] if videoMode in VIDEO_MODES_INV else "unknown " + str(videoMode)
        frameText = FRAMERATES_INV[framerate] if framerate in FRAMERATES_INV else "unknown " + str(framerate)
        colorText = COLOR_CODINGS_INV[colorCoding] if colorCoding in COLOR_CODINGS_INV else "unknown " + str(colorCoding)
        
        if (videoMode < DC1394_VIDEO_MODE_FORMAT7_0):    
            print "setVideoMode: {0}, {1}".format(videoText, frameText)
            videolib.FV_setVideoMode(self.fcd, videoMode, framerate)
            self.colorCoding = DC1394_COLOR_CODING_RGB8
        else:
            print "setVideoMode: {0}, {1}, {2} fps".format(videoText, colorText, framerate)
            print "         ROI: {0} x {1} + ({2} , {3})".format(width, height, posX, posY)
            videolib.FV_setVideoModeF7(self.fcd, videoMode, colorCoding, 
                                       width, height, posX, posY, ctypes.c_float(framerate), rawProcess)
            
            if (colorCoding < DC1394_COLOR_CODING_RAW8 or (rawProcess & DEMOSAIC_RGB)):
                # camera library converts to RGB8 for anything but raw
                self.colorCoding = DC1394_COLOR_CODING_RGB8
            else:
                self.colorCoding = colorCoding
    
    def startTransmission(self):
        videolib.FV_startTransmission(self.fcd)
    
    def stopTransmission(self, flush=False):
        videolib.FV_stopTransmission(self.fcd, flush)
    
    def printFeatures(self, full=False):
        '''
        Prints current camera feature values.  If full is True, this prints
        complete camera information for each feature using dc1394_feature_print_all().
        Otherwise, the features that are controllable through the
        FirewireVideo interface are printed out along with their bounds.
        Absolute values are printed when available.
        '''
        if full:
            videolib.FV_printFeatures(self.fcd)
        else:
            # get current feature values
            self.updateFeatures()
            
            fields = [self.shutter,
                      self.gain,
                      self.brightness,
                      self.gamma,
                      self.exposure,
                      self.framerate,
                      self.pan,
                      self.tilt,
                      self.hue,
                      self.saturation]
            
            # print features
            print "------ Features (cam {0}) ------".format(self.device)
            for field in fields:
                feature = field.absolute if field.absolute.available else field.register
                power = "on" if field.power else "off"
                print "\t{0:20} ({1:>3}) => {2:>12} [{3:>12} -- {4:>12}]".format(
                    FEATURES_INV[field.id], power, feature.value, feature.minimum, feature.maximum)
            
            power = "on" if self.whiteBalance.power else "off"
            print "\t{0:20} ({1:>3}) => {2:>12} [{3:>12} -- {4:>12}]".format(
                    FEATURES_INV[self.whiteBalance.id] + " blue", power, 
                    self.whiteBalance.register.getBlueValue(), 
                    self.whiteBalance.register.minimum, self.whiteBalance.register.maximum)
            print "\t{0:20} ({1:>3}) => {2:>12} [{3:>12} -- {4:>12}]".format(
                    FEATURES_INV[self.whiteBalance.id] + " red", power, 
                    self.whiteBalance.register.getRedValue(), 
                    self.whiteBalance.register.minimum, self.whiteBalance.register.maximum)
    
    def updateFeatures(self):
        '''
        Updates camera features and sets values on the appropriate fields. After
        calling this method, any of these values can be obtained directly from
        the fields on this object.

        Each field is a tuple where the the first element stores the feature
        value and the second and third elements store that feature's minimum and
        maximum value, respectively.  Each feature has register and absolute
        value fields.  For example:
        
        camobj.shutter[0] provides the shutter register value while
        camobj.shutterAbs[0] provides the shutter speed in seconds.
        '''
        count = DC1394_FEATURE_CAPTURE_QUALITY - DC1394_FEATURE_MIN + 1
        registerValues = numpy.zeros((count, 5), dtype=numpy.uint32)
        absoluteValues = numpy.zeros((count, 4), dtype=numpy.float32)
        ptrReg = registerValues.ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
        ptrAbs = absoluteValues.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) 
        videolib.FV_getAllFeatureInfo(self.fcd, ptrReg, ptrAbs)
        
        # set up list of features to update
        fields = [self.brightness,
                  self.exposure,
                  self.gamma,
                  self.shutter,
                  self.gain,
                  self.framerate,
                  self.pan,
                  self.tilt,
                  self.hue,
                  self.saturation,
                  self.whiteBalance]
        
        # copy values obtained from the camera into fields
        for field in fields:
            base = field.id - DC1394_FEATURE_MIN
#            print FEATURES_INV[fid], registerValues[base, :], absoluteValues[base, :]
            field.available = bool(registerValues[base, 3])
            field.register.available = bool(registerValues[base, 3])
            field.power = bool(registerValues[base, 4])
            if field.register.available:
                field.register.value, field.register.minimum, field.register.maximum = registerValues[base, 0:3]
            field.absolute.available = bool(absoluteValues[base, 3])
            if field.absolute.available:
                field.absolute.value, field.absolute.minimum, field.absolute.maximum = absoluteValues[base, 0:3]

        # update white balance values separately
        whiteBlue, whiteRed = ctypes.c_uint(0), ctypes.c_uint(0)
        videolib.FV_getWhiteBalanceRegister(self.fcd, ctypes.byref(whiteBlue), ctypes.byref(whiteRed))
        self.whiteBalance.register.blueValue = whiteBlue.value
        self.whiteBalance.register.redValue = whiteRed.value
        
    def setExposureRegister(self, shutter=None, gain=None, brightness=None, gamma=None):
        '''
        Set camera exposure features, any of which can be None to 
        preserve current settings.  The values provided are register
        values.
        '''
        if shutter is not None:
            self.shutter.register.value = shutter
            videolib.FV_setFeatureRegister(self.fcd, self.shutter.id, ctypes.c_uint(self.shutter.register.value))
        if gain is not None:
            self.gain.register.value = gain
            videolib.FV_setFeatureRegister(self.fcd, self.gain.id, ctypes.c_uint(self.gain.register.value))
        if brightness is not None:
            self.brightness.register.value = brightness
            videolib.FV_setFeatureRegister(self.fcd, self.brightness.id, ctypes.c_uint(self.brightness.register.value))
        if gamma is not None:
            self.gamma.register.value = gamma
            videolib.FV_setFeatureRegister(self.fcd, self.gamma.id, ctypes.c_uint(self.gamma.register.value))
                
    def setExposureAbsolute(self, shutter=None, gain=None, brightness=None, gamma=None):
        '''
        Set camera exposure features, any of which can be None to 
        preserve current settings.  The values provided are absolute
        values, e.g. seconds for shutter speed.
        '''        
        if shutter is not None:
            self.shutter.absolute.value = shutter
            videolib.FV_setFeatureAbsolute(self.fcd, self.shutter.id, ctypes.c_float(self.shutter.absolute.value))
        if gain is not None:
            self.gain.absolute.value = gain
            videolib.FV_setFeatureAbsolute(self.fcd, self.gain.id, ctypes.c_float(self.gain.absolute.value))
        if brightness is not None:
            self.brightness.absolute.value = brightness
            videolib.FV_setFeatureAbsolute(self.fcd, self.brightness.id, ctypes.c_float(self.brightness.absolute.value))
        if gamma is not None:
            self.gamma.absolute.value = gamma
            videolib.FV_setFeatureAbsolute(self.fcd, self.gamma.id, ctypes.c_float(self.gamma.absolute.value))
    
    def setColorRegister(self, hue=None, saturation=None, whiteBlue=None, whiteRed=None):
        '''
        Set the color control registers on the camera.  White balance is
        set with two values, the relative gains of the blue and red pixels
        relative to the green pixels in the Bayer mask.
        '''
        if hue is not None:
            self.hue.register.value = hue
            videolib.FV_setFeatureRegister(self.fcd, self.hue.id, ctypes.c_uint(self.hue.register.value))
        if saturation is not None:
            self.saturation.register.value = saturation
            videolib.FV_setFeatureRegister(self.fcd, self.saturation.id, ctypes.c_uint(self.saturation.register.value))
        
        if whiteBlue is not None or whiteRed is not None:
            # need to know current white balance values in case one is not provided
            currBlue, currRed = ctypes.c_uint(0), ctypes.c_uint(0)
            videolib.FV_getWhiteBalanceRegister(self.fcd, ctypes.byref(currBlue), ctypes.byref(currRed))
            self.whiteBalance.register.blueValue = currBlue.value
            self.whiteBalance.register.redValue = currRed.value
            
            if whiteBlue is not None:
                self.whiteBalance.register.blueValue = whiteBlue
            if whiteRed is not None:
                self.whiteBalance.register.redValue = whiteRed
                                
            videolib.FV_setWhiteBalanceRegister(self.fcd, 
                ctypes.c_uint(self.whiteBalance.register.blueValue), 
                ctypes.c_uint(self.whiteBalance.register.redValue))
        
    def setColorAbsolute(self, hue=None, saturation=None, whiteBlue=None, whiteRed=None):
        '''
        Set the color control registers on the camera using absolute values.
        White balance is set with two values, the relative gains of the 
        blue and red pixels relative to the green pixels in the Bayer mask.
        White balance values need to be provided as register values.
        '''
        if hue is not None:
            self.hue.absolute.value = hue
            videolib.FV_setFeatureAbsolute(self.fcd, self.hue.id, ctypes.c_float(self.hue.absolute.value))
        if saturation is not None:
            self.saturation.absolute.value = saturation
            videolib.FV_setFeatureAbsolute(self.fcd, self.saturation.id, ctypes.c_float(self.saturation.absolute.value))
        
        if whiteBlue is not None or whiteRed is not None:
            # need to know current white balance values in case one is not provided
            currBlue, currRed = ctypes.c_uint(0), ctypes.c_uint(0)
            videolib.FV_getWhiteBalanceRegister(self.fcd, ctypes.byref(currBlue), ctypes.byref(currRed))
            self.whiteBalance.register.blueValue = currBlue.value
            self.whiteBalance.register.redValue = currRed.value
            
            if whiteBlue is not None:
                self.whiteBalance.register.blueValue = whiteBlue
            if whiteRed is not None:
                self.whiteBalance.register.redValue = whiteRed
                                
            videolib.FV_setWhiteBalanceRegister(self.fcd, 
                ctypes.c_uint(self.whiteBalance.register.blueValue), 
                ctypes.c_uint(self.whiteBalance.register.redValue))

    def setAutoExposure(self, enable=True):
        '''
        Enables/disables auto exposure for the camera.  Auto exposure
        selects the shutter speed and gain to obtain an average exposure.
        '''
        mode = DC1394_FEATURE_MODE_AUTO if enable else DC1394_FEATURE_MODE_MANUAL
        videolib.FV_setFeatureMode(self.fcd, self.exposure.id, mode)
        videolib.FV_setFeatureMode(self.fcd, self.shutter.id, mode)
        videolib.FV_setFeatureMode(self.fcd, self.gain.id, mode)
        
    def setAutoColor(self, enable=True):
        '''
        Enables or disables auto color settings for the camera.  This 
        includes white balance, hue, and saturation settings.
        '''
        mode = DC1394_FEATURE_MODE_AUTO if enable else DC1394_FEATURE_MODE_MANUAL
        videolib.FV_setFeatureMode(self.fcd, self.hue.id, mode)
        videolib.FV_setFeatureMode(self.fcd, self.saturation.id, mode)
        videolib.FV_setFeatureMode(self.fcd, self.whiteBalance.id, mode)
            
    def setFrameRegister(self, framerate=None, pan=None, tilt=None):
        '''
        Sets frame properties of framerate, pan, and tilt.  Any of these
        can be None to keep their current values.  All values should be
        specified in absolute units; i.e. framerate in frames per second,
        pan and tilt in pixels.
        '''
        self.updateFeatures()
        
        if framerate is not None:
            self.framerate.register.value = framerate
            videolib.FV_setFeatureRegister(self.fcd, self.framerate.id, ctypes.c_uint(self.framerate.register.value))
        if pan is not None:
            self.pan.register.value = pan
            videolib.FV_setFeatureRegister(self.fcd, self.pan.id, ctypes.c_uint(self.pan.register.value))
        if tilt is not None:
            self.tilt.register.value = tilt
            videolib.FV_setFeatureRegister(self.fcd, self.tilt.id, ctypes.c_uint(self.tilt.register.value))
        
    def setFrameAbsolute(self, framerate=None, pan=None, tilt=None):
        '''
        Sets frame properties of framerate, pan, and tilt.  Any of these
        can be None to keep their current values.  All values should be
        specified in absolute units; i.e. framerate in frames per second,
        pan and tilt in pixels.
        '''
        self.updateFeatures()
        
        if framerate is not None:
            self.framerate.absolute.value = framerate
            videolib.FV_setFeatureAbsolute(self.fcd, 
                self.framerate.id, ctypes.c_float(self.framerate.absolute.value))
            
        # for pan and tilt, register and absolute values are the same
        if pan is not None:
            self.pan.register.value = pan
            videolib.FV_setFeatureRegister(self.fcd,
                self.pan.id, ctypes.c_uint(self.pan.register.value))
        if tilt is not None:
            self.tilt.register.value = tilt
            videolib.FV_setFeatureRegister(self.fcd,
                self.tilt.id, ctypes.c_uint(self.tilt.register.value))
        
    def allocateImage(self, shape, bytesPerPixel):
        '''
        Allocates new space for an rgb numpy image in this camera device's
        frame set.  Returns a c_void_p pointer to the image's data buffer.
        '''
#        print "allocateImage(): entering with shape: ", shape[0:3]
        # make single channel images 2D, not 3D...this is important.
        if shape[2] == 1:
            useShape = shape[0:2]
        else:
            useShape = shape[0:3]
    
        # pixel flag determines whether to allocate 8 or 16 bit pixel data
        if bytesPerPixel == 1:
            image = numpy.zeros(useShape, numpy.uint8)
        else:
            image = numpy.zeros(useShape, numpy.uint16)
        
        # store image in array, and return a pointer to the data
        self.frameSet.append(image)
        ptr = image.ctypes.data_as(ctypes.c_void_p).value
        return ptr
    
    def acquireFrame(self, flush=False):
        '''
        Acquire and return a single frame from the camera.
        Returns the frame as a numpy array.
        '''
        self.cameraLock.acquire()
        if videolib is not None:
            if flush:
                videolib.FV_flushRingBuffer(self.fcd)
            
            videolib.FV_acquireFrame(self.fcd, ALLOCATOR(self.allocateImage))
            self.rgbframe = self.frameSet.pop(0)
            
        else:
            h, w = (480, 640)
            self.rgbframe = numpy.zeros((h, w, 3), dtype=numpy.uint8)
            
            yy, xx = numpy.mgrid[0:h, 0:w]
            self.rgbframe[..., 0] = 255.0 * yy / h
            self.rgbframe[..., 1] = 255.0 * xx / w
        
        self.timeIndex = (self.timeIndex + 1) % self.timeCount
        self.timeFrames[self.timeIndex] = time.time()
        self.cameraLock.release()
        
        return self.rgbframe
    
    def oneShot(self):
        '''
        Acquire a single frame from the camera.  Image transfer should
        *not* be running when this method is called.
        '''
        self.cameraLock.acquire()
        videolib.FV_oneShot(self.fcd, ALLOCATOR(self.allocateImage))
        self.rgbframe = self.frameSet.pop(0)
        self.cameraLock.release()
        
        return self.rgbframe
    
    def flushRingBuffer(self):
        '''
        Flushes the ring of image buffers associated with this camera.
        This ensures the next frames obtained will be current.
        '''
        if self.fcd is not None:
            videolib.FV_flushRingBuffer(self.fcd)
    
    def closeVideoDevice(self):
        '''
        Close the video device associated with this object.
        '''
        if self.fcd is not None:
            videolib.FV_closeVideoDevice(self.fcd)
            self.fcd = None;
            
    def computeFramerate(self):
        '''
        Returns the frame rate for the most recent set of frames acquired
        from this camera.  In other words, this is a running framerate,
        not a cummulative framerate.
        '''
        # the current time index points to the most recent frame,
        # so the next time index is the least recent frame
        diff = self.timeFrames[self.timeIndex] - self.timeFrames[(self.timeIndex + 1) % self.timeCount]
        return self.timeCount / diff
    
    def settleAutoExposure(self, numFrames=30, display=True):
        '''
        Run image acquisition for a while in auto exposure mode.
        '''
        self.setAutoExposure(True)
        self.startTransmission()
        for i in range(numFrames):
            npimg = self.acquireFrame()
            if display:
                imgutil.imageShow(npimg, "Auto", False, wait=10)
        
        self.setAutoExposure(False)
        
    def splitRaw(self, input, filter=None):
        '''
        Splits a raw image into the four Bayer channel components.  Returns
        an image in RGBG order.  The Bayer pattern ordering is determined
        from the camera if the filter parameter is None.
        '''
        # determine the new image size
        h, w = input.shape[:2]
        split = numpy.zeros((h // 2, w // 2, 4), dtype=input.dtype)
        
        # split channels irrespective of color filter; the channel index
        # maps to the possible Bayer patterns as follows:
        # 0 1  |  R G    G B    G R    B G
        # 2 3  |  G B    R G    B G    G R
        split[..., 0] = input[0::2, 0::2]
        split[..., 1] = input[0::2, 1::2]
        split[..., 2] = input[1::2, 0::2]
        split[..., 3] = input[1::2, 1::2]
        
        if filter is None:
            # determine the color filter ordering
            filterCint = ctypes.c_int(0) 
            videolib.FV_getColorFilter(self.fcd, ctypes.byref(filterCint))
            filter = filterCint.value
        
        # reorder the split channels
        if filter == DC1394_COLOR_FILTER_RGGB:
            output = split[..., numpy.r_[0, 1, 3, 2]]
        elif filter == DC1394_COLOR_FILTER_GBRG:
            output = split[..., numpy.r_[2, 0, 1, 3]]
        elif filter == DC1394_COLOR_FILTER_GRBG:
            output = split[..., numpy.r_[1, 0, 2, 3]]
        elif filter == DC1394_COLOR_FILTER_BGGR:
            output = split[..., numpy.r_[3, 1, 0, 2]]
        return output
    
    def demosaic(self, input, filter=None):
        '''
        Demosaics a four channel Bayer image in RGBG ordering into a
        three channel RGB image.  Each Bayer color channel is upsampled;
        the green channels are averaged and the resulting frames are
        assembled into a single image.
        '''
#        for d in range(input.shape[2]):
#            imgutil.imageInfo(input[..., d], "input {0}".format(d))
        
        # build an image to hold the upsampled image data in RGBG format
        h, w = input.shape[0:2]
        channels = numpy.zeros((2 * h, 2 * w, 4), dtype=numpy.float32)
        
        if filter is None:
            # fill in image data in the appropriate spot based on Bayer pattern mask
            # determine the color filter ordering
            filterCint = ctypes.c_int(0) 
            videolib.FV_getColorFilter(self.fcd, ctypes.byref(filterCint))
            filter = filterCint.value
            
        if filter not in FILTERS_INV:
            print "FirewireVideo.demosaic: Warning, the color filter pattern from the camera was invalid ({0}), assuming BGRG".format(filter)
            filter = DC1394_COLOR_FILTER_GBRG            
                
        # reorder the split channels
        if filter == DC1394_COLOR_FILTER_RGGB:
            channels[0::2, 0::2, 0] = input[..., 0]
            channels[0::2, 1::2, 1] = input[..., 1]
            channels[1::2, 1::2, 2] = input[..., 2]
            channels[1::2, 0::2, 3] = input[..., 3]
        elif filter == DC1394_COLOR_FILTER_GBRG:
            channels[1::2, 0::2, 0] = input[..., 0]
            channels[0::2, 0::2, 1] = input[..., 1]
            channels[0::2, 1::2, 2] = input[..., 2]
            channels[1::2, 1::2, 3] = input[..., 3]
        elif filter == DC1394_COLOR_FILTER_GRBG:
            channels[0::2, 1::2, 0] = input[..., 0]
            channels[0::2, 0::2, 1] = input[..., 1]
            channels[1::2, 0::2, 2] = input[..., 2]
            channels[1::2, 1::2, 3] = input[..., 3]
        elif filter == DC1394_COLOR_FILTER_BGGR:
            channels[1::2, 1::2, 0] = input[..., 0]
            channels[0::2, 1::2, 1] = input[..., 1]
            channels[0::2, 0::2, 2] = input[..., 2]
            channels[1::2, 0::2, 3] = input[..., 3]
        
        # interpolate missing data values on each channel
        kernel = 1.0 / 8 * numpy.array([1, 4, 6, 4, 1])
        channels = filters.correlate1d(channels, kernel, 0, mode="constant")
        channels = filters.correlate1d(channels, kernel, 1, mode="constant")
                
        # build RGB result
        output = numpy.zeros((2 * h, 2 * w, 3), dtype=input.dtype)
        output[..., 0] = channels[..., 0]
        output[..., 2] = channels[..., 2]
        output[..., 1] = channels[..., numpy.r_[1, 3]].mean(2)

#        for d in range(output.shape[2]):
#            imgutil.imageInfo(output[..., d], "output {0}".format(d))
        
        return output
    
def shutter2seconds(value):
    '''
    Finds the shutter duration in seconds from the shutter speed number for a
    Unibrain Fire-i camera.  Valid shutter values are within [1 ... 3843] and 
    shutter durations range from 1 us to 3600 s.  All durations are returned in
    seconds.
    
    Mapping function comes from page 32 of the Unibrain Fire-i series manual.
    '''
    duration = 0.
    if value <= 500:
        duration = value * 1e-6
    elif value <= 1000:
        duration = ((value - 500) * 10 + 500) * 1e-6
    elif value <= 1705:
        duration = ((value - 1000) * 100 + 5500) * 1e-6
    elif value <= 2399:
        duration = ((value - 1705) + 76) * 1e-3
    elif value <= 2902:
        duration = ((value - 2399) * 10 + 770) * 1e-3
    elif value <= 3304:
        duration = ((value - 2902) * 100 + 5800) * 1e-3
    elif value <= 3508:
        duration = ((value - 3304) * 1000 + 46000) * 1e-3
    elif value <= 3843:
        duration = ((value - 3508) * 10 + 250)
    
    return duration

def gain2decibels(value):
    '''
    Converts a gain value to a decibel value.
    '''
    return 25. * value / 723.

class FeatureValue(object):
    '''
    Represents a camera feature that can have a value and a minimum
    and maximum possible value.  The available flag indicates whether
    these values are available on the camera.
    '''
    def __init__(self):
        self.available = False
        self.value = 0
        self.minimum = 0
        self.maximum = 0
        
        # these fields are used only for the white balance feature
        self.blueValue = 0
        self.redValue = 0
    
    def getBlueValue(self):
        '''
        The white balance register incorporates both a blue and red value;
        this provides access to the blue value.
        '''
        return self.blueValue
    
    def getRedValue(self):
        '''
        The white balance register incorporates both a blue and red value;
        this provides access to the red value.
        '''
        return self.redValue
    
class Feature(object):
    '''
    Represents a camera feature that can have a register and absolute
    value.  Each of these is represented by a FeatureValue, encapsulating
    the actual feature value and min/max bounds.
    '''
    def __init__(self, id=0):
        self.id = id
        self.available = False;
        self.power = False;
        self.register = FeatureValue()
        self.absolute = FeatureValue()

def testStart():
    # connect to firewire camera, select frame format
    fwcam = FirewireVideo(0, DC1394_ISO_SPEED_800)
    fwcam.setVideoMode(DC1394_VIDEO_MODE_1024x768_RGB8, DC1394_FRAMERATE_15)
    
    # set up camera parameters
    fwcam.setExposureAbsolute(brightness=0, gamma=1.0)
    fwcam.setAutoExposure(True)
    fwcam.setColorAbsolute(whiteBlue=1023, whiteRed=276)
    
    # start grabbing video frames
    fwcam.startTransmission()
    
    # display frames forever
    index = 0
    poll = 100
    t0 = time.time()
    
    key = None
    while key != 27:
        # grab a frame
        frame = fwcam.acquireFrame()
        key = imgutil.imageShow(frame, "fwvideo", False, 10)
        
        index += 1
        t1 = time.time()
        if index % poll == 0:
            fwcam.printFeatures(False)
            print "{0:8}: {1:8.3f} fps".format(index, float(poll) / (t1-t0))
            t0 = t1
    
    # disconnect from camera
    fwcam.setAutoExposure(False)
    fwcam.stopTransmission()
    fwcam.closeVideoDevice()

if __name__ == "__main__":
    testStart()