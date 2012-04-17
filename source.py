'''
Created on May 12, 2010
Classes that act as image sources--pipeline objects that generate or load
images.  These are typically upstream pipeline objects.

@author: bseastwo
'''

import cv
import numpy
import os.path
import time

import imgutil
import pipeline

import FirewireVideo

def readImageFile(filename):
    '''
    Reads an image from file, using an appropriate method based on the extension
    of the file.  Numpy is used to handle npy and raw images.  OpenCV handles
    common image formats (tif, jpg, png), with proper red-blue channel swapping.
    
    Returns the image as a numpy array.
    '''
    
    if filename.split(".")[-1] == "npy":
        # load it using numpy
        npimg = numpy.load(filename)
        
    elif filename.split(".")[-1] == "raw":
        # handle a few different raw frame sizes, based on the file size
        npimg = numpy.fromfile(filename, dtype=numpy.uint16)
        if npimg.size == 1036 * 1388:
            npimg = npimg.reshape((1036, 1388))
        elif npimg.size == 518 * 692:
            npimg = npimg.reshape((518, 692))
        elif npimg.size == 484 * 648:
            npimg = npimg.reshape((484, 648))
        else:
            print "Unknown raw image size:", npimg.shape
    
    else:
        # load the image using OpenCV
        cvimg = cv.LoadImage(filename)
        npimg = imgutil.cv2array(cvimg)[:,:,numpy.r_[2, 1, 0]]
        
    return npimg

class ImageSource(pipeline.ProcessObject):
    '''
    A ProcessObject that is the source of images.  This represents the most
    upstream object in an image processing pipeline--the object that acquires
    images from a camera or reads images from disk.
    '''
    
    # define play modes as binary flags
    (pause,
     play, 
     reverse,
     beginning,
     end,
     seek) = map(lambda x: 2**x, range(6))
    
    allmodes = (pause|play|reverse|beginning|end|seek)
    
    def __init__(self):
        '''
        Initialize an ImageSource.  ImageSources have no input because they 
        occupy the beginning of a pipeline.
        '''
        super(ImageSource, self).__init__()
        
        # index starts at 0
        self._index = 0
        
        # initialize play modes
        self._allowedModes = ImageSource.pause|ImageSource.play
        self._loop = False
        self._playMode = 0
        self.setPlayMode(ImageSource.pause)
        
        # number of times generateData is called and time when initialized
        self._generateDataCount = 0
        self._timeInit = 0
        self.resetDataFrameRate()
        
    def generateData(self):
        '''
        The implementation in ImageSource keeps track of calls to generateData
        to record actual data generation frame rate.  Override this method in
        subclasses, but call the superclass to ensure this framerate is
        properly computed.
        '''
        self._generateDataCount += 1
        
    def __del__(self):
        '''
        Default deconstructor.  Override if image source needs to be 
        turned off or cleaned up at the end.
        '''
        pass
                    
    def getDataFrameRate(self):
        '''
        Get the average frame rate based on the number of times generateData has
        been called since the last call to resetDataFrameRate.
        '''
        return self._generateDataCount / (time.time() - self._timeInit)
    
    def resetDataFrameRate(self):
        '''
        Resets the frame rate calculation.
        '''
        self._generateDataCount = 0
        self._timeInit = time.time()
        
    def getIndex(self):
        '''
        Gets the current index.
        '''
        return self._index
    
    def setIndex(self, index):
        '''
        Sets the index to 'index' if it can. If larger than the last possible
        value: pause if loop is False, or go back to beginning if loop is True.
        '''
        if (self.getLength() > 0):
            # length has meaning
        
            if self.isLoop():
                # loop back to beginning
                self._index = index % self.getLength()
            
            elif (index < self.getLength()) and index >= 0:
                # index is within valid range
                self._index = index
                
            elif index < 0:
                self._index = 0
                self.setPlayMode(ImageSource.pause)
                
            else:
                # pause if not looping and index invalid
                self._index = self.getLength() - 1
                self.setPlayMode(ImageSource.pause)
        
        else:
            # length has no meaning (camera, filereader, etc)
            self._index = index
        
        self.modified()
        
    def getLength(self):
        '''
        Default implementation. Returns the number of frames in the file source
        and will return 0 for any file source that has no concept of length.
        '''
        return 0
        
    def getFrameName(self):
        '''
        Returns a str containing an appropriate name or label for the current
        frame.  The default implementation returns "image-<index>", but the
        implementation for a file stack reader would probably return the
        name of the underlying image file, without extension.
        '''
        return "image-{0:06d}".format(self.getIndex())

    def getFilename(self):
        '''
        Returns a string containing the source filename.  The default
        implementation returns "notafile"
        '''
        return "notafile"

    def getAllowedModes(self):
        '''
        The allowed play modes for the ImageSource.  Some binary mask of
        pause | play | reverse | beginning | end
        '''
        return self._allowedModes
    
    def setAllowedModes(self, modeFlags):
        self._allowedModes = modeFlags & ImageSource.allmodes
        
    def getPlayMode(self):
        '''
        Returns current play mode.
        '''
        return self._playMode
    
    def setPlayMode(self, mode):
        '''
        Sets the play mode to 'mode'.
        '''
        if self._allowedModes & mode == mode:
            self._playMode = mode
    
    def isLoop(self):
        '''
        Indicates whether this reader loops to the beginning at the end of the
        image source.
        '''
        return self._loop
        
    def setLoop(self, loop=True):
        '''
        Sets whether this reader loops to the beginning at the end of the
        image source.
        '''
        self._loop = loop
    
    def updatePlayMode(self):
        '''
        Modifies the image index on this ImageSource according to the
        current playMode.  This may advance, reverse, or reset the image
        index.  Note that if the image index changes, the ImageSource is
        modified and will generate new data on update().
        '''
        if self._playMode == ImageSource.pause:
            # don't need to update if paused
            pass
        
        elif self._playMode == ImageSource.play:
            self.setIndex(self.getIndex() + 1)
        
        elif self._playMode == ImageSource.reverse:
            self.setIndex(self.getIndex() - 1)
            
        elif self._playMode == ImageSource.beginning:
            self.setIndex(0)
            self.setPlayMode(ImageSource.pause)
            
        elif self._playMode == ImageSource.end:
            self.setIndex(self.getLength() - 1)
            self.setPlayMode(ImageSource.pause)        
        
class FileReader(ImageSource):
    '''
    An image source that reads an image in by filename.
    
    Note that the images returned by this image source are in RGB format; that
    is, the red and blue channels have been swapped from the original OpenCV
    input.
    
    Note that each time a file is read, the output image for this image source
    is modified.  An image can be copied with image = source.getOutput().copy().
    '''
    
    def __init__(self, filename=None):
        # setup output (no input)
        super(FileReader, self).__init__()
        self.setAllowedModes(ImageSource.pause)
        self.setPlayMode(ImageSource.pause)
        
        self._filename = filename
                
    def generateData(self):
        '''
        Load the image using OpenCV, converting to RGB.
        '''
        super(FileReader, self).generateData()
        
        output = readImageFile(self._filename)
        self.getOutput(0).setData(output)
        
    def getFilename(self):
        return self._filename
    
    def setFilename(self, filename):
        self._filename = filename
        self.modified()
        
    def getFrameName(self):
        filename = os.path.split(self._filename)[1]
        return os.path.splitext(filename)[0]
    
class FileStackReader(ImageSource):
    '''
    An image source that reads images from a stack of files.  An index keeps
    track of which image should be read in at the next update.

    When the index overruns the end of the file list, the stack reader can
    either loop through all the image files or stick at the last image file.
    
    Note that the images returned by this image source are in RGB format; that
    is, the red and blue channels have been swapped from the original OpenCV
    input.
    
    Note that each time an image is read, the output image data for this image
    source is overwritten.  An image can be copied to separate it from the
    pipeline: image = source.getOutput().copy()
    '''
    
    def __init__(self, files=[]):
        # setup output (no input)
        super(FileStackReader, self).__init__()
        self.setAllowedModes(ImageSource.allmodes)
        self._files = files
        
    def getFiles(self):
        '''
        The file names accessed by this image reader, a list of strings.
        '''
        return self._files

    def setFiles(self, files):
        '''
        Set the file names accessed by this image reader.
        '''
        self._files = files
        self.modified()
        
    def increment(self):
        '''
        Advances the image index to the next in the list.
        '''
        self.setIndex(self.getIndex() + 1)
        self.modified()
    
    def getLength(self):
        '''
        Get the number of files that this file stack will read from. 
        '''
        return len(self._files)
        
    def generateData(self):
        '''
        Load the image file with the current index into the output of
        this stack reader.
        '''
        super(FileStackReader, self).generateData()
        
        filename = self._files[self._index]
#        print "FileStackReader.generateData:", filename
        output = readImageFile(filename)
        self.getOutput(0).setData(output)
        
    def getFrameName(self):
        filename = os.path.split(self._files[self._index])[1]
        return os.path.splitext(filename)[0]
    
    def getFilename(self):
        '''
        Returns the filename for the current index.
        '''
        return self._files[self._index]
            
class CameraCV(ImageSource):
    '''
    Image source that captures images from a camera using OpenCV.  OpenCV
    cameras are accessed by a camera index.  
    
    The updateDelay parameter indicates how often the update() method should
    acquire new data from the camera.  After a frame acquisition, this camera
    class will wait at least updateDelay milliseconds before acquiring another
    frame.  It is a good idea to set this parameter to match the real update
    rate of the underlying camera, or even provide a value longer than the
    update cycle.  Note that get/setUpdateFramerate are an equivalent way to
    set this parameter.
    
    Note that the images returned by this image source are in RGB format; that
    is, the red and blue channels have been swapped from the original OpenCV
    input.
    '''
    
    def __init__(self, cameraId=0):
        '''
        Initialize a camera image source with a default updateDelay of 33 ms.
        '''
        # initialize output and input from camera
        super(CameraCV, self).__init__()
        self.setAllowedModes(ImageSource.pause|ImageSource.play)
        self.setPlayMode(ImageSource.play)
                
        self._camera = None
        self._cameraId = cameraId
        
        # connect to camera
        self.reconnect(cameraId)
    
    def reconnect(self, cameraId=0):
        '''
        Reconnect to camera.
        '''
        # disconnect from camera first
        self._camera = None
        
        # connect to it
        try:
            self._camera = cv.CaptureFromCAM(cameraId)
            self._cameraId = cameraId
        except cv.error:
            print "Could not open camera {0}".format(cameraId)
            self._camera = None
            
    def getCameraId(self):
        '''
        Returns camera device ID.
        '''
        return self._cameraId
        
    def generateData(self):
        '''
        Acquires image data from the camera.
        '''
        super(CameraCV, self).generateData()
        
        if self._camera is not None:
            cvimg = cv.QueryFrame(self._camera)
#            cv.CvtColor(cvimg, cvimg, cv.CV_BGR2RGB)
            output = imgutil.cv2array(cvimg)[..., numpy.r_[2, 1, 0]]
        else:
            # test image if camera doesn't work
            output = numpy.zeros((480,640,3), numpy.int8)
            
        self.getOutput(0).setData(output)
    
    def getFilename(self):
        return "camera{0}".format(self._cameraId)
    
    def __del__(self):
        self._camera = None

class CameraFW(ImageSource):
    '''
    Image source that captures images from a firewire camera.
    '''
    
    def __init__(self, cameraId=0,
                 videoMode=FirewireVideo.DC1394_VIDEO_MODE_640x480_YUV422,
                 framerate=FirewireVideo.DC1394_FRAMERATE_7_5):
        '''
        Open a video device with a set of camera parameters.  User can set the 
        firewire device number, the video mode (image size and color), shutter
        speed, and whether to use demosaicing (for raw video mode only).

        Other camera options can only be set through direct access to the 
        underlying camera object, from self.getCamera().
        '''
        super(CameraFW, self).__init__()
        self.setAllowedModes(ImageSource.pause|ImageSource.play)
        self.setPlayMode(ImageSource.play)
        
        self._cameraId = None
        self._camera = None
        self._flush = False
        self.reconnect(cameraId, videoMode, framerate)
        
    def disconnectPipeline(self):
        # when a camera is disconnected, disconnect from the firewire camera, too
        if self._camera is not None:
            self._camera.closeVideoDevice()
            self._camera = None
        super(CameraFW, self).disconnectPipeline()

    def __del__(self):
        if self._camera is not None:
            self._camera.closeVideoDevice()
            self._camera = None            
        
    def reconnect(self, cameraId=0, 
                  videoMode=FirewireVideo.DC1394_VIDEO_MODE_640x480_YUV422,
                  framerate=FirewireVideo.DC1394_FRAMERATE_7_5):
        if self._camera is not None:
            self._camera.closeVideoDevice()
        
        self._cameraId = cameraId
        self._camera = FirewireVideo.FirewireVideo(self._cameraId,
                                                   FirewireVideo.DC1394_ISO_SPEED_800)
        self._camera.setVideoMode(videoMode, framerate)
        self._camera.startTransmission()
        
    def getCameraId(self):
        '''
        Returns camera device ID.
        '''
        return self._cameraId
    
    def getCamera(self):
        '''
        Returns the firewire camera.
        '''
        return self._camera
    
    def isFlushing(self):
        return self._flush
    
    def setFlush(self, value):
        self._flush = value
        
    def generateData(self):
        '''
        Acquires image data from the camera.
        '''
        super(CameraFW, self).generateData()
        output = self._camera.acquireFrame(flush=self._flush)
        self.getOutput(0).setData(output)
        
    def getFilename(self):
        return "camera{0}".format(self._cameraId)

class VideoCV(ImageSource):
    '''
    Image source that acquires images from a video file using OpenCV.
    '''
    
    def __init__(self, filename=None):
        super(VideoCV, self).__init__()
        self.setAllowedModes(ImageSource.allmodes)
        self.setPlayMode(ImageSource.pause)
        
        self._filename = None
        self._video = None
        self._lastIndex = None
        self._length = 0
        
        self.setFilename(filename)
    
    def getLength(self):
        '''
        Returns the video frame count.
        '''
        # try to get the length from the file
        if self._length == 0:
            self.length = cv.GetCaptureProperty(self._video, cv.CV_CAP_PROP_FRAME_COUNT)
        
        return self.length
    
    def getVideoFPS(self):
        '''
        Returns the frames per second that the video was recorded at.
        '''
        return cv.GetCaptureProperty(self._video, cv.CV_CAP_PROP_FPS)
    
    def getFilename(self):
        return self._filename
    
    def setFilename(self, filename):
        self._filename = filename
        
        if filename is not None:
            self._video = cv.CaptureFromFile(self._filename)
        else:
            self._video = None
            
        self.modified()
    
    def generateData(self):
        '''
        Acquires frame data from the video.
        '''
        super(VideoCV, self).generateData()
        
        # the ui will try to load a single frame multiple times
        # TODO: fix the playback model to use a single clock.
        if self._lastIndex == self.getIndex():
            return
        
        # try to set the frame position if we can't just grab the next one
        if self._lastIndex is not None and self._lastIndex != self.getIndex() - 1:
            retval = cv.SetCaptureProperty(self._video, cv.CV_CAP_PROP_POS_FRAMES, self.getIndex())
            # print("seeking frame {0:4} from {1:4} => {2}".format(self.getIndex(), self._lastIndex, retval))

            # reload video file if there was a problem (often happens after reaching the end of a video)
            if retval == 0:
                self.setFilename(self._filename)
        
        # grab the next frame
        cvimg = cv.QueryFrame(self._video)
        self._lastIndex = self.getIndex()
        
        # set as output
        output = imgutil.cv2array(cvimg)[:,:,numpy.r_[2, 1, 0]]
        self.getOutput(0).setData(output)
        
    def getFrameName(self):
        filename = os.path.split(self.getFilename())
        filebase = os.path.splitext(filename[-1])[0]
        return "{0:s}-{1:06d}".format(filebase, int(self.getIndex()))
    
    def disconnectPipeline(self):
        # clean up the video file
        if self._video is not None:
            self._video = None
        super(VideoCV, self).disconnectPipeline()
        
    def __del__(self):
        if self._video is not None:
            self._video = None
    
def testCameraFW():    
    # acquire images from the firewire camera
    camfw = CameraFW(0, 
                     FirewireVideo.DC1394_VIDEO_MODE_800x600_RGB8,
                     FirewireVideo.DC1394_FRAMERATE_15)
    
    # cycle through images until escape is pressed
    key = None
    i = 0
    while key != 27:
        camfw.updatePlayMode()
        camfw.update()
        key = imgutil.imageShow(camfw.getOutput(0).getData(), "pipeline", False, 10)
        
        i += 1
        if i % 20 == 0:
            print "{0}: {1:8.3f} fps".format(i, camfw.getCamera().computeFramerate())
        
if __name__ == "__main__":    
    testCameraFW()
