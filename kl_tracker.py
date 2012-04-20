#!/usr/bin/env python

# David Cain
# Justin Sperry
# 2012-04-17
# CS365, Brian Eastwood

import cv
import cv2
import glob
import numpy
from scipy import ndimage
from scipy.ndimage import filters
from scipy.ndimage import interpolation

import imgutil
import pipeline
from source  import FileStackReader


class HarrisDetection(pipeline.ProcessObject):
    """
        Given an input image, and some adjustment parameters, find the
        highest-scoring Harris corner features for tracking.

        Outputs two objects- the initial image, and an array of (x,y)
        feature locations and their scores
    """
    
    def __init__(self, tensor=None, sigmaD=1.0, sigmaI=1.5, numFeatures=12,
            filter_by_mean=False):
        """
            Pass in a tensor, set parameters for corner detection,
            and specify max number of features.
        """
        pipeline.ProcessObject.__init__(self, tensor, outputCount = 2)
        self.sigma_D = sigmaD
        self.sigma_I = sigmaI
        self.numFeatures = numFeatures
        self.filter_by_mean = filter_by_mean
        
    def generateData(self):
        """
            Find the highest scoring features for the image, return a
            sorted array of coordinates and feature scores.
        """
        inpt = self.getInput(0).getData()#.astype(numpy.float32)
        Ixx, Iyy, Ixy = self.getInput(0).getData()
        
        imgH = (Ixx * Iyy - Ixy**2) / (Ixx + Iyy + 1e-8)
        (h, w) = imgH.shape[:2]
            
        # exclude points near the image border
        imgH[:16, :] = 0
        imgH[-16:, :] = 0
        imgH[:, :16] = 0
        imgH[:, -16:] = 0

        #non-max suppression
        localMax = filters.maximum_filter(imgH, (5,5))
        imgH = imgH * (imgH == localMax)
        
        #sort features by strength
        sortIdx = numpy.argsort(imgH.flatten())[::-1]
        sortIdx = sortIdx[:self.numFeatures]
        
        #find x,y positions
        yy = sortIdx / w
        xx = sortIdx % w
            
        #add together
        features = numpy.vstack((xx, yy, imgH.flatten()[sortIdx])).transpose()
        assert len(features) == self.numFeatures

        # Filter out Harris corner strengths that aren't some percentage
        # of mean strength (if the option was enabled)
        if self.filter_by_mean:
            strengths =  features[:,2]
            print "Mean: %s"  % numpy.mean(strengths)
            print "Median: %s" % numpy.median(strengths)
            print "Range: [%.2f-%.2f] " % (numpy.min(strengths), numpy.max(strengths))

            threshold = numpy.mean(strengths)

            features = numpy.array([f for f in features if f[2] > threshold])
            print "%i features found with threshold %i" % (len(features), threshold)

        # for (x, y, value) in features, tack on the active flag as true
        self.getOutput(0).setData(inpt)
        self.getOutput(1).setData(numpy.hstack((features, numpy.ones((features.shape[0],1)))))
        


class KLTracker(pipeline.ProcessObject):
    """
        Track features within an image using the KLT Tracker discussed
        in Shi & Tomasi.
        
        Requires a list of corner features, and the structure tensor.
    """
    
    def __init__(self, I=None, features=None, tensor=None, spdev=None ):
        """
            Reads in two frames (i0, i1), Harris corner features, the
            tensor, and spatial derivatives from the tensor.
        """
            
        pipeline.ProcessObject.__init__(self, I, 4,2) # 4 inputs, 2 outputs
        self.setInput(features, 1)
        self.setInput(tensor, 2)
        self.setInput(spdev, 3)
        self.frame_number = 0
        self.last_frame = None
        self.framelist = []
    
    def generateData(self):
    
        print "On frame %d"%(self.frame_number)
        
        #first frame setup
        if self.last_frame == None:
            self.last_frame = self.getInput(0).getData()
            self.framelist.append(self.getInput(1).getData().astype(numpy.float32))
            self.getOutput(0).setData(self.last_frame)
            self.getOutput(1).setData(self.framelist[0])
        
        #all others
        else:
            I1 = self.getInput(0).getData()
            
            Ixx, Iyy, Ixy = self.getInput(2).getData()
            Ix, Iy = self.getInput(3).getData()
            
            features = self.framelist[self.frame_number-1]
            # new frame to put this feature data in
            newFrame = numpy.zeros(features.shape)
            
            num_lost = 0
            #loop through features
            for i in range(features.shape[0]):
                # if the feature is active
                if features[i,3] > 0:
                    #pull x and y from the feature
                    x = features[i,0]
                    y = features[i,1]
                    s = features[i,2]
                    
                    #compute A^T*A
                    A = numpy.matrix([[Ixx[y,x],Ixy[y,x]],
                                      [Ixy[y,x],Iyy[y,x]]])
                    
                    # hardcode sigmaI right in there(#djykstrawouldntlikeit)
                    g= imgutil.gaussian(1.5)[0]
                    g = g[:,None]
                    gg = numpy.dot(g, g.transpose()).flatten() 
                    r = g.size/2
                    
                     #create x, y pairs for the patch
                    iyy, ixx = numpy.mgrid[-r:r+1,-r:r+1]
                    ryy = y + iyy
                    rxx = x + ixx
                    
                    patchIx = interpolation.map_coordinates(Ix, numpy.array([ryy.flatten(), rxx.flatten()]))
                    patchIy = interpolation.map_coordinates(Iy, numpy.array([ryy.flatten(), rxx.flatten()]))
                    
                    duv = numpy.array([100.0,100.0]) 
                    
                    iterations = 10
                    epsilon = float('1.0e-3')**2
                    U = 0
                    V = 0
                    
                    # iterates to find the temporal derivative multiple times
                    #change to have distance threshold as opposed to simple number iterations
                    while iterations > 0 and numpy.dot(duv, duv) > epsilon:
                        
                        #grab patches from each of the Images
                        patchI1 = interpolation.map_coordinates(I1, numpy.array([ryy.flatten(), rxx.flatten()]))
                        patchI0 = interpolation.map_coordinates(self.last_frame, numpy.array([ryy.flatten(), rxx.flatten()]))
                        
                        #calculate It and a new ATb
                        patchIt = patchI1 - patchI0
                        GIxIt = (patchIt * patchIx * gg).sum()
                        GIyIt = (patchIt * patchIy * gg).sum()
                        ATb = -numpy.array([GIxIt,GIyIt])
                        
                        #solve for Av = ATb
                        duv = numpy.linalg.lstsq(A, ATb)[0]
                        
                        U = U + duv[0]
                        V = V + duv[1]
                        
                        iterations -= 1
                    
                    #update X and Y positions for object
                    newX = x + U
                    newY = y + V
                    
                    #if feature is still in frame, keep as active
                    active = 1
                    if newX > I1.shape[1] or newX < 0 or newY > I1.shape[0] or newY < 0:
                        active = 0
                        num_lost += 1
                        
                    newFrame[i]  = numpy.array([newX, newY, s, active])
            
            print "%d Features lost in frame %d " % (num_lost, self.frame_number)
            self.framelist.append(newFrame)
            self.last_frame = I1
            self.getOutput(0).setData(I1)
            self.getOutput(1).setData(newFrame)
        self.frame_number += 1
        #replaces last frame
        
    def getFrameList(self):
        """
            Returns the frame list for use in plotting, etc
        """
        return self.framelist
        
class DisplayLabeled(pipeline.ProcessObject):
    """
        Display the image with features outlined by a red box.
    """
    def __init__(self, inpt=None, features=None):
        pipeline.ProcessObject.__init__(self, inpt, inputCount=2)
        self.setInput(features, 1)

    def generateData(self):
        """
            For each feature, draw a rectangle around its x,y point.
        """
        inpt = numpy.copy(self.getInput(0).getData())
        features = self.getInput(1).getData()

        box_color = (255, 0, 0) # red
        r = 5 # half the width of the rectangle
        for (x, y, val, a) in features:
            top_left = ( int(x-r), int(y-r) )
            bottom_right = ( int(x+r), int(y+r) )
            cv2.rectangle(inpt, top_left, bottom_right, box_color, thickness=2)
        self.getOutput(0).setData(inpt)

        
class Grayscale(pipeline.ProcessObject):
    """
        Convert a color image to grayscale
    """
    def __init__(self, inpt=None):
        pipeline.ProcessObject.__init__(self, inpt)
        
    def generateData(self):
        inpt = self.getInput(0).getData()
        
        if inpt.ndim == 3 and inpt.shape[2] == 3:
            output = inpt[...,0]*0.114 + inpt[...,1]*0.587 + inpt[...,2]*0.229

        self.getOutput(0).setData(output)

        
class StructureTensor(pipeline.ProcessObject):
    """
        Takes in an image pipeline object, and outputs a tuple of the
        components of the structure tensor.
    """

    def __init__(self, inpt = None, sigmaD=1.0, sigmaI=1.5):
        pipeline.ProcessObject.__init__(self, inpt, outputCount = 3)
        self.sigma_D = sigmaD
        self.sigma_I = sigmaI
    
    def generateData(self):
        """
            Generate the structure tensor data.
        """
        inpt = self.getInput(0).getData()
         
        Ix = ndimage.filters.gaussian_filter1d(inpt, self.sigma_D, 0, 0)
        Ix = ndimage.filters.gaussian_filter1d(Ix, self.sigma_D, 1, 1)
        Iy = ndimage.filters.gaussian_filter1d(inpt, self.sigma_D, 1, 0)
        Iy = ndimage.filters.gaussian_filter1d(Iy, self.sigma_D, 0, 1)
        
        Ixx = ndimage.filters.gaussian_filter1d(Ix**2, self.sigma_I, 0, 0)
        Ixx = ndimage.filters.gaussian_filter1d(Ixx, self.sigma_I, 1, 1)
        Iyy = ndimage.filters.gaussian_filter1d(Iy**2, self.sigma_I, 0, 0)
        Iyy = ndimage.filters.gaussian_filter1d(Iyy, self.sigma_I, 1, 1)
        Ixy = ndimage.filters.gaussian_filter1d(Ix * Iy, self.sigma_I, 0,0)
        Ixy = ndimage.filters.gaussian_filter1d(Ixy, self.sigma_I, 1, 1)
        
        self.getOutput(0).setData(inpt)
        self.getOutput(1).setData((Ixx,Iyy,Ixy))
        self.getOutput(2).setData((Ix,Iy))
        
class Display(pipeline.ProcessObject):
    """
        Displays the pipeline object in a CV window, flipping the
        channels for proper display
    """
    
    def __init__(self, inpt = None, name = "pipeline"):
        """
            Initialize a named CV window to show the image in
        """
        pipeline.ProcessObject.__init__(self, inpt)
        cv2.namedWindow(name, cv.CV_WINDOW_NORMAL)
        self.name = name
        
    def generateData(self):
        """
            Flip the channels, display the image in the created window.
        """
        inpt = self.getInput(0).getData()
        # output here so channels don't get flipped
        self.getOutput(0).setData(inpt)

        # Convert back to OpenCV BGR from RGB
        if inpt.ndim == 3 and inpt.shape[2] == 3:
            inpt = inpt[..., ::-1]
        
        cv2.imshow(self.name, inpt.astype(numpy.uint8))

    def destroy(self):
        """
            Destroy the created CV window.
        """
        cv2.destroyWindow(self.name)

def main():
    """
        Obtain a time sequence of microscope slides, track the stage's
        movement by tracking image movement.
    """
    key = None
    image_dir = "images_100"
    images = sorted(glob.glob("%s/*.npy" % image_dir))
    assert len(images) > 0, "No .npy images found in '%s'" % image_dir
    fileStackReader  = FileStackReader(images)

    grayscale = Grayscale(fileStackReader.getOutput())

    tensor = StructureTensor(grayscale.getOutput())
    harris = HarrisDetection(tensor.getOutput(1)) # pass Harris the tensor

    labeled = DisplayLabeled(fileStackReader.getOutput(), harris.getOutput(1))
    display = Display(labeled.getOutput(), "Harris")
    
    tracker = KLTracker(grayscale.getOutput(), harris.getOutput(1),
                        tensor.getOutput(1), tensor.getOutput(2))
                        
    # Displays color outputs
    track_labeled = DisplayLabeled(fileStackReader.getOutput(), tracker.getOutput(1))
    display2 = Display(track_labeled.getOutput(), "Tracking" )
    

    # Note the time of the first capture
    first_frame = fileStackReader.getFrameName()
    start_time = int(first_frame)

    while key != 27:
        # Print the elapsed time since capture start
        fileStackReader.increment()
        capture_time = int(fileStackReader.getFrameName())
        print "  +%ims" % (capture_time - start_time)

        # Update all the pipeline objects
        tensor.update()
        harris.update()
        display.update()
        labeled.update()
        tracker.update()
        track_labeled.update()
        display2.update()

        # Save the keypress (make sure converted to ASCII value)
        key = cv2.waitKey(10)
        key &= 255

    display.destroy()
    display2.destroy()


if __name__ == "__main__":
    main()
