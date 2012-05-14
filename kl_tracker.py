#!/usr/bin/env python

# David Cain
# Justin Sperry
# 2012-04-17
# CS365, Brian Eastwood

import cv
import cv2
import glob
import numpy
import pylab
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
        


'''
Basic implementation of the KLT Tracker discussed in Shi & Tomasi

'''
class KLTracker(pipeline.ProcessObject):

    def __init__(self, inpt=None, features=None, tensor=None, spdev=None):
        pipeline.ProcessObject.__init__(self, inpt, inputCount=4)
        self.last_features = None
        self.last_img = None
        self.patches = None
        self.frame_number = 0
        self.iterations = 10
        self.feature_list = []
        self.sigma_d = 2.0
        
        #initialize inputs
        self.setInput(features, 1)
        self.setInput(tensor, 2)
        self.setInput(spdev, 3)
        
    def generateData(self):
        print "Frame: %d" % (self.frame_number)
        
        # if on the first frame
        if self.frame_number == 0:
            img = self.getInput(0).getData()
            self.last_img = img
            
            harris_corners = self.getInput(1).getData()
            #NumFeaturesx4 array containing the position, strength & active flag
            features = numpy.ones((harris_corners.shape[0],4), dtype=numpy.float32)
            features[:,:2]=harris_corners[:,:2] #->initialize features to harris corners
            self.last_features = features # 
            self.getOutput(0).setData(features)
            self.feature_list.append(features)
            
            
            #make a bank of patches
            r = numpy.floor(self.sigma_d * 5.0/2)
            self.patches = numpy.zeros((harris_corners.shape[0], (2*r+1)**2), dtype=numpy.float32)
            for i, (x,y, _, _) in enumerate(features):
                self.patches[i,...] = img[y-r:y+r+1, x-r:x+r+1].flatten()
        else:
            features = numpy.copy(self.last_features)
            img = self.getInput(0).getData()
            Ixx, Iyy, Ixy = self.getInput(2).getData()
            Ix, Iy = self.getInput(3).getData()
            
            #initialize features to be stationary
            v = numpy.array([0.0, 0.0]) 
            
            #loop through all features
            for i in range(features.shape[0]):
                
               #skip if not active 
               if features[i][3] != 0:
                    x,y = features[i][:2]
                    
                    #compute A^T*A
                    A = numpy.array([[Ixx[y,x],Ixy[y,x]],
                                      [Ixy[y,x],Iyy[y,x]]])            
                    
                    #velocity difference
                    delta_v = numpy.array([100.0,100.0])
                    
                    g = imgutil.gaussian(self.sigma_d)[0]
                    g = g[:,None]
                    gg = numpy.dot(g, g.transpose()).flatten()
                    r = g.size/2
                    iyy, ixx = numpy.mgrid[-r:r+1, -r:r+1]
                    ryy = y+iyy
                    rxx = x+ixx
                    
                    patchIx = interpolation.map_coordinates(Ix, numpy.array([ryy.flatten(), rxx.flatten()]))
                    patchIy = interpolation.map_coordinates(Iy, numpy.array([ryy.flatten(), rxx.flatten()]))
                    
                    #loop through to calculate velocity
                    iterations = 10
                    eps = float('1.0e-3')**2
                    while iterations > 0 and numpy.dot(delta_v, delta_v)> eps: 
                    
                        curr_patch = interpolation.map_coordinates(img, numpy.array([ryy.flatten(), rxx.flatten()]))
                        prev_patch = interpolation.map_coordinates(self.last_img,
                            numpy.array([(ryy-v[1]).flatten(), (rxx-v[0]).flatten()]))
                        
                        #find temporal derivative
                        patch_tderiv = curr_patch-prev_patch
                        pIxt = (patch_tderiv*patchIx*gg).sum()
                        pIyt = (patch_tderiv*patchIy*gg).sum()
                        
                        #ATA = -ATb
                        ATb = -numpy.array([pIxt, pIyt])
                        delta_v = numpy.linalg.lstsq(A, ATb)[0]
                        
                        #update velocities
                        v += delta_v
                        iterations -= 1 
                    
                    
                    # find positions of new features
                    features[i][:2]+= v
                    
                    # weight for similarity to initial feature patch
                    features[i][2] = imgutil.ncc(curr_patch, self.patches[i,...])
                    
                    # make inactive if out of frame
                    h,w = img.shape
                    if features[i][0] > w or features[i][1] > h or features[i][0] < 0 or features[i][1] < 0:
                        features[i][3] = 0
                
            self.last_features = features
            self.last_img = img
            self.getOutput(0).setData(features)
            self.feature_list.append(features)
            
        self.frame_number += 1
    
    def plotFeatureList(self):
        features = numpy.array(self.feature_list)
        weights = features[:,2]
        for i in range(weights.shape[0]):
            weight_time = pylab.plot(weights[i,...])
        pylab.show()
        
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

        
#returns a tuple of the components of the structure tensor
class StructureTensor(pipeline.ProcessObject):

    def __init__(self, inpt=None, sigmaD=2.0, sigmaI=3.0):
        pipeline.ProcessObject.__init__(self, inpt, outputCount = 3)
        self.sigma_D = sigmaD
        self.sigma_I = sigmaI
    
    def generateData(self):
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
    
    def __init__(self, inpt=None, name="pipeline"):
        pipeline.ProcessObject.__init__(self, inpt)
        cv2.namedWindow(name, cv.CV_WINDOW_NORMAL)
        self.name = name
        
    def generateData(self):
        inpt = self.getInput(0).getData()
        # output here so channels don't get flipped
        self.getOutput(0).setData(inpt)

        # Convert back to OpenCV BGR from RGB
        if inpt.ndim == 3 and inpt.shape[2] == 3:
            inpt = inpt[..., ::-1]
        
        cv2.imshow(self.name, inpt.astype(numpy.uint8))

    def destroy(self):
        cv2.destroyWindow(self.name)
        

def main():
    """
        Obtain a time sequence of microscope slides, track the stage's movement
        by tracking image movement.
    """
    key = None
    image_dir = "images_100"
    images = sorted(glob.glob("%s/*.npy" % image_dir))
    fileStackReader  = FileStackReader(images)

    grayscale = Grayscale(fileStackReader.getOutput())

    tensor = StructureTensor(grayscale.getOutput())
    harris = HarrisDetection(tensor.getOutput(1), numFeatures = 15) # pass Harris the tensor

    labeled = DisplayLabeled(fileStackReader.getOutput(), harris.getOutput(1))
    display = Display(labeled.getOutput(), "Harris")
    
    # NOTE/TODO: tensor output is no longer color
    tracker = KLTracker(grayscale.getOutput(), harris.getOutput(1),
                        tensor.getOutput(1), tensor.getOutput(2))
                        
    # Displays color outputs
    track_labeled = DisplayLabeled(fileStackReader.getOutput(), tracker.getOutput())
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

        # Save the keypress (make sure converted to byte size)
        key = cv2.waitKey(10)
        key &= 255

    display.destroy()
    display2.destroy()
    tracker.plotFeatureList()


if __name__ == "__main__":
    main()
