import imgutil
import pipeline
import source
import numpy
import os
import glob
import optparse
import cv2
import cv
import shelve
from scipy import ndimage



class HarrisDetection(pipeline.ProcessObject):
    
    def __init__(self, input = None, sigmaD=1.0, sigmaI=1.5, numFeatures = 100):
        pipeline.ProcessObject.__init__(self, input)
        self.setOutputCount(2)
        self.sigma_D = sigmaD
        self.sigma_I = sigmaI
        self.numFeatures = numFeatures
        
    def generateData(self):
        Ixx, Iyy, Ixy = self.getInput(0).getData()
        
        
        imgH = (Ixx * Iyy - Ixy**2) / (Ixx + Iyy + 1e-8)
        self.getOutput(0).setData(input)
            
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
        self.getOutput(1).setData(features)
        
    


class KLTracker(pipeline.ProcessObject):
    
    def __init__(self, input = None):
        pipeline.ProcessObject.__init__(self, input)
        

        
#returns a tuple of the components of the structure tensor
class StructureTensor(pipeline.ProcessObject):

    def __init__(self, input = None, sigmaD=1.0, sigmaI=1.5):
        pipeline.ProcessObject.__init__(self, input)
        self.setOutputCount(2)
        self.sigma_D = sigmaD
        self.sigma_I = sigmaI
    
     def generateData(self):
        input = self.getInput(0).getData().astype(numpy.float32)
         
        Ix = ndimage.filters.gaussian_filter1d(input, self.sigma_D, 0, 0)
        Ix = ndimage.filters.gaussian_filter1d(Ix, self.sigma_D, 1, 1)
        Iy = ndimage.filters.gaussian_filter1d(input, self.sigma_D, 1, 0)
        Iy = ndimage.filters.gaussian_filter1d(Iy, self.sigma_D, 0, 1)
        
        
        Ixx = ndimage.filters.gaussian_filter1d(Ix**2, self.sigma_I, 0, 0)
        Ixx = ndimage.filters.gaussian_filter1d(Ixx, self.sigma_I, 1, 1)
        Iyy = ndimage.filters.gaussian_filter1d(Iy**2, self.sigma_I, 0, 0)
        Iyy = ndimage.filters.gaussian_filter1d(Iyy, self.sigma_I, 1, 1)
        Ixy = ndimage.filters.gaussian_filter1d(Ix * Iy, self.sigma_I, 0,0)
        Ixy = ndimage.filters.gaussian_filter1d(Ixy, self.sigma_I, 1, 1)
        
        self.getOutput(0).setData(input)
        self.getOutput(1).setData((Ixx,Iyy,Ixy))