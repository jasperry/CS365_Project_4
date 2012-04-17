from scipy import ndimage
from scipy.ndimage import filters
import cv
import cv2
import numpy
import pipeline


class HarrisDetection(pipeline.ProcessObject):
    
    def __init__(self, input = None, sigmaD=1.0, sigmaI=1.5, numFeatures = 100):
        pipeline.ProcessObject.__init__(self, input, outputCount = 2)
        self.sigma_D = sigmaD
        self.sigma_I = sigmaI
        self.numFeatures = numFeatures
        
    def generateData(self):
        Ixx, Iyy, Ixy = self.getInput(0).getData()
        
        imgH = (Ixx * Iyy - Ixy**2) / (Ixx + Iyy + 1e-8)
            
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
        self.getOutput(0).setData(input)
        self.getOutput(1).setData(features)
        
    

'''
Basic implementation of the KLT Tracker discussed in shi & tomasi

'''
class KLTracker(pipeline.ProcessObject):
    
    def __init__(self, ik = None, ikplusone = None, features = None, tensor = None,
            patchn = 5, ):
            
        pipeline.ProcessObject.__init__(self, ik, 4)
        self.setInput(ikplusone, 1)
        self.setInput(features, 2)
        self.setInput(tensor, 3)
        self.frame = 0
    
    def generateData(self):
        I0 = self.getInput(0).getData()
        I1 = self.getInput(1).getData()
        features = self.getInput(2).getData()
        Ixx, Iyy, Ixy = self.getInput(3).getData()
        It = Ikplusone - Ik
        
        for i in range(features.shape[0]):
            if features[i,3] == True:
            	#pull x and y from the feature
                y = features[i,0]
                x = features[i,1]
                
                #compute A^T*A
                A = numpy.matrix([[Ixx,Ixy],[Ixy, Iyy]])
                
                count = 0
                while count < 5
                #start with a single iteration
                # hardcode sigmaI right in there(#djykstrawouldntlikeit)
                g = imgutil.gaussian(1.5)
                gg = numpy.dot(g.transpose(),g).flatten() 
                r = g.size/2
                
                iyy, ixx = numpy.mgrid[-r:r+1,-r:r+1]
                ryy = iyy + y
                rxx = ixx +x
                
                patchcoords  = numpy.vstack((ryy.flatten(), rxx.flatten()))
                
                patchI1 = interpolation.map_coordinates(I, (ryy, rxx))
                patchIt = patch
                
        

        
class DisplayLabeled(pipeline.ProcessObject):
    def __init__(self, input = None, features = None):
        pipeline.ProcessObject.__init__(self, input, 2)
        self.setInput(features, 1)
        
    def generateData(self):
        input = self.getInput(0).getData()
        features = self.getInput(1).getData()

        
#returns a tuple of the components of the structure tensor
class StructureTensor(pipeline.ProcessObject):

    def __init__(self, input = None, sigmaD=1.0, sigmaI=1.5):
        pipeline.ProcessObject.__init__(self, input, outputCount = 2)
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
        

class Display(pipeline.ProcessObject):
    
    def __init__(self, input = None, name = "pipeline"):
        pipeline.ProcessObject.__init__(self, input)
        cv2.namedWindow(name, cv.CV_WINDOW_NORMAL)
        self.name = name
        
    def generateData(self):
        input = self.getInput(0).getData()
        # output here so channels don't get flipped
        self.getOutput(0).setData(input)

        # Convert back to OpenCV BGR from RGB
        if input.ndim == 3 and input.shape[2] == 3:
            input = input[..., ::-1]
        
        cv2.imshow(self.name, input.astype(numpy.uint8))
        
        
if __name__ == "__main__":
    key = cv2.waitKey(10)
    while key != 27:
        fileStackReader.increment()
        #print fileStackReader.getFrameName()
        display1.update()
        #display2.update()
        #display3.update()
        cv2.waitKey(10)
