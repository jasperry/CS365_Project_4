from scipy import ndimage
from scipy.ndimage import filters
import cv
import cv2
import glob
import numpy

import pipeline
from source  import FileStackReader


class HarrisDetection(pipeline.ProcessObject):
    
    def __init__(self, inpt=None, sigmaD=1.0, sigmaI=1.5, numFeatures=100):
        pipeline.ProcessObject.__init__(self, inpt, outputCount = 2)
        self.sigma_D = sigmaD
        self.sigma_I = sigmaI
        self.numFeatures = numFeatures
        
    def generateData(self):
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
        # for (x, y, value) in features
        self.getOutput(0).setData(inpt)
        self.getOutput(1).setData(features)
        

'''
Basic implementation of the KLT Tracker discussed in Shi & Tomasi

'''
class KLTracker(pipeline.ProcessObject):
    
    def __init__(self, I=None, features=None, tensor=None, spdev=None ):
        """
            Reads in two frames (i0, i1), Harris corner features, the
            tensor, and spatial derivatives from the tensor.
        """
            
        pipeline.ProcessObject.__init__(self, I, 4,2) # 5 inputs, 2 outputs
        self.setInput(features, 1)
        self.setInput(tensor, 2)
        self.setInput(spdev, 3)
        self.frame_number = 0
        self.last_frame = None
        self.framelist = []
    
    def generateData(self):
        
        #first frame setup
        if self.last_frame == None:
            self.last_frame = self.getInput(0).getData()
            self.framelist.append(self.getInput(1).getData())
            self.getOutput(0).setData(self.last_frame)
            self.getOutput(1).setData(self.framelist[0])
        
        #all others
        else:
        
            I0 = self.last_frame
            I1 = self.getInput(0).getData()
            
            #replaces last frame
            self.last_frame = I1
            
            Ixx, Iyy, Ixy = self.getInput(2).getData()
            Ix, Iy = self.getInput(3).getData()
            
            features = self.framelist[self.frame_number-1]
            # new frame to put this feature data in
            newFrame = numpy.zeros(features.shape)
            
            #loop through features
            for i in range(features.shape[0]):
                # if the feature is active
                if features[i,2] == 1:
                    #pull x and y from the feature
                    y = features[i,0]
                    x = features[i,1]
                    
                    #compute A^T*A
                    A = numpy.matrix([[Ixx,Ixy],[Ixy, Iyy]])
                    
                    # hardcode sigmaI right in there(#djykstrawouldntlikeit)
                    g = imgutil.gaussian(1.5)
                    gg = numpy.dot(g.transpose(),g).flatten() 
                    r = g.size/2
                    
                    count = 0
                    U, V = 0
                    
                    # iterates to find the temporal derivative multiple times
                    #change to have distance threshold as opposed to simple number iterations
                    while count < 5:
                        
                        #create x, y pairs for the patch
                        iyy, ixx = numpy.mgrid[-r:r+1,-r:r+1]
                        ryy = iyy + y
                        rxx = ixx +x
                        patchcoords  = numpy.vstack((ryy.flatten(), rxx.flatten()))
                        
                        #grab patches from each of the Images
                        patchI1 = interpolation.map_coordinates(I1, patchcoords)
                        patchI0 = interpolation.map_coordinates(I0, patchcoords)
                        patchIx = interpolation.map_coordinates(Ix, patchcoords)
                        patchIy = interpolation.map_coordinates(Ix, patchcoords)
                        
                        #calculate It and a new ATb
                        patchIt = patchI1 - patchI0
                        GIxIt = (patchIt * patchIx * gg).sum()
                        GIyIt = (patchIt * patchIy * gg).sum()
                        ATb = numpy.matrix([[GixIt],
                                            [GiyIt]])
                        
                        #solve for Av = ATb
                        duv = numpy.linalg.lstsq(A, ATb)
                        
                        U = U + duv[0]
                        V = v + duv[1]
                        
                        count += 1
                    
                    #update X and Y positions for object
                    newX = x + U
                    newY = y + V
                    
                    #if feature is still in frame, keep as active
                    active = 0
                    if newX < I1.shape[1] and newY < I1.shape[0]:
                        active = 1
            
                    newFrame[i]  = np.array([newX, newY, active])
            
            
            self.framelist.append(newFrame)     
            self.getOutput(0).setData(I1)
            self.getOutput(1).setData(newFrame)
        self.frame_number += 1
        
    #returns the frame list for use plotting, etc
    def getFrameList(self):
        return self.framelist
        
class DisplayLabeled(pipeline.ProcessObject):
    def __init__(self, inpt = None, features = None):
        pipeline.ProcessObject.__init__(self, inpt, inputCount=2)
        self.setInput(features, 1)

    def generateData(self):
        inpt = numpy.copy(self.getInput(0).getData()) # TODO: numpy copy here
        features = self.getInput(1).getData()

        box_color = (255, 0, 0) # red
        r = 5 # half the width of the rectangle
        for (x, y, val) in features:
            top_left = ( int(x-r), int(y-r))
            bottom_right = ( int(x+r), int(y+r))
            cv2.rectangle(inpt, top_left, bottom_right, box_color, thickness=2)
        self.getOutput(0).setData(inpt)

        
#returns a tuple of the components of the structure tensor
class StructureTensor(pipeline.ProcessObject):

    def __init__(self, inpt = None, sigmaD=1.0, sigmaI=1.5):
        pipeline.ProcessObject.__init__(self, inpt, outputCount = 3)
        self.sigma_D = sigmaD
        self.sigma_I = sigmaI
    
    def generateData(self):
        inpt = self.getInput(0).getData().astype(numpy.float32)
        grayscale = inpt[..., 1]
         
        Ix = ndimage.filters.gaussian_filter1d(grayscale, self.sigma_D, 0, 0)
        Ix = ndimage.filters.gaussian_filter1d(Ix, self.sigma_D, 1, 1)
        Iy = ndimage.filters.gaussian_filter1d(grayscale, self.sigma_D, 1, 0)
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
    
    def __init__(self, inpt = None, name = "pipeline"):
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
    key = None
    image_dir = "images_100"
    images = sorted(glob.glob("%s/*.npy" % image_dir))
    fileStackReader  = FileStackReader(images)

    tensor = StructureTensor(fileStackReader.getOutput())
    harris = HarrisDetection(tensor.getOutput(1)) # pass Harris the tensor
    #display = Display(fileStackReader.getOutput()) # display the raw image

    labeled = DisplayLabeled(fileStackReader.getOutput(), harris.getOutput(1))
    display = Display(labeled.getOutput(), "Harris")
    
    tracker = KLTracker(fileStackReader.getOutput(), harris.getOutput(1),
                        tensor.getOutput(1), tensor.getOutput(2))
                        
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

        tensor.update()
        harris.update()
        display.update()
        labeled.update()
        tracker.update()
        track_labeled.update()
        display2.update()

        key = cv2.waitKey(10)
        key &= 255
    display.destroy()
    display2.destroy()


if __name__ == "__main__":
    main()
