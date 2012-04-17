'''
Created on May 11, 2010

Defines the fundamental elements of an image processing pipeline.  This pipeline
architecture is modeled after the one found in the Insight Toolkit (ITK), 
although this one is more basic.  In the pipeline model, image processing
networks are created by linking together a chain of processing components.  When
one component changes (e.g. via a parameter change), its output gets
regenerated and passed to all downstream components.

Changes are propagated downstream because all components ask for changes from 
upstream components before updating themselves.

The mechanism that triggers whether data needs to be regenerated is based on
timestamps for when an object is modified and updated.  If a process object
has been modified since the last time it was updated, data will be regenerated.

@author: bseastwo
'''

import time

class ProcessObject(object):
    '''
    ProcessObject is the base for classes that process images.  This is the 
    foundation of a basic pipelining architecture.  ProcessObjects perform
    operations on image data in a lazy manner--changes are made only when
    needed.  When a change is made to the parameters of a ProcessObject, it
    should be marked as modified() to signal that the data should be
    updated.  On update, all upstream (input) ProcessObjects are updated before
    the data for this ProcessObject is updated.  In this way, the update uses
    the most recent data.
    
    A ProcessObject can have any number of inputs and outputs.
    
    Note that because this object is used so heavily, a lot of the methods
    do not use much data verification, so it is expected that one will get
    exceptions if they have hooked up process objects that are not properly
    initialized.
    '''
    
    def __init__(self, input=None, inputCount=1, outputCount=1):
        '''
        Initialize this ProcessObject with empty input and output lists.
        '''
        self._inputs = {}
        self._outputs = {}
        
        self._inputCount = 0
        self.setInputCount(inputCount)
        self._outputCount = 0
        self.setOutputCount(outputCount)
        
        # ensure all pipeline objects start off modified
        self._timeModified = time.time()
        self._timeUpdated = self._timeModified - 1
        
        if input is not None:
            self.setInput(input, 0)
        
    def modified(self):
        '''
        Mark this ProcessObject as modified so that data gets regenerated the
        next time update() is called.
        '''
        self._timeModified = time.time()
        
    def update(self):
        '''
        Update the data held by this ProcessObject, calling for upstream
        updates first.
        '''
#        print "{0:s}.update()".format(self.__class__.__name__)
        
        # update upstream ProcessObjects
        # we need to update if our modified time or any upstream updated time
        # is greater than our updated time
        maxUpTime = 0
        for key in self._inputs.keys():
            if self._inputs[key] is not None:
                upstreamTime = self._inputs[key].update()
                maxUpTime = max(maxUpTime, upstreamTime)
        
        # check timestamps to see if we need to generate data
        if maxUpTime > self._timeUpdated or self._timeModified > self._timeUpdated:
            self.ensureOutput()
            self.generateData()
            self._timeUpdated = time.time()
            
        maxUpTime = max(self._timeUpdated, maxUpTime)
        return maxUpTime
        
    def generateData(self):
        '''
        Generate the data for this ProcessObject using the input and any other
        parameters.  Override this method in all subclasses.
        '''
        print "ProcessObject::generateData() does nothing; you may want to overload it in {0:s}.".format(self.__class__.__name__)
        
    def getInput(self, index=0):
        '''
        Returns the input at the given index, if it exists.
        '''
        return self._inputs[index] if index in self._inputs else None

    def setInput(self, object, index=0):
        '''
        Sets the object as an input to this ProcessObject at the given index.
        '''
        self._inputs[index] = object
        self.modified()

    def getOutput(self, index=0):
        '''
        Returns the output at the given index, if it exists.  To ensure an
        output exist, use ensureOutput.
        '''
        return self._outputs[index]
    
    def setOutput(self, object, index=0):
        '''
        Sets an output for this ProcessObject.
        '''
        self._outputs[index] = object
                
    def ensureOutput(self, count=None):
        '''
        Ensures that Image objects exist for the number of outputs provided
        by this ProcessObject.  If an Image does not exist, this method
        creates the space for it.  By default, this sets up the number of 
        Images in outputCount, but this can be overridden (for backwards 
        compatibility).
        '''
        if count is None:
            count = self._outputCount
            
        for index in range(count):
            if not self._outputs.has_key(index):
                self._outputs[index] = Image(self)
                
    def getInputCount(self):
        '''
        The number of input images expected by this ProcessObject.
        '''
        return self._inputCount
    
    def setInputCount(self, count):
        self._inputCount = count
        
    def getOutputCount(self):
        '''
        The number of output images provided by this ProcessObject.
        '''
        return self._outputCount
    
    def setOutputCount(self, count):
        self._outputCount = count
        self.ensureOutput()
        
    def disconnectInput(self):
        '''
        Disconnect this ProcessObject from all upstream (input) ProcessObjects. 
        '''
        # disconnect from upstream objects
        self._inputs = {}
        
    def disconnectOutput(self):
        '''
        Disconnect this Process object from downstream (output) ProcessObjects.
        '''
        # disconnect ourself from all downstream objects
        # this requires finding instances of ourselves in the inputs dictionary
        # for all of our outputs
        for output in self._outputs.values():
            keysToRemove = []
            for key in output._inputs:
                if output._inputs[key] == self:
                    keysToRemove.append(key)
#                    print "found myself in the output's inputs"
            for key in keysToRemove:
                del output._inputs[key]
        
        
    def disconnectPipeline(self):
        '''
        Disconnect this ProcessObject from upstream (input) and downstream 
        (output) ProcessObjects. 
        '''
        self.disconnectInput()
        self.disconnectOutput()
                
class Image(ProcessObject):
    '''
    Represents an image in an image processing pipeline.  This class is the
    output from ProcessObjects, such as filters.  The Image is generally tied
    to the ProcessObject that created it, so that it can update the data used
    to generate it when necessary.
    '''
    
    def __init__(self, parent=None, data=None):
        '''
        Initialize the image.  The parent is a ProcessObject responsible for
        updating the data for this Image.
        '''
        super(Image, self).__init__(input=parent, outputCount=0)
        self._data = data
    
    def getData(self):
        '''
        Get the numpy array that holds image data for this Image.  To ensure
        the latest data is provided, call update() first.
        '''
        return self._data
        
    def setData(self, data):
        '''
        Set the image data for this Image, a numpy array.
        '''
        self._data = data
        
    def copy(self):
        '''
        Create a copy of this Image.  This copy is not linked to the pipeline
        that generated the data.
        '''
        newcopy = Image(None)
        newcopy.setData(self._data.copy())
        return newcopy

    def generateData(self):
        '''
        Images do not generate their own data.
        '''
        pass