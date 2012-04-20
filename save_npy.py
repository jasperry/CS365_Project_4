#!/usr/bin/env python

# David Cain
# Justin Sperry
# 2012-04-18
# CS365, Brian Eastwood

"""
    A script to convert .npy images to .png
"""

import os
import sys

import Image
import numpy as np

def convert(fn):
    """
        Converts the numpy image saved to 'fn' into a .png
        The resulting image is saved with the same filename, except with
        the .png extension (input image must have extension .npy).
    """
    assert fn[-4:] == ".npy", "%s: File extension should match '.npy'" % fn

    print "Converting file '%s' to .png" % fn
    numpy_img = np.load(fn)

    #Rescale to 0-255 and convert to uint8
    rescaled = (255.0 / numpy_img.max() * (numpy_img - numpy_img.min())).astype(np.uint8)

    im = Image.fromarray(rescaled)
    im.save(fn[:-4] + ".png")


if __name__ == "__main__":
    """
        Sys args should be paths to .npy files
    """
    usage = """
    Usage: python save_npy.py file1 [file2] [file3]...
    """
    if len(sys.argv) <= 1:
        print usage
    for filename in sys.argv[1:]:
        assert os.path.isfile(filename), "File '%s' doesn't exist" % filename
        convert(filename)
