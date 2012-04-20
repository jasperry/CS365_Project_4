CS365 Project 4: Tracking
=========================

David Cain and Justin Sperry

A write-up of this project is available on the [CS365 course
wiki](https://wiki.colby.edu/display/CS365/CS365+Project+4).


Image format
------------
This project relies on still images taken from a microscope. The images are
saved in the numpy image format, which is highly uncompressed.

So, not all of the images used in this project are online. However, two
tarballs contain images that should demonstrate functionality:

 - `images.tar.gz` - images of a slide as the stage is being adjusted
 - `pond3.tar.gz` - images of organisms in pond water

Simply extract the tarball to the working directory for the program to
access them. To view the images in a standard image viewer, use the
`save_npy.py` script to convert from the .npy format to .png.
