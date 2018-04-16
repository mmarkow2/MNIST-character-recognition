from mnist import MNIST
import numpy

mndata = MNIST('Data')

#load images
trainImages, trainLabels = mndata.load_training()

#convert images to array in order to simplify computations
trainImages = numpy.asarray(trainImages)

# Simple classifier
# First perform pca

#compute the singular value decomposition of the matrix
u, s, vt = numpy.linalg.svd(trainImages - trainImages.mean(axis=0), full_matrices=False);

#compute the coordinate matrix
coords = numpy.matmul(u, numpy.diag(s))

#sample coordinate conversion
#numpy.matmul(numpy.transpose(vt) , numpy.transpose(coords[100])) + images.mean(axis=0)

#training
#none

#testing