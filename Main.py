from mnist import MNIST
import numpy

mndata = MNIST('Data')

#load images
trainImages, trainLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()

#convert images to array in order to simplify computations
trainImages = numpy.asarray(trainImages)
testImages = numpy.asarray(testImages)

#Simple classifier
#compute the singular value decomposition of the matrix
u, s, vt = numpy.linalg.svd(trainImages - trainImages.mean(axis=0), full_matrices=False);

#compute the coordinate matrix
trainCoords = numpy.matmul(u, numpy.diag(s))

#sample coordinate conversion
#numpy.matmul(numpy.transpose(vt) , numpy.transpose(coords[100])) + images.mean(axis=0)

#training
k = 10

#testing
correct = 0
wrong = 0

#compute the projection of the testing set onto the pca found before
testCoords = numpy.matmul(testImages - testImages.mean(axis=0), numpy.linalg.inv(vt))

for i in range(len(testCoords)):
  #array to track the k nearest neighbors
  #first column is distance
  #second column is index in trainImages
  nearestNeighbors = [[0 for i in range(2)] for j in range(k)]
  
  #fill nearestNeighbors in with the first numbers in the training set
  for j in range(len(nearestNeighbors[:][0])):
    nearestNeighbors[j][0] = numpy.linalg.norm(trainCoords[j] - testCoords[i])
    nearestNeighbors[j][1] = j
    
  for j in range(len(trainCoords)):
    if numpy.linalg.norm(trainCoords[j] - testCoords[i]) < max(nearestNeighbors[:][0]):
      index = nearestNeighbors[:][0].index(max(nearestNeighbors[:][0]))
      nearestNeighbors[index][0] = numpy.linalg.norm(trainCoords[j] - testCoords[i])
      nearestNeighbors[index][1] = j
  
  #take vote of the kth closest neighbors
  vote = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  
  for j in range(len(nearestNeighbors[:][1])):
    if (trainLabels[nearestNeighbors[j][1]] == 0):
      vote[0] += 1
    elif (trainLabels[nearestNeighbors[j][1]] == 1):
      vote[1] += 1
    elif (trainLabels[nearestNeighbors[j][1]] == 2):
      vote[2] += 1
    elif (trainLabels[nearestNeighbors[j][1]] == 3):
      vote[3] += 1
    elif (trainLabels[nearestNeighbors[j][1]] == 4):
      vote[4] += 1
    elif (trainLabels[nearestNeighbors[j][1]] == 5):
      vote[5] += 1
    elif (trainLabels[nearestNeighbors[j][1]] == 6):
      vote[6] += 1
    elif (trainLabels[nearestNeighbors[j][1]] == 7):
      vote[7] += 1
    elif (trainLabels[nearestNeighbors[j][1]] == 8):
      vote[8] += 1
    elif (trainLabels[nearestNeighbors[j][1]] == 9):
      vote[9] += 1
  #the result of the vote
  guess = vote.index(max(vote))
    
  #display number and guess
  print(mndata.display(testImages[i]))
  print("Guess: ")
  print(guess)
    
  #add to stats
  if (guess == testLabels[i]):
    correct += 1
  else:
    wrong += 1
print("Accuracy: ")
print(correct / (correct + wrong))