from mnist import MNIST
import numpy

def sigmoid(vector):
  for i in range(len(vector)):
    vector[i] = 1/numpy.exp(-(vector[i]))
  return vector

mndata = MNIST('Data')

#load images
trainImages, trainLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()

#convert images to matrices in order to simplify computations
trainImages = numpy.asmatrix(trainImages)
testImages = numpy.asmatrix(testImages)

#Advanced classifier

#training
#initialize the two hidden layers
layer1weights = numpy.zeros((16, trainImages[0].shape[1]))
layer1biases = numpy.zeros((16, 1))

layer2weights = numpy.zeros((16, 16))
layer2biases = numpy.zeros((16, 1))

#initialize output layer
finalLayerWeights = numpy.zeros((10, 16))
finalLayerbiases = numpy.zeros((10, 1))

#testing
correct = 0
wrong = 0

for i in range(testImages.shape[0]):
  #compute output
  layer1output = sigmoid(numpy.add(numpy.matmul(layer1weights, numpy.transpose(testImages[i])), layer1biases))
  layer2output = sigmoid(numpy.add(numpy.matmul(layer2weights, layer1output), layer2biases))
  output = sigmoid(numpy.add(numpy.matmul(finalLayerWeights, layer2output), finalLayerbiases))
  
  guess = output.argmax()
  
  #display number and guess
  print(mndata.display(testImages.tolist()[i]))
  print("Guess: ")
  print(guess)
    
  #add to stats
  if (guess == testLabels[i]):
    correct += 1
  else:
    wrong += 1
    
print("Accuracy: ")
print(correct / (correct + wrong))