from mnist import MNIST
import numpy
import numpy.matlib

# sigmoid(x) = 1 / (1 + e^(-x))
def sigmoid(vector):
  for i in range(len(vector)):
    vector[i] = 1/numpy.exp(-(vector[i]))
  return vector

# sigmoidDerivative(x) = e^(-x) / (1 + e^(-x))^2
def sigmoidDerivative(vector):
  for i in range(len(vector)):
    vector[i] = numpy.exp(-(vector[i]))/((1 + numpy.exp(-(vector[i]))) * (1 + numpy.exp(-(vector[i]))))
  return vector

mndata = MNIST('Data')

#load images
trainImages, trainLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()

#convert images to arrays in order to simplify computations
trainImages = numpy.asarray(trainImages)
testImages = numpy.asarray(testImages)

#Advanced classifier

#training
#initialize the two hidden layers
layer1weights = numpy.zeros((16, len(trainImages[0])))
layer1biases = numpy.zeros((16, 1))

layer2weights = numpy.zeros((16, 16))
layer2biases = numpy.zeros((16, 1))

#initialize output layer
finalLayerWeights = numpy.zeros((10, 16))
finalLayerbiases = numpy.zeros((10, 1))

for i in range(len(trainImages)):
  #convert the current image to a column vector
  curImage = numpy.transpose(numpy.asmatrix(trainImages[i]))
  
  #compute output
  layer1activations = numpy.add(numpy.matmul(layer1weights, curImage), layer1biases)
  layer2activations = numpy.add(numpy.matmul(layer2weights, sigmoid(layer1activations)), layer2biases)
  outputactivations = numpy.add(numpy.matmul(finalLayerWeights, sigmoid(layer2activations)), finalLayerbiases)
  
  #expected output (all elements 0 except the correct output)
  outputExpected = numpy.zeros((10, 1))
  outputExpected[trainLabels[i]] = 1
  
  #compute the gradient of the previous layer's weights
  outputActivationDerivatives = numpy.subtract(sigmoid(outputactivations), outputExpected)
  outputSigmoidDerivatives = sigmoidDerivative(outputactivations)
  
  #results for output layer
  outputBiasGradient = numpy.multiply(outputActivationDerivatives, outputSigmoidDerivatives)
  outputWeightGradient = numpy.multiply(numpy.matlib.repmat(outputBiasGradient, 1, finalLayerWeights.shape[1]), finalLayerWeights)
  
  #backpropagate to layer 2
  #compute the gradient of the previous layer's weights
  layer2ActivationDerivatives = numpy.transpose(numpy.matmul(numpy.transpose(outputBiasGradient), finalLayerWeights))
  layer2SigmoidDerivatives = sigmoidDerivative(layer2activations)
  
  #results for layer 2
  layer2BiasGradient = numpy.multiply(layer2ActivationDerivatives, layer2SigmoidDerivatives)
  layer2WeightGradient = numpy.multiply(numpy.matlib.repmat(layer2BiasGradient, 1, layer2weights.shape[1]), layer2weights)
  
  #backpropagate to layer 1
  layer1ActivationDerivatives = numpy.transpose(numpy.matmul(numpy.transpose(layer2BiasGradient), layer2weights))
  layer1SigmoidDerivatives = sigmoidDerivative(layer1activations)
  
  #results for layer 1
  layer1BiasGradient = numpy.multiply(layer1ActivationDerivatives, layer1SigmoidDerivatives)
  layer1WeightGradient = numpy.multiply(numpy.matlib.repmat(layer1BiasGradient, 1, layer1weights.shape[1]), layer1weights)
  
  #add the gradients to the neurons
  layer1weights = numpy.subtract(layer1weights, layer1WeightGradient)
  layer1biases = numpy.subtract(layer1biases, layer1BiasGradient)

  layer2weights = numpy.subtract(layer2weights, layer2WeightGradient)
  layer2biases = numpy.subtract(layer2biases, layer2BiasGradient)

  #initialize output layer
  finalLayerWeights = numpy.subtract(finalLayerWeights, outputWeightGradient)
  finalLayerbiases = numpy.subtract(finalLayerbiases, outputBiasGradient)
  
  

#testing
correct = 0
wrong = 0

for i in range(len(testImages)):
  #convert the current image to a column vector
  curImage = numpy.transpose(numpy.asmatrix(testImages[i]))
  
  #compute output
  layer1output = sigmoid(numpy.add(numpy.matmul(layer1weights, curImage), layer1biases))
  layer2output = sigmoid(numpy.add(numpy.matmul(layer2weights, layer1output), layer2biases))
  output = sigmoid(numpy.add(numpy.matmul(finalLayerWeights, layer2output), finalLayerbiases))
  
  guess = output.argmax()
  
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