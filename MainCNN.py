from mnist import MNIST
import numpy
import numpy.matlib
from scipy.special import expit

# sigmoid(x) = 1 / (1 + e^(-x))
def sigmoid(vector):
  for i in range(len(vector)):
    vector[i] = expit(vector[i])
  return vector

# sigmoidDerivative(x) = e^(-x) / (1 + e^(-x))^2
def sigmoidDerivative(vector):
  for i in range(len(vector)):
    vector[i] = expit(vector[i]) * (1-expit(vector[i]))
  return vector

mndata = MNIST('Data')

LEARNING_RATE = 0.01
BATCH_SIZE = 32

#load images
trainImages, trainLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()

#convert images to arrays in order to simplify computations
trainImages = numpy.asarray(trainImages)
testImages = numpy.asarray(testImages)

#Advanced classifier

#training
#initialize the two hidden layers
layer1weights = 0.01 * numpy.random.randn(16, len(trainImages[0]))
layer1biases = numpy.zeros((16, 1))

layer2weights = 0.01 * numpy.random.randn(16, 16)
layer2biases = numpy.zeros((16, 1))

#initialize output layer
finalLayerWeights = 0.01 * numpy.random.randn(10, 16)
finalLayerbiases = numpy.zeros((10, 1))

for i in range(0, len(trainImages), BATCH_SIZE):
  outputBiasGradientSUM = 0
  outputWeightGradientSUM = 0
  layer2BiasGradientSUM = 0
  layer2WeightGradientSUM = 0
  layer1BiasGradientSUM = 0
  layer1WeightGradientSUM = 0
  
  for j in range(i, i + BATCH_SIZE):
    #progress bar
    percentComplete = int(float(j) / len(trainImages) * 100)
  
    print("-" * percentComplete + " " + str(percentComplete) + "%", end="\r")
  
    #convert the current image to a column vector
    curImage = numpy.transpose(numpy.asmatrix(trainImages[j]))
  
    #compute output
    layer1activations = numpy.add(numpy.matmul(layer1weights, curImage), layer1biases)
    layer2activations = numpy.add(numpy.matmul(layer2weights, sigmoid(layer1activations)), layer2biases)
    outputactivations = numpy.add(numpy.matmul(finalLayerWeights, sigmoid(layer2activations)), finalLayerbiases)
  
    #expected output (all elements 0 except the correct output)
    outputExpected = numpy.zeros((10, 1))
    outputExpected[trainLabels[j]] = 1
  
    #compute the gradient of the previous layer's weights
    outputActivationDerivatives = numpy.subtract(sigmoid(outputactivations), outputExpected)
    outputSigmoidDerivatives = sigmoidDerivative(outputactivations)
  
    #results for output layer
    outputBiasGradient = numpy.multiply(outputActivationDerivatives, outputSigmoidDerivatives)
    outputWeightGradient = numpy.matmul(outputBiasGradient, numpy.transpose(sigmoid(layer2activations)))
  
    #backpropagate to layer 2
    #compute the gradient of the previous layer's weights
    layer2ActivationDerivatives = numpy.matmul(numpy.transpose(finalLayerWeights), outputBiasGradient)
    layer2SigmoidDerivatives = sigmoidDerivative(layer2activations)

    #results for layer 2
    layer2BiasGradient = numpy.multiply(layer2ActivationDerivatives, layer2SigmoidDerivatives)
    layer2WeightGradient = numpy.matmul(layer2BiasGradient, numpy.transpose(sigmoid(layer1activations)))

    #backpropagate to layer 1
    layer1ActivationDerivatives = numpy.matmul(numpy.transpose(layer2weights), layer2BiasGradient)
    layer1SigmoidDerivatives = sigmoidDerivative(layer1activations)

    #results for layer 1
    layer1BiasGradient = numpy.multiply(layer1ActivationDerivatives, layer1SigmoidDerivatives)
    layer1WeightGradient = numpy.matmul(layer1BiasGradient, numpy.transpose(sigmoid(curImage)))
    
    #sum up the gradient
    outputBiasGradientSUM = outputBiasGradientSUM + outputBiasGradient
    outputWeightGradientSUM = outputWeightGradientSUM + outputWeightGradient
    layer2BiasGradientSUM = layer2BiasGradientSUM + layer2BiasGradient
    layer2WeightGradientSUM = layer2WeightGradientSUM + layer2WeightGradient
    layer1BiasGradientSUM = layer1BiasGradientSUM + layer1BiasGradient
    layer1WeightGradientSUM = layer1WeightGradientSUM + layer1WeightGradient

  #subtract the gradients from the neurons
  layer1weights = numpy.subtract(layer1weights, LEARNING_RATE * layer1WeightGradientSUM)
  layer1biases = numpy.subtract(layer1biases, LEARNING_RATE * layer1BiasGradientSUM)

  layer2weights = numpy.subtract(layer2weights, LEARNING_RATE * layer2WeightGradientSUM)
  layer2biases = numpy.subtract(layer2biases, LEARNING_RATE * layer2BiasGradientSUM)

  finalLayerWeights = numpy.subtract(finalLayerWeights, LEARNING_RATE * outputWeightGradientSUM)
  finalLayerbiases = numpy.subtract(finalLayerbiases, LEARNING_RATE * outputBiasGradientSUM)
  
  

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