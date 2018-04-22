from mnist import MNIST
import numpy
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

LEARNING_RATE = 0.02
BATCH_SIZE = 32
LAYER_ARRAY = [500, 150, 10]
TARGET_ACCURACY = 0.90
MAXIMUM_EPOCHS = 5

#load images
trainImages, trainLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()

#convert images to arrays in order to simplify computations
trainImages = numpy.asarray(trainImages)
testImages = numpy.asarray(testImages)

#Advanced classifier

#training
#initialize the layers
layerNum = 0
layerweights = [0] * len(LAYER_ARRAY)
layerbiases = [0] * len(LAYER_ARRAY)
for num in LAYER_ARRAY:
  if layerNum == 0:
    layerweights[layerNum] = 0.5 * numpy.random.randn(num, len(trainImages[0]))
  else:
    layerweights[layerNum] = 0.5 * numpy.random.randn(num, layerweights[layerNum - 1].shape[0])
  layerbiases[layerNum] = numpy.zeros((num, 1))
  layerNum = layerNum + 1

#initialize activations array
activations = [0] * len(LAYER_ARRAY)

#gradients
biasGradient = [0] * len(LAYER_ARRAY)
weightGradient = [0] * len(LAYER_ARRAY)
activationDerivatives = [0] * len(LAYER_ARRAY)

shouldTrain = True

#number of epochs
numEpochs = 0

while shouldTrain:
  order = numpy.random.choice(trainImages.shape[0], size=trainImages.shape[0], replace=False)
  for i in range(0, len(order) - BATCH_SIZE, BATCH_SIZE):
    #gradient sums (used for batches)
    biasGradientSUM = [0] * len(LAYER_ARRAY)
    weightGradientSUM = [0] * len(LAYER_ARRAY)

    for j in range(i, i + BATCH_SIZE):
      #progress bar
      percentComplete = int(j / len(trainImages) * 100)

      print("-" * percentComplete + " " + str(percentComplete) + "%", end="\r")

      #convert the current image to a column vector
      curImage = numpy.transpose(numpy.asmatrix(trainImages[order[j]]))

      #compute output
      activations[0] = numpy.add(numpy.matmul(layerweights[0], curImage), layerbiases[0])
      for k in range(1, len(LAYER_ARRAY)):
        activations[k] = numpy.add(numpy.matmul(layerweights[k], sigmoid(activations[k - 1])), layerbiases[k])

      #expected output (all elements 0 except the correct output)
      outputExpected = numpy.zeros((LAYER_ARRAY[len(LAYER_ARRAY) - 1], 1))
      outputExpected[trainLabels[j]] = 1

      for k in reversed(range(0, len(LAYER_ARRAY))):
        #compute the gradient of the previous layer's weights
        if k == len(LAYER_ARRAY) - 1:
          #compute the expected minus the output
          activationDerivatives[k] = numpy.subtract(sigmoid(activations[k]), outputExpected)
        else:
          #backpropagate
          activationDerivatives[k] = numpy.matmul(numpy.transpose(layerweights[k + 1]), biasGradient[k + 1])

        sigmoidDerivatives = sigmoidDerivative(activations[k])
        biasGradient[k] = numpy.multiply(activationDerivatives[k], sigmoidDerivatives[k])
        if (k == 0):
          weightGradient[k] = numpy.matmul(biasGradient[k], numpy.transpose(sigmoid(curImage)))
        else:
          weightGradient[k] = numpy.matmul(biasGradient[k], numpy.transpose(sigmoid(activations[k-1])))

        #sum up the gradient
        biasGradientSUM[k] = biasGradientSUM[k] + biasGradient[k]
        weightGradientSUM[k] = weightGradientSUM[k] + weightGradient[k]

    #subtract the gradients from the neurons
    for j in range(len(LAYER_ARRAY)):
      layerweights[j] = numpy.subtract(layerweights[j], LEARNING_RATE * weightGradientSUM[j])
      layerbiases[j] = numpy.subtract(layerbiases[j], LEARNING_RATE * biasGradientSUM[j])
  
  numEpochs = numEpochs + 1
  print("Epoch " + str(numEpochs) + " Complete")
  print("Testing accuracy")
  
  #Get a subset of test images to test accuracy
  indexes = numpy.random.choice(trainImages.shape[0], size=1000, replace=False)
  
  #testing accuracy
  correct = 0
  wrong = 0
  
  for num in indexes:
    curImage = numpy.transpose(numpy.asmatrix(trainImages[num]))
    output = numpy.add(numpy.matmul(layerweights[0], curImage), layerbiases[0])
    for k in range(1, len(LAYER_ARRAY)):
      output = numpy.add(numpy.matmul(layerweights[k], sigmoid(output)), layerbiases[k])
    guess = sigmoid(output).argmax()
    if (guess == trainLabels[num]):
      correct += 1
    else:
      wrong += 1
      
  #if the accuracy was achieved or if we have taken over the maximum number of times, terminate
  if (correct/(correct + wrong) > TARGET_ACCURACY or numEpochs >= MAXIMUM_EPOCHS):
    shouldTrain = False
    print("Target accuracy achieved or max time reached")
  else:
    print("Target accuracy not achieved (" + str(correct/(correct+wrong)) + " < " + str(TARGET_ACCURACY) + ")")

#testing
correct = 0
wrong = 0

for i in range(len(testImages)):
  #convert the current image to a column vector
  curImage = numpy.transpose(numpy.asmatrix(testImages[i]))
  
  #compute output
  output = numpy.add(numpy.matmul(layerweights[0], curImage), layerbiases[0])
  for k in range(1, len(LAYER_ARRAY)):
    output = numpy.add(numpy.matmul(layerweights[k], sigmoid(output)), layerbiases[k])
  
  guess = sigmoid(output).argmax()
  
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