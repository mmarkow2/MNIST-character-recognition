from mnist import MNIST
from random import randrange

mndata = MNIST('Data')

images, labels = mndata.load_training()

index = randrange(0, len(images))  # choose an index ;-)
print(mndata.display(images[index]))