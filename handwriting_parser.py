import numpy as np
import random as rd
from imageio.v3 import imread
from sklearn import svm
from skimage.transform import resize

def boundaries(binarized,axis):
    # Variables named assuming axis = 0; algorithm valid for axis=1
    # [1,0][axis] effectively swaps axes for summing
    rows = np.sum(binarized, axis = [1,0][axis]) > 0
    rows[1:] = np.logical_xor(rows[1:], rows[:-1])
    change = np.nonzero(rows)[0]
    ymin = change[::2]
    ymax = change[1::2]
    height = ymax-ymin
    too_small = 10 # Real letters will be bigger than 10px by 10px.
    ymin = ymin[height>too_small]
    ymax = ymax[height>too_small]
    return list(zip(ymin,ymax))

def separate(img):
    orig_img = img.copy()
    pure_white = 255
    white = np.max(img)
    black = np.min(img)
    thresh = (white+black)/2.0
    binarized = img<thresh
    row_bounds = boundaries(binarized, axis = 0) 
    cropped = []
    for r1,r2 in row_bounds:
        img = binarized[r1:r2,:]
        col_bounds = boundaries(img,axis=1)
        print("COL BOUNDS", col_bounds)
        rects = [r1,r2,col_bounds[0][0],col_bounds[0][1]]
        cropped.append(np.array(orig_img[rects[0]:rects[1],rects[2]:rects[3]]/pure_white))
    return cropped

# Data, target, percentage used for training vs testing.
def partition(data,target,p):
    m = int(p * len(data))
    train_data = data[:m,:]
    train_target = target[:m]
    test_data = data[m:]
    test_target = target[m:]
    return train_data, train_target, test_data, test_target

# Import big images:
big_a = imread("a.png", mode="L")#, flatten = True)
big_b = imread("b.png", mode="L")#, flatten = True)
big_c = imread("c.png", mode="L")#, flatten = True)

# Separate the images into lists of small images
a_imgs = separate(big_a)
b_imgs = separate(big_b)
c_imgs = separate(big_c)

# Create a list of all of the images and another list with their corresponding targets.
images = a_imgs + b_imgs + c_imgs
targets = [0 for _ in range(len(a_imgs))] + [1 for _ in range(len(b_imgs))] + [2 for _ in range(len(c_imgs))]

# resize the images to 10x10
for i in range(len(images)):
    images[i] = resize(images[i], (10,10))

# shuffle the images and targets but keep the targets corresponding to the correct
# image after reshuffling
images_and_labels = list(zip(images,targets))
rd.shuffle(images_and_labels)
images = []
targets = []
for i in range(len(images_and_labels)):
    images.append(images_and_labels[i][0])
    targets.append(images_and_labels[i][1])

# convert to numpy array
images = np.array(images)
targets = np.array(targets)

# reshape the images numpy array so that it is the correct shape for the classifier
n_samples = len(images)
images = images.reshape((n_samples, -1))

# Partition the data and targets into training and test data and targets
partition = partition(images,targets,0.5)
train_data = partition[0]
train_target = partition[1]
test_data = partition[2]
test_target = partition[3]

# Create the classified and fit the training data to the target data
clf = svm.SVC(gamma = 0.001, C= 100)
clf.fit(train_data,train_target)

# Create lists of expected and predicted values using the targets of the
# test images and the classifier on the images respectively
expected = test_target
predicted = clf.predict(test_data)

# Output the predicted targets versus actual targets to the console
print("Predicted:", predicted)
print("Truth:    ", expected)

# Calculate the accuracy by counting the number of correct guesses
# and dividing by the number of total guesses and then print
test_size = len(test_data)
correct_predictions = 0
for i in range(test_size):
    if predicted[i] == expected[i]:
        correct_predictions += 1
print("Accuracy", 100 * correct_predictions / float(test_size), "%")
