"""A simple application of feature extraction on images using PCA, by
working with face images from the Labeled Faces in the Wild Dataset.
This dataset contains face images of celebrities downloaded from the
Internet, and it includes faces of politicians, singers, actors and
athletes from the early 2000s. We use gray scale versions of these
images, and scale them down for faster processing.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape
print("Shape of each image: {}".format(image_shape))
print("Hence total images shape: {}".format(people.images.shape))
print("Number of classes: {}".format(len(people.target_names)))

fix, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])
plt.show(block=True)

# This dataset consist of lot of images of George Bush and Colon Powell.
# Hence the data is more skewed
# We can count of each target as it appears
counts = np.bincount(people.target)
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end=' ')
    if (i+1) % 3 == 0:
        print()
print("\n")
# To make data less skewed, we will consider only few 50 images of each
# class.
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

# Preprocessing ie. scale the grayscale values to be between 0 and 1
# instead of 0 and 255 for better numeric stability
X_people = X_people/255

# Let us implement kNeighboursClassifier where k = 1
# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people,
                                                    stratify=y_people, random_state=4)
print(X_train.shape)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Test set score of 1-nn: {}".format(knn.score(X_test, y_test)))

# The test score seems to be very less with 1-NN classifier
# So let us apply PCA before 1-NN
pca = PCA(n_components=100, whiten=True, random_state=4)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print("Test set score of 1-nn after pca: {}".format(knn.score(X_test_pca, y_test)))

# Let us check the components shape
print("pca components shape: {}".format(pca.components_.shape))

fix, axes = plt.subplots(10, 10, figsize=(15, 12),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape), cmap='viridis')
    ax.set_title("{}. component".format((i+1)))
plt.show(block=True)