
import os
import glob

import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

'''
def parse_dataset(num_points=2048):

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))
        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )

DATA_DIR = tf.keras.utils.get_file(
    "modelnet.zip",
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    extract=True,
)
DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")


mesh = trimesh.load(os.path.join(DATA_DIR, "chair/train/chair_0001.off"))
#mesh.show()
points = mesh.sample(2048)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2])
ax.set_axis_off()
plt.show()


NUM_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE = 32

train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(
    NUM_POINTS
)

print(train_points.shape)
print(type(train_points))
print(train_labels.shape)
print(CLASS_MAP)

data = {'train_x':train_points, 'test_x':test_points, 'train_y':train_labels, 'test_y':test_labels, 'class':CLASS_MAP}


np.save("./mat_files/modelnet-10", data)


# Run this command in terminal to compress the data if this line doesn't work
# gzip ./mat_files/modelnet-10.npy

'''
import gzip

with gzip.open('./mat_files/modelnet-10.npy.gz','rb') as data_zip:
    data = np.load(data_zip, allow_pickle=True)
    X_train = data.item()['train_x']
    T_train = data.item()['train_y']
    X_test = data.item()['test_x']
    T_test = data.item()['test_y']

print("Train data shape: ", X_train.shape)
print("Test data shape: ", X_test.shape)
print("Train data label: ", T_train.shape)
print("Train data label: ", T_test.shape)


# Take an example to visualize
points = X_train[3000]

print(points[1:1000, 0][1:10])
print(points[1:1000, 1][1:10])
print(points[1:1000, 2][1:10])

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
ax.grid(True)
# Example of first n points chosen
ax.scatter(points[1:1000, 0], points[1:1000, 1], points[1:1000, 2])
ax.set_axis_off()
plt.show()

