
import os
import glob
import copy
import gzip
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

class PointNet:
    def __init__(self, num_classes=10, input_shape=(2048, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape

    def conv_bn(x, filters):
        x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation("relu")(x)

    def dense_bn(x, filters):
        x = layers.Dense(filters)(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation("relu")(x)

    def tnet(self, inputs, num_features):
        # Initalise bias as the indentity matrix
        bias = keras.initializers.Constant(np.eye(num_features).flatten())
        reg = OrthogonalRegularizer(num_features)

        x = self.conv_bn(inputs, 32)
        x = self.conv_bn(x, 64)
        x = self.conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)
        x = self.dense_bn(x, 256)
        x = self.dense_bn(x, 128)
        x = layers.Dense(
            num_features * num_features,
            kernel_initializer="zeros",
            bias_initializer=bias,
            activity_regularizer=reg,
        )(x)
        feat_T = layers.Reshape((num_features, num_features))(x)
        # Apply affine transformation to input features
        return layers.Dot(axes=(2, 1))([inputs, feat_T])

    def create_pointnet_model(self):
        """
        The main network can be then implemented in the same manner where the t-net mini models
        can be dropped in a layers in the graph. Here we replicate the network architecture
        published in the original paper but with half the number of weights at each layer as we
        are using the smaller 10 class ModelNet dataset.
        """
        inputs = keras.Input(shape=self.input_shape)

        x = self.tnet(inputs, 3)
        x = self.conv_bn(x, 32)
        x = self.conv_bn(x, 32)
        x = self.tnet(x, 32)
        x = self.conv_bn(x, 32)
        x = self.conv_bn(x, 64)
        x = self.conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)
        x = self.dense_bn(x, 256)
        x = layers.Dropout(0.3)(x)
        x = self.dense_bn(x, 128)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
        model.summary()
        return model

    def run_cnn_inference(self, X_train, T_train, X_test, T_test):
        """
            X_train ([float]): [The matrix of training data. Each row contains one sample.]
            X_test ([float]): [The matrix of testing data. Each row contains one sample.]
            T_train ([float]): [The matrix of training target. Each row contains one sample.]
            T_test ([float]): [The matrix of testing target. Each row contains one sample.]
        """
        # Make sure images have shape (28, 28, 1)
        X_train = X_train.reshape(X_train.shape[0], 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 28, 28)
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)
        
        model = self.create_pointnet_model()
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(X_train, T_train, epochs=10, batch_size=128)
        # Sanity checks
        score_train = model.evaluate(X_train, T_train, verbose=0)
        score_test = model.evaluate(X_test, T_test, verbose=0)
        print("Test loss:", score_test[0])
        print("Test accuracy:", score_test[1])
        t_hat_test = model.predict(X_test).reshape(T_test.shape)
        t_hat = model.predict(X_train).reshape(T_train.shape)
        return compute_nme(T_train,t_hat), compute_nme(T_test,t_hat_test),calculate_accuracy(T_train.T,t_hat.T), calculate_accuracy(T_test.T,t_hat_test.T)

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


def create_modelnet10_data():
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

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def load_modelnet10_data():
    
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
    
    X_train[:,0:3] = pc_normalize(X_train[:,0:3])
    X_test[:,0:3] = pc_normalize(X_test[:,0:3])

    return X_train, T_train, X_test, T_test
    
def cuboid_data(o, size=(1,1,1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

def plotCubeAt(positions,sizes=None,colors=None, **kwargs):
    if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
    g = []
    for p,s,c in zip(positions,sizes,colors):
        g.append( cuboid_data(p, size=s) )
    return Poly3DCollection(np.concatenate(g),  
                            facecolors=np.repeat(colors,6), **kwargs)

def draw_points(ax, points):
    ax.set_xlabel('X')
    ax.set_xlim(-50, 50)
    ax.set_ylabel('Y')
    ax.set_ylim(-50, 50)
    ax.set_zlabel('Z')
    ax.set_zlim(-50, 50)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])

def visualize_window_movement(points, search_range=20, radius=5):
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    draw_points(ax, points)
    # Example of first 100 points chosen
    # ax.scatter(points[1:100, 0], points[1:100, 1], points[1:100, 2])
    
    
    ############################################
    data = copy.deepcopy(points)
    for k in range(-search_range, search_range, radius):
        for i in range(-search_range, search_range, radius):
            for j in range(-search_range, search_range, radius):
                
                positions = [(i,j,k)]
                sizes = [(radius, radius, radius)]
                colors = ["crimson"]
                pc = plotCubeAt(positions, sizes, colors=colors, edgecolor="k",  alpha=0.1)
                ax.add_collection3d(pc)  

                plt.ion() # turn on interactive mode
                plt.show()

                data[:,0] = np.logical_and(data[:,0] > i, data[:,0] <= i+radius)
                data[:,1] = np.logical_and(data[:,1] > j, data[:,1] <= j+radius)
                data[:,2] = np.logical_and(data[:,2] > k, data[:,2] <= k+radius)

                indx = (np.sum(data, axis=1) == 3)
                
                if len(points[indx,:])>0:
                    print("*******We got ", len(points[indx,:]), " points in the window")
                plt.pause(0.1)
                data = copy.deepcopy(points)
    ############################################

    #ax.set_axis_off()

def pc_rescale(pc, scaling_factor=40):
    pc[:,0] = scaling_factor*(pc[:,0] - 0)/(max(pc[:,0]) - min(pc[:,0]))
    pc[:,1] = scaling_factor*(pc[:,1] - 0)/(max(pc[:,1]) - min(pc[:,1]))
    pc[:,2] = scaling_factor*(pc[:,2] - 0)/(max(pc[:,2]) - min(pc[:,2]))
    return pc

def training_loop(X_train, T_train, X_test, T_test):    
    data = X_train[np.random.randint(2048)]
    visualize_window_movement(pc_rescale(data))

def main():
    X_train, T_train, X_test, T_test = load_modelnet10_data()
    training_loop(X_train, T_train, X_test, T_test)

if __name__ == '__main__':
    main()