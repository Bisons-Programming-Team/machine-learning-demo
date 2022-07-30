import tensorflow.keras.backend as K
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
import tensorflow as tf
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import json

def FullyConn():

    model = models.Sequential()
    #model.add(layers.Reshape((32,32,3), input_shape=(3072,)))
    model.add(layers.Dense(3072, activation=tf.nn.relu, input_shape=(3072,)))
    model.add(layers.Dense(768, activation=tf.nn.relu))
    model.add(layers.Dense(192, activation=tf.nn.relu))
    model.add(layers.Dense(50, activation=tf.nn.relu))
    model.add(layers.Dense(10))
    model.add(layers.Softmax())

    return model

def AlexNet():

    model = models.Sequential()
    model.add(layers.Reshape((32, 32, 3), input_shape=(3072,))) # real AlexNet has 228x228x3 images
    model.add(layers.Conv2D(96, (11, 11), strides=4, padding='same', activation=tf.nn.relu))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(layers.Conv2D(256, (5, 5), strides=2, padding='same', activation=tf.nn.relu))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(layers.Conv2D(384, (3, 3), strides=1, padding='same', activation=tf.nn.relu))
    model.add(layers.Conv2D(384, (3, 3), strides=1, padding='same', activation=tf.nn.relu))
    model.add(layers.Conv2D(256, (3, 3), strides=1, padding='same', activation=tf.nn.relu))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation=tf.nn.relu))
    model.add(layers.Dense(4096, activation=tf.nn.relu))
    model.add(layers.Dense(10))
    model.add(layers.Softmax())

    return model

def load_cifar(BATCH_FLAG, cifar_loc):

    data_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_files = 'test_batch'
    meta_file = 'batches.meta'

    if not BATCH_FLAG:

        train_labels = []
        train_data = []
        for file in data_files:
            with open(os.path.join(cifar_loc, file), 'rb') as f:
                batch_dict = pickle.load(f, encoding='bytes')
            train_labels.append(batch_dict[b'labels'])
            train_data.append(batch_dict[b'data'])

    with open(os.path.join(cifar_loc, test_files), 'rb') as f:
        test_dict = pickle.load(f, encoding='bytes')

    test_labels = test_dict[b'labels']
    test_data = test_dict[b'data']

    with open(os.path.join(cifar_loc, meta_file), 'rb') as f:
        label_names = [line for line in f]

    train_labels = np.array(train_labels)
    train_labels = np.reshape(train_labels, (train_labels.shape[0]*train_labels.shape[1],))
    train_data = np.array(train_data)
    train_data = np.reshape(train_data, (train_data.shape[0]*train_data.shape[1], train_data.shape[2]))

    test_labels = np.array(test_labels)

    for ii, input in enumerate(np.linspace(1, np.size(train_data, 0), np.size(train_data, 0))):
        index_arr = train_data[ii, :]
        index_arr = np.reshape(index_arr, (1, 32*32*3))
        index_arr = np.reshape(index_arr, (3, 32*32))
        index_arr = index_arr.flatten('F')
        index_arr = np.reshape(index_arr, (1, 32*32*3))
        train_data[ii, :] = index_arr

    for ii, input in enumerate(np.linspace(1, np.size(test_data, 0), np.size(test_data, 0))):
        index_arr = test_data[ii, :]
        index_arr = np.reshape(index_arr, (1, 32 * 32 * 3))
        index_arr = np.reshape(index_arr, (3, 32 * 32))
        index_arr = index_arr.flatten('F')
        index_arr = np.reshape(index_arr, (1, 32 * 32 * 3))
        test_data[ii, :] = index_arr

    #train_data = train_data.astype(float)/255
    #test_data = test_data.astype(float)/255
    size_train = int(0.9*np.shape(train_data)[0])
    size_val = int(np.shape(train_data)[0]) - size_train

    train_set = train_data[:size_train, :]
    train_set_labels = train_labels[:size_train]

    val_set = train_data[size_train:size_train+size_val, :]
    val_labels = train_labels[size_train:size_train+size_val]

    train_data_mean = np.mean(train_set, axis=0)
    train_data_std = np.std(train_set, axis=0)

    train_set = (train_set - train_data_mean)/train_data_std
    val_set = (val_set - train_data_mean)/train_data_std
    test_data = (test_data - train_data_mean)/train_data_std

    return train_set_labels, train_set, test_labels, test_data, label_names, val_set, val_labels


def get_confusion_mat(true_labels, predictions, label_names):
    confusion_mat = np.empty([len(label_names), len(label_names)], dtype=int)

    for ii in np.linspace(0, len(label_names)-1, len(label_names), dtype=int):
        for jj in np.linspace(0, len(label_names)-1, len(label_names), dtype=int):
            confusion_mat[ii, jj] = sum((true_labels == ii) & (predictions == jj))

    return confusion_mat


def plot_loss(hist_dict, model_type):
    fig = plt.figure()
    plt.plot(hist_dict['loss'], 'r')
    plt.plot(hist_dict['val_loss'], 'c')
    plt.title('Loss Function for '+model_type+' Network')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    fig.savefig('loss_historyFull.png')
    plt.show()
    plt.close(fig)


def train_nn(model_str, optimizer_settings,train_data,train_labels,val_data,val_labels):
    
    # Set strings
    if model_str == 'Fully Connected':
        model = FullyConn()
        model_h5_str = 'modelfull.h5'
        model_json_str = 'modelfull.json'
    elif model_str == 'Convolutional Neural':
        model = AlexNet()
        model_h5_str = 'modelAlex.h5'
        model_json_str = 'modelAlex.json'

    # Print model summary    
    model.summary()

    # Build the model
    model.compile(optimizer=optimizer_settings, loss=losses.sparse_categorical_crossentropy)
    
    # Train the model (history saves information during training)
    history = model.fit(train_data, train_labels, batch_size=1000, epochs=50, shuffle=True, validation_data=(val_data, val_labels))
    hist_dict = history.history

    # Save the model
    model.save(model_h5_str)
    model_json = model.to_json()
    with open(model_json_str, 'wt') as f:
        json.dump(json.loads(model_json), f, indent=4)

    #plot_loss(hist_dict, model_str)


def predict(model_str, test_data):
    
    # Set strings
    if model_str == 'Fully Connected':
        model = FullyConn()
        model_h5_str = 'modelfull.h5'
    elif model_str == 'Convolutional Neural':
        model = AlexNet()
        model_h5_str = 'modelAlex.h5'

    # Load model and make predictions
    model = models.load_model(model_h5_str)
    softmax_predictions = model.predict(test_data)
    predictions = np.argmax(softmax_predictions, axis=1)    # collapse softmax probabilities to single integer prediction

    return predictions

def main():
    BATCH_FLAG = 0
    AUGMENT_FLAG = 0
    cifar_loc = 'cifar-10-batches-py'

    # Load dataset
    [train_labels, train_data, test_labels, test_data, label_names, val_data, val_labels] = load_cifar(BATCH_FLAG, cifar_loc)
    label_names = ['airplane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Configure optimizer parameters
    adam = optimizers.Adam(learning_rate=0.0007, beta_1=0.9, beta_2=0.995, amsgrad=False)

    #####  Train FullyConn #####
    fc_model_str = 'Fully Connected'
    
    # If a model doesn't exist, train the model
    if not os.path.exists('modelfull.h5'):
        train_nn(fc_model_str, adam, train_data, train_labels, val_data, val_labels)
        
    # Get predictions
    fc_predictions = predict(fc_model_str, test_data)

    fc_confusion_mat = get_confusion_mat(test_labels, fc_predictions, label_names)
    print(fc_confusion_mat)


    #####  Train AlexNet #####
    cnn_model_str = 'Convolutional Neural'
    
    # If a model doesn't exist, train the model
    if not os.path.exists('modelAlex.h5'):
        train_nn(cnn_model_str, adam, train_data, train_labels, val_data, val_labels)
    
    # Get predictions
    cnn_predictions = predict(cnn_model_str, test_data)
    
    cnn_confusion_mat = get_confusion_mat(test_labels, cnn_predictions, label_names)
    print(cnn_confusion_mat)


if __name__ == "__main__":
    main()