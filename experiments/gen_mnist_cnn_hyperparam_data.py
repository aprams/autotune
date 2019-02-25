from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import param_space
import optimizers.grid_search
import config
import pickle
import os


def get_mnist_data():
    """
    Download and preprocess MNIST data
    :return: train/validation data and meta parameters for data
    """
    # input image dimensions
    img_rows, img_cols = 28, 28
    num_classes = 10

    # the data, split between train and test sets
    (x_train, y_train), (x_val, y_val) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_train /= 255
    x_val /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    return x_train, y_train, x_val, y_val, input_shape, num_classes

def eval_mnist_cnn(params, data_tuple):
    K.clear_session()
    x_train, y_train, x_val, y_val, input_shape, num_classes = data_tuple

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train[:100], y_train[:100],
              batch_size=params['batch_size'],
              epochs=params['epochs'],
              verbose=1,
              validation_data=(x_val[:100], y_val[:100]))
    val_score = model.evaluate(x_val[:100], y_val[:100], verbose=0)
    print('Val loss:', val_score[0])
    print('Val accuracy:', val_score[1])

    return val_score[0]


if __name__ == '__main__':
    # Prep data
    x_train, y_train, x_val, y_val, input_shape, num_classes = get_mnist_data()

    # Set parameter space
    dropout_param = param_space.Real([0, 0.5], name='dropout', n_points_to_sample=1)
    learning_rate_param = param_space.Real([-5, -2], projection_fn=lambda x: 10 ** x, name='learning_rate', n_points_to_sample=1)

    batch_size_param = param_space.Integer([8, 32, 128], name='batch_size')
    epochs_param = param_space.Integer([1], name='epochs')

    hyper_param_list = [dropout_param, learning_rate_param, batch_size_param, epochs_param]

    def eval_fn(params):
        return -eval_mnist_cnn(params, (x_train, y_train, x_val, y_val, input_shape, num_classes))


    def sample_callback_fn(**params):
        print(params)

    grid_search_optimizer = optimizers.grid_search.GridSearchOptimizer(hyper_param_list=hyper_param_list,
                                                                       eval_fn=eval_fn,
                                                                       callback_fn=sample_callback_fn)

    results = grid_search_optimizer.maximize()
    print(results)

    with open(os.path.join(config.EXPERIMENT_RESULTS_FOLDER, 'mnist_cnn.p'), 'wb') as fp:
        pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL)


