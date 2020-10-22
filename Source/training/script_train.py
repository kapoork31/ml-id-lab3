import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import argparse
import codecs
import json
import numpy as np
import os
import tensorflow as tf
#from tf.keras.models import Sequential
#from tf.keras.layers import Dense, Conv2D, Flatten, Conv1D,MaxPooling2D,Dropout

max_features = 20000
maxlen = 400
embedding_dims = 300
filters = 256
kernel_size = 3
hidden_dims = 256

os.system('mkdir /opt/ml/model/code')
os.system('cp inference.py /opt/ml/model/code')
os.system('cp requirements.txt /opt/ml/model/code')


def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.01)

    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))

    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    return parser.parse_known_args()

def save_history(path, history):

    history_for_json = {}
    # transform float values that aren't json-serializable
    for key in list(history.history.keys()):
        if type(history.history[key]) == np.ndarray:
            history_for_json[key] == history.history[key].tolist()
        elif type(history.history[key]) == list:
           if  type(history.history[key][0]) == np.float32 or type(history.history[key][0]) == np.float64:
               history_for_json[key] = list(map(float, history.history[key]))

    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(history_for_json, f, separators=(',', ':'), sort_keys=True, indent=4)
        
def get_train_data(train_dir):

    x_train = np.load(os.path.join(train_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(train_dir, 'y_train.npy'))
    print('x train', x_train.shape,'y train', y_train.shape)
    return x_train, y_train


def get_test_data(test_dir):

    x_test = np.load(os.path.join(test_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(test_dir, 'y_test.npy'))
    print('x test', x_test.shape,'y test', y_test.shape)
    return x_test, y_test

def get_model_parameter_distributed(learning_rate):

    model = tf.keras.models.Sequential()
    #add model layers
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2, activation='softmax')) # set to 2 cause 2 outputs via softmax. 
    # softmax pushes values between 0 and 1. and all probs add up to 1. 
    #compile model using accuracy to measure model performance
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # categorical_crossentropy as y is one hot encoded
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == "__main__":

    args, _ = parse_args()

    x_train, y_train = get_train_data(args.train)
    x_test, y_test = get_test_data(args.train)

    model = get_model_parameter_distributed(args.learning_rate)

    history = model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=(x_test, y_test),
              verbose = 0)

    save_history(args.model_dir + "/history.p", history)
    
    # create a TensorFlow SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
    #model.save(args.model_dir + '/1')
    tf.contrib.saved_model.save_keras_model(model, args.model_dir)
    