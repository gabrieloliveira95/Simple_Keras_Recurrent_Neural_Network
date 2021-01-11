import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb

tf.__version__


def rnnModel():
    model = tf.keras.Sequential()
    # print(x_train.shape)
    # print(x_train.shape[1])
    model.add(tf.keras.layers.Embedding(input_dim=number_of_words,
                                        output_dim=128, input_shape=[x_train.shape[1]]))
    model.add(tf.keras.layers.LSTM(units=128, activation='tanh'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def loadModelAndWeights():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # Evaluate loaded model on test data
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    print(f'Test Loss = {test_loss}')
    print(f'Test Accuracy = {test_accuracy}')


if __name__ == "__main__":
    number_of_words = 20000
    max_len = 100

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=number_of_words)
    print(f'Train Shape: {x_train.shape}')  # 25.000 Texts
    # print(x_train)
    # print(x_train[0])
    # print(y_train)
    print(f'Number of Words Position [0]: {len(x_train[0])}')  # 218 Words
    print(f'Number of Words Position [1]: {len(x_train[1])}')  # 189 Words

    # PRE PROCESSING
    x_train = tf.keras.preprocessing.sequence.pad_sequences(
        x_train, maxlen=max_len)
    print('Pre Processing...')
    print('Now:')
    print(f'Number of Words Position [0]: {len(x_train[0])}')  # 218 Words
    print(f'Number of Words Position [1]: {len(x_train[1])}')  # 189 Words
    x_test = tf.keras.preprocessing.sequence.pad_sequences(
        x_test, maxlen=max_len)

    model = model()
    print('Model Summary:')
    print(model.summary())

    print('\n\nDo You Want To Train This Model??')
    confirm = input(
        "It may take a long time... [y or n]: ")

    if confirm.upper() == 'Y':

        model.fit(x_train, y_train, epochs=3, batch_size=128)

        test_loss, test_accuracy = model.evaluate(x_test, y_test)

        print(f'Loss = {test_loss}')
        print(f'Accuracy = {test_accuracy}')

        save = input(
            "Do You Want to Save This Model and Weights? [y or n]: ")
        if save.upper() == 'Y':
            # Model to JSON
            model_json = model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)

            # Weights to HDF5
            model.save_weights("model.h5")
            print("Saved model to disk")

            # Save Complete Model
            # model.save('model.h5')

            testSavedModel = input(
                "You want to Test The Saved Models and Weights? [y or n]: ")
            if testSavedModel.upper() == 'Y':
                loadModelAndWeights()
