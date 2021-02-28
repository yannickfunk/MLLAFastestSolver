#  csv processing
import pandas as pd
import numpy as np

# keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint

# other imports
from time import time


# method to create a dataset returning features x and labels y
def create_dataset(df):
    df = df.replace(True, 1).replace(False, 0)
    features = []
    labels = []
    counter = 0

    for index, row in df.sample(frac=1).iterrows():  # iterate over shuffled dataframe
        feature_array = np.array(row[7:])  # consider all columns but to last one as features

        label_raw = np.array(row[1:7])
        label = np.zeros(len(label_raw))
        label[np.argmin(label_raw)] = 1

        features.append(feature_array)
        labels.append(label)

    features = np.array(features).astype("float64")
    labels = np.array(labels).astype("float64")

    # shuffle created dataset
    p = np.random.permutation(len(features))
    return features[p], labels[p]


# method for setting up deep learning architecture
def create_model():
    model = Sequential()

    model.add(Dense(32, input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))

    # adam as optimizer, mean squared error as loss function
    model.compile(optimizer=Adam(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model


# method returning a list of all callbacks
def define_callbacks():
    """
    return [TensorBoard(log_dir='logs/{}'.format(time())),
            ModelCheckpoint(filepath="nets/{epoch:02d}-{val_acc:.2f}.h5", monitor='val_acc',
                            verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)]
    """
    return []


x, y = create_dataset(pd.read_csv('../dataset.csv'))  # loading features x and labels y
# constants used for training
input_dim = x.shape[1]
num_samples = x.shape[0]
batch_size = 8
epochs = 2000
validation_split = 0.3
lr = 0.001

classifier = create_model()  # define classifier
classifier.fit(x, y,  # fit classifier
               epochs=epochs,
               validation_split=validation_split,
               shuffle=True,
               callbacks=define_callbacks())