# common
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;plt.style.use('ggplot')
import seaborn as sns; sns.set()
import random
# sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import PolynomialFeatures
# keras
from keras.callbacks import Callback
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional

print(tf.__version__)

# load data
data_train = pd.read_csv('./data/train.csv')
data_tornm = pd.read_csv('./data/test1.csv')
X, y = np.array(data_train.ix[:,:-1]), np.array(data_train.ix[:,-1]) # Read 1 column as X, Read 2nd column as y
print("X=",X)
print("y=",y)

# callback
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
# divide train / test
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.15,
                                                    random_state=random.randint(0, 100))

# model
print('X_train:', X_train)
print('X_test:', X_test)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(100, 128, input_length=X_train.shape[1]))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

# train model
losshist = LossHistory()
print(model.summary())
model.fit(X_train, y_train, verbose=2,
          batch_size=50,
          nb_epoch=4,
          validation_data=[X_test, y_test],
          callbacks=[losshist])
