
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt

(train_x, train_y), (test_x, test_y) = mnist.load_data()

tarin_itr = 10


train_x = train_x.reshape(train_x.shape[0], 28*28)
test_x = test_x.reshape(test_x.shape[0], 28*28)

train_y = np_utils.to_categorical(train_y, tarin_itr)
test_y = np_utils.to_categorical(test_y, tarin_itr)


model = Sequential()

model.add(Dense(100, input_dim=28*28))

model.add(Activation('sigmoid'))
model.add(Dense(tarin_itr))
model.add(Activation('sigmoid'))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics = ['accuracy'])

model.summary()


history = model.fit(train_x, train_y, batch_size=128, nb_epoch=10, verbose=2, validation_data = (test_x, test_y))


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



