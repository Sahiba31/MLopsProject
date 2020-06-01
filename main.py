from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))
mod = model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

trained_model = model.fit(X_train, Y_train,
         epochs=10,batch_size=32,
          validation_data=(X_test, Y_test),
          )

final_acc=int(trained_model.history['accuracy'][-1]*100)


final_acc

loss , acc = model.evaluate(X_test, Y_test)


acc

f = open("output.txt", "w")
f.write(str(final_acc))
f.close()