from keras.datasets import fashion_mnist
from keras.utils.np_utils import to_categorical
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential

count=1

filt=64
epoch=1
unit=100

def data():
	(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
	X_train = X_train.reshape((-1, 28, 28, 1)).astype('float32')
	X_test = X_test.reshape((-1, 28, 28, 1)).astype('float32')
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)
	X_train_norm = X_train / 255
	X_test_norm = X_test / 255
	return X_train_norm,y_train,X_test_norm,y_test

def create_model(f,u):
	model = Sequential()
	model.add(Convolution2D(filters=f, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
	f=int(f/2)
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(units=u, activation='relu'))
	u=int(u/2)
	model.add(Dense(10, activation='softmax'))
	h = model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

X_train_norm,y_train,X_test_norm,y_test = data()
model=create_model(filt,unit)
print(model.summary())

trained_model = model.fit(X_train_norm, y_train,
         epochs=epoch,batch_size=32,
          validation_data=(X_test_norm, y_test),
          )

final_acc=int(trained_model.history['accuracy'][-1]*100)

f = open("accuracy.txt", "w")
f.write(str(final_acc))
f.close()
