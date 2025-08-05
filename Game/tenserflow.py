import numpy as np
import tenserflow as tf
import keras 
# from keras.layers import Dense
# from keras.models import Sequential


model = tf.keras.Sequential([tf.keras.Input(shape=(1,)),
                             tf.keras.layers.Dense(units=1)])

model.complie(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0,0.0,1.0,2.0,3.0,4.0], dtype=float)
ys = np.array([-3.0,-1.0,1.0,3.0,.0,7.0], dtype=float)

model.fit(xs, ys, epochs=500) #epochs is how many itereations

model.predict(np.array([10.0]))

fashion_mnist = tf.keras.datasets.fashion_mnist(train_images,train_labels), (test_images, test_labels)=
fashion_mnist.load_data()

index = 42

np.set_printoptions(linewidth=320)

print(f'label: {training_labels[index]}')
print(f'\nimage pixel array :\n\n{traing_images[index]}\n\n')

plt.imshow(training_images[index])
plt.colorbar()
plt.show()

#to make the value either a zero or a one
traing_images = training_images /255
test_images = test_images/255

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28,28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              matrics=['accuracy'])

model.fit(training_images,training_labels,epochs=5)
model.evaluate(test_images, test_labels)