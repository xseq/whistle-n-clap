# This is a simple realization of CNN based on this article: https://zhuanlan.zhihu.com/p/52298361
# The key: six steps of building a machine learning application
# 0) Proprocessing
# 1) Define inputs and outputs
# 2) Modeling
# 3) Define the cost function
# 4) Define the optimization method
# 5) Train and validate the model
# 6) Check the results

# environment: (spoken_numbers) python 3.9.7, tensorflow 2.8.0
# notes for tf 2.8 https://www.tensorflow.org/tutorials/quickstart/beginner


import os
import logging
import tensorflow as tf


os.system('clear')
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.random.set_seed(111)


# Step 0: Feature preprocessing, needed for audio but not for images
mnist = tf.keras.datasets.mnist


# Step 1: Shapes of inputs and outputs
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
input_shape = x_train.shape[1] * x_train.shape[2]  # row * column
output_shape = y_test.max() - y_train.min() + 1    # 10
print("Input Shape: " + str(input_shape))
print("Output Shape: " + str(output_shape))


# Step 2: Modeling
model = tf.keras.models.Sequential([    

    # conv1
    tf.keras.layers.Conv2D(
        data_format='channels_last',
        input_shape=(28, 28, 1),
        filters=32,
        kernel_size=(5, 5),
        padding='same',
        activation=tf.nn.relu),  # output size: [n_samples,28,28,32]

    # maxpooling1
    tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=2,
        padding='same'),  # output size: [n_samples,14,14,32]

    # fc1
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        units=256, 
        activation=tf.nn.relu
    ),

    # dropout 1
    tf.keras.layers.Dropout(
        rate=0.3
    ),

    # fc2
    tf.keras.layers.Dense(
        units=32, 
        activation=tf.nn.relu
    ),

    # dropout 2
    tf.keras.layers.Dropout(
        rate=0.3
    ),

    # fc3
    tf.keras.layers.Dense(
        units=10, 
        activation=tf.nn.relu
    )

])


# Step 3: Define a loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


# Step 4: Optimization method
model.compile(optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy'])


# Step 5: Training
model.fit(x_train, y_train, epochs=50)


# Step 6: Evaluation
model.evaluate(x_test,  y_test)


# probability_model = tf.keras.Sequential([
#     model,
#     tf.keras.layers.Softmax()
# ])
# print(probability_model(x_test[:5]))


