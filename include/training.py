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


import numpy as np
import os
import logging
import tensorflow as tf


os.system('clear')
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.random.set_seed(888)


# Step 0: Feature preprocessing, needed for audio but not for images
proj_path = os.path.abspath(os.getcwd())
npz_path = proj_path + '/data/npz/'
npz_file_name = npz_path + 'data_20220802_2037.npz'
data = np.load(npz_file_name)
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

# Step 1: Shapes of inputs and outputs
# x_train, x_test = x_train / 255.0, x_test / 255.0
input_shape = x_train.shape  # row * column
output_shape = y_train.shape    # 10
print("Input Shape: " + str(input_shape))
print("Output Shape: " + str(output_shape))

# Step 2: Modeling
model = tf.keras.models.Sequential([    

    # conv1
    tf.keras.layers.Conv2D(
        data_format='channels_last',
        input_shape=(128, 130, 1),
        filters=32,
        kernel_size=(5, 5),
        padding='same',
        activation=tf.nn.relu),  # output size: [n_samples,28,28,32]

    # maxpooling1
    tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=2,
        padding='same'),  # output size: [n_samples,14,14,32]

    # conv2
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
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

model.summary()

# Step 3: Define a loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


# Step 4: Optimization method
model.compile(optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy'])


# Step 5: Training
model.fit(x_train, y_train, epochs=100,)
# saving model and weights
model_path = proj_path + '/include/model/'
model_name = 'cnn_20220802.h5'
model_full_name = model_path + model_name
model.save(model_full_name)
print('model saved.')
weight_name = 'cnn_weight_20220802'
weight_full_name = model_path + weight_name
model.save_weights(weight_full_name)
print('weights saved.')


# Step 6: Evaluation
print('Start evaluation: ')
model.evaluate(x_test,  y_test)



