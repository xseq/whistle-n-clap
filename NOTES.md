
=== Notes ===
I encountered the following problems during the development:

* [preprocessing] Saving the preprocessed audio features to a single file for training purposes. Solution: numpy.savez, see https://numpy.org/doc/stable/reference/generated/numpy.savez.html 

* [preprocessing] Audio features from pyaudio. Solution: log mel spectrogram, see https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html

* [training] Set random seeds in tensorflow 2.9. Solution: tf.random.set_seed, see https://www.tensorflow.org/api_docs/python/tf/random/set_seed

* [training] Tensorflow 2 quick start. See https://www.tensorflow.org/tutorials/quickstart/beginner

* [training] CNN layer configuration. see https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

* [training] Maxpooling layer configuration. see https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D

* [training] Fully connected layer configuration. see https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense

* [training] Multiple class classificiation loss function. Solution: tf.keras.losses.SparseCategoricalCrossentropy, Use this crossentropy loss function when there are two or more label classes. We expect labels to be provided as integers. see https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy

* [training] Saving and loading a model. Solution: model.save, see https://www.tensorflow.org/tutorials/keras/save_and_load

* [training] Optimizer configuration. see https://www.tensorflow.org/api_docs/python/tf/keras/optimizers

* [training] CNN model framework. see https://zhuanlan.zhihu.com/p/52298361
        # The key: six steps of building a machine learning application
        # 0) Proprocessing
        # 1) Define inputs and outputs
        # 2) Modeling
        # 3) Define the cost function
        # 4) Define the optimization method
        # 5) Train and validate the model
        # 6) Check the results

* [inference] Keras complains about input size; the reason is that Keras wants batched data; Solution: numpy.expand_dims, see https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html, somehow reshape didn't work, see https://stackoverflow.com/questions/52322794/keras-model-predict-error-when-checking-input-shape

* [inference] Show detection score; Solution: only showing the top score above a huristic threshold

* [audio streaming] read and write audio data; Solution: pyaudio, see https://people.csail.mit.edu/hubert/pyaudio/docs/ 

* [audio streaming] audio recording device ID changes after replugging the USB headset. Solution: get_device_info_by_index, see https://stackoverflow.com/questions/40704026/voice-recording-using-pyaudio

* [audio streaming] Input overflowed in real time processing; used a frame size of 32768 which has less issue. Solution: ignore overflow errors, see https://stackoverflow.com/questions/10733903/pyaudio-input-overflowed

* [audio streaming] Data formatting for wav files. Solution: WAV files can specify several data types, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html

* [audio streaming] Circular buffer; using a circular buffer would save processing time and improve efficiency; not fixed for now. Solution: queue and dqueue https://stackoverflow.com/questions/4151320/efficient-circular-buffer

* [audio streaming] Buffer data type in pyaudio; each buffer frame comes as a string; has a lot of difficulty transforming it to float numbers. Solution: PyAudio is giving you binary-encoded audio frames as bytes in a string.  unpack it with struct, see https://stackoverflow.com/questions/36413567/pyaudio-convert-stream-read-into-int-to-get-amplitude and https://stackoverflow.com/questions/19629496/get-an-audio-sample-as-float-number-from-pyaudio-stream
