# whistle-n-clap
This is a sound event detection (SED) demo for sounds that you can make around your computer: clapping, breathing, knocking and more. A public domain audio data set is preprocessed for training purposes. The train SED algorithm uses audio features (in this case, log-mel spctrogram) to do inference based on recorded sounds. 


=== Usage ===
'realtime.py' is the main entrance. It loads a trained model, streams audio and do an inference every half second.
'training.py' trains (and retrains) the SED model. Use '/util/wav_to_npz.py' to generate local numpy array first.


=== Data Set ===
This project uses a subset of the FSD50k data set.
https://annotator.freesound.org/fsd/release/FSD50K/

Note: I started the project with the name "whistle-n-clap" without realizing that the data set does not contain whistling sounds. So sorry, no whistling can be detected.


=== Notes ===
I encountered the following problems during the development:

* [audio streaming] read and write audio data; Solution: pyaudio, see https://people.csail.mit.edu/hubert/pyaudio/docs/ 

* [audio streaming] audio recording device ID changes after replugging the USB headset. Solution: get_device_info_by_index, see https://stackoverflow.com/questions/40704026/voice-recording-using-pyaudio

* [audio streaming] Input overflowed in real time processing; used a frame size of 32768 which has less issue. Solution: ignore overflow errors, see https://stackoverflow.com/questions/10733903/pyaudio-input-overflowed

* [audio streaming] Data formatting for wav files. Solution: WAV files can specify several data types, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html

* [audio streaming] Circular buffer; using a circular buffer would save processing time and improve efficiency; not fixed for now. Solution: queue and dqueue https://stackoverflow.com/questions/4151320/efficient-circular-buffer

* [audio streaming] Buffer data type in pyaudio; each buffer frame comes as a string; has a lot of difficulty transforming it to float numbers. Solution: PyAudio is giving you binary-encoded audio frames as bytes in a string.  unpack it with struct, see https://stackoverflow.com/questions/36413567/pyaudio-convert-stream-read-into-int-to-get-amplitude and https://stackoverflow.com/questions/19629496/get-an-audio-sample-as-float-number-from-pyaudio-stream

* [preprocessing] Saving the preprocessed audio features to a single file for training purposes. Solution: numpy.savez, see https://numpy.org/doc/stable/reference/generated/numpy.savez.html 

* [preprocessing] Audio features from pyaudio. Solution: log mel spectrogram, see https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html

* [training] Set random seeds in tensorflow 2.9. Solution: tf.random.set_seed, see https://www.tensorflow.org/api_docs/python/tf/random/set_seed

* [training] Tensorflow 2 quick start. See https://www.tensorflow.org/tutorials/quickstart/beginner



* [inference] Keras complains about input size; the reason is that Keras wants batched data; Solution: numpy.expand_dims, see https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html, somehow reshape didn't work, see https://stackoverflow.com/questions/52322794/keras-model-predict-error-when-checking-input-shape

* [inference] Show detection score; Solution: only showing the top score above a huristic threshold





=== Reference ===
>Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, Xavier Serra. "FSD50K: an Open Dataset of Human-Labeled Sound Events", arXiv 2020.
Purwins, H., Li, B., Virtanen, T., Schlüter, J., Chang, S. Y., & Sainath, T. (2019). Deep learning for audio signal processing. IEEE Journal of Selected Topics in Signal Processing, 13(2), 206-219.
