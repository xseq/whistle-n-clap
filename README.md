# whistle-n-clap
This is a sound event detection (SED) demo for sounds that you can make around your computer: clapping, breathing, knocking and more. A public domain audio data set is preprocessed for training purposes. The train SED algorithm uses audio features (in this case, log-mel spctrogram) to do inference based on recorded sounds. 


=== Usage ===
'realtime.py' is the main entrance. It loads a trained model, streams audio and do an inference every half second.
'training.py' trains (and retrains) the SED model. Use '/util/wav_to_npz.py' to generate local numpy array first.


=== Data Set ===
This project uses a subset of the FSD50k data set.
https://annotator.freesound.org/fsd/release/FSD50K/

Note: I started the project with the name "whistle-n-clap" without realizing that the data set does not contain whistling sounds. So sorry, no whistling can be detected.


=== Reference ===
>Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, Xavier Serra. "FSD50K: an Open Dataset of Human-Labeled Sound Events", arXiv 2020.
Purwins, H., Li, B., Virtanen, T., Schlüter, J., Chang, S. Y., & Sainath, T. (2019). Deep learning for audio signal processing. IEEE Journal of Selected Topics in Signal Processing, 13(2), 206-219.
