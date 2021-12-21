# Music Generation RNN
This repository is on a project to generate music using a biaxial recurrent neural network. The project is inspired by the [research paper of Daniel J. Johnson](https://link.springer.com/chapter/10.1007/978-3-319-55750-2_9). The code is intended to replicate the orginial model as closely as possible using TensorFlow 2.0. The model uses LSTM layers and a convolution-like reshaping of the input to predict which notes will be played when. Additionally, we implement different approaches to data processing and model architectures. We report on the results of these experiments.

## Original Paper
Original paper: https://link.springer.com/chapter/10.1007/978-3-319-55750-2_9

Blog: https://www.danieldjohnson.com/2015/08/03/composing-music-with-recurrent-neural-networks/

Github: https://github.com/danieldjohnson/biaxial-rnn-music-composition (The original code is written using Theano, a library that we did not use)


## Instruction on How to Run the Code
Running the code only requires running the jupyter notebook called **[rnn-music-generation.ipynb](/rnn-music-generation.ipynb)**.

Most of the code relies on base Python, TensorFlow 2.0, Pandas, and Numpy.

The dataset will be installed using the jupyter notebook.

Some not so common libraries will be installed by the notebook. Should this not work, they should be installed using the following commands:
```
pip install pretty-midi
sudo apt install -y fluidsynth
pip install --upgrade pyfluidsynth
```

## Data
The Maestro dataset by Google Magenta is used. MAESTRO (MIDI and Audio Edited for Synchronous TRacks and Organization) is a dataset composed of about 200 hours of virtuosic piano performances captured with fine alignment (~3 ms) between note labels and audio waveforms.

Link: https://magenta.tensorflow.org/datasets/maestro

## Organization of the Directory

audio:  Stores output audio files
data:   Stores input audio files
plots:  Stores output plots of the created music and graphs
src:    Stores all source code
models: Stores the classes and functions for the differen neural networks
utils:  Stores support function like those needed to convert and visualize midi-files

```
./
├── audio
├── data
├── plots
├── run_scripts
└── src
    ├── models
    └── utils
```

## Authors
Trevor Gordon: tjg2148@columbia.edu
Jannik Wiedenhaupt: j.wiedenhaupt@columbia.edu
