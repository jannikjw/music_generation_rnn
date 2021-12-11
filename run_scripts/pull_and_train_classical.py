"""Pull and train classical music
"""
import pandas as pd
import glob
import pathlib
import requests
import tensorflow as tf

import sys
sys.path.insert(0, "/Users/loreliegordon/Library/Mobile Documents/com~apple~CloudDocs/Documents/root/Columbia/Fall2021/ECBM4040/Project/rnn_music/music_generation_rnn/")
from src.utils.midi_support import MidiSupport

def save_music(num_files=15):

    # pm = mf
    data_dir = pathlib.Path("music_generation_rnn/training_data/classical/")
    if not data_dir.exists():
        tf.keras.utils.get_file(
            'maestro-v2.0.0-midi.zip',
            origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
            extract=True,
            cache_dir='.', cache_subdir='data',
        )
    

def load_music(num_files=15):

    filenames = glob.glob(str("music_generation_rnn/training_data/classical/**/**/*.mid*"))

    ms = MidiSupport()

    num_files = 15
    song_dfs = []
    for f in filenames[:num_files]:
        mf_i = ms.load_midi_file(f)
        # song_dfs.append(pd.DataFrame(mf.get_piano_roll(fs=10)).T)
        song_dfs.append(ms.prepare_song(mf_i))

    all_song_dfs = pd.concat(song_dfs)
    all_song_dfs = (all_song_dfs > 0) * 1

    return all_song_dfs

def save_and_load_music(num_files=15):
    save_music(num_files=num_files)
    return load_music(num_files=num_files)


if __name__ == "__main__":
    load_music(num_files=15)