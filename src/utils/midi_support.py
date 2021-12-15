"""Module for support with midi files

"""
import pandas as pd
import numpy as np
import glob
import pathlib
#import pygame
import pretty_midi
import tensorflow as tf


class MidiSupport():

    def __init__(self):
        pass

    def play_midi_file(self, file_path):

        
        def play_music(midi_filename):
            '''Stream music_file in a blocking manner'''
            clock = pygame.time.Clock()
            pygame.mixer.music.load(midi_filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                clock.tick(30) # check if playback has finished
            
        midi_filename = file_path

        # mixer config
        freq = 44100  # audio CD quality
        bitsize = -16   # unsigned 16 bit
        channels = 1  # 1 is mono, 2 is stereo
        buffer = 1024   # number of samples
        pygame.mixer.init(freq, bitsize, channels, buffer)

        # optional volume 0 to 1.0
        pygame.mixer.music.set_volume(0.8)

        # listen for interruptions
        try:
            # use the midi file you just saved
            play_music(midi_filename)
        except KeyboardInterrupt:
            # if user hits Ctrl/C then exit
            # (works only in console mode)
            pygame.mixer.music.fadeout(1000)
            pygame.mixer.music.stop()
            raise SystemExit

    def load_midi_file(self, file_path):
        return pretty_midi.PrettyMIDI(file_path)

    def midi_to_16_beats_processed(self, midi_file):
        return self.prepare_song(midi_file)

    def add_beat_location(self, input_arr_df):
        input_arr_df["ind"] = range(len(input_arr_df))
        input_arr_df["id"] = input_arr_df["ind"].apply(lambda x: x%16)

        input_arr_df[['id_0','id_1', 'id_2', 'id_3']] = input_arr_df['id'].apply(lambda x: pd.Series(list(bin(x)[2:].zfill(4))))
        input_arr_df = input_arr_df.drop(["ind", "id"], axis=1)
        for col in ["id_0", "id_1", "id_2", "id_3"]:
            input_arr_df[col] = input_arr_df[col].astype(float)
        return input_arr_df

    def song_to_beats(self, song_df, beats_expanded):
        resampled = song_df.loc[beats_expanded]
        return resampled.values

    def song_to_beat_articulation(self, song_df, beats_expanded):
        print("starting")
        
        note_sub_df_np = song_df.values
        changed = (note_sub_df_np[1:] > note_sub_df_np[:-1])
        
        total_notes, not_pitches = changed.shape
        # TODO: This is ugly
        size_of_group = len(song_df.loc[beats_expanded[0]:beats_expanded[1],:]) - 1
        divisible_len = (total_notes // size_of_group) * size_of_group
        num_groups = divisible_len // size_of_group
        split_arr = np.dstack(np.split(changed[:divisible_len], num_groups))
        note_changes = split_arr.any(axis=0).astype(np.int32)

        # Note: adding zeros at the beginning to register the first note increase.
        aa = (note_sub_df_np[0:1] > 0).astype(np.int32)
        note_changes = np.concatenate([aa.T, note_changes], axis=1)
        return note_changes

    def prepare_song(self, midi_obj):
        samp_f = 100
        beats_together = [np.arange(x,y, (y-x)/4) for x, y in zip(midi_obj.get_beats(), midi_obj.get_beats()[1:])]
        beats_expanded = [x for y in beats_together for x in y]
        beats_expanded = np.around(beats_expanded, 2)

        song_df = pd.DataFrame(midi_obj.get_piano_roll(fs=samp_f)).T
        song_df["time"] = np.around([x*(1/samp_f) for x in range(len(song_df))], 2)
        song_df.index = song_df["time"]
        song_df = song_df.drop("time", axis=1)
        note_articulated = self.song_to_beat_articulation(song_df, beats_expanded).T
        notes_held = self.song_to_beats(song_df, beats_expanded)

        min_length = min(notes_held.shape[0], note_articulated.shape[0])
        note_articulated = note_articulated[0:min_length, :].astype(np.float32)
        notes_held = notes_held[0:min_length, :]

        # prepare_song(mf_that_song)

        play_articulated_concat = np.concatenate([notes_held.T, note_articulated.T], axis=1)
        play_articulated = play_articulated_concat.flatten().reshape(-1, min_length)
        
        
        return self.add_beat_location(pd.DataFrame(play_articulated).T)


    def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
        '''Convert a Piano Roll array into a PrettyMidi object
        with a single instrument.
        Parameters
        ----------
        piano_roll : np.ndarray, shape=(128,frames), dtype=int
            Piano roll of one instrument
        fs : int
            Sampling frequency of the columns, i.e. each column is spaced apart
            by ``1./fs`` seconds.
        program : int
            The program number of the instrument.
        Returns
        -------
        midi_object : pretty_midi.PrettyMIDI
            A pretty_midi.PrettyMIDI class instance describing
            the piano roll.
        '''
        notes, frames = piano_roll.shape
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=program)

        # pad 1 column of zeros so we can acknowledge inital and ending events
        piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

        # use changes in velocities to find note on / note off events
        velocity_changes = np.nonzero(np.diff(piano_roll).T)

        # keep track on velocities and note on times
        prev_velocities = np.zeros(notes, dtype=int)
        note_on_time = np.zeros(notes)

        for time, note in zip(*velocity_changes):
            # use time + 1 because of padding above
            velocity = piano_roll[note, time + 1]
            time = time / fs
            if velocity > 0:
                if prev_velocities[note] == 0:
                    note_on_time[note] = time
                    prev_velocities[note] = velocity
            else:
                pm_note = pretty_midi.Note(
                    velocity=prev_velocities[note],
                    pitch=note,
                    start=note_on_time[note],
                    end=time)
                instrument.notes.append(pm_note)
                prev_velocities[note] = 0
        pm.instruments.append(instrument)
        return pm


class RNNMusicDataSetPreparer():

    def __init__(self) -> None:
        pass


    def prepare(self, all_song_dfs, seq_length=15):

        print(f"all_song_dfs.shape in is {all_song_dfs.shape}")
        song_tensor=tf.convert_to_tensor(all_song_dfs)
        # song_tensor_reshape = tf.reshape(song_tensor, (6261, 128))
        dataset = tf.data.Dataset.from_tensor_slices(song_tensor)
        vocab_size = 128
        """Returns TF Dataset of sequence and label examples."""
        seq_length = seq_length+1

        # Take 1 extra for the labels
        windows = dataset.window(seq_length, shift=1, stride=1,
                                    drop_remainder=True)

        # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
        flatten = lambda x: x.batch(seq_length, drop_remainder=True)
        sequences = windows.flat_map(flatten)

        # Normalize note pitch
        def scale_pitch(x):
            x = x/vocab_size
            return x

        # Split the labels
        def split_labels(sequences):
            inputs = sequences[:-1]
            print(f"sequences.shape is {sequences.shape}")
            # labels_dense = sequences[-1][0]
            labels_dense = tf.reshape(sequences[-1][0:256], (256, 1))
            return inputs, {"pitch": labels_dense}

        seq_ds = sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)


        X_tst, y_tst = list(seq_ds.take(1))[0]
        print(f"X_tst.shape out is {X_tst.shape}")
        print(f"y_tst.shape out is {X_tst.shape}")

        return seq_ds


def download_and_save_data():

    data_dir = pathlib.Path('data/maestro-v2.0.0')
    if not data_dir.exists():
        tf.keras.utils.get_file(
            'maestro-v2.0.0-midi.zip',
            origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
            extract=True,
            cache_dir='.', cache_subdir='data',
        )

def load_music(data_dir="", num_files=15, seq_length=15):

    download_and_save_data()

    # filenames = glob.glob(str("music_generation_rnn/training_data/classical/**/**/*.mid*"))
    filenames = glob.glob(str('data/maestro-v2.0.0/**/*.mid*'))
    if len(filenames) == 0:
        raise Exception("Couldn't find the downloaded data :(")

    ms = MidiSupport()

    num_files = num_files
    song_dfs = []
    for f in filenames[100:100+num_files]:
        mf_i = ms.load_midi_file(f)
        
        song_dfs.append(ms.prepare_song(mf_i))

    all_song_dfs = pd.concat(song_dfs)
    all_song_dfs = (all_song_dfs > 0) * 1
    return all_song_dfs

if __name__ == "__main__":


    ms = MidiSupport()

    mf = ms.load_midi_file("music_generation_rnn/training_data/Alan_Walker_Faded.midi")

    prepared = ms.midi_to_16_beats_processed(mf)

    print(prepared.T.iloc[2*50:2*50+40, 0:16])

    
