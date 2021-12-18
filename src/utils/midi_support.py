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

    def load_midi_file(self, file_path):
        return pretty_midi.PrettyMIDI(file_path)

    def midi_to_16_beats_processed(self, midi_file):
        return self.prepare_song(midi_file)

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

    def add_beat_location(self, input_arr_df):
        '''
        Add four rows to a dataframe where each row is either the played or the articulation 
        binary of a note and the columns are the timestepts
        Inputs:
            - arr_df: Dataframe. Shape of (2*n_notes, timesteps*n_notes)
            - repeat_amount: TODO:Explain
        '''
        beats_per_bar = 16
        num_notes = 128
        # num_notes = 4
        total_len, _ = input_arr_df.shape

        num_beats = total_len//num_notes

        # Add an extra beat then trim it to be same shape as input later
        num_bars = num_beats//16 + 1

        bin_pattern_tst = np.array([bin(x)[2:].zfill(4) for x in np.arange(0, beats_per_bar)])
        repeated_str = np.repeat(bin_pattern_tst, num_notes, axis=0)
        repeated_str = np.tile(repeated_str, num_bars)
        all_beats = np.array([list(x) for x in repeated_str], dtype=np.int32)

        all_beats = all_beats[:total_len, :]

        all_beats = pd.DataFrame(all_beats)
        ret = pd.concat([input_arr_df, all_beats], axis=1)
        return ret

    def add_midi_value(arr, n_notes):
        '''
        Create a row of the note values by creating a range from the column indices % 12. 
        Shift by 1 because the first column does not have previous vicinity and is omitted.
        Inputs:
            - arr: One-hot array of the notes and articulations previous vicinity of each timestep.  
                  Shape of (2*n_notes, timesteps*n_notes)
        Outputs:
            - midi_row: Row of midi values for each of the columns
        '''
        midi_row = (np.arange(arr.shape[1]) + 1) % n_notes
        return midi_row

    def calculate_pitchclass(midi_row, arr):
        '''
        Create an array of 12 one-hot encoded pitchclasses. Will be 1 at the position 
        of the current note, starting at A for 0 and increasing by 1 per half-step, 
        and 0 for all the others.     Used to allow selection of more common chords 
        (i.e. it's more common to have a C major chord than an E-flat major chord).
        Inputs:
            - arr: One-hot array of the notes and articulations previous vicinity of each timestep. 
                Shape of (2*n_notes, timesteps*n_notes)
        Outputs:
            - pitchclasses: One-hot encoded array of the pitchclass of the current note.
        '''
        pitchclasses = np.array([[int(i == pitch % 12) for i in range(12)] for pitch in midi_row]).T
        return pitchclasses
    
    def build_context(arr, midi_row, pitchclass_rows):
        '''
        Create an array of 12 one-hot encoded pitchclasses. Value at index i will be 
        the number of times any note x where (x-i-pitchclass) mod 12 was played last 
        timestep. Thus if current note is C and there were 2 E's last timestep, the 
        value at index 4 (since E is 4 half steps above C) would be 2.
        Inputs:
            - arr: One-hot array of the notes and articulations previous vicinity of each timestep.  
                Shape of (2*n_notes, timesteps*n_notes)
        Outputs:
            - pitchclasses: One-hot encoded array of the pitchclass of the current note.
        '''
        df = pd.DataFrame(arr)
        df["index"] = df.index
        df = df.loc[df["index"] % 2 == 0].reset_index(drop=True)
        df["index"] = df.index
        df["pitchclass"] = df["index"].apply(lambda x: x % 12)
        df = df.groupby("pitchclass").sum()
        df = df.drop(["index"], axis=1)
        return np.array(df)

    def song_to_beats(self, song_df, beats_expanded):
        resampled = song_df.loc[beats_expanded]
        return resampled.values

    def windowed_data_across_notes_time(self, in_data, mask_length_x=6, return_labels=True):
        length, num_in_time = in_data.shape

        pad_val = mask_length_x//2

        num_notes = length
        stride_length_x = 2

        elements_per_time_step = ((num_notes + 2*pad_val) - mask_length_x)//stride_length_x

        padded = np.pad(in_data, pad_val)[:, pad_val:pad_val+num_in_time]

        # padval_offset = pad_val * num_in_time

        row_offset = 1 * num_in_time
        # print(f"row_offset is {row_offset}")

        stride = stride_length_x
        mask_width = mask_length_x

        row_width = num_in_time
        horizontal_indexing_inner = num_in_time * stride * np.arange(elements_per_time_step)[None, :]
        horizontal_indexing_increment = np.repeat(np.arange(num_in_time), elements_per_time_step, axis=0).flatten()
        horizontal_indexing = np.repeat(horizontal_indexing_inner, num_in_time, axis=0).flatten() + horizontal_indexing_increment
        vertical_indexing = row_offset + row_width * np.arange(mask_width)[:, None]
        indexer = horizontal_indexing + vertical_indexing
        indexer = indexer.astype(np.int)
        X = padded.flatten()[indexer]


        if mask_length_x % 2 != 0:
            half_mask_offset = (mask_length_x//2 + 1 )*num_in_time
        else:
            half_mask_offset = (mask_length_x//2 )*num_in_time


        mask_width = 2
        row_width = num_in_time
        horizontal_indexing_inner = num_in_time * stride * np.arange(elements_per_time_step)[None, :]
        horizontal_indexing_increment = np.repeat(np.arange(num_in_time), elements_per_time_step, axis=0).flatten()
        horizontal_indexing = np.repeat(horizontal_indexing_inner, num_in_time, axis=0).flatten() + horizontal_indexing_increment
        vertical_indexing = half_mask_offset + row_width * np.arange(mask_width)[:, None]
        label_indexer = horizontal_indexing + vertical_indexing

        y = padded.flatten()[label_indexer].astype(np.int32)

        if return_labels:
            X = X[:, 0:-(elements_per_time_step)]
            y = y[: ,elements_per_time_step:]
            return X, y, elements_per_time_step
        else:
            return X, None, elements_per_time_step

    def midi_obj_to_play_articulate(self, midi_obj):

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

        play_articulated_concat = np.concatenate([notes_held.T, note_articulated.T], axis=1)
        play_articulated = play_articulated_concat.flatten().reshape(-1, min_length)
        return play_articulated

    def all_midi_obj_to_play_articulate(self, midi_obj_list):
        loaded_data = pd.concat([self.midi_obj_to_play_articulate(m) for m in midi_obj_list])
        return loaded_data

    def transform_beats_to_batch(self, X, y, elements_per_time_step):
        """Transform beat groups to new dimension

        Takes elements_per_time_step columns then moves the 2d portion into a new dimension

        Args:
            X (np.array): X coming from windowed_data_across_notes_time with additional information added
            y (np.array): 
            elements_per_time_step ([type]): [description]

        Returns:
            X: X 3d shape now
            u: y 3D shape
        """
        num_groups = X.shape[1]//elements_per_time_step
        X = np.dstack(np.split(X, num_groups, axis=1))
        y = np.dstack(np.split(y, num_groups, axis=1))

        # Initially, the 0 dimension indexes to different input per note for a given time. For example taking the last four elements gives beat info -5:-1
        # Initially, the 1 dimension indexes to different notes.
        # Initially, the 2 dimension indexes to different beats.
        X = np.swapaxes(X, 0, 2)
        y = np.swapaxes(y, 0, 2)
        # After swapping axes the 0 dimension indexes to different beats.
        # After swapping axes the 1 dimension indexes to different notes.
        # After swapping axes the 2 dimension indexes to different input per note for a given time. For example taking the last four elements gives beat info -5:-1

        return X, y

    def prepare_song_time_invariant(self, midi_obj, vicinity=50):
        '''
        Convert a given array of one-hot encoded midi notes into an array with the 
        following values (the number in brackets is the number of elements in the 
        input vector that correspond to each part). 
            - Position[1] - the MIDI note value
            - Pitchclass[12]
            - Previous vicinity[50]
            - Previous context[12]
            - Beat[4]
        Inputs:
            - midi_obj: One-hot array of the notes and articulations of each timestep. 
                        Shape of (2*n_notes, timesteps)
            - vicinity: Integer. Number of notes considered around specified note.
        Outputs:
            - X: Training data. Shape of (2*n_notes, timesteps*n_notes)
            - y: Labels. For each note, this is a list of [played, articulated]
            - elements_per_time_step:
        '''
        play_articulated = self.midi_obj_to_play_articulate(midi_obj=midi_obj, vicinity=vicinity)
        
        n_notes, timesteps = play_articulated.shape

        # window across each note vicinity
        X, y, elements_per_time_step = self.windowed_data_across_notes_time(play_articulated, mask_length_x=vicinity)

        # Get array of Midi values for each note value
        midi_row = self.add_midi_value(X, n_notes)

        # Get array of one hot encoded pitch values for each note value
        pitchclass_rows = self.calculate_pitchclass(midi_row, X)

        # Add total_pitch count repeated for each note window
        previous_context = self.build_context(X, midi_row, pitchclass_rows)

        X = np.vstack((midi_row, pitchclass_rows, X, previous_context))

        # Add beats repeated for each note window
        X = self.add_beat_location(pd.DataFrame(X.T), repeat_amount=elements_per_time_step).T
        
        return X, pd.DataFrame(y), elements_per_time_step

    def prepare_song_note_invariant_plus_beats(self, all_midi_objs, vicinity=50):
        '''
        Convert a given array of one-hot encoded midi notes into an array with the 
        following values (the number in brackets is the number of elements in the 
        input vector that correspond to each part). 
            - Previous vicinity[50]
            - Beat[4]
        Inputs:
            - songs_df: One-hot array of the notes and articulations of each timestep. 
                        Shape of (2*n_notes, timesteps)
            - vicinity: Integer. Number of notes considered around specified note.
        Outputs:
            - X: Training data. Shape of (2*n_notes, timesteps*n_notes)
            - y: Labels. For each note, this is a list of [played, articulated]
            - elements_per_time_step:
        '''
        play_articulated = self.all_midi_obj_to_play_articulate(all_midi_objs)
        X, y, elements_per_time_step = self.windowed_data_across_notes_time(play_articulated, mask_length_x=vicinity)
        X = self.add_beat_location(pd.DataFrame(X.T), repeat_amount=elements_per_time_step).T
        X, y = self.transform_beats_to_batch(X, y)
        return X, y

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
        
        return play_articulated


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

def load_midi_objs(data_dir="", num_files=15, seq_length=15):

    download_and_save_data()

    # filenames = glob.glob(str("music_generation_rnn/training_data/classical/**/**/*.mid*"))
    filenames = glob.glob(str('data/maestro-v2.0.0/**/*.mid*'))
    if len(filenames) == 0:
        raise Exception("Couldn't find the downloaded data :(")

    ms = MidiSupport()

    num_files = num_files
    midi_objs = []
    for f in filenames[:num_files]:
        mf_i = ms.load_midi_file(f)
        
        midi_objs.append(mf_i)

    return midi_objs

if __name__ == "__main__":


    ms = MidiSupport()

    mf = ms.load_midi_file("music_generation_rnn/training_data/Alan_Walker_Faded.midi")

    prepared = ms.midi_to_16_beats_processed(mf)

    print(prepared.T.iloc[2*50:2*50+40, 0:16])

    
