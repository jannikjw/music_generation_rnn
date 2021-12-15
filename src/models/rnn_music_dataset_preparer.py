import pandas as pd
import numpy as np
import tensorflow as tf


class RNNMusicDataSetPreparer():
  def __init__(self) -> None:
    pass
  
  def windowed_data_across_notes_time(in_data, mask_length_x=25):

    length, num_in_time = in_data.shape
    flat_in_data = in_data.flatten()

    pad_val = mask_length_x//2

    num_notes = length
    stride_length_x = 1

    len_out_per_time_step = 1 + ((num_notes + 2*pad_val) - mask_length_x)//stride_length_x
    
    padded = np.pad(in_data, pad_val)[:, pad_val:pad_val+num_in_time]

    stride = stride_length_x
    mask_width = mask_length_x
    elements_per_time_step = 1 + (num_notes - mask_length_x)//stride_length_x
    row_width = num_in_time
    horizontal_indexing_inner = num_in_time * stride * np.arange(elements_per_time_step)[None, :]
    horizontal_indexing_increment = np.repeat(np.arange(num_in_time), elements_per_time_step, axis=0).flatten()
    horizontal_indexing = np.repeat(horizontal_indexing_inner, num_in_time, axis=0).flatten() + horizontal_indexing_increment
    vertical_indexing = row_width * np.arange(mask_width)[:, None]
    indexer = horizontal_indexing + vertical_indexing
    indexer = indexer.astype(np.int)

    print(f"in_data.shape is {in_data.shape}")
    print(f"padded.shape is {padded.shape}")

    X = padded.flatten()[indexer]

    len_out_per_time_step_y_label = 1

    half_mask_offset = (mask_length_x//2 + 1)*num_in_time
    
    row_width = num_in_time
    horizontal_indexing_inner = num_in_time * stride * np.arange(elements_per_time_step)[None, :]
    horizontal_indexing_increment = np.repeat(np.arange(num_in_time), elements_per_time_step, axis=0).flatten()
    horizontal_indexing = np.repeat(horizontal_indexing_inner, num_in_time, axis=0).flatten() + horizontal_indexing_increment
    vertical_indexing = half_mask_offset + row_width * np.arange(mask_width)[:, None]
    label_indexer = horizontal_indexing + vertical_indexing

    y = padded.flatten()[label_indexer].astype(np.int32)

    X = X[:, 0:-elements_per_time_step]
    y = y[: ,elements_per_time_step:]

    return X, y, elements_per_time_step

  def add_beat_location(arr_df, repeat_amount=2):
      '''
      Add four rows to a dataframe where each row is either the played or the articulation 
      binary of a note and the columns are the timestepts
      Inputs:
          - arr_df: Dataframe. Shape of (2*n_notes, timesteps*n_notes)
          - repeat_amount: TODO:Explain
      '''
      arr_df["ind"] = range(len(arr_df))

      arr_df["ind"] = [i for i in range(len(arr_df)//repeat_amount + 1) for _ in range(repeat_amount)][:len(arr_df)]
      
      arr_df["id"] = arr_df["ind"].apply(lambda x: x%16)

      arr_df[['id_0','id_1', 'id_2', 'id_3']] = arr_df['id'].apply(lambda x: pd.Series(list(bin(x)[2:].zfill(4))))
      arr_df = arr_df.drop(["ind", "id"], axis=1)
      for col in ["id_0", "id_1", "id_2", "id_3"]:
          arr_df[col] = arr_df[col].astype(int)
      return arr_df

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

  def prepare(song_arr, vicinity=12):
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
          - song_arr: One-hot array of the notes and articulations of each timestep. 
                      Shape of (2*n_notes, timesteps)
          - vicinity: Integer. Number of notes considered around specified note.
      Outputs:
          - X: Lraining data. Shape of (2*n_notes, timesteps*n_notes)
          - y: Labels. For each note, this is a list of [played, articulated]
          - elements_per_time_step:
      '''
      n_notes, timesteps = song_arr.shape

      # window across each note vicinity
      X, y, elements_per_time_step = windowed_data_across_notes_time(song_arr, mask_length_x=vicinity)

      # Get array of Midi values for each note value
      midi_row = add_midi_value(X, n_notes)

      # Get array of one hot encoded pitch values for each note value
      pitchclass_rows = calculate_pitchclass(midi_row, X)

      # Add total_pitch count repeated for each note window
      previous_context = build_context(X, midi_row, pitchclass_rows)

      X = np.vstack((midi_row, pitchclass_rows, X, previous_context))

      # Add beats repeated for each note window
      X = add_beat_location(pd.DataFrame(X.T), repeat_amount=elements_per_time_step).T

      # Add 
      return X, pd.DataFrame(y), elements_per_time_step

if __name__ == "__main__":

    dsp = RNNMusicDataSetPreparer()
    vicinity = 50

    X_prepared, y_prepared, elements_per_time_step = dsp.prepare_notes(in_data, vicinity=vicinity)

    print(X_prepared.head())
