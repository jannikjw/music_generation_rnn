"""Module for support with midi files

"""
import pandas as pd
import numpy as np
import pygame
import pretty_midi


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


if __name__ == "__main__":


    ms = MidiSupport()

    mf = ms.load_midi_file("music_generation_rnn/training_data/Alan_Walker_Faded.midi")

    prepared = ms.midi_to_16_beats_processed(mf)

    print(prepared.T.iloc[2*50:2*50+40, 0:16])

    