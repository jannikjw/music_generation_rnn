import matplotlib.pyplot as plt
from src.utils.midi_support import MidiSupport
from IPython import display

def plot_piano_roll(note_df, file_path):

    plt.rcParams["figure.figsize"] = (40,10)
    plt.imshow(note_df.T.values, cmap='hot', interpolation='nearest', aspect="auto")
    plt.savefig(file_path)


def save_audio_file(predicted, filepath):
    tst = 80 * predicted.T.values
    tst = tst[range(1, 257, 2), :]
    new_midi = MidiSupport().piano_roll_to_pretty_midi(tst, fs=5)
    _SAMPLING_RATE = 16000
    seconds = 30
    waveform = new_midi.fluidsynth(fs=_SAMPLING_RATE)
    # Take a sample of the generated waveform to mitigate kernel resets
    waveform_short = waveform[:seconds*_SAMPLING_RATE]
    audtst = display.Audio(waveform_short, rate=_SAMPLING_RATE)
    with open(filepath, "wb") as f:
        f.write(audtst.data)