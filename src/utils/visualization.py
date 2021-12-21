import matplotlib.pyplot as plt
from src.utils.midi_support import MidiSupport
from IPython import display

def note_and_artic_to_one(data, what="artic"):
    with_volume = 80 * data.T.values
    if what == "artic":
        ret = with_volume[range(1, 257, 2), :]
    elif what == "note_hold":
        ret = with_volume[range(0, 256, 2), :]
    elif what == "both":
        ret = with_volume
    else:
        raise KeyError("what needs to be artic or note_hold or both")
    return ret

def plot_piano_roll(note_df, file_path, plot_type="both"):
    data = note_and_artic_to_one(note_df.T.values, what=plot_type)
    fig=plt.figure()
    plt.rcParams["figure.figsize"] = (40,10)
    plt.imshow(data, cmap='hot', interpolation='nearest', aspect="auto")
    plt.close(fig)
    plt.savefig(file_path)

def save_audio_file(predicted, filepath, audio_type="artic"):
    if audio_type not in ["artic", "note_hold"]:
        raise KeyError("what needs to be artic or note_hold")
    data = note_and_artic_to_one(predicted.T.values, what=audio_type)
    new_midi = MidiSupport().piano_roll_to_pretty_midi(data, fs=5)
    _SAMPLING_RATE = 16000
    seconds = 30
    waveform = new_midi.fluidsynth(fs=_SAMPLING_RATE)
    # Take a sample of the generated waveform to mitigate kernel resets
    waveform_short = waveform[:seconds*_SAMPLING_RATE]
    audtst = display.Audio(waveform_short, rate=_SAMPLING_RATE)
    with open(filepath, "wb") as f:
        f.write(audtst.data)
    
def plot_histories(histories, labels, parameter='loss'):
    
    for i, hist in enumerate(histories):
        plt.plot(hist.history[parameter], label=labels[i])
    plt.title(parameter.capitalize())
    plt.ylabel(parameter)
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()