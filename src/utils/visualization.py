import matplotlib.pyplot as plt


def plot_piano_roll(note_df, file_path):

    plt.rcParams["figure.figsize"] = (40,10)
    plt.imshow(note_df.T.values, cmap='hot', interpolation='nearest', aspect="auto")
    plt.savefig(file_path)
    
def plot_parameter_training(histories, labels, parameter='loss'):
    for i, hist in enumerate(histories):
        plt.plot(hist.history[parameter], label=labels[i])
    plt.title(parameter.capitalize())
    plt.ylabel(parameter)
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()