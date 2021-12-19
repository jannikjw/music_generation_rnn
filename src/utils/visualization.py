import matplotlib.pyplot as plt


def plot_piano_roll(note_df, file_path):

    plt.rcParams["figure.figsize"] = (40,10)
    plt.imshow(note_df.T.values, cmap='hot', interpolation='nearest', aspect="auto")
    plt.save(file_path)