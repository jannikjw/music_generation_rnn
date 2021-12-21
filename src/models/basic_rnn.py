import pandas as pd
import numpy as np
import tensorflow as tf
from src.utils.midi_support import MidiSupport, RNNMusicDataSetPreparer, load_midi_objs, load_just_that_one_test_song, download_and_save_data
from src.utils.visualization import plot_piano_roll, save_audio_file
import pdb
from datetime import datetime

def get_fat_diagonal_mat(mat_len, one_dis):
    ones = np.ones((mat_len, mat_len), dtype=np.float32)
    for i in range(mat_len):
        for j in range(mat_len):
            if abs(i-j) > one_dis:
                ones[i, j] = 0
    return tf.convert_to_tensor(ones)

# Kernel Regularizers for limited connectivity
#@tf.keras.utils.register_keras_serializable(package='Custom', name='kernel_regularizer_r')
def kernel_regularizer_a(kernel_weights):
   len_0, len_1 = kernel_weights.shape
   len_2 = int(len_1/4)
   print(len_0, len_1)
   mask = tf.concat([get_fat_diagonal_mat(len_2, 12), tf.ones((len_2, 4))], axis=1)
   mask_stacked = 1 - tf.concat([mask, mask, mask, mask], axis=0)
   print(kernel_weights.shape)
   print(mask_stacked.shape)
   penality = tf.math.reduce_sum((tf.math.abs(kernel_weights) * tf.transpose(mask_stacked)))
   return 0.01 * penality

#@tf.keras.utils.register_keras_serializable(package='Custom', name='recurrent_kernel_regularizer_b')
def recurrent_kernel_regularizer_a(recurrent_kernel_weights):
   len_0, len_1 = recurrent_kernel_weights.shape
   len_2 = int(len_1/4)
   print(len_0, len_1)
   mask = get_fat_diagonal_mat(len_2, 12)
   mask_stacked = 1 - tf.concat([mask, mask, mask, mask], axis=0)
   penality = tf.math.reduce_sum((tf.math.abs(recurrent_kernel_weights) * tf.transpose(mask_stacked)))
   return 0.01 * penality

def model_4_lstm_layer_limited_connectivity(seq_length=15, learning_rate = 0.0005):
    """This model has:

    Model
    - 4 LSTM Layers
    - Model expects input shape to be 260 x seq length
        - 260 being 128 notes + 128 articulations + 4 beats
    - Model has a kernel regularizer to encourage limited connectivity

    Loss:
    - This model uses CategoricalCrossentropy

    Optimizer
    - This model uses the ADAM optimizer

    Args:
        seq_length (int, optional): [description]. Defaults to 15.
    """
    
    input_shape = (seq_length, 260)
    

    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.LSTM(256, kernel_regularizer=kernel_regularizer_a, return_sequences=True)(inputs)
    y = tf.keras.layers.LSTM(256, kernel_regularizer=recurrent_kernel_regularizer_a, return_sequences=True)(x)
    z = tf.keras.layers.LSTM(256, recurrent_regularizer=tf.keras.regularizers.L2(1), return_sequences=True)(y)
    k = tf.keras.layers.LSTM(256, recurrent_regularizer=tf.keras.regularizers.L2(1))(z)

    outputs = {
    'pitch': tf.keras.layers.Dense(256, activation="relu", name='pitch')(k),
    }

    model = tf.keras.Model(inputs, outputs)

    loss = {
        'pitch': tf.keras.losses.CategoricalCrossentropy(
            from_logits=True)
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss=loss,
        loss_weights={
            'pitch': 1,
        },
        optimizer=optimizer,
    )
    # model.evaluate(train_ds, return_dict=True)


    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./training_checkpoints/ckpt_{epoch}',
            save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            verbose=1,
            restore_best_weights=True)
    ]
    return model,  callbacks

def model_5_lstm_layer_with_artic(seq_length=15, learning_rate = 0.0005):
    """This model has:

    Model
    - 4 LSTM Layers
    - Model expects input shape to be 260 x seq length
        - 260 being 128 notes + 128 articulations + 4 beats

    Loss:
    - This model uses CategoricalCrossentropy

    Optimizer
    - This model uses the ADAM optimizer

    Args:
        seq_length (int, optional): [description]. Defaults to 15.
    """
    
    input_shape = (seq_length, 260)
    
    print("in get model")
    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.LSTM(256, return_sequences=True)(inputs)
    y = tf.keras.layers.LSTM(256, return_sequences=True)(x)
    z = tf.keras.layers.LSTM(256, return_sequences=True)(y)
    k = tf.keras.layers.LSTM(256, return_sequences=False)(z)

    outputs = {
    'pitch': tf.keras.layers.Dense(256, activation="relu", name='pitch')(k),
    }

    model = tf.keras.Model(inputs, outputs)

    loss = {
        'pitch': tf.keras.losses.CategoricalCrossentropy(
            from_logits=True)
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    #model.compile(loss=loss, optimizer=optimizer)

    model.summary()


    model.compile(
        loss=loss,
        loss_weights={
            'pitch': 1,
        },
        optimizer=optimizer,
    )
    # model.evaluate(train_ds, return_dict=True)


    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./training_checkpoints/ckpt_{epoch}',
            save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            verbose=1,
            restore_best_weights=True)
    ]
    return model,  callbacks

def model_6_note_invariant(learning_rate=0.01, num_hidden_nodes=50, total_vicinity=28):

    elements_per_time_step = 128
    input_shape = (elements_per_time_step, total_vicinity)

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    model.add(tf.keras.layers.LSTM(num_hidden_nodes,  return_sequences=True))
    model.add(tf.keras.layers.LSTM(num_hidden_nodes,  return_sequences=True))
    model.add(tf.keras.layers.LSTM(num_hidden_nodes,  return_sequences=True))
    model.add(tf.keras.layers.Dense(2, activation="sigmoid"))

    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['mse']
    )
    return model, None

def model_7_google(learning_rate=0.005, seq_length=25):
    '''
    This function is copied from Tensorflow and is only used to replicate the music prediction model in the
    TensorFlow tutorial (https://www.tensorflow.org/tutorials/audio/music_generation).
    '''
    def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
        mse = (y_true - y_pred) ** 2
        positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
        return tf.reduce_mean(mse + positive_pressure)
    
    input_shape = (seq_length, 3)
    learning_rate = 0.005

    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.LSTM(128)(inputs)

    outputs = {
        'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
        'step': tf.keras.layers.Dense(1, name='step')(x),
        'duration': tf.keras.layers.Dense(1, name='duration')(x),
    }

    model = tf.keras.Model(inputs, outputs)

    model.summary()
    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
    loss=loss,
    loss_weights={
        'pitch': 0.05,
        'step': 1.0,
        'duration':1.0,
        },
        optimizer=optimizer,
    )
    
    return model, None

def predict_notes_256_sigmoid(model, train_data, size=10):
    """Predict notes

    Expects model predictioin to be sigmoid output 0 or 1
    Plays notes if probability is greater than 0.5

    Args:
        model (tf.keras.model): The model
            Expects the model to give output in dictionary that has "pitch"
        size (int): Number of notes to predict Defaults to 10.
        training data pd.DataFrame: Training data to pull random data from for start of prediction

    Returns:
        pd.DataFrame: The predicted notes as shape (128 x size)
    """
    outputs = []
    all_probs = []
    offset = np.random.choice(range(len(train_data)-size))
    # offset= 1233
    input_notes = train_data.iloc[offset: offset+size]
    input_notes_reshape = input_notes.values.reshape((1, size, 260))
    last_beats =[str(x) for x in input_notes_reshape[:, -1, -4:][0]]
    last_beats_int = int("".join(last_beats), 2)
    for l in range(size):
        probs = model.predict(input_notes_reshape)["pitch"].flatten()
        # probs = probs/probs.sum()
        # selected = np.random.choice(range(len(probs)), p=probs, size=2)
        out = np.zeros_like(probs)
        # out[selected] = 1
        out[probs > 0.5 ] = 1
        outputs.append(out)
        all_probs.append(probs)

        last_beats_int += 1
        last_beats_int = last_beats_int%16
        next_beats_ind = [int(x) for x in bin(last_beats_int)[2:].zfill(4)]

        last_new_note = np.concatenate([out, next_beats_ind]).reshape((1,1,260))
        input_notes_reshape = np.concatenate([input_notes_reshape[:, 1:, :], last_new_note], axis=1)

    return pd.DataFrame(outputs), pd.DataFrame(all_probs)

def predict_notes_note_invariant(model, reshaped_train_data, size=10):
    """Predict notes

    Expects model predictioin to be sigmoid output 0 or 1
    Expects model that predict sequence of 128 at a time which amount to "the next note"
    Plays notes if probability is greater than 0.5

    Args:
        reshaped_train_data: Needs to already be X_prepared unreavelled form
        model (tf.keras.model): The model
            Expects the model to give output in dictionary that has "pitch"
        size (int): Number of notes to predict Defaults to 10.
        training data pd.DataFrame: Training data to pull random data from for start of prediction

    Returns:
        pd.DataFrame: The predicted notes as shape (128 x size)
    """
    outputs = []
    all_probs = []
    num_notes = 128
    num_beats = 1
    elements_per_time_step = 128
    offset = np.random.choice(range(len(reshaped_train_data)))
    input_notes = reshaped_train_data[offset:offset+num_beats, :, :]
    # input_notes_reshape = input_notes.reshape(1, num_notes, total_vicinity)
    input_notes_reshape = input_notes
    last_beats =[str(x) for x in input_notes_reshape[0, -1, -4:]]
    last_beats_int = int("".join(last_beats), 2)
    for l in range(size):

        # probs = model.predict(input_notes_reshape)["pitch"].flatten()
        #*******
        probs = model.predict(input_notes_reshape)
        # probs shape should be something like (256, beats)
        probs = pd.DataFrame(probs.reshape(num_beats, elements_per_time_step*2, order="C").T) 


        # probs = probs[:, -1]
        # output sequence will be the same length as the input. Try to either take the first or the last beat

      
        play_bias = 0.01
        probs = probs + play_bias
        probs[probs > 1] = 1
        probs[probs < 0] = 0
        out = np.zeros_like(probs)
        # out[tst_output_reshaped > 0.5 ] = 1

        # This part fails when we have multiple output.
        out = np.random.binomial(1, probs, size=None).reshape(num_notes*2)

        # Need to window across out to get the input form again.
        out = out.reshape((256,1))
        probs = probs.values.reshape((256,))

        outputs.append(out.reshape(256,))
        all_probs.append(probs)

        next_pred, _, _ = MidiSupport().windowed_data_across_notes_time(out, mask_length_x=24, return_labels=False)# Return (24, 128)

        #*******

        # pdb.set_trace()
        last_beats_int += 1
        last_beats_int = last_beats_int%16
        next_beats_ind = np.array([int(x) for x in bin(last_beats_int)[2:].zfill(4)])
        next_beats_ind = next_beats_ind.reshape((4, 1))
        next_beats_ind = np.repeat(next_beats_ind, num_notes, axis=1)



        # TODO, check if beat is correctly increasing: might need to flip it before adding
        last_new_note = np.concatenate([next_pred, next_beats_ind])
        last_new_note = last_new_note[np.newaxis, :, :] # Shape now  (1, 28, 128)
        last_new_note = np.swapaxes(last_new_note, 1, 2) # Shape now  (1, 128, 28)
        input_notes_reshape = last_new_note

    outputs_joined = pd.DataFrame(outputs)
    all_probs_joined = pd.DataFrame(all_probs)
    return outputs_joined, all_probs_joined

def predict_notes_note_invariant_plus_extras(model, reshaped_train_data, size=10):
    """Predict notes

    Expects model predictioin to be sigmoid output 0 or 1
    Expects model that predict sequence of 128 at a time which amount to "the next note"
    Plays notes if probability is greater than 0.5

    Args:
        reshaped_train_data: Needs to already be X_prepared unreavelled form
        model (tf.keras.model): The model
            Expects the model to give output in dictionary that has "pitch"
        size (int): Number of notes to predict Defaults to 10.
        training data pd.DataFrame: Training data to pull random data from for start of prediction

    Returns:
        pd.DataFrame: The predicted notes as shape (128 x size)
    """
    outputs = []
    all_probs = []
    num_notes = 128
    elements_per_time_step = 128
    num_beats = 1
    offset = np.random.choice(range(len(reshaped_train_data)))
    input_notes = reshaped_train_data[offset:offset+num_beats, :, :]
    # input_notes_reshape = input_notes.reshape(1, num_notes, total_vicinity)
    input_notes_reshape = input_notes
    last_beats =[str(x) for x in input_notes_reshape[0, -1, -4:]]
    last_beats_int = int("".join(last_beats), 2)

    last_beats_int = 0

    for l in range(size):

        # probs = model.predict(input_notes_reshape)["pitch"].flatten()
        #*******
        # model.reset_states()
        probs = model.predict(input_notes_reshape)

        # probs shape should be something like (256, beats)
        probs = pd.DataFrame(probs.reshape(num_beats, elements_per_time_step*2, order="C").T) 

        play_bias = 0
        probs = probs + play_bias
        probs[probs > 1] = 1
        probs[probs < 0] = 0
        out = np.zeros_like(probs)
        # out[tst_output_reshaped > 0.5 ] = 1

        # This part fails when we have multiple output.
        out = np.random.binomial(1, probs, size=None).reshape(num_notes*2)

        # Need to window across out to get the input form again.
        out = out.reshape((256,1))
        probs = probs.values.reshape((256,))

        outputs.append(out.reshape(256,))
        all_probs.append(probs)

        # this_vicin = total_vicinity-4-12-12-1
        this_vicin = 24
        next_pred, _, _ = MidiSupport().windowed_data_across_notes_time(out, mask_length_x=this_vicin, return_labels=False)# Return (total_vicinity, 128)

        # Get array of Midi values for each note value
        n_notes = 128
        midi_row = MidiSupport().add_midi_value(next_pred, n_notes)

        # Get array of one hot encoded pitch values for each note value
        pitchclass_rows = MidiSupport().calculate_pitchclass(midi_row, next_pred)

        # Add total_pitch count repeated for each note window
        previous_context = MidiSupport().build_context(next_pred, midi_row, pitchclass_rows)

        midi_row = midi_row.reshape((1, -1))
        next_pred = np.vstack((next_pred, midi_row, pitchclass_rows, previous_context))
        
        last_beats_int += 1
        last_beats_int = last_beats_int%16
        next_beats_ind = np.array([int(x) for x in bin(last_beats_int)[2:].zfill(4)])
        next_beats_ind = next_beats_ind.reshape((4, 1))
        next_beats_ind = np.repeat(next_beats_ind, num_notes, axis=1)

        last_new_note = np.concatenate([next_pred, next_beats_ind])
        last_new_note = last_new_note[np.newaxis, :, :] # Shape now  (1, 28, 128)
        last_new_note = np.swapaxes(last_new_note, 1, 2) # Shape now  (1, 128, 28)
        input_notes_reshape = last_new_note

        

    outputs_joined = pd.DataFrame(outputs)
    all_probs_joined = pd.DataFrame(all_probs)
    return outputs_joined, all_probs_joined

def predict_notes_note_invariant_plus_extras_multiple_time_steps(model, reshaped_train_data, num_beats=15, size=10, note_vicinity=24):
    """Predict notes
    Same as before but uses sequence_length number of beats to predict output
    """
    outputs = []
    all_probs = []
    num_notes = 128
    total_vicinity = 53
    offset = np.random.choice(range(len(reshaped_train_data)))
    input_notes = reshaped_train_data[offset:offset+num_beats, :, :]
    input_notes_reshape = input_notes
    last_beats =[str(x) for x in input_notes_reshape[0, -1, -4:]]
    last_beats_int = int("".join(last_beats), 2)

    for l in range(size):

        probs = model.predict(input_notes_reshape)
        # probs shape should be something like (256, beats)
        probs = probs.reshape(num_beats, num_notes*2, order="C").T

        probs = probs[:, -1:]
        # output sequence will be the same length as the input. Try to either take the first or the last beat

        play_bias = 0
        probs = probs + play_bias
        probs[probs > 1] = 1
        probs[probs < 0] = 0
        out = np.zeros_like(probs)

        out = np.random.binomial(1, probs, size=None).reshape(num_notes*2)

        # Need to window across out to get the input form again.
        out = out.reshape((256,1))
        probs = probs.reshape((256,))
        outputs.append(out.reshape(256,))
        all_probs.append(probs)

        # note_vicinity = total_vicinity-4-12-12-1
        next_pred, _, _ = MidiSupport().windowed_data_across_notes_time(out, mask_length_x=note_vicinity, return_labels=False)# Return (total_vicinity, 128)

        # Get array of Midi values for each note value
        n_notes = 128
        midi_row = MidiSupport().add_midi_value(next_pred, n_notes)

        # Get array of one hot encoded pitch values for each note value
        pitchclass_rows = MidiSupport().calculate_pitchclass(midi_row, next_pred)

        # Add total_pitch count repeated for each note window
        previous_context = MidiSupport().build_context(next_pred, midi_row, pitchclass_rows)

        midi_row = midi_row.reshape((1, -1))
        next_pred = np.vstack((next_pred, midi_row, pitchclass_rows, previous_context))
        
        last_beats_int += 1
        last_beats_int = last_beats_int%16
        next_beats_ind = np.array([int(x) for x in bin(last_beats_int)[2:].zfill(4)])
        next_beats_ind = next_beats_ind.reshape((4, 1))
        next_beats_ind = np.repeat(next_beats_ind, num_notes, axis=1)

        # TODO, check if beat is correctly increasing: might need to flip it before adding
        last_new_note = np.concatenate([next_pred, next_beats_ind])
        last_new_note = last_new_note[np.newaxis, :, :] # Shape now  (1, 28, 128)
        last_new_note = np.swapaxes(last_new_note, 1, 2) # Shape now  (1, 128, 28)
        
        together = np.concatenate([input_notes_reshape[1:, :, :], last_new_note], axis=0)
        input_notes_reshape = together

    outputs_joined = pd.DataFrame(outputs)
    all_probs_joined = pd.DataFrame(all_probs)
    return outputs_joined, all_probs_joined


class RNNMusicExperiment():
    '''
    Super class for all experiments
    '''

    def __init__(self, sequence_length=15, epochs=10, learning_rate=0.001, batch_size=64, num_music_files=2, vocab_size=128) -> None:
        self.common_config = {
            "seq_length": sequence_length,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_music_files": num_music_files,
            "vocab_size": vocab_size,
        }
        return
    
    def get_model(self):
        return self.model
        
    def get_history(self):
        return self.history
    
    def get_name(self):
        raise NotImplementedError
        
    def set_model(self):
        self.model, self.callbacks = model_5_lstm_layer_with_artic(
            learning_rate=self.common_config["learning_rate"],
            seq_length=self.common_config["seq_length"]
        )

    def run(self):
        self.set_model()
        loaded_data = self.load_data()
        prepared_data = self.prepare_data(loaded_data)
        print("Training...")
        self.model, self.history = self.train_model(prepared_data)
        # Save training stats?
        # Pickle the model?

        print("Predicting data...")
        predicted, probs = self.predict_data(self.model, loaded_data)
        
        print("Saving data...")
        self.plot_and_save_predicted_data(predicted)
        self.plot_and_save_predicted_data(probs)
        self.create_and_save_predicted_audio(predicted, "_music_")
        # Save music file?
        # Save music output plot?

    def basic_load_data(self):
        loaded = load_midi_objs(
            num_files=self.common_config["num_music_files"],
            seq_length=self.common_config["seq_length"]
            )
        return loaded

    def load_data(self):
        raise NotImplementedError

    def prepare_data(self):
        # TODO: Can the be in the form seq_ds or (X_train, y_train)
        raise NotImplementedError

    def train_model(self, prepared_data):
        model, callbacks = self.model, self.callbacks
        print(f"type prepared_data is {type(prepared_data)}")
        # seq_length, _ = song_df.shape
        # buffer_size = n_notes - seq_length  # the number of items in the dataset
        buffer_size = 100
        train_ds = (prepared_data
                    .shuffle(buffer_size)
                    .batch(self.common_config["batch_size"], drop_remainder=True)
                    .cache()
                    .prefetch(tf.data.experimental.AUTOTUNE))

        history = model.fit(
            train_ds,
            epochs=self.common_config["epochs"],
            callbacks=callbacks,
        )
        return model, history

    def predict_data(self):
        raise NotImplementedError

    def get_save_plot_path(self, str_ind=""):
        out = "plots/"
        out += self.get_name()
        out += str_ind
        out += "_".join([str(x).replace(".", "dot") for x in self.common_config.values()])
        out += "__"
        out += datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        out += ".png"
        return out

    def get_save_audio_path(self, str_ind=""):
        out = "audio/"
        out += self.get_name()
        out += str_ind
        out += "_".join([str(x).replace(".", "dot") for x in self.common_config.values()])
        out += datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        out += ".wav"
        return out

    def plot_and_save_predicted_data(self, predicted, str_ind=""):
        if predicted is not None:
            plot_piano_roll(predicted, self.get_save_plot_path("_both_" + str_ind), plot_type="both")
            plot_piano_roll(predicted, self.get_save_plot_path("_artic_" + str_ind), plot_type="artic")
            plot_piano_roll(predicted, self.get_save_plot_path("_note_hold_" + str_ind), plot_type="note_hold")

    def create_and_save_predicted_audio(self, predicted, str_ind=""):
        save_audio_file(predicted, self.get_save_audio_path("_artic_" + str_ind), audio_type="artic")
        save_audio_file(predicted, self.get_save_audio_path("_note_hold_" + str_ind), audio_type="note_hold")


class RNNMusicExperimentOne(RNNMusicExperiment):

    def get_name(self):
        return "Exp1"
    
    def load_data(self):
        return self.basic_load_data()

    def prepare_data(self, midi_objs):
        play_articulated = MidiSupport().all_midi_obj_to_play_articulate(midi_objs)
        seq_ds = RNNMusicDataSetPreparer().prepare(play_articulated)
        # TODO: Some models return a DataSet and some return X_train, y_train
        return seq_ds
        
    def predict_data(self, model, loaded_data):
        return predict_notes_256_sigmoid(model=model, train_data=loaded_data, size=self.common_config["seq_length"])


class RNNMusicExperimentTwo(RNNMusicExperiment):
    """Builds on Exp 1 by adding limited connectivity

    Args:
        RNNMusicExperiment ([type]): [description]
    """
    
    def get_name(self):
        return "Exp2"
              
    def set_model(self):
        self.model, self.callbacks = model_4_lstm_layer_limited_connectivity(
            learning_rate=self.common_config["learning_rate"],
            seq_length=self.common_config["seq_length"]
        )

    def load_data(self):
        return self.basic_load_data()

    def prepare_data(self, loaded_data):
        seq_ds = RNNMusicDataSetPreparer().prepare(loaded_data)
        # TODO: Some models return a DataSet and some return X_train, y_train
        return seq_ds
        
    def predict_data(self, model, loaded_data):
        return predict_notes_256_sigmoid(model=model, train_data=loaded_data, size=self.common_config["seq_length"])


class RNNMusicExperimentThree(RNNMusicExperiment):
    """Note invariance with articularion

    Args:
        RNNMusicExperiment ([type]): [description]
    """ 
    def run(self):
        self.set_model()
        loaded_data = self.load_data()
        prepared_data = self.prepare_data(loaded_data)
        print("Training...")
        self.model, self.history = self.train_model(prepared_data)

        print("Predicting data...")
        predicted, probs = self.predict_data(self.model, prepared_data)
        
        print("Saving data...")
        self.plot_and_save_predicted_data(predicted, "_predicted_")
        self.plot_and_save_predicted_data(probs, "_probs_")
    
    def get_name(self):
        return "Exp3"
    
    def set_model(self):
        self.model, self.callbacks = model_6_note_invariant(
            learning_rate=self.common_config["learning_rate"]
        )

    def train_model(self, prepared_data):
        """Train model overwrite

        Args:
            prepared_data (tuple): X, y

        """
        model, callbacks = self.model, self.callbacks
        history = model.fit(prepared_data[0],
            prepared_data[1],
            epochs=self.common_config["epochs"],
            callbacks=callbacks,
            batch_size=1,
        )
        return model, history
    
    def load_data(self):
        return self.basic_load_data()

    def prepare_data(self, loaded_data):
        #seq_ds is in form X, y here
        seq_ds = MidiSupport().prepare_song_note_invariant_plus_beats(loaded_data)
        # TODO: Some models return a DataSet and some return X_train, y_train
        return seq_ds
        
    def predict_data(self, model, prepared_data):
        return predict_notes_note_invariant(model, prepared_data[0])


class RNNMusicExperimentFour(RNNMusicExperiment):
    """Note invariance with articularion and beats and extras


    Args:
        RNNMusicExperiment ([type]): [description]
    """

    def __init__(self, *args, num_beats_for_prediction=1, note_vicinity=24, **kwargs):
        super().__init__(*args, **kwargs)
        self.common_config["num_beats_for_prediction"] = num_beats_for_prediction
        self.common_config["note_vicinity"] = note_vicinity

    def get_name(self):
        return "Exp4"
    
    def set_model(self):
        self.model, self.callbacks = model_6_note_invariant(
            learning_rate=self.common_config["learning_rate"],
            # Total vicinity 24 notes + 4 beats + 1 midi + 12 context + 12 pitchclass 
            total_vicinity=self.common_config["note_vicinity"]+4+1+12+12,
        )

    def run(self):
        self.set_model()
        loaded_midi = self.load_data()
        self.prepared_data = self.prepare_data(loaded_midi)
        print("Training...")
        self.model, self.history = self.train_model(self.prepared_data)
        self.predict_and_save_data()

    def predict_and_save_data(self, str_id=""):

        print("Predicting data...")
        predicted, probs = self.predict_data(self.model, self.prepared_data)
        print("Saving data...")
        self.plot_and_save_predicted_data(predicted, str_id + "_predicted_")
        self.plot_and_save_predicted_data(probs, str_id + "_probs_")
        self.create_and_save_predicted_audio(predicted, str_id + "_music_")
        # Save music file?
        # Save music output plot?

    def train_model(self, prepared_data):
        model, callbacks = self.model, self.callbacks
        history = model.fit(prepared_data[0],
            prepared_data[1],
            epochs=self.common_config["epochs"],
            callbacks=callbacks,
            batch_size=self.common_config["batch_size"],
        )
        return model, history
    
    def load_data(self):
        return self.basic_load_data()

    def prepare_data(self, loaded_data):
        #seq_ds is in form X, y here
        seq_ds = MidiSupport().prepare_song_note_invariant_plus_beats_and_more(loaded_data)
        # TODO: Some models return a DataSet and some return X_train, y_train
        return seq_ds
        
    def predict_data(self, model, prepared_data):
        return predict_notes_note_invariant_plus_extras_multiple_time_steps(
            model,
            prepared_data[0],
            size=200,
            num_beats=self.common_config["num_beats_for_prediction"])


class RNNMusicExperimentFive(RNNMusicExperimentFour):
    """Note invariance with articularion and beats and extras


    Same as experiment Four but just run on a single song

    Args:
        RNNMusicExperiment ([type]): [description]
    """

    def get_name(self):
        return "Exp5"

    def basic_load_data(self):
        loaded = load_just_that_one_test_song(
            num_files=self.common_config["num_music_files"],
            seq_length=self.common_config["seq_length"]
            )
        return loaded


class RNNMusicExperimentSeven(RNNMusicExperimentFive):
    """Study on number of notes to have in the vicinity

    Args:
        RNNMusicExperimentFour ([type]): [description]
    """

    def get_name(self):
        return "Exp7"

    def prepare_data(self, loaded_data):
        #seq_ds is in form X, y here
        seq_ds = MidiSupport().prepare_song_note_invariant_plus_beats_and_more(loaded_data, vicinity=self.common_config["note_vicinity"])
        # TODO: Some models return a DataSet and some return X_train, y_train
        return seq_ds

    def predict_data(self, model, prepared_data):
        return predict_notes_note_invariant_plus_extras_multiple_time_steps(
            model,
            prepared_data[0],
            size=200,
            note_vicinity=self.common_config["note_vicinity"],
            num_beats=self.common_config["num_beats_for_prediction"])


class RNNMusicExperimentEight(RNNMusicExperimentSeven):
    """Study on increasing model capacitty

    Args:
        RNNMusicExperimentFour ([type]): [description]
    """

    def __init__(self, *args, num_hidden_nodes=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.common_config["num_hidden_nodes"] = num_hidden_nodes

    def get_name(self):
        return "Exp8"

    def set_model(self):
        self.model, self.callbacks = model_6_note_invariant(
            learning_rate=self.common_config["learning_rate"],
            num_hidden_nodes=self.common_config["num_hidden_nodes"],
            # Total vicinity 24 notes + 4 beats + 1 midi + 12 context + 12 pitchclass 
            total_vicinity=self.common_config["note_vicinity"]+4+1+12+12,
        )
    

class RNNMusicExperimentNine(RNNMusicExperimentEight):
    """Build on Experiment Eight but trys to train on many songs

    Args:
        RNNMusicExperimentEight ([type]): [description]
    """

    def basic_load_data(self):
        loaded = load_midi_objs(
            num_files=self.common_config["num_music_files"],
            seq_length=self.common_config["seq_length"]
            )
        return loaded


class RNNMusicExperimentTFRef(RNNMusicExperiment):
    """Google Tutorial Version
    RNNMusicExperiment ([type]): Implements the model described by Google here. Only used for performance 
    comparisons: https://www.tensorflow.org/tutorials/audio/music_generation
    """

    def get_name(self):
        return "ExpTFRef"
    
    def set_model(self):
        print(f"in get_model self is {self}")
        self.model, self.callbacks = model_7_google(
            learning_rate=self.common_config["learning_rate"],
            seq_length=self.common_config["seq_length"]
        )
        

    def run(self):
        self.key_order = ['pitch', 'step', 'duration']
        super().run()

    def load_data(self):
        return self.basic_load_data()

    def prepare_data(self, loaded_data):
        #seq_ds is in form X, y here
        seq_length = self.common_config["seq_length"]
        num_files = self.common_config["num_music_files"]
        batch_size = self.common_config["batch_size"]
        
        all_notes = []
        for f in loaded_data[:num_files]:
            notes = MidiSupport().midi_to_notes(f)
            all_notes.append(notes)

        all_notes = pd.concat(all_notes)
        self.all_notes = all_notes
        n_notes = len(all_notes)
        
        train_notes = np.stack([all_notes[key] for key in self.key_order], axis=1)
        
        notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
        
        seq_ds = RNNMusicDataSetPreparer().create_sequences(notes_ds, seq_length=seq_length, key_order=self.key_order)
        
        buffer_size = n_notes - seq_length  # the number of items in the dataset
        train_ds = (seq_ds
            .shuffle(buffer_size)
            .batch(batch_size, drop_remainder=True)
            .cache()
            .prefetch(tf.data.experimental.AUTOTUNE))
        
        return train_ds
    
    def train_model(self, prepared_data):
        """Train model overwrite

        Args:
            prepared_data (Tensorflow dataset): prepared_data

        """
        model, callbacks = self.model, self.callbacks
        history = model.fit(
            prepared_data,
            epochs=self.common_config["epochs"],
            callbacks=callbacks,
        )
        return model, history

    def predict_data(self, model, prepared_data):
        key_order = self.key_order
        all_notes = self.all_notes
        seq_length = self.common_config["seq_length"]
        vocab_size = self.common_config["vocab_size"]

        
        def predict_next_note(
            notes: np.ndarray, 
            keras_model: tf.keras.Model, 
            temperature: float = 1.0) -> int:
            """Generates a note IDs using a trained sequence model."""

            assert temperature > 0

            # Add batch dimension
            inputs = tf.expand_dims(notes, 0)

            predictions = model.predict(inputs)
            pitch_logits = predictions['pitch']
            step = predictions['step']
            duration = predictions['duration']

            pitch_logits /= temperature
            pitch = tf.random.categorical(pitch_logits, num_samples=1)
            pitch = tf.squeeze(pitch, axis=-1)
            duration = tf.squeeze(duration, axis=-1)
            step = tf.squeeze(step, axis=-1)

            # `step` and `duration` values should be non-negative
            step = tf.maximum(0, step)
            duration = tf.maximum(0, duration)

            return int(pitch), float(step), float(duration)
    
        temperature = 2.0
        num_predictions = 120

        sample_notes = np.stack([all_notes[key] for key in key_order], axis=1)

        # The initial sequence of notes; pitch is normalized similar to training
        # sequences
        input_notes = (
        sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

        generated_notes = []
        prev_start = 0
        for _ in range(num_predictions):
            pitch, step, duration = predict_next_note(input_notes, model, temperature)
            start = prev_start + step
            end = start + duration
            input_note = (pitch, step, duration)
            generated_notes.append((*input_note, start, end))
            input_notes = np.delete(input_notes, 0, axis=0)
            input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
            prev_start = start

        generated_notes = pd.DataFrame(generated_notes, columns=(*self.key_order, 'start', 'end'))
        return generated_notes, None


# Main training loop
if __name__ == "__main__":
    
    learning_rate = 0.001
    training_epoch = 100
    for sequence_length in [15]:#, 32]:
        # print("Trying Exp 1")
        # exp = RNNMusicExperimentOne(
        #     sequence_length=sequence_length,
        #     learning_rate=learning_rate,
        #     epochs=training_epoch)
        # exp.run()

        
        # print("Trying Exp 2")
        # exp = RNNMusicExperimentTwo(
        #     sequence_length=sequence_length,
        #     learning_rate=learning_rate,
        #     epochs=training_epoch)
        # exp.run()

        # print("Trying Exp 3")
        # exp = RNNMusicExperimentThreee(
        #     learning_rate=0.01,
        #     epochs=1,
        #     batch_size=1,
        #     num_music_files=1)
        # exp.run()

        # print("Trying Exp 6")
        # exp = RNNMusicExperimentTFRef(
        #     learning_rate=0.005,
        #     epochs=3,
        #     batch_size=64,
        #     num_music_files=5,
        #     sequence_length=25,
        # )
        # exp.run()
        
        print("Trying Exp 5")
        exp = RNNMusicExperimentFive(
            learning_rate=0.01,
            epochs=3,
            batch_size=1,
            num_music_files=1)
        exp.run()

        print("Exp 5 predicting with 31 beats")
        exp.common_config["num_beats_for_prediction"] = 31
        exp.predict_and_save_data()

        print("Exp 5 predicting with 15 beats")
        exp.common_config["num_beats_for_prediction"] = 15
        exp.predict_and_save_data()

        print("Exp 5 predicting with 3 beats")
        exp.common_config["num_beats_for_prediction"] = 3
        exp.predict_and_save_data()

        print("Trying Exp 4")
        exp = RNNMusicExperimentFour(
            learning_rate=0.01,
            epochs=3,
            batch_size=1,
            num_music_files=5,
        )
        exp.run()

        print("Exp 5 predicting with 31 beats")
        exp.common_config["num_beats_for_prediction"] = 31
        exp.predict_and_save_data()

        print("Exp 5 predicting with 15 beats")
        exp.common_config["num_beats_for_prediction"] = 15
        exp.predict_and_save_data()

        print("Exp 5 predicting with 3 beats")
        exp.common_config["num_beats_for_prediction"] = 3
        exp.predict_and_save_data()
        
        


