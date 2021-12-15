import pandas as pd
import numpy as np
import tensorflow as tf
from src.utils.midi_support import RNNMusicDataSetPreparer, load_music

def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)


def get_fat_diagonal_mat(mat_len, one_dis):
    ones = np.ones((mat_len, mat_len), dtype=np.float32)
    for i in range(mat_len):
        for j in range(mat_len):
            if abs(i-j) > one_dis:
                ones[i, j] = 0
    return tf.convert_to_tensor(ones)

# Kernel Regularizers for limited connectivity
@tf.keras.utils.register_keras_serializable(package='Custom', name='kernel_regularizer_r')
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

@tf.keras.utils.register_keras_serializable(package='Custom', name='recurrent_kernel_regularizer_b')
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

    model.compile(loss=loss, optimizer=optimizer)

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

   
# predicted2, pred_probs = predict_notes(100)

# predicted2, pred_probs = predict_notes(100)

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


class RNNMusicExperiment():

    def __init__(self, sequence_length=15, epochs=10, learning_rate=0.001, batch_size=64, num_music_files=2) -> None:
        self.common_config = {
            "seq_length": sequence_length,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_music_files": num_music_files,
        }
        return

    def run(self):

        loaded_data = self.load_data()
        prepared_data = self.prepare_data(loaded_data)
        model, history = self.train_model(prepared_data)
        # Save training stats?
        # Pickle the model?

        print("Trying to predict some data")
        print(self.predict_data(model, loaded_data))
        # Save music file?
        # Save music output plot?

    def basic_load_data(self):
        loaded = load_music(
            num_files=self.common_config["num_music_files"],
            seq_length=self.common_config["seq_length"]
            )
        return loaded

    def load_data(self):
        raise NotImplementedError

    def prepare_data(self):
        # TODO: Can the be in the form seq_ds or (X_train, y_train)
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError

    def train_model(self, prepared_data):

        
        model, callbacks = self.get_model()
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


class RNNMusicExperimentOne(RNNMusicExperiment):
    
    def load_data(self):
        return self.basic_load_data()

    def prepare_data(self, loaded_data):
        seq_ds = RNNMusicDataSetPreparer().prepare(loaded_data)
        # TODO: Some models return a DataSet and some return X_train, y_train
        return seq_ds

    def get_model(self):
        print(f"in get_model self is {self}")
        model, callbacks = model_5_lstm_layer_with_artic(
            learning_rate=self.common_config["learning_rate"],
            seq_length=self.common_config["seq_length"]
        )
        return model, callbacks
        
    def predict_data(self, model, loaded_data):
        return predict_notes_256_sigmoid(model=model, train_data=loaded_data)


class RNNMusicExperimentTwo(RNNMusicExperiment):
    """Builds on Exp 1 by adding limited connectivity

    Args:
        RNNMusicExperiment ([type]): [description]
    """
    
    def load_data(self):
        return self.basic_load_data()

    def prepare_data(self, loaded_data):
        seq_ds = RNNMusicDataSetPreparer().prepare(loaded_data)
        # TODO: Some models return a DataSet and some return X_train, y_train
        return seq_ds

    def get_model(self):
        print(f"in get_model self is {self}")
        model, callbacks = model_4_lstm_layer_limited_connectivity(
            learning_rate=self.common_config["learning_rate"],
            seq_length=self.common_config["seq_length"]
        )
        return model, callbacks
        
    def predict_data(self, model, loaded_data):
        return predict_notes_256_sigmoid(model=model, train_data=loaded_data)



# Main training loop
if __name__ == "__main__":
    
    learning_rate = 0.001
    training_epoch = 100
    for sequence_length in [15]:#, 32]:
        print("Trying Exp 1")
        exp = RNNMusicExperimentOne(
            sequence_length=sequence_length,
            learning_rate=learning_rate,
            epochs=training_epoch)
        exp.run()

        
        print("Trying Exp 2")
        exp = RNNMusicExperimentTwo(
            sequence_length=sequence_length,
            learning_rate=learning_rate,
            epochs=training_epoch)
        exp.run()

