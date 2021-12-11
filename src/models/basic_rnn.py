import pandas as pd
import numpy as np
import tensorflow as tf

from utils.midi_support import RNNMusicDataSetPreparer

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



class RNNMusicPredictor():

    def __init__(self, seq_length=15, learning_rate=0.0005) -> None:
        self.seq_length = seq_length
        self.learning_rate = learning_rate

    def setup_model(self):

        seq_length = 15
        input_shape = (seq_length, 132)
        learning_rate = 0.0005

        inputs = tf.keras.Input(input_shape)
        x = tf.keras.layers.LSTM(128, kernel_regularizer=kernel_regularizer_a, return_sequences=True)(inputs)
        y = tf.keras.layers.LSTM(128, kernel_regularizer=recurrent_kernel_regularizer_a, return_sequences=True)(x)
        z = tf.keras.layers.LSTM(128, recurrent_regularizer=tf.keras.regularizers.L2(1), return_sequences=True)(y)
        k = tf.keras.layers.LSTM(128, recurrent_regularizer=tf.keras.regularizers.L2(1))(z)

        outputs = {
        'pitch': tf.keras.layers.Dense(128, activation="relu", name='pitch')(k),
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


        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath='./training_checkpoints/ckpt_{epoch}',
                save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                verbose=1,
                restore_best_weights=True)
        ]

    def train(self, all_song_dfs):

        epochs = 5

        seq_ds = RNNMusicDataSetPreparer().prepare(all_song_dfs)

        batch_size = 64
        # seq_length, _ = song_df.shape
        # buffer_size = n_notes - seq_length  # the number of items in the dataset
        buffer_size = 100
        train_ds = (seq_ds
                    .shuffle(buffer_size)
                    .batch(batch_size, drop_remainder=True)
                    .cache()
                    .prefetch(tf.data.experimental.AUTOTUNE))

        history = self.model.fit(
            train_ds,
            epochs=epochs,
            callbacks=self.callbacks,
        )

    def predict_notes(all_song_dfs, seq_length, size=10):
        outputs = []
        all_probs = []
        offset = np.random.choice(range(len(all_song_dfs)-seq_length))
        # offset= 1233
        input_notes = all_song_dfs.iloc[offset: offset+seq_length]
        input_notes_reshape = input_notes.values.reshape((1, seq_length, 132))
        last_beats =[str(x) for x in input_notes_reshape[:, -1, -4:][0]]
        last_beats_int = int("".join(last_beats), 2)
        print(input_notes_reshape.shape)
        for l in range(size):
            probs = self.model.predict(input_notes_reshape)["pitch"].flatten()
            probs = probs/probs.sum()
            selected = np.random.choice(range(len(probs)), p=probs, size=2)
            out = np.zeros_like(probs)
            out[selected] = 1
            # out[probs > 0.5 ] = 1
            outputs.append(out)
            all_probs.append(probs)
            
            last_beats_int += 1
            last_beats_int = last_beats_int%16
            next_beats_ind = [int(x) for x in bin(last_beats_int)[2:].zfill(4)]
            
            
            last_new_note = np.concatenate([out, next_beats_ind]).reshape((1,1,132))
            input_notes_reshape = np.concatenate([input_notes_reshape[:, 1:, :], last_new_note], axis=1)
            
        return pd.DataFrame(outputs), pd.DataFrame(all_probs)
# predicted2, pred_probs = predict_notes(100)

    def predict_notes(size=10, all_song_dfs, seq_length):
        outputs = []
        all_probs = []
        offset = np.random.choice(range(len(all_song_dfs)-seq_length))
        # offset= 1233
        input_notes = all_song_dfs.iloc[offset: offset+seq_length]
        input_notes_reshape = input_notes.values.reshape((1, seq_length, 132))
        last_beats =[str(x) for x in input_notes_reshape[:, -1, -4:][0]]
        last_beats_int = int("".join(last_beats), 2)
        print(input_notes_reshape.shape)
        for l in range(size):
            probs = self.model.predict(input_notes_reshape)["pitch"].flatten()
            probs = probs/probs.sum()
            selected = np.random.choice(range(len(probs)), p=probs, size=2)
            out = np.zeros_like(probs)
            out[selected] = 1
            # out[probs > 0.5 ] = 1
            outputs.append(out)
            all_probs.append(probs)
            
            last_beats_int += 1
            last_beats_int = last_beats_int%16
            next_beats_ind = [int(x) for x in bin(last_beats_int)[2:].zfill(4)]
            
            
            last_new_note = np.concatenate([out, next_beats_ind]).reshape((1,1,132))
            input_notes_reshape = np.concatenate([input_notes_reshape[:, 1:, :], last_new_note], axis=1)
            
        return pd.DataFrame(outputs), pd.DataFrame(all_probs)

# predicted2, pred_probs = predict_notes(100)


