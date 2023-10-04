import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split


def add_dense_layer(input_layer, node_count):
    layer = tf.keras.layers.Dense(node_count, activation='relu')(input_layer)
    output = tf.keras.layers.Dropout(0.2)(layer)
    return output


def load_dataframe(root, datapath):
    df = pd.read_csv(root+datapath+"data_frame.csv")

    data = df.loc[:, ["Ridge_Height", "Ridge_Width"]]
    labels = df.loc[:, ["True_cd", "True_ntg"]]
    return data, labels


# filepaths
root = "F:/College_UCC/AM6021- Dissertation/Depth Map Numpy Files/"  # location of all processed files
training_datapath = "Simulated data/2_15000_0.2_100_2.5/temp/"
version_diff = "unscaled"

shared_layer_counts = [64, 64]
cd_layer_counts = [32, 32, 32]
ntg_layer_counts = [32, 32, 32]
model_name = "_".join([str(i) for i in shared_layer_counts])+\
             "."+"_".join([str(i) for i in cd_layer_counts])+\
             "."+"_".join([str(i) for i in ntg_layer_counts])+\
             "."+version_diff


"""model architecture"""
start = tf.keras.layers.Input(shape=(2, 1))
x = tf.keras.layers.Dropout(0., name="Spacer_1")(start)

for val in shared_layer_counts:
    x = add_dense_layer(input_layer=x, node_count=val)
cd = tf.keras.layers.Dropout(0., name="Spacer_2")(x)

for val in cd_layer_counts:
    cd = add_dense_layer(input_layer=cd, node_count=val)
cd_output = tf.keras.layers.Dense(1, activation='linear', name="cd_output")(cd)

ntg = tf.keras.layers.Dropout(0., name="Spacer_3")(x)
for val in ntg_layer_counts:
    ntg = add_dense_layer(input_layer=ntg, node_count=val)
ntg_output = tf.keras.layers.Dense(1, activation='linear', name="ntg_output")(ntg)

model = tf.keras.models.Model(inputs=start, outputs=[cd_output, ntg_output], name=model_name)

optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.1)
model.compile(optimizer=optimizer, loss={'cd_output': 'mae', 'ntg_output': 'mae'})
model.summary()

# track the weights of the model
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=root + training_datapath + f'/{model.name}.weights.hdf5', monitor='loss', save_best_only=True)
board = tf.keras.callbacks.TensorBoard(log_dir=f'./tmp/{model.name}')

# training data
data, labels = load_dataframe(root=root, datapath=training_datapath)
print(data.shape, labels.shape)
train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.5, random_state=0)
# load saved weights, if they exist, so to continue training
if os.path.isfile(root + training_datapath + f'/{model.name}.weights.hdf5'):
    model.load_weights(root + training_datapath + f'/{model.name}.weights.hdf5')

model_history = model.fit(
    x=train_data,
    y=train_label,
    epochs=10000,
    validation_data=(test_data, test_label),
    callbacks=[checkpointer, board],
    verbose=1,
)

plt.title("Loss over time(epochs)")
plt.plot(model_history.history['loss'], label="loss")
plt.plot(model_history.history['val_loss'], label="val loss")
plt.legend()
plt.savefig(root + training_datapath + model.name + "training.png", bbox_inches='tight')
plt.show()
