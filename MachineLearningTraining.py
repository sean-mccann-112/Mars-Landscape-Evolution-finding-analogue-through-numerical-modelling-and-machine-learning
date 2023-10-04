import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf


def load_data(root, path, filename, samplereduction=1):
    # open the name_list.txt file, and turn it into list of filenames within the folder of the same path
    # take a certain amount of samples, according to sample_reduction:
    # bare in mind, there's already an inherent sample skip, usually 1 in 100
    with open(root + path + filename, 'r') as file:
        file_name_string = file.read()
    file_name_list = file_name_string.split(sep="\n")
    # print(file_name_list)
    file_name_list = list(
        filter(lambda string: int(string.replace(".npy", "").split("_")[3]) % samplereduction == 0, file_name_list)
    )
    return file_name_list


class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, path, filenamelist, shuffle=True, batch_size=10, imagesize=(200, 200), usable_indices=None, scale_labels=None):
        """
        Initializes a data generator object
        path: the File Path, as a string, that is the path to the common location
        filenamelist: a list of strings, where each one is a compressed file array
        shuffle: shuffle the data after each epoch
        batch_size: The size of each batch returned by __getitem__
        imagesize: size of the arrays being processed
        usable_indices: a way to specify what labels in the filenamelist are usable in the training,
        they would otherwise be used as testing data.
        """
        self.base_dir = path
        self.filenamelist = filenamelist
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.imagesize = imagesize
        self.usable_indices=usable_indices
        self.scale_labels =scale_labels
        self.on_epoch_end()

    def on_epoch_end(self):
        # total amount of datapoints allowed to be used = usable_indices,
        # unless no usable_indices was provided, then assume full dataset is used.
        if self.usable_indices is None:
            self.indices = np.arange(len(self.filenamelist))
        else:
            self.indices = self.usable_indices

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        # returns no. of batches for the data
        if self.usable_indices is None:
            return int(len(self.filenamelist) / self.batch_size)
        else:
            return int(len(self.usable_indices) / self.batch_size)

    def __getitem__(self, index):
        # Initialize batch
        # data = array of batch_size, imagesize[0], imagesize[1]
        X = np.empty((self.batch_size, *self.imagesize))
        # labels = (batch_size, cd, ntg, 1)
        y = np.empty((self.batch_size, 2))

        # get the indices of the requested batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # for each indices selected:
        for i, data_index in enumerate(indices):

            # load the selected numpy array
            imagePath = self.base_dir+self.filenamelist[data_index]
            image = np.load(file=imagePath)

            # extract the labels, ie first two values in the file name
            label_str = self.filenamelist[data_index].split(sep="_")
            label = list(map(int, label_str[0:2]))

            # this should scale the labels to between given values, for example: [15, 50]
            # as long as the given scalers are the max value found
            if self.scale_labels is not None:
                if len(self.scale_labels) == 2:
                    label[0] = label[0] / self.scale_labels[0]
                    label[1] = label[1] / self.scale_labels[1]

            # place the values/labels into their respective batch arrays
            X[i,] = image
            y[i] = label

        return X, y


def add_cnn_layer(input_layer, cnn_filter, cnn_kernal_size):
    layer = tf.keras.layers.Conv2D(cnn_filter, (cnn_kernal_size, cnn_kernal_size), activation='relu', padding='same')(input_layer)  # increase amount of layers
    output = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(layer)
    return output


def add_dense_layer(input_layer, node_count):
    layer = tf.keras.layers.Dense(node_count, activation='relu')(input_layer)
    output = tf.keras.layers.Dropout(0.2)(layer)
    return output


# remake in a way that automates model creation
def create_model(
        input_shape,
        split_point,
        cnn_layer_count,
        cnn_filter_count,
        cnn_kernal_size,
        dense_layer_count,
        dense_node_count,
        output_activation,
        version_diff: str = "",
):
    """
    :param version_diff: string to differentiate models of same shape, but different elsewhere
    :param output_activation: string determining the output layer(s) activation function.
    :param dense_node_count: number of nodes in dense layers, this works similar to cnn filter count
    :param dense_layer_count: int value that determines number of dense layers before end
        (Dropoff layers is an auto add on, and doesn't count as a second layer for splitting purposes)
    :param cnn_kernal_size: length of a side of the kernal (always square)
    :param cnn_filter_count: number of kernals per layer, can also pass a list, which defines the exact number per layer
    :param cnn_layer_count: int value that determines number of cnn layers added
        (max pooling is an auto add on, and doesn't count as a second layer for splitting purposes)
    :param split_point: int value determining which layer the model splits after (0 is input_img)
    :param input_shape: input data size
    :return: tf.Model object, fully built, but not compiled
    """
    model_structure = [split_point, cnn_layer_count, cnn_filter_count, cnn_kernal_size, dense_layer_count, dense_node_count, output_activation]
    structure_string = "_".join([str(i) for i in model_structure])+"_"
    structure_string = structure_string.replace(", ", "_").replace("[", ".").replace("]", ".")
    model_name = f"{structure_string}"+version_diff

    # ensure the split_point is within the model's boundries
    if split_point < 0:
        split_point = 0
    if cnn_layer_count + dense_layer_count < split_point:
        split_point = cnn_layer_count + dense_layer_count

    # make any int's always lists of ints for node/filter counts per layer
    if dense_node_count is int:
        dense_node_count = [dense_node_count]*dense_layer_count
    if cnn_filter_count is int:
        cnn_filter_count = [cnn_filter_count]*cnn_layer_count

    """input layer"""
    start = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dropout(0.05)(start)
    counter = 0
    split_passed = False
    for i in range(cnn_layer_count):
        # check if we reached the split point, or the split has passed, otherwise continue adding layer
        if counter == split_point:
            split_passed = True
            y = add_cnn_layer(input_layer=x, cnn_filter=cnn_filter_count[i], cnn_kernal_size=cnn_kernal_size)
        elif split_passed:
            y = add_cnn_layer(input_layer=y, cnn_filter=cnn_filter_count[i], cnn_kernal_size=cnn_kernal_size)

        x = add_cnn_layer(input_layer=x, cnn_filter=cnn_filter_count[i], cnn_kernal_size=cnn_kernal_size)
        counter += 1

    # check again, adding flatten layer this time
    if counter == split_point:
        split_passed = True
        y = tf.keras.layers.Flatten()(x)
    elif split_passed:
        y = tf.keras.layers.Flatten()(y)
    x = tf.keras.layers.Flatten()(x)
    counter += 1

    for i in range(dense_layer_count):
        # check if we reached the split point, or the split has passed, otherwise continue adding layer
        if counter == split_point:
            split_passed = True
            y = add_dense_layer(input_layer=x, node_count=dense_node_count[i])
        elif split_passed:
            y = add_dense_layer(input_layer=y, node_count=dense_node_count[i])

        x = add_dense_layer(input_layer=x, node_count=dense_node_count[i])
        counter += 1

    x_output = tf.keras.layers.Dense(1, activation=output_activation, name="x_output")(x)
    if not split_passed:
        y_output = tf.keras.layers.Dense(1, activation=output_activation, name="y_output")(x)
    else:
        y_output = tf.keras.layers.Dense(1, activation=output_activation, name="y_output")(y)

    model = tf.keras.models.Model(inputs=start, outputs=[x_output, y_output], name=model_name)
    return model


def main(root, data_path):
    max_labels = [15, 50]
    sample_reduction = 1
    # input with size (200, 200, 1)
    model = create_model(
        input_shape=(200, 200, 1),
        split_point=2,
        cnn_layer_count=1,
        cnn_filter_count=[128, 16],
        cnn_kernal_size=4,
        dense_layer_count=2,
        dense_node_count=[128, 16],
        output_activation="sigmoid",
        version_diff="scaled",
    )
    # compile model and save images
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss={'x_output': 'mae', 'y_output': 'mae'})
    model.summary()
    tf.keras.utils.plot_model(model, to_file=root + data_path + model.name + " Graph.png")

    # track the weights of the model
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=root + data_path + f'/{model.name}.weights.hdf5',
        monitor='loss',
        save_best_only=True,
        save_freq='epoch',
        period=1,
    )
    board = tf.keras.callbacks.TensorBoard(log_dir=f'./tmp/{model.name}')

    # open the name_list.txt file, and turn it into list of filenames within the folder of the same path
    file_name_list = load_data(root=root, path=data_path, filename="name_list.txt", samplereduction=sample_reduction)

    # create a list of indices, randomize, then split into training and testing usable_indices
    indices = np.arange(len(file_name_list))
    np.random.shuffle(indices)
    testing_percentage = 0.2
    seperation_point = int(len(file_name_list) * (1 - testing_percentage))
    training_indices = indices[:seperation_point]
    testing_indices = indices[seperation_point:]

    # training generator object
    training_generator = BatchGenerator(
        path=root + data_path,
        filenamelist=file_name_list,
        shuffle=True,
        batch_size=50,
        imagesize=(200, 200),
        usable_indices=training_indices,
        scale_labels=max_labels,
    )

    # testing generator object
    testing_generator = BatchGenerator(
        path=root + data_path,
        filenamelist=file_name_list,
        shuffle=True,
        batch_size=50,
        imagesize=(200, 200),
        usable_indices=testing_indices,
        scale_labels=max_labels,
    )

    # load saved weights, if they exist, so to continue training
    if os.path.isfile(root + data_path + f'/{model.name}.weights.hdf5'):
        model.load_weights(root + data_path + f'/{model.name}.weights.hdf5')

    model_history = model.fit(
        training_generator,
        validation_data=testing_generator,
        epochs=50,
        initial_epoch=0,
        callbacks=[checkpointer, board]
    )

    plt.title("Loss over time(epochs)")
    plt.plot(model_history.history['loss'], label="loss")
    plt.plot(model_history.history['val_loss'], label="val loss")
    plt.legend()
    plt.savefig(root + data_path + model.name + "training.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # define the path the data is travelling: root will not change, the individual folder will depend on dataset accessed
    root = "F:/College_UCC/AM6021- Dissertation/Depth Map Numpy Files/"
    data_path = "Simulated data/2_15000_0.2_100_2.5/temp/"
    main(root, data_path)
