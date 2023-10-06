import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import pandas as pd

from MachineLearningTraining import load_data, create_model


def load_unique_model():
    """5th model scaled architecture"""
    # input with size (200, 200, 1)
    input_img = tf.keras.layers.Input(shape=(200, 200, 1))  # adapt this if using `channels_first` image data format

    # CNN starter block
    x = tf.keras.layers.Conv2D(20, (5, 5), activation='relu', padding='same')(input_img)  # increase amount of layers
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(20, (4, 4), activation='relu', padding='same')(x)
    split = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    # cd branch
    cd = tf.keras.layers.Conv2D(20, (3, 3), activation='relu', padding='same')(split)
    cd = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(cd)
    cd = tf.keras.layers.Conv2D(20, (2, 2), activation='relu', padding='same')(cd)
    code_cd = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(cd)  # 12, 12, 30

    cd = tf.keras.layers.Flatten()(code_cd)
    cd = tf.keras.layers.Dense(100, activation='relu')(cd)
    cd = tf.keras.layers.Dropout(0.2)(cd)
    cd = tf.keras.layers.Dense(50, activation='relu')(cd)
    cd = tf.keras.layers.Dropout(0.2)(cd)
    cd = tf.keras.layers.Dense(50, activation='sigmoid')(cd)
    cd = tf.keras.layers.Dropout(0.2)(cd)
    cd_output = tf.keras.layers.Dense(1, activation='sigmoid', name="cd_output")(cd)

    # ntg branch
    ntg = tf.keras.layers.Conv2D(20, (3, 3), activation='relu', padding='same')(split)
    ntg = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(ntg)
    ntg = tf.keras.layers.Conv2D(20, (2, 2), activation='relu', padding='same')(ntg)
    code_ntg = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(ntg)  # 12, 12, 30

    ntg = tf.keras.layers.Flatten()(code_ntg)
    ntg = tf.keras.layers.Dense(100, activation='relu')(ntg)
    ntg = tf.keras.layers.Dropout(0.2)(ntg)
    ntg = tf.keras.layers.Dense(50, activation='relu')(ntg)
    ntg = tf.keras.layers.Dropout(0.2)(ntg)
    ntg = tf.keras.layers.Dense(50, activation='sigmoid')(ntg)
    ntg = tf.keras.layers.Dropout(0.2)(ntg)
    ntg_output = tf.keras.layers.Dense(1, activation='sigmoid', name="ntg_output")(ntg)
    model = tf.keras.models.Model(inputs=input_img, outputs=[cd_output, ntg_output], name="5th_Model_scaled")
    return model


def plot_image(prediction, true_label, img):
    """
    prediction == what the model thinks the labels are
    true_label == what the actual labels are
    img == the image that the predictions are applying to.
    """
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap="Greys", vmin=0, vmax=1)
    # true_label = "[3, 16]"  # Earth Data True Label
    plt.title(f"{prediction}", color="red")
    plt.xlabel(f"true:{true_label}", color="blue")


def plot_prediction(root, path, file_name_list, model, offset=20, num_rows=5, num_cols=5, scale_labels=(1, 1)):
    num_images = num_rows * num_cols

    image_set = np.empty((num_images, 200, 200))
    image_labels = np.empty((num_images, 2))

    splitval = int((len(file_name_list) - offset) // num_images)
    file_name_list = file_name_list[::splitval]
    print(file_name_list)

    # create the Batch to be predicted on
    for i in range(num_images):
        file_name = file_name_list[i]  # file name (also contains labels data for image)
        # load the selected numpy array
        imagePath = root + path + file_name

        validation_image = np.load(file=imagePath)

        # extract the labels, ie first two values in the file name
        label_str = file_name.split(sep="_")
        validation_label = list(map(int, label_str[0:2]))

        image_set[i] = validation_image
        image_labels[i] = validation_label

    predictions = np.array(model.predict(image_set))

    plt.figure(figsize=(2 * num_cols, 2 * num_rows))

    print(predictions.shape)
    for i in range(num_images):
        prediction = predictions[:, i, 0]

        # this rescales labels, since they were scaled previously
        prediction[0] = prediction[0] * scale_labels[0]
        prediction[1] = prediction[1] * scale_labels[1]

        validation_image = image_set[i]
        validation_label = image_labels[i]

        plt.subplot(num_rows, num_cols, i + 1)
        plot_image(prediction, validation_label, validation_image)

    plt.tight_layout()
    plt.show()


def plot_prediction_true_correlation(root, path, file_name_list, model, scale_labels):
    image_set = np.empty((len(file_name_list), 200, 200))
    image_labels = np.empty((len(file_name_list), 2))

    for i in range(len(file_name_list)):
        # print(i, len(file_name_list))
        file_name = file_name_list[i]  # file name (also contains labels data for image)
        validation_image = np.load(file=root + path + file_name)

        # extract the labels, ie first two values in the file name
        label_str = file_name.split(sep="_")
        validation_label = list(map(int, label_str[0:2]))

        image_set[i] = validation_image
        image_labels[i] = validation_label

    predictions = np.array(model.predict(image_set))

    column_names = ["True cd", "True ntg", "Predicted cd", "Predicted ntg"]
    df = pd.DataFrame(np.transpose(np.array([image_labels[:, 0], image_labels[:, 1], predictions[0, :, 0]*scale_labels[0], predictions[1, :, 0]*scale_labels[1]])), columns=column_names)
    sorted_df = df.sort_values(by=['True cd', 'True ntg'])

    cd_key = sorted(list(set(image_labels[:, 0])))
    ntg_key = sorted(list(set(image_labels[:, 1])))

    colour_list = ["blue", "green", "yellow", "orange", "red"]
    for k in range(len(ntg_key)):
        ntg = ntg_key[k]
        average_array = []
        standard_array = []
        for j in range(len(cd_key)):
            cd = cd_key[j]
            filtered_df = sorted_df[(sorted_df['True cd'] == cd) & (sorted_df['True ntg'] == ntg)]
            average_array.append(filtered_df["Predicted cd"].mean())
            standard_array.append(filtered_df["Predicted cd"].std())
            plt.scatter(x=filtered_df['True cd']+(ntg/1000), y=filtered_df["Predicted cd"], alpha=2/(len(filtered_df)+2), c=colour_list[k])

        plt.errorbar(x=cd_key, y=average_array, yerr=standard_array, capsize=2, c=colour_list[k], alpha=0.5)
        plt.plot(cd_key, average_array, c=colour_list[k], label=f"ntg={ntg}%")

    plt.plot([2, 15], [2, 15], c="black", linestyle="--", label="accurate")
    plt.title("Channel Depth Training Data Accuracy")
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.legend()
    plt.show()


def plot_precision(root, path, file_name_list, model, scale_labels=(30, 100)):
    image_set = np.empty((len(file_name_list), 200, 200))

    for i in range(len(file_name_list)):
        file_name = file_name_list[i]
        imagePath = root + path + file_name
        validation_image = np.load(file=imagePath)

        image_set[i] = validation_image

    predictions = np.array(model.predict(image_set))
    summed = predictions.sum(axis=1)[:, 0]
    print(predictions.shape)
    average = [summed[0]/predictions.shape[1]*scale_labels[0], summed[1]/predictions.shape[1]*scale_labels[1]]
    print(average)
    x = predictions[0] * scale_labels[0]
    y = predictions[1] * scale_labels[1]
    plt.scatter(x, y, color="r", marker="o", alpha=0.2, label="Predicted Results")
    # plt.scatter(3, 20, s=100, color="b", marker="x", label="True Result")
    # plt.errorbar(3, 16, xerr=1, yerr=6, fmt='x', ecolor='b', color='b', label="True Result")
    plt.scatter(average[0], average[1], s=100, color="g", marker="x", label="Average Result")
    plt.xlim(0, 15)
    plt.ylim(0, 50)
    plt.legend()
    plt.show()


def main(root, model_path, valid_path):
    max_labels = [15, 100]
    sample_reduction = 1

    weight_file_name = model_path.split(sep="/")[-1].split(sep=".")[:5]
    weight_file_name = [x.split(sep="_") for x in weight_file_name]

    sp = int(weight_file_name[0][0])
    cnnlc = int(weight_file_name[0][1])
    cnnfc = list(map(int, weight_file_name[1]))
    cnnks = int(weight_file_name[2][1])
    dlc = int(weight_file_name[2][2])
    dnc = list(map(int, weight_file_name[3]))
    oa = weight_file_name[4][1]
    vd = weight_file_name[4][2]
    
    # model = create_model(
    #         input_shape=(200, 200, 1),
    #         split_point=sp,
    #         cnn_layer_count=cnnlc,
    #         cnn_filter_count=cnnfc,
    #         cnn_kernal_size=cnnks,
    #         dense_layer_count=dlc,
    #         dense_node_count=dnc,
    #         output_activation=oa,
    #         version_diff=vd,
    #     )
    model = load_unique_model()

    # compile model and save images
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss={'x_output': 'mae', 'y_output': 'mae'})
    model.summary()

    # load saved weights, if they exist, so to continue training
    model.load_weights("C:/Users/User/Downloads/Dissertation Model Weights/unique model.weights.hdf5")
    # if os.path.isfile(root + model_path + f'/{model.name}.weights.hdf5'):
    #     model.load_weights(root + model_path + f'/{model.name}.weights.hdf5')
    # file_name_list = [f for f in os.listdir(root+valid_path) if f.endswith('.npy')]
    file_name_list = load_data(root=root, path=valid_path, filename="name_list.txt", samplereduction=sample_reduction)
    plot_prediction_true_correlation(root=root, path=valid_path, file_name_list=file_name_list, model=model, scale_labels=max_labels)
    # plot_prediction(root=root, path=valid_path, file_name_list=file_name_list, model=model, offset=20, num_rows=5, num_cols=5, scale_labels=max_labels)
    # plot_precision(root=root, path=valid_path, file_name_list=file_name_list, model=model, scale_labels=max_labels)


if __name__ == "__main__":
    # define the path the data is travelling: root will not change, the individual folder will depend on dataset accessed
    root = "F:/College_UCC/AM6021- Dissertation/Depth Map Numpy Files/"
    weight_path = "Simulated data/2_15000_0.2_100_2.5/temp/1_1_.128._3_2_.256_16._sigmoid_scaled.weights.hdf5"
    # valid_path = "Mars data/mars1/"
    valid_path = "Simulated data/2_15000_0.2_100_2.5 testing/temp/"
    main(root=root, model_path=weight_path, valid_path=valid_path)
