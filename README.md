# Dissertation-Files

"create_dataset.py"

FLUMY blocks are saved as {channel depth}_{net to gross}_{seed}_{block identifier}.csv, using gslib format, with facies id, grain size and age matrixs saved.

is used on a set of FLUMY blocks with this specific setup. 
The amount of blocks to stack, the range of values of the FLUMY blocks and the facies id groupings and Abrasion Resistance values are set inside the main() function

otherwise, the main() function have these parameters:    
param file_path: string of file path to save the landscape evolution timelines
param erosion_steps: number of timesteps to compute
param timestep: size of a single timestep, this is equivalent to the erosion constant.
param edr: erosion/diffusion ratio, divides timestep param by edr param to get the diffusion constant
param timestep_skip: no. of initial timesteps to skip over before saving
param flumy_file_path: file path string pointing towards flumy data folder

Output: 
Compressed Numpy arrays of the landscape evolution timeline.



"DEM_Parser.py"

Set of functions used to parse larger TIF DEM files into image_size squared DEM's. Along with imaging and graphing functions to observe how the methods work.
Recording Ridge Feature values in data_frame.csv, and saving all file names of the parsed DEMs as name_list.txt

param tif_file_name: path to TIF file to extract sample DEMs from
param save_path: path to save the DEM arrays to
param stride: step size for subsectioning image
param cross_section_size: length of cross section sampling, in pixel units
param threshold: determines whether a ridge occurs, by checking this length on each side of the ridge point, and if both sides height compared to the max point exceeds this value, the ridge is considered "True"
param given_cd: given values of the channel depth of paleo-river, from Literature
param given_ntg: given values of the net to gross % of paleo-river, from Literature
param image_size: size of the images to process/save
param plot_all_image: a bool that decides whether to use all the plot image functions, to display results

Output: 
name_list.txt: containing the filename of all images saved
data_frame.csv: contains ridge height and width for each image saved, along side other data.
saved arrays .npy: multitude of numpy files, as part of the machine learning earth validation or Mars Prediction Set.



"analyse timelines.py"

uses several functions from DEM_Parser, due to similar function.
Used to filter the Landscape Evolution Compressed timelines, first checking every layerresolution=100 amount of DEMs from the compressed timeline, 
then filtering out the DEMs that don't have a significant enough ridge, recording ridge feature values, and then saving as individual arrays (.npy),
finally, saving the collected ridge feature values as a pandas dataframe data_frame.csv, and recording the file names of the saved images in a name_list.txt

param root: string of file path of folder containing the compressed Landscape Evolution Timelines (.npz files)
param save_path: string of file path of folder to save the individual filtered arrays, the pandas feature dataframe, and the file name list file.
param layerresolution: how many layers to skip over, used to reduce the number of similar dems added to dataset
param cross_section_size: length of cross section of potential ridge used to check if potential ridge is a ridge
param threshold: value used as threshold in detect_ridge() function from DEM_Parser.py
param image_size: used to define DEM size, to prevent the ridge_decider() function from DEM_Parser.py from trying to check out of bounds index for cross section

Output: 
name_list.txt: containing the filename of all images saved
data_frame.csv: contains ridge height and width for each image saved, along side other data.
saved arrays .npy: multitude of numpy files, filtered to have significant ridges, as part of the machine learning training set.



"MachineLearningTraining.py"

uses dataset generated from "analyse timelines.py", using the name_list.txt to know what files to use, 
max_labels is used to scale the labels to be between 0 and 1, when the model trained has an output activation function that uses values between 0 and 1.
The model created is formed using the create_model() function. This has a specific structure, and can be defined using a name:

{split point}_{number of cnn layers used}_.{list of number of filters used per cnn layer}._
_{size of kernal used in cnn layers}_{number of dense layers used}_.{list of node count per dense layer}._
{output activator string}.{model differentiator string}

split point: describes how many layers of the model are made before a bifercation of the layers occur, in order to predict the two regression values.
list of filters: lists are between full stops in file name, and are formated 1_2_3_4. This would correspond to a 4 layer cnn, with 1st layer having 1 filter, 2nd layer having 2 etc.
list of nodes: similar to filter list, but each value corresponds to number of nodes in a dense layer.
output activator: examples, "sigmoid", "linear", with sigmoid needing the labels to be scaled using max_labels.
model_differentiator: a string used to diferenciate models with the same architecture, but differing in other ways, such as max_labels scaling, or differing optimizer.

also in main(), theres a testing_percentage = 0.2 parameter, defining the fraction split from total training data to be testing data.

param root: path to root folder, where training, testing, validation and other datasets are contained.
param data_path: string of file path, from root folder to the training dataset used.

Output: 
Model trained on selected data, using selected architecture, and the weights trained. The weights file uses the specific model architecture defining name, in order to reproduce the model using only the weight file name


"Observe_Model_Prediction.py"

Uses max_labels parameter in the main() function to rescale predicted results to be in line with the original labels used. 
loads a specified weight file, generates a model using the specified architecture from the weight file name, and predicts the characteristic values for the specified validation dataset.

param root: string of collective file path for both the weight and valid path, acting as a root directory
param weight_path: string of file path to the model weights file
param valid_path: string of file path to the folder containing the validation dataset

Output:
images showing prediction results of specified model, on specified dataset.

