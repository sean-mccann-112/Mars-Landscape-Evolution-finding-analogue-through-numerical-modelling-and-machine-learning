# Dissertation-Files


create_dataset.py 

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

Output: Compressed Numpy arrays of the landscape evolution timeline.