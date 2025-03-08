import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
from Model import ModelTest
from Modelutils import EnergyShifter, load
import os
import pandas as pd
import sys
import argparse
import torch




if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path_train',
                        help='Path of the dataset, can be an hdf5 file or a directory containing hdf5 files')
    parser.add_argument('--dataset_path_test',
                        help='Path of the dataset, can be an hdf5 file or a directory containing hdf5 files')    
    parser.add_argument('-d', '--device',
                        help='Device of modules and tensors',
                        default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('-b', '--batch_size',
                        help='Number of conformations of each batch',
                        default=256, type=int)
    parser.add_argument('-n', '--num_epochs',
                        help='Number of epochs',
                        default=1000, type=int)


    args = parser.parse_args()



if args.dataset_path_train is None:
    args.dataset_path_train = input("Enter the path of the train dataset: ")

if args.dataset_path_test is None:
    args.dataset_path_test = input("Enter the path of the test dataset: ")


#################################  loading Data ############################################
"""Tools for loading, shuffling, and batching Retrievium datasets



"""

species_order = ['H', 'C', 'N', 'O', 'S', 'Cl']

num_species = len(species_order)

energy_shifter = EnergyShifter(None)

try:
        path = os.path.dirname(os.path.realpath(__file__))
except NameError:
        path = os.getcwd()
dspath = os.path.join(args.dataset_path_train)
batch_size = args.batch_size


training, validation = load(dspath)\
                                .subtract_self_energies(energy_shifter, species_order)\
                                .remove_outliers()\
                                .species_to_indices(species_order)\
                                .shuffle()\
                                .split(0.8, None)


training = training.collate(batch_size).cache()
validation = validation.collate(batch_size).cache()
print('Self atomic energies: ', energy_shifter.self_energies)

###################################################

try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()
dspath_test = os.path.join(args.dataset_path_test)
batch_size = args.batch_size


_, test = load(dspath_test)\
                                    .subtract_self_energies(energy_shifter, species_order)\
                                    .remove_outliers()\
                                    .species_to_indices(species_order)\
                                    .shuffle()\
                                    .split(0.0, None)

# training_test = training_test.collate(batch_size).cache()
data_test = test.collate(batch_size).cache()
print('Self atomic energies: ', energy_shifter.self_energies)

########## Run Model ################################################

rmse_test, mae_test = ModelTest(training, validation, data_test,species_order,energy_shifter, args.num_epochs, args.device) 
print('RMSE_TEST', rmse_test, 'MAE_TEST', mae_test)

 





