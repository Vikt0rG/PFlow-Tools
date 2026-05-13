#!/usr/bin/env python

# Max's script

import os

import pandas as pd
import numpy as np
import h5py
import yaml
import argparse


def list_datasets(h5_fname: str) -> list[str]:
    """
    Lists all datasets contained in a given HDF5 file.

    Arguments:
        h5_fname (str): Path to the HDF5 file

    Returns:
        list[str]:      List of dataset names
    """
    datasets = []
    with h5py.File(h5_fname, 'r') as h5_file:
        def add_name_if_ds(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append(name)

        h5_file.visititems(add_name_if_ds)
    return datasets


def remove_columns(in_path: str, out_path: str, columns: list[str], dataset: str) -> None:
    """
    Removes columns from a specific dataset in a given HDF5 file and creates a new updated file.

    Args: 
        in_path (str):         Path to the HDF5 file
        out_path (str):        Path to the output file
        columns (list[str]):    List of column names to be removed 
        dataset (str):          Name of the dataset
    """
    datasets = list_datasets(in_path)

    with h5py.File(in_path, 'r') as pflow_file:
        ds = pflow_file[dataset]

        # Convert float64 to float32 if present
        dtype_f32 = [col if col[1] != '<f8' else (col[0], '<f4', *col[1:]) for col in ds.dtype.descr]
        # Filter out specified columns 
        dtype_filtered = [col for col in dtype_f32 if col[0] not in columns]

        ds_mod = np.empty(ds.shape, dtype=np.dtype(dtype_filtered))
        for column_name in map(lambda col: col[0], dtype_filtered):
            ds_mod[column_name] = ds[column_name]

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with h5py.File(out_path, 'w') as mod_file:
            for d_set in datasets:
                shape = np.array(d_set).shape
                maxshape = (h5py.h5s.UNLIMITED) if len(shape) == 1 else (h5py.h5s.UNLIMITED, *shape[1:])

                if d_set == dataset:
                    mod_file.create_dataset(d_set, data=ds_mod, maxshape=maxshape)
                else:
                    mod_file.create_dataset(d_set, data=pflow_file[d_set], maxshape=maxshape)


def split_datasets(
    in_path: str, 
    train_path: str, 
    val_path: str,
    fraction_train: float = 0.7, 
    fraction_val: float = 0.3, 
    test_path: str = None, 
    fraction_test: float = 0.0, 
    shuffle: bool = False, 
    seed: int = 42
) -> None:
    """
    Splits datasets from a single HDF5 file into training, validation, and optional testing subsets, ensuring consistency across all datasets. 
    It is assumed that all datasets contain the same number of entries. (!)

    Args:
        in_path (str):          Path to the input file
        train_path (str):       Path to the output file for the training data
        fraction_train (float): Size of the training data, will be divided by (train_size + val_size + test_size)
        val_path (str):         Path to the output file for the validation data
        fraction_val (float):   Size of the validation data, will be devided by (train_size + val_size + test_size)
        test_path (str):        Path to the output file for the testing data
        fraction_test (float):  Size of the testing data, will be devided by (train_size + val_size + test_size)
        shuffle (bool):         Whether the data should be shuffled before splitting
        seed (int):             Seed for the data shuffling
    """
    datasets = list_datasets(in_path)
    
    with h5py.File(in_path, 'r') as pflow_file:
        num_events = len(pflow_file[datasets[0]])

        # Define shuffled indices for consistent shuffling
        if shuffle:
            rng = np.random.default_rng(seed)
            indices = np.arange(num_events)
            rng.shuffle(indices)
        else:
            indices = np.arange(num_events)

        # Normalize sizes to fractions
        training_fraction = fraction_train/(fraction_train + fraction_val + fraction_test)
        validation_fraction = fraction_val/(fraction_train + fraction_val + fraction_test)

        datasets_train = []
        datasets_val = []
        datasets_test = []

        ds_maxshapes = []

        for ds_name in datasets:
            print(f'Loading dataset {ds_name}')
            ds = pflow_file[ds_name]

            shape = np.array(ds).shape
            maxshape = (h5py.h5s.UNLIMITED) if len(shape) == 1 else (h5py.h5s.UNLIMITED, *shape[1:])
            
            ds_maxshapes.append(maxshape)

            # Create subsets as slices defined by fraction * number of entries
            datasets_train.append(np.array(ds)[indices][:int(training_fraction*len(ds))])

            # Only create a test set if test size and test path are specified
            if fraction_test == 0 and test_path is not None:
                datasets_val.append(np.array(ds)[indices][int(training_fraction*len(ds)):])
            else:
                datasets_val.append(np.array(ds)[indices][int(training_fraction*len(ds)):int((training_fraction+validation_fraction)*len(ds))])
                datasets_test.append(np.array(ds)[indices][int((training_fraction+validation_fraction)*len(ds)):])

        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        with h5py.File(train_path, 'w') as training_file:
            print(f'Writing to {train_path}')
            for i, ds_name in enumerate(datasets):
                training_file.create_dataset(ds_name, data=datasets_train[i], maxshape=ds_maxshapes[i])

        os.makedirs(os.path.dirname(val_path), exist_ok=True)
        with h5py.File(val_path, 'w') as validation_file:
            print(f'Writing to {val_path}')
            for i, ds_name in enumerate(datasets):
                validation_file.create_dataset(ds_name, data=datasets_val[i], maxshape=ds_maxshapes[i])

        if test_path is not None:
            os.makedirs(os.path.dirname(test_path), exist_ok=True)
            with h5py.File(test_path, 'w') as test_file:
                print(f'Writing to {test_path}')
                for i, ds_name in enumerate(datasets):
                    test_file.create_dataset(ds_name, data=datasets_test[i], maxshape=ds_maxshapes[i])


def create_norm_dict(in_path: str, out_path: str = 'norm_dict.yaml') -> None:
    """
    Create the norm_dict.yaml file that specifies mean and standard deviation for each quantity in a HDF5 file

    Args:
        in_path (str):  Path to the input file (should be the training data)
        out_path (str): Path to the output file (norm_dict.yaml)
    """
    datasets = list_datasets(in_path)

    with h5py.File(in_path, 'r') as h5file:
        dataframes = []
        for ds_name in datasets:
            # Use pandas for mean + std calculation as casting to dict is possible 
            df = pd.DataFrame()
            for col in h5file[ds_name].dtype.names:
                df[col] = h5file[ds_name][col].flatten()
            dataframes.append(df)

    norm_dict = {ds_name:{} for ds_name in datasets}
    for i, ds_name in enumerate(datasets):
        means = dict(dataframes[i].mean())
        stds = dict(dataframes[i].std())

        for col, mean in means.items():
            norm_dict[ds_name][col] = {"mean": float(mean), "std": float(stds[col])}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as norm_file:
        yaml.dump(norm_dict, norm_file)


def load_config(filepath: str) -> dict | list:
    """
    Load yaml config to list/ dict.

    Args:
        filepath (str): Path to yaml file.

    Returns:
        dict | list:    Returns config as dict or list based on its structure. 
                        Look into example.yaml for further information.
    """
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)


def run_config(config: dict | list) -> None: 
    """
    Run commands in the order given in the config file with their 
    respective arguments.

    Args:
        config (dict | list):   Config as {commands: {args: values}} dict or list [{commands: {args: values}}, {...}]. 
                                The list form is preferred as it enables running the same command multiple times.
    """
    if isinstance(config, dict):
        for i, (command, kwargs) in enumerate(config.items()):
            print(f'Running command {i}: {command}')
            if command == "remove-cols":
                remove_columns(**kwargs)
            elif command == "split":
                split_datasets(**kwargs)
            elif command == "normdict":
                create_norm_dict(**kwargs)
    elif isinstance(config, list):
        counter = 1
        for cmd in config:
            for command, kwargs in cmd.items():
                print(f'Running command {counter}: {command}')
                if command == "remove-cols":
                    remove_columns(**kwargs)
                elif command == "split":
                    split_datasets(**kwargs)
                elif command == "normdict":
                    create_norm_dict(**kwargs)
                counter += 1


def main():
    """Main entry point for the h5_prep command-line tool."""
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, help='Config file that defines the order and arguments of commands to be run. Is ignored when directly running the commands.')
    parser.set_defaults(func=(lambda args: run_config(load_config(args.config))))

    subparsers = parser.add_subparsers()
    rc_parser = subparsers.add_parser('remove-cols', help='Remove columns from a dataset')
    rc_parser.add_argument('in_path', type=str, help='Path to the input HDF5 file')
    rc_parser.add_argument('out_path', type=str, help='Path to the output file containing the updated datasets')
    rc_parser.add_argument('-c', '--columns', nargs='+', type=str, default=['clusterParticleFlavour_Visible','clusterParticlePerFlavourEnergy_Visible','clusterParticleFlavour_Full','clusterParticlePerFlavourEnergy_Full'], help='List of column names that should be removed')
    rc_parser.add_argument('-d', '--dataset', type=str, default='clusters', help='Name of the dataset (default=clusters)')
    rc_parser.set_defaults(func=(lambda args: remove_columns(args.in_path,args.out_path,args.columns,args.dataset)))

    sd_parser = subparsers.add_parser('split', help='Split datasets into train, val, test files')
    sd_parser.add_argument('in_path', type=str, help='Path to the HDF5 input file')
    sd_parser.add_argument('train_path', type=str, help='Path to the output file for the training data')
    sd_parser.add_argument('val_path', type=str, help='Path to the output file for the validation data')
    sd_parser.add_argument('-t','--test_path', type=str, default=None, help='Path to the output file for the testing data')
    sd_parser.add_argument('--fraction_train', type=float, default=0.7, help='Size of the training data, will be divided by (train_size + val_size + test_size) (default=0.7)')
    sd_parser.add_argument('--fraction_val', type=float, default=0.3, help='Size of the validation data, will be divided by (train_size + val_size + test_size) (default=0.3)')
    sd_parser.add_argument('--fraction_test', type=float, default=0.0, help='Size of the testing data, will be divided by (train_size + val_size + test_size) (default=0.0)')
    sd_parser.add_argument('--shuffle', action='store_true', default=False, help='Shuffle the datasets')
    sd_parser.add_argument('-s', '--seed', type=int, default=42, help='Seed used for shuffling')
    sd_parser.set_defaults(func=(lambda args: split_datasets(args.in_path, args.train_path, args.val_path, 
                                                             args.fraction_train, args.fraction_val, args.test_path, 
                                                             args.fraction_test, args.shuffle, args.seed)))
    
    nd_parser = subparsers.add_parser('normdict', help='Create a normalization dictionary (norm_dict.yaml) for a given HDF5 file')
    nd_parser.add_argument('in_path', type=str, help='Path to the HDF5 input file')
    nd_parser.add_argument('-o','--out_path', type=str, default='./norm_dict.yaml', help='Path to the output norm_dict.yaml (default=./norm_dict.yaml)')
    nd_parser.set_defaults(func=(lambda args: create_norm_dict(args.in_path, args.out_path)))

    cmd = parser.parse_args()
    cmd.func(cmd)


if __name__ == '__main__':
    main()