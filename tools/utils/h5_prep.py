import os

import yaml
import argparse

import numpy as np
import awkward as ak
import h5py
import pandas as pd

# Extended Max's script(s)


def list_datasets(h5_fname: str) -> list[str]:
    """Lists all datasets contained in a given HDF5 file.

    Parameters
    ----------
    h5_fname : str
        Path to the HDF5 file

    Returns
    -------
    list[str]
        List of dataset names
    """
    datasets = []
    with h5py.File(h5_fname, 'r') as h5_file:
        def add_name_if_ds(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append(name)

        h5_file.visititems(add_name_if_ds)
    return datasets


def load_hdf(filepath: str) -> ak.Array:
    """Loads an HDF5 file into an awkward array

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file

    Returns
    -------
    ak.array:
        Awkward array with structure
            {
                'dataset0':
                    {
                        'column0' : [...],
                        ...
                        'columnN' : [...]
                    },
                ...
                'datasetN': {...}
            }

    """
    datasets = list_datasets(filepath)

    entries = { ds : {} for ds in datasets }

    with h5py.File(filepath, 'r') as test_file:
        for ds in datasets:
            dataset = test_file[ds]
            entry = {
                column[0]: dataset[column[0]][:] for column in dataset.dtype.descr
            }
            entries[ds] = entry

    return ak.Array(entries)


def remove_columns(
    in_path: str,
    out_path: str,
    columns: list[str],
    dataset: str
) -> None:
    """Removes columns from a specific dataset in a given HDF5 file.

    Creates a new modified HDF5 file with the specified columns
    removed from a specified dataset.

    Parameters
    ---------- 
    in_path : str
        Path to the HDF5 file
    out_path : str
        Path to the output file
    columns : list[str]
        List of column names to be removed 
    dataset : str
        Name of the dataset

    Returns
    -------
    None
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


def filter_events(
    in_path: str,
    out_path: str,
    target_columns: list[str] | None = None,
    prefix: str = "clusterParticle_EnergyFraction_Full_",
    dataset: str = "clusters",
    discard_unset: bool = False
) -> None:
    """Filters out events containing unphysical energy fractions.

    Parameters
    ----------
    in_path : str
        Path to the input HDF5 file
    out_path : str
        Path to the output filtered HDF5 file
    target_columns : list[str] | None
        Explicit list of target column names to check for physical energy fractions.
        If None, all columns starting with the specified prefix will be checked.
    prefix : str
        Prefix for target columns (used if target_columns is not provided)
    dataset : str
        Name of the dataset to check for unphysical energy fractions
    discard_unset : bool
        Whether to discard events with unset energy fractions (-1). If True, only
        events with all energy fractions strictly in the range [0.0, 1.0] will be kept.
    """
    datasets = list_datasets(in_path)

    with h5py.File(in_path, 'r') as pflow_file:
        main_ds = pflow_file[dataset]
        num_events = len(main_ds)

        # Determine columns to check
        if target_columns:
            columns_to_check = [col for col in target_columns if col in main_ds.dtype.names]
            missing_cols = set(target_columns) - set(columns_to_check)
            if missing_cols:
                print(f"Warning: Columns not found in '{dataset}': {missing_cols}")
        else:
            columns_to_check = [col for col in main_ds.dtype.names if col.startswith(prefix)]

        if not columns_to_check:
            print(f"Warning: No valid columns found in '{dataset}'. Copying file as-is.")
            mask_valid = np.ones(num_events, dtype=bool)
        else:
            print(f"Checking {len(columns_to_check)} energy fraction columns in '{dataset}'...")
            mask_valid = np.ones(num_events, dtype=bool)
            col_stats = {}

            # Evaluate bounds per column
            for col in columns_to_check:
                col_data = main_ds[col]

                # Mutually Exclusive Element-Level Masks
                mask_is_unset = (col_data == -1.0)
                mask_is_set = (col_data >= 0.0) & (col_data <= 1.0)
                mask_is_oob = ~(mask_is_set | mask_is_unset)  # Out of bounds (<0 or >1 and !=-1)

                # Validity rule based on user flag
                if discard_unset:
                    mask_is_valid = mask_is_set
                else:
                    mask_is_valid = mask_is_set | mask_is_unset

                total_clusters = col_data.size

                # Aggregate cluster counts
                set_cnt = int(np.sum(mask_is_set))
                unset_cnt = int(np.sum(mask_is_unset))
                oob_cnt = int(np.sum(mask_is_oob))

                # Event-level validity (Event passes only if ALL its clusters are valid)
                if col_data.ndim > 1:
                    mask_event_valid = np.all(mask_is_valid, axis=tuple(range(1, col_data.ndim)))
                else:
                    mask_event_valid = mask_is_valid

                invalid_events_cnt = int(np.sum(~mask_event_valid))

                category_name = col.replace(prefix, "") if col.startswith(prefix) else col
                col_stats[category_name] = {
                    "set": set_cnt,
                    "unset": unset_cnt,
                    "oob": oob_cnt,
                    "total": total_clusters,
                    "invalid_events": invalid_events_cnt,
                }

                # Combine with global event mask
                mask_valid &= mask_event_valid

            num_passed = int(np.sum(mask_valid))
            num_discarded = num_events - num_passed

            # Breakdown Table Printouts
            print(f"\nFiltering & Cluster Breakdown for '{in_path}':")
            if discard_unset:
                print(">>> NOTE: 'discard_unset' is True. Events with Unset (-1) clusters are being discarded.")
            print("=" * 105)
            print(
                f"{'Category/Column':<25} | {'Set [0.0, 1.0]':<18} | {'Unset (-1.0)':<18} | "
                f"{'OOB (<0 or >1)':<18} | {'Bad Events (%)'}"
            )
            print("=" * 105)

            for category, stats in col_stats.items():
                tot = stats["total"]

                # Cluster-level percentages
                set_pct = (stats["set"] / tot) * 100
                unset_pct = (stats["unset"] / tot) * 100
                oob_pct = (stats["oob"] / tot) * 100

                # Event-level percentage
                bad_events = stats["invalid_events"]
                bad_events_pct = (bad_events / num_events) * 100

                set_str = f"{stats['set']} ({set_pct:.1f}%)"
                unset_str = f"{stats['unset']} ({unset_pct:.1f}%)"
                oob_str = f"{stats['oob']} ({oob_pct:.1f}%)"
                bad_str = f"{bad_events} ({bad_events_pct:.1f}%)"

                print(
                    f"{category:<25} | {set_str:<18} | "
                    f"{unset_str:<18} | {oob_str:<18} | {bad_str}"
                )
            print("=" * 105)

            print(f"\nOverall Event Summary:")
            print(f"  Total events  : {num_events}")
            print(f"  Passed filter : {num_passed} ({num_passed / num_events * 100:.2f}%)")
            print(f"  Discarded     : {num_discarded} ({num_discarded / num_events * 100:.2f}%)")

        # Write filtered datasets
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with h5py.File(out_path, 'w') as out_file:
            for ds_name in datasets:
                ds = pflow_file[ds_name]
                filtered_data = ds[mask_valid]

                shape = filtered_data.shape
                maxshape = (h5py.h5s.UNLIMITED) if len(shape) == 1 else (h5py.h5s.UNLIMITED, *shape[1:])
                out_file.create_dataset(ds_name, data=filtered_data, maxshape=maxshape)

    print(f"\nSaved filtered dataset to: {out_path}\n")


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
    """Dataset splitting

    Splits datasets from a single HDF5 file into training, validation, and
    optional testing subsets, ensuring consistency across all datasets.

    NOTE: It is assumed that all datasets contain the same number of entries!

    Parameters
    ----------
    in_path : str
        Path to the input file
    train_path : str
        Path to the output file for the training data
    fraction_train : float
        Size of the training data, will be divided by (train_size + val_size + test_size)
    val_path : str
        Path to the output file for the validation data
    fraction_val : float
        Size of the validation data, will be devided by (train_size + val_size + test_size)
    test_path : str
        Path to the output file for the testing data
    fraction_test : float
        Size of the testing data, will be devided by (train_size + val_size + test_size)
    shuffle : bool
        Whether the data should be shuffled before splitting
    seed : int
        Seed for the data shuffling

    Returns
    -------
    None
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
                datasets_val.append(
                    np.array(ds)[indices][
                        int(training_fraction * len(ds)) : 
                        int((training_fraction + validation_fraction) * len(ds))
                    ]
                )
                datasets_test.append(
                    np.array(ds)[indices][
                        int((training_fraction + validation_fraction) * len(ds)):
                    ]
                )

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
    """norm_dict.yaml creation

    Create the norm_dict.yaml file to specify mean values and
    standard deviations for each quantity in a HDF5 file

    Parameters
    ----------
    in_path : str
        Path to the input file (should be the training data)
    out_path : str
        Path to the output file (norm_dict.yaml)
    """
    datasets = list_datasets(in_path)

    with h5py.File(in_path, 'r') as h5file:
        dataframes = []
        for ds_name in datasets:
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
    """Load yaml config to list/ dict.

    Parameters
    ----------
    filepath : str
        Path to yaml file.

    Returns
    -------
    dict | list
        Returns config as dict or list based on its structure. 
        See example.yaml for more information.
    """
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)


def run_config(config: dict | list) -> None: 
    """Run commands in the order given in the config file.

    Run commands in the order given in the config file with their 
    respective arguments.

    Parameters
    ----------
    config : dict | list
        Config as {commands: {args: values}} dict or list
        [{commands: {args: values}}, {...}]. The list form is preferred
        as it enables running the same command multiple times.
    """
    handlers = {
        "filter-events": filter_events,
        "remove-cols": remove_columns,
        "split": split_datasets,
        "normdict": create_norm_dict,
    }
    if isinstance(config, dict):
        for i, (command, kwargs) in enumerate(config.items()):
            print(f'Running command {i}: {command}')
            if command in handlers:
                handlers[command](**kwargs)
    elif isinstance(config, list):
        counter = 1
        for cmd in config:
            for command, kwargs in cmd.items():
                print(f'Running command {counter}: {command}')
                if command in handlers:
                    handlers[command](**kwargs)
                counter += 1


def get_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config', type=str,
        help='Config file that defines the order and arguments of commands to be run. Is ignored when directly running the commands.'
    )
    parser.set_defaults(func=(lambda args: run_config(load_config(args.config))))

    subparsers = parser.add_subparsers()

    # Subparser: remove-cols
    rc_parser = subparsers.add_parser('remove-cols', help='Remove columns from a dataset')
    rc_parser.add_argument('in_path', type=str, help='Path to the input HDF5 file')
    rc_parser.add_argument('out_path', type=str, help='Path to the output file containing the updated datasets')
    rc_parser.add_argument(
        '-c', '--columns',
        nargs='+',
        type=str,
        default=[
            'clusterParticleFlavour_Visible',
            'clusterParticlePerFlavourEnergy_Visible',
            'clusterParticleFlavour_Full',
            'clusterParticlePerFlavourEnergy_Full',
        ],
        help='List of column names that should be removed',
    )
    rc_parser.add_argument('-d', '--dataset', type=str, default='clusters', help='Name of the dataset (default=clusters)')
    rc_parser.set_defaults(
        func=(lambda args: remove_columns(args.in_path, args.out_path, args.columns, args.dataset))
    )

    # Subparser: split
    sd_parser = subparsers.add_parser('split', help='Split datasets into train, val, test files')
    sd_parser.add_argument('in_path', type=str, help='Path to the HDF5 input file')
    sd_parser.add_argument('train_path', type=str, help='Path to the output file for the training data')
    sd_parser.add_argument('val_path', type=str, help='Path to the output file for the validation data')
    sd_parser.add_argument('-t', '--test_path', type=str, default=None, help='Path to the output file for the testing data')
    sd_parser.add_argument('--fraction_train', type=float, default=0.7, help='Size of the training data (default=0.7)')
    sd_parser.add_argument('--fraction_val', type=float, default=0.3, help='Size of the validation data (default=0.3)')
    sd_parser.add_argument('--fraction_test', type=float, default=0.0, help='Size of the testing data (default=0.0)')
    sd_parser.add_argument('--shuffle', action='store_true', default=False, help='Shuffle the datasets')
    sd_parser.add_argument('-s', '--seed', type=int, default=42, help='Seed used for shuffling')
    sd_parser.set_defaults(
        func=(
            lambda args: split_datasets(
                args.in_path,
                args.train_path,
                args.val_path,
                args.fraction_train,
                args.fraction_val,
                args.test_path,
                args.fraction_test,
                args.shuffle,
                args.seed,
            )
        )
    )

    # Subparser: normdict
    nd_parser = subparsers.add_parser('normdict', help='Create a normalization dictionary (norm_dict.yaml) for a given HDF5 file')
    nd_parser.add_argument('in_path', type=str, help='Path to the HDF5 input file')
    nd_parser.add_argument('-o', '--out_path', type=str, default='./norm_dict.yaml', help='Path to the output norm_dict.yaml (default=./norm_dict.yaml)')
    nd_parser.set_defaults(func=(lambda args: create_norm_dict(args.in_path, args.out_path)))

    # Subparser: filter-events
    fe_parser = subparsers.add_parser('filter-events', help='Filter out events with unphysical energy fractions')
    fe_parser.add_argument('in_path', type=str, help='Path to the input HDF5 file')
    fe_parser.add_argument('out_path', type=str, help='Path to the output filtered HDF5 file')
    fe_parser.add_argument(
        '-t', '--target_columns',
        nargs='+',
        type=str,
        default=None,
        help='Explicit list of target column names to check for physical energy fractions.'
    )
    fe_parser.add_argument('-p', '--prefix', type=str, default='clusterParticle_EnergyFraction_Full_',
                           help='Prefix for target columns (used if target_columns is not provided)')
    fe_parser.add_argument('-d', '--dataset', type=str, default='clusters',
                           help='Name of the dataset (default=clusters)')
    fe_parser.add_argument('--discard-unset', action='store_true', 
                           help='Discard events containing unset energy fractions (-1).')
    fe_parser.set_defaults(
        func=(lambda args: filter_events(args.in_path, args.out_path, args.target_columns, args.prefix, args.dataset, args.discard_unset))
    )

    return parser


def main():
    """Main entry point for the h5_prep command-line tool."""
    parser = get_parser()
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()