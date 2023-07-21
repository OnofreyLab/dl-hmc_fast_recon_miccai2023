import pandas as pd
import numpy as np



def concatenate_vicra(
    df, 
    vicra_keys=['VC_11', 'VC_12', 'VC_13', 'VC_14', 'VC_21', 'VC_22','VC_23', 'VC_24', 'VC_31', 'VC_32', 'VC_33', 'VC_34'],
):
    """Concatenate all vicra parameters into a single 12x1 array.

    """
    V = df[vicra_keys].values.tolist()
    return V


def split_dataset(
    df, 
    num_subsample=None, 
    test_size_percent = None,
    validation_size_percent = None,
    random_state=42,
):

    df_sample = df.copy()
    if num_subsample is not None:
        df_sample = df_sample.sample(n=num_subsample, random_state=random_state)
    df_sample = df_sample.sort_index()

    n = len(df_sample)
    X = list(np.arange(n))

    train_idx = []
    val_idx = []
    test_idx = []

    # Set the numpy rng
    np.random.seed(seed=random_state)

    if test_size_percent is not None and test_size_percent > 0.0:
        n_test = int(np.floor(n*test_size_percent))
        test_idx = np.random.choice(X, size=n_test, replace=False)
        X = list(set(X) - set(test_idx))
    if validation_size_percent is not None and validation_size_percent > 0.0:
        n_val = int(np.floor(n*validation_size_percent))
        val_idx = np.random.choice(X, size=n_val, replace=False)
        X = list(set(X) - set(val_idx))

    train_idx = X

    df_train = df_sample.iloc[np.sort(train_idx)].copy()
    df_val = df_sample.iloc[np.sort(val_idx)].copy()
    df_test = df_sample.iloc[np.sort(test_idx)].copy()

    return df_train, df_val, df_test