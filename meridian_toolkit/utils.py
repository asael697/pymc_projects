import numpy as np
import pandas as pd

def create_media_mapping(media_list, suffix='_spend'):
    """
    Create a mapping dictionary from media list with suffix removed.
    
    Parameters
    ----------
    media_list : list
        List of media channel names (e.g., ['meta_spend', 'google_spend'])
    suffix : str, default '_spend'
        Suffix to remove from channel names
    
    Returns
    -------
    dict
        Mapping dictionary with suffix removed
    
    Examples
    --------
    >>> media = ['meta_spend', 'google_spend']
    >>> create_media_mapping(media)
    {'meta_spend': 'meta', 'google_spend': 'google'}
    """
    return {
        channel: channel.replace(suffix, '')
        for channel in media_list
    }

def compute_holdout_id(data_real:pd.DataFrame, n_test:int = 12):
    """
    Create a holdout mask for the last n_test observations.
    
    Parameters
    ----------
    data_real : pd.DataFrame
        Dataset with temporal observations
    n_test : int
        Number of observations to hold out (from the end)
    
    Returns
    -------
    np.ndarray
        Boolean array where True indicates holdout observations
    
    Examples
    --------
    >>> holdout_id = compute_holdout_id(data, n_test=12)
    >>> print(f"Train: {np.sum(~holdout_id)}, Test: {np.sum(holdout_id)}")
    """
    n_times = len(data_real)
    
    # Create array of False
    holdout_id = np.full(n_times, False)
    
    # Mark last n_test as True
    holdout_id[-n_test:] = True
    
    return holdout_id