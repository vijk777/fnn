from functools import reduce
from operator import getitem
import numpy as np
from scipy import stats
from tqdm import tqdm

def pad_responses(responses: list):
    """
    Concatenate the arrays, fill in NaNs for data with uneven repeats

    Parameters
    ----------
    responses : list of lists of arrays (n_video x n_repeats x n_samples x n_units)
        len(responses) = n_video
        len(responses[i]) = n_repeats
        responses[i][j].shape = n_samples x n_units 

    Returns
    -------
    responses : np.ndarray
        Shape: n_video x n_repeats x n_samples x n_units
    """
    max_repeats = max([len(repeats) for repeats in responses])
    return np.array([
        np.pad(
                repeats,
                ((0, max_repeats - len(repeats)), (0, 0), (0, 0)),
                mode="constant",
                constant_values=np.nan
            ) for repeats in responses
    ]) # n_video x n_repeats x n_samples x n_units


def format_responses(responses, burnin_frames=0, pad=True):
    """
    Removes burn-in frames, optionally pads, and reshapes the responses.

    Parameters
    ----------
    responses : list of lists of arrays (n_video x n_repeats x n_samples x n_units)
        len(responses) = n_video
        len(responses[i]) = n_repeats
        responses[i][j].shape = n_samples x n_units 
    burnin_frames : int
        Number of frames to remove from the beginning of each response.
    pad : bool
        If True, pads the responses to have the same number of repeats.

    Returns
    -------
    responses : np.ndarray
        Shape: n_units x n_repeats x n_samples_concat
            where n_samples_concat = (n_samples - burnin_frames) * n_video
    """
    if pad:
        responses = pad_responses(responses)
    responses = responses.transpose(3, 1, 0, 2) # n_units x n_repeats x n_video x n_samples
    responses = responses[..., burnin_frames:] # n_units x n_repeats x n_video x (n_samples - burnin_frames)
    return responses.reshape(responses.shape[0], responses.shape[1], -1) # n_units x n_repeats x (n_samples - burnin_frames) * n_video


def compute_cc_max_unit(unit_responses: np.ndarray):
    """
    Compute the upper bound of signal correlation for a single unit (CC_max).

    Parameters
    ----------
    unit_responses : np.ndarray
        Shape: n_repeats x n_samples_concat
            where n_samples_concat = (n_samples - burnin_frames) * n_video
    Returns
    -------
    float
        CC_max value for the unit.
    """
    x = unit_responses
    
    # number of repeats per sample
    n_repeats, n_samples_concat = x.shape
    t = n_repeats - np.isnan(x).sum(axis=0)

    # pooled variance -> n
    v = 1 / t**2
    w = t - 1
    z = t.sum() - n_samples_concat
    n = np.sqrt(z / (w * v).sum())

    # response mean
    y_m = np.nanmean(x, axis=0)

    # signal power
    P = np.var(y_m, axis=0, ddof=1)
    TP = np.mean(np.nanvar(x, axis=1, ddof=1), axis=0)
    SP = (n * P - TP) / (n - 1)

    # variance of response mean
    y_m_v = np.var(y_m, axis=0, ddof=0)

    # correlation coefficient ceiling
    return np.sqrt(SP / y_m_v)


def compute_cc_max(responses: np.ndarray):
    """
    Compute the upper bound of signal correlation (CC_max) for all units.
    Parameters
    ----------
    responses : np.ndarray
        Shape: n_units x n_repeats x n_samples_concat
            where n_samples_concat = (n_samples - burnin_frames) * n_video  
    Returns
    -------
    np.ndarray
        CC_max values for each unit.
    """
    return np.array([compute_cc_max_unit(arr) for arr in responses])


def compute_cc_abs_unit(unit_predictions, unit_responses):
    """
    Compute model test correlation (CC_abs) for a single unit.
    
    Parameters
    ----------
    unit_predictions : np.ndarray
        Shape: n_repeats x n_samples_concat
            where n_samples_concat = (n_samples - burnin_frames) * n_video
    unit_responses : np.ndarray
        Shape: n_repeats x n_samples_concat
            where n_samples_concat = (n_samples - burnin_frames) * n_video
    Returns
    -------
    float
        CC_abs for the unit.
    """
    return stats.pearsonr(
            np.nanmean(unit_predictions, axis=0), 
            np.nanmean(unit_responses, axis=0)
        )[0] 


def compute_cc_abs(predictions: np.ndarray, responses: np.ndarray):
    """
    Compute model test correlation (CC_abs) for all units.

    Parameters
    ----------
    predictions : np.ndarray
        Shape: n_units x n_repeats x n_samples_concat
            where n_samples_concat = (n_samples - burnin_frames) * n_video
    responses : np.ndarray
        Shape: n_units x n_repeats x n_samples_concat
            where n_samples_concat = (n_samples - burnin_frames) * n_video
    Returns
    -------
    np.ndarray
        CC_abs values for each unit.
    """
    return np.array([
        compute_cc_abs_unit(pred, resp) for pred, resp in zip(predictions, responses)
    ])
