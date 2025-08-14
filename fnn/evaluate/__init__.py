from functools import reduce
from operator import getitem
import numpy as np
from scipy import stats
from tqdm import tqdm

def pad_responses(responses):
    """
    concatenate the arrays, fill in NaNs for uneven repeats and clip lengths
    """
    len_repeats = []
    len_samples = []
    for repeats in responses:
        len_repeats.append(len(repeats))
        for samples in repeats:
            len_samples.append(len(samples))
    max_repeats = np.max(len_repeats)
    max_samples = np.max(len_samples)

    repeats_padded = []
    for repeats in responses:
        samples_padded = []
        for samples in repeats:
            samples_padded.append(
                np.pad(
                    samples, 
                    ((0, max_samples - len(samples)), (0, 0)), 
                     mode='constant', 
                     constant_values=np.nan
                )
            )
        repeats_padded.append(
            np.pad(
                samples_padded,
                ((0, max_repeats - len(repeats)), (0, 0), (0, 0)),
                mode="constant",
                constant_values=np.nan
            )
        )
    return np.stack(repeats_padded)

def format_responses(responses, burnin_frames=0, pad=True):
    if pad:
        responses = pad_responses(responses)
    responses = responses.transpose(3, 1, 0, 2)
    responses = responses[..., burnin_frames:]
    return responses.reshape(responses.shape[0], responses.shape[1], -1)


def compute_cc_max_unit(unit_responses: np.ndarray):
    """
    Compute the upper bound of signal correlation for a single unit.
    """
    x = unit_responses
    
    # number of trials per sample
    trials, samples = x.shape
    t = trials - np.isnan(x).sum(axis=0)

    # pooled variance -> n
    v = 1 / t**2
    w = t - 1
    z = t.sum() - samples
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
    return np.array([compute_cc_max_unit(arr) for arr in responses])

def compute_cc_abs_unit(unit_predictions, unit_responses):
    return stats.pearsonr(
            np.nanmean(unit_predictions, axis=0), 
            np.nanmean(unit_responses, axis=0)
        )[0] 

def compute_cc_abs(predictions: np.ndarray, responses: np.ndarray):
    return np.array([
        compute_cc_abs_unit(pred, resp) for pred, resp in zip(predictions, responses)
    ])
