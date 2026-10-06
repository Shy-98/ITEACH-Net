"""The IEMOCAP host/guest mask generation used by the training and UME paths."""

import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder
import torch


def generate_inputs(audio_host, text_host, visual_host,
                    audio_guest, text_guest, visual_guest, qmask):
    host = torch.cat([audio_host, text_host, visual_host], dim=2)
    guest = torch.cat([audio_guest, text_guest, visual_guest], dim=2)
    featdim = host.size(-1)
    speaker_mask = qmask.transpose(0, 1).unsqueeze(2).repeat(1, 1, featdim)
    return [torch.where(speaker_mask == 0, host, guest)]


def random_mask(view_num: int, input_len: int, missing_rate: float):
    """Reference retry sampler for fixed-rate masks, including its padded-cell policy."""
    assert missing_rate is not None
    one_rate = 1 - missing_rate

    if one_rate <= (1 / view_num):
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        return enc.fit_transform(randint(0, view_num, size=(input_len, 1))).toarray()

    if one_rate == 1:
        return randint(1, 2, size=(input_len, view_num))

    alldata_len = 32 if input_len < 32 else input_len
    error = 1
    while error >= 0.005:
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(
            randint(0, view_num, size=(alldata_len, 1))).toarray()
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num))
                       < int(ratio * 100)).astype(int)
        overlap = np.sum(((matrix_iter + view_preserve) > 1).astype(int))
        one_num_iter = one_num / (1 - overlap / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num))
                       < int(ratio * 100)).astype(int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    return matrix[:input_len, :]


def continuous_random_mask(view_num: int, input_len: int, missing_rate: float):
    """Finite-count Random training sampler, separate from fixed-rate eval masks."""
    matrix = np.zeros((input_len, view_num), dtype=np.int64)
    matrix[np.arange(input_len), np.random.randint(view_num, size=input_len)] = 1
    retained = min(max(round(input_len * view_num * (1 - missing_rate)), input_len),
                   input_len * view_num)
    available = np.flatnonzero(matrix.ravel() == 0)
    chosen = np.random.choice(available, size=retained - input_len, replace=False)
    matrix.ravel()[chosen] = 1
    return matrix


def select_training_mask_rate(strategy: str, epoch_index: int, fixed_rate: float) -> float:
    if strategy == "random":
        return float(np.random.uniform(0.0, 0.7))
    if strategy == "progressive":
        return min(0.1 * (epoch_index // 10), 0.7)
    return fixed_rate
