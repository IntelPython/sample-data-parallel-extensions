import numpy as np
import dpctl

from kde_setuptools import pybind11_kde_eval as kde_eval

import pytest

def ref_kde(x, data, h):
    """
    Reference NumPy implementation for KDE evaluation
    """
    assert x.ndim == 2 and data.ndim == 2
    assert x.shape[1] == data.shape[1]
    dim = x.shape[1]
    n_data = data.shape[0]
    return np.exp(
        np.square(x[:, np.newaxis, :]-data).sum(axis=-1)/(-2*h*h)
    ).sum(axis=1)/(np.sqrt(2*np.pi)*h)**dim / n_data


def test_1d():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Execution queue could not be created, skipping...")

    x = np.linspace(0.1, 0.9, num=15).reshape((-1, 1))
    data = np.random.rand(128,1)

    f_cy = kde_eval(q, x, data, 0.1)
    f_ref = ref_kde(x, data, 0.1)

    assert np.allclose(f_cy, f_ref)


def test_2d():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Execution queue could not be created, skipping...")

    x = np.dstack(
        np.meshgrid(
            np.linspace(0.1, 0.9, num=7),
            np.linspace(0.1, 0.9, num=7)
        )
    ).reshape(-1, 2)

    data = np.random.rand(128*128, 2)

    f_cy = kde_eval(q, x, data, 0.05)
    f_ref = ref_kde(x, data, 0.05)

    assert np.allclose(f_cy, f_ref)


def test_3d():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Execution queue could not be created, skipping...")

    x = np.dstack(
        np.meshgrid(
            np.linspace(0.1, 0.9, num=7),
            np.linspace(0.1, 0.9, num=7),
            np.linspace(0.1, 0.9, num=7)
        )
    ).reshape(-1, 3)

    data = np.random.rand(16000,3)

    f_cy = kde_eval(q, x, data, 0.01)
    f_ref = ref_kde(x, data, 0.01)

    assert np.allclose(f_cy, f_ref)
