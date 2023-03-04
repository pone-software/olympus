"""Collection of classes implementing a detector."""
import numpy as np


def sample_direction(n_samples, rng=np.random.RandomState(1337)):
    """Sample uniform directions."""
    cos_theta = rng.uniform(-1, 1, size=n_samples)
    theta = np.arccos(cos_theta)
    phi = rng.uniform(0, 2 * np.pi)

    samples = np.empty((n_samples, 3))
    samples[:, 0] = np.sin(theta) * np.cos(phi)
    samples[:, 1] = np.sin(theta) * np.sin(phi)
    samples[:, 2] = np.cos(theta)

    return samples
