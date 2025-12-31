"""Velocity generation tests."""

import pytest
import numpy as np
from nasa_sbm import explosion
from nasa_sbm.core import generate_isotropic_velocities


def test_isotropic_distribution():
    fragments = explosion(mass=100.0, cutoff=0.1, seed=42)
    
    mean_vel = fragments.delta_velocity.mean(dim='fragment').values
    
    assert np.abs(mean_vel[0]) < 0.1
    assert np.abs(mean_vel[1]) < 0.1
    assert np.abs(mean_vel[2]) < 0.1


def test_velocity_magnitudes():
    fragments = explosion(mass=100.0, cutoff=0.1, seed=42)
    
    magnitudes = np.linalg.norm(fragments.delta_velocity. values, axis=1)
    
    assert np.all(magnitudes > 0)


def test_velocity_reproducible():
    frag1 = explosion(mass=100.0, cutoff=0.1, seed=42)
    frag2 = explosion(mass=100.0, cutoff=0.1, seed=42)
    
    np.testing.assert_array_equal(
        frag1.delta_velocity.values, 
        frag2.delta_velocity.values
    )


def test_generate_isotropic_velocities_shape():
    parent_vels = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    
    result = generate_isotropic_velocities(parent_vels, seed=42)
    
    assert result.shape == (3, 3)
    
    expected_mags = np.array([1.0, 2.0, 3.0])
    actual_mags = np.linalg.norm(result, axis=1)
    
    np.testing.assert_array_almost_equal(actual_mags, expected_mags)
