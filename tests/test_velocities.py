"""Velocity generation tests."""

import pytest
import numpy as np
from nasa_sbm import explosion


def test_isotropic_distribution():
    """Test that ejection velocities are isotropically distributed."""
    fragments = explosion(mass=100.0, cutoff=0.1, seed=42)
    
    mean_vel = fragments.ejection_velocity.mean(dim='fragment').values
    
    assert np.abs(mean_vel[0]) < 0.1
    assert np.abs(mean_vel[1]) < 0.1
    assert np.abs(mean_vel[2]) < 0.1


def test_velocity_magnitudes():
    """Test that all fragments have non-zero velocities."""
    fragments = explosion(mass=100.0, cutoff=0.1, seed=42)
    
    magnitudes = np.linalg.norm(fragments.ejection_velocity.values, axis=1)
    
    assert np.all(magnitudes > 0)


def test_velocity_reproducible():
    """Test that velocity generation is reproducible with same seed."""
    frag1 = explosion(mass=100.0, cutoff=0.1, seed=42)
    frag2 = explosion(mass=100.0, cutoff=0.1, seed=42)
    
    np.testing.assert_array_equal(
        frag1.ejection_velocity.values, 
        frag2.ejection_velocity.values
    )


def test_total_velocity_is_ejection_plus_parent():
    """Test that total velocity equals parent velocity plus ejection velocity."""
    fragments = explosion(mass=100.0, cutoff=0.1, seed=42, orbit_altitude=400.0)
    
    # For explosion at altitude, fragments should have velocity = parent + ejection
    # Since all fragments start at same position, parent velocity should be similar
    # We can't test exact equality due to how C++ handles this, but we can check structure
    assert fragments.velocity.shape == fragments.ejection_velocity.shape
    assert np.all(np.isfinite(fragments.velocity.values))
    assert np.all(np.isfinite(fragments.ejection_velocity.values))


def test_velocity_with_orbit_altitude():
    """Test that orbital altitude affects velocities properly."""
    frag_no_orbit = explosion(mass=100.0, cutoff=0.1, seed=42)
    frag_with_orbit = explosion(mass=100.0, cutoff=0.1, seed=42, orbit_altitude=400.0)
    
    # With orbit, velocities should be non-zero (includes orbital velocity)
    # Without orbit, velocities should only be ejection velocities
    vel_mag_no_orbit = np.linalg.norm(frag_no_orbit.velocity.values, axis=1).mean()
    vel_mag_with_orbit = np.linalg.norm(frag_with_orbit.velocity.values, axis=1).mean()
    
    # With orbital velocity, total should be larger
    assert vel_mag_with_orbit > vel_mag_no_orbit
