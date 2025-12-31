"""Core functionality tests."""

import pytest
import numpy as np
from nasa_sbm import explosion, collision


def test_explosion_basic():
    fragments = explosion(mass=100.0, cutoff=0.1, seed=42)
    
    assert len(fragments.fragment) > 0
    assert (fragments.fragment_mass > 0).all()
    assert (fragments.fragment_size >= 0.1).all()
    assert fragments.attrs['simulation_type'] == 'explosion'


def test_explosion_reproducible():
    frag1 = explosion(mass=100.0, cutoff=0.1, seed=42)
    frag2 = explosion(mass=100.0, cutoff=0.1, seed=42)
    
    assert len(frag1.fragment) == len(frag2.fragment)
    np.testing.assert_array_equal(frag1.fragment_mass.values, frag2.fragment_mass.values)


def test_explosion_satellite_types():
    sc_frags = explosion(mass=100.0, sat_type="spacecraft", cutoff=0.1, seed=42)
    rb_frags = explosion(mass=100.0, sat_type="rocket_body", cutoff=0.1, seed=42)
    
    assert len(sc_frags.fragment) > 0
    assert len(rb_frags.fragment) > 0
    assert sc_frags.attrs['satellite_type'] == 'spacecraft'
    assert rb_frags.attrs['satellite_type'] == 'rocket_body'


def test_collision_basic():
    fragments = collision(mass1=100.0, mass2=200.0, velocity=10.0, cutoff=0.1, seed=42)
    
    assert len(fragments.fragment) > 0
    assert (fragments.fragment_mass > 0).all()
    assert (fragments.fragment_size >= 0.1).all()
    assert fragments.attrs['simulation_type'] == 'collision'


def test_collision_reproducible():
    frag1 = collision(mass1=100.0, mass2=200.0, velocity=10.0, cutoff=0.1, seed=42)
    frag2 = collision(mass1=100.0, mass2=200.0, velocity=10.0, cutoff=0.1, seed=42)
    
    assert len(frag1.fragment) == len(frag2.fragment)
    np.testing.assert_array_equal(frag1.fragment_mass.values, frag2.fragment_mass.values)


def test_mass_conservation():
    fragments = explosion(mass=100.0, cutoff=0.1, seed=42, enforce_mass_conservation=True)
    total_mass = fragments.fragment_mass.sum().values
    
    assert total_mass <= 100.0
    assert fragments.attrs['enforce_mass_conservation'] == 1


def test_cutoff_respected():
    fragments = explosion(mass=100.0, cutoff=0.05, seed=42)
    
    assert (fragments.fragment_size >= 0.05).all()


def test_dataset_structure():
    fragments = explosion(mass=100.0, cutoff=0.1, seed=42)
    
    assert 'fragment' in fragments.dims
    assert 'xyz' in fragments.dims
    assert 'fragment_size' in fragments.data_vars
    assert 'fragment_mass' in fragments.data_vars
    assert 'delta_velocity' in fragments.data_vars
    assert fragments.delta_velocity.shape[1] == 3
