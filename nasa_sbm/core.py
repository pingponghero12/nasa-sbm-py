"""Core Python API for NASA Standard Breakup Model."""

import numpy as np
import xarray as xr
from datetime import datetime
from typing import Optional, Literal

try:
    from nasa_sbm import _core
except ImportError:
    # During development, try relative import
    from . import _core


def explosion(
    mass: float,
    sat_type: Literal["spacecraft", "rocket_body"] = "spacecraft",
    cutoff: float = 0.01,
    seed: Optional[int] = None,
    enforce_mass_conservation: bool = False,
) -> xr.Dataset:
    """
    Generate debris cloud from satellite explosion. 
    
    Parameters
    ----------
    mass : float
        Parent satellite mass in kg
    sat_type : str
        Satellite type:  'spacecraft' or 'rocket_body'
    cutoff : float
        Minimum fragment characteristic length in meters
    seed : int, optional
        Random seed for reproducibility
    enforce_mass_conservation :  bool
        Whether to enforce mass conservation
        
    Returns
    -------
    xr.Dataset
        Fragment data including sizes, masses, velocities
    """
    # Create simulation
    sim = _core.BreakupSimulation()
    
    # Run explosion
    sim.run_explosion(
        mass=mass,
        sat_type=sat_type,
        min_lc=cutoff,
        seed=seed,
        enforce_mass_conservation=enforce_mass_conservation
    )
    
    # Get fragments
    fragments = sim.get_fragments()
    n_fragments = sim.get_fragment_count()
    
    # Generate isotropic delta-velocity directions
    delta_v = _generate_isotropic_velocities(fragments["velocity"], seed)
    
    # Create xarray Dataset
    coords = {
        'fragment': np.arange(n_fragments),
        'xyz': ['x', 'y', 'z']
    }
    
    data_vars = {
        'fragment_size': (['fragment'], fragments['characteristic_length']),
        'area_to_mass_ratio': (['fragment'], fragments['area_to_mass_ratio']),
        'cross_sectional_area': (['fragment'], fragments['area']),
        'fragment_mass': (['fragment'], fragments['mass']),
        'initial_position': (['fragment', 'xyz'], fragments['position']),
        'delta_velocity': (['fragment', 'xyz'], delta_v),
        'parent_velocity': (['fragment', 'xyz'], fragments['velocity']),
    }
    
    attrs = {
        'title': 'NASA SBM Explosion Debris Cloud',
        'simulation_type': 'explosion',
        'parent_mass_kg': mass,
        'satellite_type': sat_type,
        'size_cutoff_m': cutoff,
        'n_fragments': n_fragments,
        'creation_time': datetime.utcnow().isoformat(),
        'enforce_mass_conservation': int(enforce_mass_conservation),
        'random_seed': seed if seed is not None else -1,
    }
    
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def collision(
    mass1: float,
    mass2: float,
    velocity: float,
    sat_type1: Literal["spacecraft", "rocket_body"] = "spacecraft",
    sat_type2: Literal["spacecraft", "rocket_body"] = "spacecraft",
    cutoff: float = 0.01,
    seed: Optional[int] = None,
    enforce_mass_conservation: bool = False,
) -> xr.Dataset:
    """
    Generate debris cloud from satellite collision.
    
    Parameters
    ----------
    mass1 : float
        First satellite mass in kg
    mass2 : float
        Second satellite mass in kg
    velocity : float
        Relative collision velocity in km/s
    sat_type1 : str
        First satellite type: 'spacecraft' or 'rocket_body'
    sat_type2 : str
        Second satellite type: 'spacecraft' or 'rocket_body'
    cutoff : float
        Minimum fragment characteristic length in meters
    seed : int, optional
        Random seed for reproducibility
    enforce_mass_conservation : bool
        Whether to enforce mass conservation
        
    Returns
    -------
    xr.Dataset
        Fragment data including sizes, masses, velocities
    """
    # Create simulation
    sim = _core.BreakupSimulation()
    
    # Run collision
    sim.run_collision(
        mass1=mass1,
        mass2=mass2,
        velocity=velocity,
        sat_type1=sat_type1,
        sat_type2=sat_type2,
        min_lc=cutoff,
        seed=seed,
        enforce_mass_conservation=enforce_mass_conservation
    )
    
    # Get fragments
    fragments = sim.get_fragments()
    n_fragments = sim.get_fragment_count()
    
    # Generate isotropic delta-velocity directions
    delta_v = _generate_isotropic_velocities(fragments["velocity"], seed)
    
    # Create xarray Dataset
    coords = {
        'fragment': np.arange(n_fragments),
        'xyz': ['x', 'y', 'z']
    }
    
    data_vars = {
        'fragment_size': (['fragment'], fragments['characteristic_length']),
        'area_to_mass_ratio': (['fragment'], fragments['area_to_mass_ratio']),
        'cross_sectional_area': (['fragment'], fragments['area']),
        'fragment_mass': (['fragment'], fragments['mass']),
        'initial_position': (['fragment', 'xyz'], fragments['position']),
        'delta_velocity': (['fragment', 'xyz'], delta_v),
        'parent_velocity': (['fragment', 'xyz'], fragments['velocity']),
    }
    
    attrs = {
        'title':  'NASA SBM Collision Debris Cloud',
        'simulation_type': 'collision',
        'mass1_kg': mass1,
        'mass2_kg': mass2,
        'relative_velocity_km_s': velocity,
        'satellite_type1': sat_type1,
        'satellite_type2': sat_type2,
        'size_cutoff_m': cutoff,
        'n_fragments': n_fragments,
        'creation_time': datetime.utcnow().isoformat(),
        'enforce_mass_conservation': int(enforce_mass_conservation),
        'random_seed': seed if seed is not None else -1,
    }
    
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def _generate_isotropic_velocities(parent_velocities: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate isotropic velocity directions on unit sphere.
    
    This replaces the parent velocities with isotropic delta-v vectors
    while keeping the magnitudes from the C++ calculation.
    """
    rng = np.random.default_rng(seed)
    n = parent_velocities.shape[0]
    
    # Calculate magnitudes from parent velocities
    magnitudes = np.linalg.norm(parent_velocities, axis=1)
    
    # Generate isotropic directions
    phi = rng.uniform(0, 2 * np.pi, n)
    cos_theta = rng.uniform(-1, 1, n)
    sin_theta = np.sqrt(1 - cos_theta**2)
    
    # Create direction vectors
    directions = np.column_stack([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ])
    
    # Scale by magnitudes
    return directions * magnitudes[: , np.newaxis]
