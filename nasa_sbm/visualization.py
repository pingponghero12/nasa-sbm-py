"""Visualization tools for NASA Standard Breakup Model."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import xarray as xr
from typing import Optional, Tuple

try:
    from poliastro.core.elements import rv2coe
    POLIASTRO_AVAILABLE = True
except ImportError:
    POLIASTRO_AVAILABLE = False

def plot_gabbard_diagram(
    fragments:  xr.Dataset,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show:  bool = True
) -> plt.Figure:
    """
    Plot Gabbard diagram (altitude vs orbital period).
    
    Shows apogee (blue) and perigee (red) altitudes vs orbital period for debris fragments.
    Uses parent orbit + ejection velocity to calculate perturbed orbits. 
    """
    if not POLIASTRO_AVAILABLE: 
        raise ImportError("poliastro is required for Gabbard diagrams.  Install with: pip install poliastro")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Constants
    mu = 398600.4418e9  # Earth gravitational parameter (m^3/s^2)
    R_earth = 6378137.0  # Earth radius (m)
    
    # Get parent orbit data
    if 'orbit_altitude_km' not in fragments.attrs:
        ax. text(0.5, 0.5, 'No orbital data available\n(use orbit_altitude parameter)',
                ha='center', va='center', transform=ax. transAxes, fontsize=12)
        plt.tight_layout()
        return fig
    
    parent_alt = fragments.attrs['orbit_altitude_km']
    r_parent = R_earth + parent_alt * 1000.0  # Parent orbital radius in m
    v_parent = np.sqrt(mu / r_parent)  # Parent circular velocity in m/s
    
    # Parent position and velocity (circular orbit)
    r_vec_parent = np.array([r_parent, 0.0, 0.0])
    v_vec_parent = np.array([0.0, v_parent, 0.0])
    
    # Get fragment data
    ejection_velocities = fragments.ejection_velocity.values * 1000.0  # km/s to m/s
    masses = fragments.fragment_mass.values
    
    n_fragments = len(fragments. fragment)
    
    # Calculate orbital elements for each fragment
    apogees = []
    perigees = []
    periods = []
    valid_masses = []
    
    for i in range(n_fragments):
        try:
            # Fragment velocity = parent velocity + ejection velocity
            v_fragment = v_vec_parent + ejection_velocities[i]
            
            # Calculate orbital elements using poliastro
            # Fragment starts at parent position with perturbed velocity
            p, ecc, inc, raan, argp, nu = rv2coe(mu, r_vec_parent, v_fragment)
            
            # Semi-major axis
            a = p / (1 - ecc**2)
            
            # Skip hyperbolic/parabolic orbits
            if ecc >= 1.0 or a <= 0:
                continue
            
            # Clamp eccentricity for numerical stability
            ecc = min(ecc, 0.99)
            
            # Apogee and perigee radii
            r_apogee = a * (1 + ecc)
            r_perigee = a * (1 - ecc)
            
            # Convert to altitudes (m to km)
            h_apogee = (r_apogee - R_earth) / 1000.0
            h_perigee = (r_perigee - R_earth) / 1000.0
            
            # Skip re-entry orbits (perigee below 100 km)
            if h_perigee < 100:
                continue
            
            # Orbital period (minutes)
            period = 2 * np.pi * np.sqrt(a**3 / mu) / 60.0  # minutes
            
            apogees. append(h_apogee)
            perigees.append(h_perigee)
            periods.append(period)
            valid_masses.append(masses[i])
            
        except Exception: 
            # Skip fragments with calculation issues
            continue
    
    if len(periods) == 0:
        ax.text(0.5, 0.5, 'No valid orbits calculated\n(all fragments may have re-entered or escaped)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        plt.tight_layout()
        return fig
    
    apogees = np.array(apogees)
    perigees = np. array(perigees)
    periods = np.array(periods)
    valid_masses = np. array(valid_masses)
    
    # Calculate axis limits using 5-95 percentiles
    period_min = np.percentile(periods, 5)
    period_max = np.percentile(periods, 95)
    alt_min = np.percentile(np.concatenate([apogees, perigees]), 5)
    alt_max = np.percentile(np.concatenate([apogees, perigees]), 95)
    
    # Add 5% margin to limits
    period_margin = (period_max - period_min) * 0.02
    alt_margin = (alt_max - alt_min) * 0.02
    
    # Plot apogees (blue dots)
    sizes = np.sqrt(valid_masses) * 10
    
    scatter_apo = ax. scatter(
        periods,
        apogees,
        s=sizes,
        c='blue',
        alpha=0.6,
        label='Apogee',
        marker='o'
    )
    
    # Plot perigees (red dots)
    scatter_per = ax.scatter(
        periods,
        perigees,
        s=sizes,
        c='red',
        alpha=0.6,
        marker='o',
        label='Perigee'
    )

    # Set axis limits to 5-95 percentile range
    ax.set_xlim(period_min - period_margin, period_max + period_margin)
    ax.set_ylim(alt_min - alt_margin, alt_max + alt_margin)

    ax.set_xlabel('Orbital Period (minutes)', fontsize=12)
    ax.set_ylabel('Altitude (km)', fontsize=12)
    ax.set_title(title or 'Gabbard Diagram', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    n_total = len(fragments.fragment)
    n_valid = len(periods)
    total_mass = fragments.fragment_mass.sum().values

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig

def plot_velocity_field_3d(
    fragments: xr.Dataset,
    max_fragments: int = 500,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot 3D velocity field with colored lines and mass-scaled endpoints.
    
    Shows fragment trajectories as lines with colors representing velocity magnitude
    and endpoint sizes representing fragment mass.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use ejection_velocity for the 3D field visualization
    velocities = fragments.ejection_velocity.values
    masses = fragments.fragment_mass.values
    
    n_frags = len(fragments.fragment)
    if n_frags > max_fragments:
        indices = np.random.choice(n_frags, max_fragments, replace=False)
        velocities = velocities[indices]
        masses = masses[indices]
        n_display = max_fragments
    else: 
        n_display = n_frags
    
    origins = np.zeros((n_display, 3))
    
    v_magnitudes = np.linalg.norm(velocities, axis=1)
    
    for i in range(n_display):
        points = np.array([origins[i], velocities[i]])
        
        line = ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=plt.cm.plasma(v_magnitudes[i] / v_magnitudes.max()),
            alpha=0.6,
            linewidth=1.5
        )[0]
    
    endpoint_sizes = np.sqrt(masses) * 50
    
    scatter = ax.scatter(
        velocities[:, 0],
        velocities[:, 1],
        velocities[:, 2],
        s=endpoint_sizes,
        c=v_magnitudes,
        cmap='plasma',
        alpha=0.8,
        edgecolors='black',
        linewidth=0.5
    )
    
    ax.scatter([0], [0], [0], color='red', s=200, marker='*', edgecolors='black', linewidth=2)
    
    ax.set_xlabel('Vx (km/s)', fontsize=11)
    ax.set_ylabel('Vy (km/s)', fontsize=11)
    ax.set_zlabel('Vz (km/s)', fontsize=11)
    ax.set_title(title or '3D Velocity Field', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, label='Velocity Magnitude (km/s)')
    
    max_range = np.abs(velocities).max()
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    ax.view_init(elev=20, azim=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show: 
        plt.show()
    
    return fig


def plot_size_distribution(
    fragments: xr.Dataset,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show:  bool = True
) -> plt.Figure:
    """Plot fragment size distribution (cumulative)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sizes = fragments.fragment_size.values
    sizes_sorted = np.sort(sizes)
    
    n_total = len(sizes)
    cumulative = np.arange(n_total, 0, -1)
    
    ax.loglog(sizes_sorted, cumulative, 'b-', linewidth=2)
    
    ax.set_xlabel('Characteristic Length (m)', fontsize=12)
    ax.set_ylabel('N(L >= Lc)', fontsize=12)
    ax.set_title(title or 'Cumulative Size Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    cutoff = fragments.attrs.get('size_cutoff_m', 0.01)
    ax.axvline(cutoff, color='r', linestyle='--', label=f'Cutoff: {cutoff} m')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show: 
        plt.show()
    
    return fig


def visualize_all(
    fragments: xr.Dataset,
    output_dir: Optional[str] = None,
    show: bool = True
) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Generate all visualization plots."""
    sim_type = fragments.attrs.get('simulation_type', 'breakup')
    
    gabbard_path = f"{output_dir}/gabbard.png" if output_dir else None
    velocity_path = f"{output_dir}/velocity_3d.png" if output_dir else None
    size_path = f"{output_dir}/size_dist.png" if output_dir else None
    
    fig1 = plot_gabbard_diagram(
        fragments,
        title=f'Gabbard Diagram - {sim_type.capitalize()}',
        save_path=gabbard_path,
        show=show
    )
    
    fig2 = plot_velocity_field_3d(
        fragments,
        title=f'3D Velocity Field - {sim_type.capitalize()}',
        save_path=velocity_path,
        show=show
    )
    
    fig3 = plot_size_distribution(
        fragments,
        title=f'Size Distribution - {sim_type.capitalize()}',
        save_path=size_path,
        show=show
    )
    
    return fig1, fig2, fig3
