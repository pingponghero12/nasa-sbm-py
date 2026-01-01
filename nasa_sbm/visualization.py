"""Visualization tools for NASA Standard Breakup Model."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import xarray as xr
from typing import Optional, Tuple


def plot_gabbard_diagram(
    fragments:  xr.Dataset,
    orbit_altitude: float = 400.0,  # Add orbit altitude parameter
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Plot Gabbard diagram (altitude vs velocity)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate velocity magnitude
    velocities = np.linalg.norm(fragments.delta_velocity.values, axis=1)
    
    # Calculate radial velocity component (altitude change rate)
    # For simplicity, use Z-component as proxy for radial direction
    radial_velocities = fragments.delta_velocity.values[:, 2]
    
    # Estimate altitude change after some time (e.g., 1 hour = 3600 seconds)
    time_step = 3600  # seconds
    altitude_changes = radial_velocities * time_step  # km
    altitudes = orbit_altitude + altitude_changes
    
    masses = fragments.fragment_mass.values
    sizes = np.sqrt(masses) * 10
    
    scatter = ax.scatter(
        velocities,
        altitudes,
        s=sizes,
        c=masses,
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    
    ax.set_xlabel('Delta-V Magnitude (km/s)', fontsize=12)
    ax.set_ylabel('Altitude (km)', fontsize=12)
    ax.set_title(title or 'Gabbard Diagram', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add reference line for original orbit
    ax.axhline(orbit_altitude, color='r', linestyle='--', 
               label=f'Original orbit:  {orbit_altitude} km', alpha=0.5)
    ax.legend()
    
    cbar = plt.colorbar(scatter, ax=ax, label='Fragment Mass (kg)')
    
    n_frags = len(fragments.fragment)
    total_mass = fragments.fragment_mass.sum().values
    
    ax.text(
        0.02, 0.98,
        f'Fragments: {n_frags}\nTotal Mass: {total_mass:.2f} kg',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=10
    )
    
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
    
    velocities = fragments.delta_velocity.values
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
