"""Command-line interface for NASA Standard Breakup Model."""

import argparse
import sys
from nasa_sbm import explosion, collision
from nasa_sbm.visualization import visualize_all


def main():
    parser = argparse.ArgumentParser(
        description="NASA Standard Breakup Model - Generate debris clouds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:  
  nasa-sbm explosion --mass 839 --sat-type rocket_body --cutoff 0.05 --out debris.nc
  nasa-sbm collision --mass1 560 --mass2 950 --velocity 11.7 --out debris.nc
  nasa-sbm explosion --mass 839 --cutoff 0.05 --out debris.nc --vis
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Simulation type')
    
    # Explosion subcommand
    exp_parser = subparsers.add_parser('explosion', help='Satellite explosion')
    exp_parser.add_argument('--mass', type=float, required=True, help='Satellite mass (kg)')
    exp_parser.add_argument('--sat-type', type=str, default='spacecraft',
                           choices=['spacecraft', 'rocket_body'],
                           help='Satellite type')
    exp_parser.add_argument('--cutoff', type=float, default=0.01,
                           help='Minimum fragment size (m)')
    exp_parser.add_argument('--out', type=str, required=True,
                           help='Output NetCDF file')
    exp_parser.add_argument('--seed', type=int, help='Random seed')
    exp_parser.add_argument('--enforce-mass-conservation', action='store_true',
                           help='Enforce mass conservation')
    exp_parser.add_argument('--orbit-altitude', type=float,
                           help='Orbital altitude (km) for circular orbit')
    exp_parser.add_argument('--vis', action='store_true',
                           help='Generate visualization plots')
    
    # Collision subcommand
    col_parser = subparsers.add_parser('collision', help='Satellite collision')
    col_parser.add_argument('--mass1', type=float, required=True, help='First satellite mass (kg)')
    col_parser.add_argument('--mass2', type=float, required=True, help='Second satellite mass (kg)')
    col_parser.add_argument('--velocity', type=float, required=True,
                           help='Relative velocity (km/s)')
    col_parser.add_argument('--sat-type1', type=str, default='spacecraft',
                           choices=['spacecraft', 'rocket_body'],
                           help='First satellite type')
    col_parser.add_argument('--sat-type2', type=str, default='spacecraft',
                           choices=['spacecraft', 'rocket_body'],
                           help='Second satellite type')
    col_parser.add_argument('--cutoff', type=float, default=0.01,
                           help='Minimum fragment size (m)')
    col_parser.add_argument('--out', type=str, required=True,
                           help='Output NetCDF file')
    col_parser.add_argument('--seed', type=int, help='Random seed')
    col_parser.add_argument('--enforce-mass-conservation', action='store_true',
                           help='Enforce mass conservation')
    col_parser.add_argument('--orbit-altitude', type=float,
                           help='Orbital altitude (km) for circular orbit')
    col_parser.add_argument('--vis', action='store_true',
                           help='Generate visualization plots')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'explosion':  
        print(f"Running explosion simulation...")
        print(f"  Mass: {args.mass} kg")
        print(f"  Type: {args.sat_type}")
        print(f"  Cutoff: {args.cutoff} m")
        if args.orbit_altitude:
            print(f"  Orbit altitude: {args.orbit_altitude} km")
        
        fragments = explosion(
            mass=args.mass,
            sat_type=args.sat_type,
            cutoff=args.cutoff,
            seed=args.seed,
            enforce_mass_conservation=args.enforce_mass_conservation,
            orbit_altitude=args.orbit_altitude
        )
        
        print(f"Generated {len(fragments.fragment)} fragments")
        fragments.to_netcdf(args.out)
        print(f"Saved to {args.out}")
        
        if args.vis:
            print("Generating visualizations...")
            visualize_all(fragments, show=True)
        
    elif args.command == 'collision':
        print(f"Running collision simulation...")
        print(f"  Mass 1: {args.mass1} kg ({args.sat_type1})")
        print(f"  Mass 2: {args.mass2} kg ({args.sat_type2})")
        print(f"  Velocity: {args.velocity} km/s")
        print(f"  Cutoff: {args.cutoff} m")
        if args.orbit_altitude:
            print(f"  Orbit altitude: {args.orbit_altitude} km")
        
        fragments = collision(
            mass1=args.mass1,
            mass2=args.mass2,
            velocity_relative=args.velocity,  # Map CLI --velocity to velocity_relative
            sat_type1=args.sat_type1,
            sat_type2=args.sat_type2,
            cutoff=args.cutoff,
            seed=args.seed,
            enforce_mass_conservation=args.enforce_mass_conservation,
            orbit_altitude=args.orbit_altitude
        )
        
        print(f"Generated {len(fragments.fragment)} fragments")
        fragments.to_netcdf(args.out)
        print(f"Saved to {args.out}")
        
        if args.vis:
            print("Generating visualizations...")
            visualize_all(fragments, show=True)


if __name__ == '__main__':
    main()
