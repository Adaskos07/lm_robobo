#!/usr/bin/env python3
import sys
from argparse import ArgumentParser

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import robot_run

def main():
    parser = ArgumentParser(prog='Robobo')

    parser.add_argument('--hardware', action='store_true',
                        help='if specified run on real robot, else simulation')
    parser.add_argument('--test_run', action='store_true',
                        help='if specified simulation runs in a test mode, no effect when running on hardware')
    parser.add_argument('--max_steps', type=int, default=50)
    args = parser.parse_args()

    if args.hardware:
        rob = HardwareRobobo(camera=True)
    else:
        rob = SimulationRobobo()
    
    robot_run(rob, max_steps=args.max_steps, test_run=args.test_run)

if __name__ == "__main__":
    main()
    # You can do better argument parsing than this!
    # if len(sys.argv) < 2:
    #     raise ValueError(
    #         """To run, we need to know if we are running on hardware of simulation
    #         Pass `--hardware` or `--simulation` to specify."""
    #     )
    # elif sys.argv[1] == "--hardware":
    #     rob = HardwareRobobo(camera=True)
    # elif sys.argv[1] == "--simulation":
    #     rob = SimulationRobobo()
    # else:
    #     raise ValueError(f"{sys.argv[1]} is not a valid argument.")

    # robot_run(rob)
