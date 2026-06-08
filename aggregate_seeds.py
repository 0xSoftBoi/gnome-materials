#!/usr/bin/env python3
"""Aggregate multi-seed WBM results and produce final paper figures.

Usage:
    python3 aggregate_seeds.py
    python3 aggregate_seeds.py --seeds 0 1 2 3 4
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from evaluation.wbm_metrics import aggregate_seeds, plot_multiseed_results, print_multiseed_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Seeds to aggregate (default: all found in results/)')
    parser.add_argument('--output', default='results/wbm_multiseed.png')
    args = parser.parse_args()

    if args.seeds is not None:
        json_paths = [f'results/wbm_al_results_seed{s}.json' for s in args.seeds]
    else:
        json_paths = sorted(glob.glob('results/wbm_al_results_seed*.json'))

    missing = [p for p in json_paths if not os.path.exists(p)]
    if missing:
        print(f"Missing result files: {missing}")
        print("Run the missing seeds first:")
        for p in missing:
            seed = p.split('seed')[1].split('.')[0]
            print(f"  python3 main.py --experiment wbm --seed {seed}")
        sys.exit(1)

    print(f"Aggregating {len(json_paths)} seed(s): {json_paths}")
    agg = aggregate_seeds(json_paths)
    print_multiseed_summary(agg)
    plot_multiseed_results(agg, output_path=args.output)
    print(f"\nDone. Figure saved to {args.output}")


if __name__ == '__main__':
    main()
