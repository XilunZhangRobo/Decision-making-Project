#!/usr/bin/env python3
"""
Debug script to check multi-seed data collection
"""

import json

def debug_data():
    print("=== DEBUGGING MULTI-SEED DATA ===")

    # Load the data
    with open('examples/multi_seed_training_curves.json', 'r') as f:
        data = json.load(f)

    print(f"Total baseline seeds: {len(data['baseline'])}")
    print(f"Total enhanced seeds: {len(data['enhanced'])}")

    print("\n=== BASELINE SEEDS ===")
    for i, result in enumerate(data['baseline']):
        curve = result['training_curve']
        print(f"Seed {result['seed']}: {len(curve)} points")
        for j, point in enumerate(curve):
            print(f"  Point {j}: timestep {point['timestep']}, profit {point['mean_profit']:.3f}")

    print("\n=== ENHANCED SEEDS ===")
    for i, result in enumerate(data['enhanced']):
        curve = result['training_curve']
        print(f"Seed {result['seed']}: {len(curve)} points")
        for j, point in enumerate(curve):
            print(f"  Point {j}: timestep {point['timestep']}, profit {point['mean_profit']:.3f}")

    print("\n=== ANALYSIS ===")
    if len(data['baseline']) > 0 and len(data['baseline'][0]['training_curve']) > 0:
        first_seed = data['baseline'][0]
        expected_points = (first_seed['training_curve'][-1]['timestep']) // 5000
        print(f"Training ran to timestep: {first_seed['training_curve'][-1]['timestep']}")
        print(f"Expected evaluation points: {expected_points} (every 5000 steps)")
        print(f"Actual evaluation points: {len(first_seed['training_curve'])}")
        if expected_points != len(first_seed['training_curve']):
            print("❌ MISMATCH: Expected more evaluation points!")

if __name__ == "__main__":
    debug_data()
