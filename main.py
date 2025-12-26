#!/usr/bin/env python3
"""
QAGNN Main Execution Script
Run phases sequentially or individually
"""

import argparse
import yaml
import sys
from pathlib import Path

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def run_phase1():
    """Generate circuit simulations"""
    print("Running Phase 1: Circuit Simulation")
    from src.circuits.ode_solver import generate_simulations
    generate_simulations()

def run_phase2():
    """Train deep learning model"""
    print("Running Phase 2: Deep Learning Training")
    from src.ai.train import train_model
    train_model()

def run_phase3():
    """Quantum optimization"""
    print("Running Phase 3: Quantum Optimization")
    from src.quantum.qaoa import run_qaoa
    run_qaoa()

def run_phase4():
    """Analysis and figures"""
    print("Running Phase 4: Analysis")
    from src.analysis.paper_figures import create_all_figures
    create_all_figures()

def main():
    parser = argparse.ArgumentParser(description='QAGNN Project Execution')
    parser.add_argument('--phase', type=int, choices=[1,2,3,4], 
                       help='Run specific phase (1-4)')
    parser.add_argument('--all', action='store_true', 
                       help='Run all phases sequentially')
    
    args = parser.parse_args()
    
    if args.all:
        run_phase1()
        run_phase2()
        run_phase3()
        run_phase4()
    elif args.phase:
        phases = {1: run_phase1, 2: run_phase2, 3: run_phase3, 4: run_phase4}
        phases[args.phase]()
    else:
        print("Please specify --phase N or --all")

if __name__ == "__main__":
    main()
