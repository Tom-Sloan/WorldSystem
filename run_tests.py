#!/usr/bin/env python3
"""Main test runner for WorldSystem project"""

import subprocess
import sys
import os

def run_test_suite(suite_name, path):
    """Run a test suite"""
    print(f"\n{'='*60}")
    print(f"Running {suite_name} Tests")
    print('='*60)
    
    if os.path.exists(path):
        subprocess.run([sys.executable, path])
    else:
        print(f"Test suite not found: {path}")

def main():
    """Run all test suites"""
    print("WorldSystem Test Runner")
    print("=" * 60)
    
    # Test suites to run
    suites = [
        ("Mesh Service", "mesh_service/tests/run_all_tests.py"),
        # Add other test suites here as they are created
    ]
    
    for suite_name, suite_path in suites:
        if os.path.exists(suite_path):
            run_test_suite(suite_name, suite_path)
        else:
            print(f"\nSkipping {suite_name} - test suite not found at {suite_path}")
    
    print("\n" + "="*60)
    print("All test suites completed")
    print("="*60)

if __name__ == "__main__":
    main()