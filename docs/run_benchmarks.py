#!/usr/bin/env python3
"""
Run the advanced benchmark suite and generate comprehensive results
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("⚠️  Some dependencies failed to install - continuing anyway")

def run_benchmarks():
    """Run the benchmark suite"""
    print("\n🚀 Running advanced benchmarks...")
    try:
        result = subprocess.run([sys.executable, "advanced_benchmark.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running benchmarks: {e}")
        return False

def main():
    # Check if we need to install dependencies
    if "--install" in sys.argv:
        install_dependencies()
    
    # Run benchmarks
    success = run_benchmarks()
    
    if success:
        print("\n✅ Benchmarks completed successfully!")
        print("📊 Check 'advanced_benchmark_results.json' for detailed results")
        print("📈 Check 'advanced_benchmark_plots.png' for visualizations")
    else:
        print("\n⚠️  Benchmarks completed with some errors")
        print("Results may be partial - check output above")

if __name__ == "__main__":
    main()