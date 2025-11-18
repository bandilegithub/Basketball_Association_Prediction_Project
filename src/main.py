"""
Main Script - NBA Basketball Association Analysis
Runs all analysis components: visualization, outlier detection, and game prediction
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from outlier_detection import main as outlier_main
from game_prediction import main as prediction_main


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def main():
    """Run all analysis components"""
    print_header("NBA BASKETBALL ASSOCIATION - MACHINE LEARNING ANALYSIS")
    
    print("This analysis includes:")
    print("  1. Outstanding Players Detection (Outlier Analysis)")
    print("  2. Game Outcome Prediction (Multiple ML Models)")
    
    input("\nPress Enter to start the analysis...")
    
    # Run outlier detection
    print_header("PART 1: OUTSTANDING PLAYERS - OUTLIER DETECTION")
    try:
        outlier_main()
    except Exception as e:
        print(f"Error in outlier detection: {e}")
    
    input("\nPress Enter to continue to game prediction...")
    
    # Run game prediction
    print_header("PART 2: GAME OUTCOME PREDICTION")
    try:
        prediction_main()
    except Exception as e:
        print(f"Error in game prediction: {e}")
    
    # Final summary
    print_header("ANALYSIS COMPLETE")
    print("All results have been saved to the 'output' folder:")
    print("\nOutlier Detection Results:")
    print("  • outstanding_players_results.csv")
    print("\nGame Prediction Results:")
    print("  • prediction_results.csv")
    
    print("\n" + "="*80)
    print("Thank you for using the NBA Basketball Association Analysis System!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
