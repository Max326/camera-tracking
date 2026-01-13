import os
import pandas as pd
import glob

def aggregate_results():
    # Define paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results-rpi")
    
    # Find all summary files matching the pattern
    pattern = os.path.join(RESULTS_DIR, "*_benchmark_summary.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No benchmark summary files found in {RESULTS_DIR}")
        return

    print(f"Found {len(files)} summary files. Aggregating...")
    
    # Read and concatenate all CSVs
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    if not dfs:
        print("No data loaded.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    
    # Columns to aggregate (Added "Valid FPS")
    numeric_cols = ["Avg FPS", "Valid FPS", "Avg IoU", "Precision (CLE < 20px)", "Success Rate (IoU > 0.5)"]
    
    # Ensure columns exist before aggregating
    available_cols = [c for c in numeric_cols if c in full_df.columns]
    
    if not available_cols:
        print("No numeric columns found to aggregate.")
        print("Columns found:", full_df.columns)
        return

    # Group by Tracker and calculate means
    aggregated = full_df.groupby("Tracker")[available_cols].mean().reset_index()
    
    # Add a count of sequences per tracker (to see if any tracker missed a sequence)
    counts = full_df.groupby("Tracker")["Sequence"].count().reset_index(name="Sequences Count")
    aggregated = pd.merge(aggregated, counts, on="Tracker")
    
    # Sort by Precision (descending), then Avg IoU (descending)
    sort_cols = ["Precision (CLE < 20px)", "Avg IoU"]
    # Ensure columns exist just in case
    sort_cols = [c for c in sort_cols if c in aggregated.columns]
    
    aggregated = aggregated.sort_values(by=sort_cols, ascending=False)
    
    # Round numeric columns to 4 decimal places for cleaner output
    aggregated = aggregated.round(4)

    # Display results
    print("\n--- Aggregated Benchmark Results (Mean across all sequences) ---")
    print(aggregated.to_string(index=False)) # float_format is now redundant but harmless
    
    # Save to CSV
    output_path = os.path.join(RESULTS_DIR, "aggregated_benchmark_results.csv")
    aggregated.to_csv(output_path, index=False)
    print(f"\nAggregated results saved to: {output_path}")

if __name__ == "__main__":
    aggregate_results()