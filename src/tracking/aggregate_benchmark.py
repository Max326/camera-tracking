import os
import pandas as pd
import glob

# --- SCENARIO MAPPING ---
# Define which attributes apply to which sequence.
# Keys: Sequence Name
# Values: List of attributes (SV: Scale Var, OCC: Occlusion, FM: Fast Motion, 
#                            BC: Background Clutter, LR: Low Res, CM: Camera Motion)


# Example attributes:   SV - scale variation, OCC - occlusion, FM - fast motion, CM - calm motion, 
#                       SO - small objects, BO - big objects, 1O - one object, MO - multiple objects
SCENARIO_MAP = {
    "bike1": ["CM", "MO"],
    "bike3": ["MO", "SO", "FM"],
    
    "boat1": ["SV", "CM", "1O", "BO"],
    "boat2": ["CM", "1O"],
    "boat3": ["CM", "1O"],
    
    # Cars: Traffic (Clutter), Bridges/Trees (Occlusion)
    "car1": ["SO", "SV", "OCC", "CM", "MO"],
    "car2": ["SO", "MO", "OCC"],
    "car3": ["SO", "MO", "OCC"],
    "car4": ["SO", "MO", "OCC"],
    "car5": ["CM", "1O"],
    "car6": ["SV", "OCC"],
    "car7": ["OCC", "SO"],
    "car8": ["CM", "SV"],
    "car16": ["SO", "SV", "FM", "1O"],
    "car17": ["SV", "FM", "1O"],
    "car18": ["SV", "FM", "1O"],

    "person2": ["CM", "1O"],
    "person3": ["CM", "1O"],
    
    "truck1": ["BO", "SV", "MO", "OCC"],
    "truck2": ["SO", "MO", "OCC"],
    "truck3": ["SO", "FM", "MO"],
    
    "wakeboard1": ["FM", "SV", "OCC", "1O"],
    "wakeboard2": ["FM", "SO", "1O"],
    "wakeboard3": ["SV", "CM", "SO", "1O"]
}

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

    # --- CATEGORY BASED BREAKDOWN ---
    print("\n--- Category Breakdown ---")
    
    # 1. Expand the DataFrame: Create a row for each (Sequence, Category) pair
    # We first map sequences to their categories
    seq_to_cats = []
    unique_sequences = full_df['Sequence'].unique()
    
    for seq in unique_sequences:
        tags = SCENARIO_MAP.get(seq, ["Uncategorized"])
        for tag in tags:
            seq_to_cats.append({"Sequence": seq, "Category": tag})
            
    cat_map_df = pd.DataFrame(seq_to_cats)
    
    # Merge with the full results
    # Each original row (Tracker X on Seq Y) will be duplicated for each category Seq Y belongs to
    merged_df = pd.merge(full_df, cat_map_df, on="Sequence", how="inner")
    
    if merged_df.empty:
        print("No category mapping matched the processed sequences.")
        return

    # 2. Group by Category and Tracker
    cat_grouped = merged_df.groupby(["Category", "Tracker"])[available_cols].mean().reset_index()
    cat_grouped = cat_grouped.round(4)
    
    # 3. Pivot for readability: Rows=Tracker, Cols=Category (Metric: Precision)
    # We'll focus on Precision for the pivot table, but save full data to CSV
    pivot_metric = "Precision (CLE < 20px)"
    if pivot_metric not in cat_grouped.columns:
        pivot_metric = "Avg IoU" # Fallback

    pivot_table = cat_grouped.pivot(index="Tracker", columns="Category", values=pivot_metric)
    
    # Add an 'All' column (Mean of the breakdown) for sorting
    # Note: This might differ slightly from the global mean if categories overlap unevenly, so we use the global aggregated val if possible
    # But for simple sorting here:
    pivot_table["Mean"] = pivot_table.mean(axis=1)
    pivot_table = pivot_table.sort_values("Mean", ascending=False)
    pivot_table = pivot_table.drop(columns=["Mean"]) # Remove helper

    print(f"\nMetric: {pivot_metric} per Category")
    print(pivot_table.fillna("-").to_string())
    
    # Save detailed category stats
    detailed_cat_path = os.path.join(RESULTS_DIR, "benchmark_by_category.csv")
    cat_grouped.sort_values(by=["Category", pivot_metric], ascending=[True, False]).to_csv(detailed_cat_path, index=False)
    print(f"\nDetailed category results saved to: {detailed_cat_path}")


if __name__ == "__main__":
    aggregate_results()