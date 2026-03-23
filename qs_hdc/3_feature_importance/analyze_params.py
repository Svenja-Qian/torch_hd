import os
import pandas as pd
import glob
import numpy as np

base_dir = "/Users/qianshen/torch_hd/torch_hd/qs_hdc/3_feature_importance"
datasets = ["isolet", "har", "mnist", "emg", "cardio"]

# 1. Main Table Data Collection
main_results = []
target_suffix = "D6000_E3_boosting_rr30_margin0.02_run200_main.csv"

# 2. Parameter Analysis Data Collection
param_analysis = {}

def get_data(filepath):
    try:
        df = pd.read_csv(filepath)
        if not df.empty:
            return df.iloc[0]
    except:
        return None
    return None

for ds in datasets:
    # 1. Main Table
    # Construct expected filename
    # Note: EMG and others might have slight variations, checking glob if exact match fails
    expected_file = os.path.join(base_dir, f"fi_{ds}", "results", f"{ds}_fi_{target_suffix}")
    if not os.path.exists(expected_file):
        # Try to find a close match if exact not found (e.g. emg might be slightly different)
        # Actually based on LS, they seem consistent: cardio, emg, har, isolet, mnist all have this file.
        # Double check EMG: emg_fi_D6000_E3_boosting_rr30_margin0.02_run200_main.csv - YES.
        # Double check MNIST: mnist_fi_D6000_E3_boosting_rr30_margin0.02_run200_main.csv - YES.
        pass
    
    row = get_data(expected_file)
    if row is not None:
        main_results.append({
            "Dataset": ds.upper(),
            "Baseline": row["acc_baseline_mean"],
            "Retrain": row["acc_ensemble_mean"],
            "Improvement": row["acc_ensemble_mean"] - row["acc_baseline_mean"],
            "StdDev": row["acc_ensemble_std"]
        })
    else:
        print(f"Warning: Main file not found for {ds}")

    # 2. Parameter Analysis
    # Get all D6000 boosting files
    pattern = os.path.join(base_dir, f"fi_{ds}", "results", f"*{ds}_fi_D6000*_boosting*_main.csv")
    files = glob.glob(pattern)
    ds_data = []
    for f in files:
        r = get_data(f)
        if r is not None:
            # Parse filename for params
            fname = os.path.basename(f)
            params = []
            if "margin" in fname:
                # extract margin value
                try:
                    margin = fname.split("margin")[1].split("_")[0]
                    params.append(f"m={margin}")
                except:
                    pass
            else:
                params.append("m=0")
            
            if "run" in fname:
                try:
                    run = fname.split("run")[1].split("_")[0]
                    params.append(f"run={run}")
                except:
                    pass
            
            if "rr" in fname:
                 try:
                    rr = fname.split("rr")[1].split("_")[0]
                    params.append(f"rr={rr}")
                 except:
                    pass

            ds_data.append({
                "Params": ", ".join(params),
                "Retrain": r["acc_ensemble_mean"],
                "Baseline": r["acc_baseline_mean"],
                "Imp": r["acc_ensemble_mean"] - r["acc_baseline_mean"],
                "File": fname
            })
    
    # Sort by Retrain accuracy descending
    ds_data.sort(key=lambda x: x["Retrain"], reverse=True)
    param_analysis[ds] = ds_data

# --- Output Generation ---

print("### 1. Main Results Table (D=6000, m=0.02, run=200, Boosting)")
print(f"{'Dataset':<10} | {'Baseline':<10} | {'Retrain':<10} | {'Improvement':<12} | {'StdDev':<10}")
print("-" * 65)
for r in main_results:
    print(f"{r['Dataset']:<10} | {r['Baseline']:.4f}     | {r['Retrain']:.4f}     | {r['Improvement']:+.4f}       | {r['StdDev']:.4f}")

print("\n\n### 2. Parameter Sensitivity Analysis")

for ds in datasets:
    print(f"\n#### {ds.upper()}")
    print(f"{'Parameters':<30} | {'Retrain':<10} | {'Baseline':<10} | {'Imp':<10}")
    print("-" * 70)
    for row in param_analysis[ds]:
        print(f"{row['Params']:<30} | {row['Retrain']:.4f}     | {row['Baseline']:.4f}     | {row['Imp']:+.4f}")
