import subprocess
import re
import csv
import sys
import os
import time
from tqdm import tqdm
from datetime import datetime


# ================= Configuration =================
# List of scripts to test. Uncomment to enable.
TARGET_SCRIPTS = [
    # '/Users/qianshen/torch_hd/torch_hd/qs_hdc/baseline_mnist.py',
     '/Users/qianshen/torch_hd/torch_hd/qs_hdc/baseline_ucihar.py',
    # '/Users/qianshen/torch_hd/torch_hd/qs_hdc/baseline_emg.py',
    # '/Users/qianshen/torch_hd/torch_hd/qs_hdc/baseline_EuroLanguages.py',
    # '/Users/qianshen/torch_hd/torch_hd/qs_hdc/baseline_isolet.py',
]

# Phase Control
RUN_PHASE_DIM =  True
RUN_PHASE_BS =  True

# Parameter Sweep
DIMENSIONS_SWEEP = [1000, 2000, 4000, 6000, 8000, 10000]
BATCH_SIZE_SWEEP = [1, 8, 32, 64, 128, 256]

# Fixed Parameters
FIXED_DIMENSION = 4000
FIXED_BATCH_SIZE = 1

# Experiment Settings
RUNS_PER_CONFIG = 1
# Generate timestamped output file in qs_hdc directory (Modification 1 & 2)
# Get the directory where the script is located
output_dir = os.path.dirname(os.path.abspath(__file__))
# Use fixed filename: experiment_results.csv
OUTPUT_FILE = os.path.join(output_dir, "experiment_results.csv")

# Cooldown Settings (Seconds)
COOLDOWN_PER_RUN = 2
COOLDOWN_PER_SET = 10

# ================= Helper Functions =================

def parse_results(output):
    """
    Parses the accuracy and std dev from the script output.
    Returns: (accuracy, std_dev) where accuracy can be None if not found.
    """
    acc = None
    std = 0.0
    
    # Match Accuracy
    acc_match = re.search(r"(?:Testing accuracy of|Average Accuracy over \d+ runs:) (\d+\.\d+)%", output)
    if acc_match:
        acc = float(acc_match.group(1))
        
    # Match Std Dev (if available)
    std_match = re.search(r"Std Dev: (\d+\.\d+)", output)
    if std_match:
        std = float(std_match.group(1))
        
    return acc, std

def run_script_realtime(script_path, dim, batch_size):
    """
    Runs the python script with specified arguments and streams output in real-time.
    Returns: (accuracy, std_dev) or None if failed.
    """
    cmd = [sys.executable, script_path, "--dim", str(dim), "--batch_size", str(batch_size)]
    full_output = []
    
    print(f"\n> Running: {os.path.basename(script_path)} | Dim={dim} | Batch={batch_size}")
    
    try:
        # Use Popen to stream output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1, 
            universal_newlines=True
        )
        
        # Stream output to console
        for line in process.stdout:
            print(line, end='')  # Real-time print
            full_output.append(line)
            
        process.wait()
        
        if process.returncode != 0:
            print(f"\n[Error] Script failed with exit code {process.returncode}")
            return None
            
        # Parse accuracy from the collected output
        full_text = "".join(full_output)
        return parse_results(full_text)
        
    except Exception as e:
        print(f"\n[Exception] Failed to run script: {e}")
        return None

def cooldown(duration, message="Cooling down"):
    if duration <= 0:
        return
    
    # Use tqdm for cooldown visualization
    for _ in tqdm(range(duration), desc=message, leave=False, bar_format='{desc}: {remaining}s'):
        time.sleep(1)

def run_experiment_suite():
    print(f"Starting experiment suite at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results will be saved to: {OUTPUT_FILE}")

    # Ensure CSV exists with headers if not already present (Modification 2)
    # Check if file exists and is not empty to determine if header is needed
    file_exists = os.path.isfile(OUTPUT_FILE)
    write_header = not file_exists or os.path.getsize(OUTPUT_FILE) == 0
    
    if write_header:
        with open(OUTPUT_FILE, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'Script', 'Dimensions', 'Batch_Size', 'Mean_Accuracy', 'Std_Dev', 'Runs'])

    total_steps = 0
    if RUN_PHASE_DIM:
        total_steps += len(TARGET_SCRIPTS) * len(DIMENSIONS_SWEEP)
    if RUN_PHASE_BS:
        total_steps += len(TARGET_SCRIPTS) * len(BATCH_SIZE_SWEEP)
        
    if total_steps == 0:
        print("No phases selected. Exiting.")
        return

    # Overall experiment progress bar
    pbar_overall = tqdm(total=total_steps, desc="Total Progress", position=0)

    for script in TARGET_SCRIPTS:
        script_name = os.path.basename(script)
        
        # --- Phase 1: Dimension Sweep ---
        if RUN_PHASE_DIM:
            print(f"\n\n=== Phase 1: Dimension Sweep for {script_name} ===")
            for dim in DIMENSIONS_SWEEP:
                # Run script ONCE (internal loop handles repetition)
                result = run_script_realtime(script, dim, FIXED_BATCH_SIZE)
                
                if result is not None:
                    acc, std = result
                    if acc is not None:
                        # Save results
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open(OUTPUT_FILE, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            # Note: Std Dev is unknown here as script only reports mean. Using 0.000 placeholder.
                            # Runs is reported as 10 (hardcoded based on known internal logic)
                            writer.writerow([current_time, script_name, dim, FIXED_BATCH_SIZE, f"{acc:.3f}", f"{std:.3f}", "10 (Internal)"])
                        
                        print(f"\n>>> Result: Dim={dim} | Mean Acc={acc:.3f}% | Std Dev={std:.3f}")

                pbar_overall.update(1)
                
                # Cooldown after a set of runs (prevent thermal throttling)
                cooldown(COOLDOWN_PER_SET, "Set Cooldown")

        # --- Phase 2: Batch Size Sweep ---
        if RUN_PHASE_BS:
            print(f"\n\n=== Phase 2: Batch Size Sweep for {script_name} ===")
            for batch in BATCH_SIZE_SWEEP:
                result = run_script_realtime(script, FIXED_DIMENSION, batch)
                
                if result is not None:
                    acc, std = result
                    if acc is not None:
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open(OUTPUT_FILE, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([current_time, script_name, FIXED_DIMENSION, batch, f"{acc:.3f}", f"{std:.3f}", "10 (Internal)"])
                        
                        print(f"\n>>> Result: Batch={batch} | Mean Acc={acc:.3f}% | Std Dev={std:.3f}")
                
                pbar_overall.update(1)
                
                cooldown(COOLDOWN_PER_SET, "Set Cooldown")

    pbar_overall.close()
    print(f"\nAll Experiments Finished. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_experiment_suite()
