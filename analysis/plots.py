import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Configuration
# ============================================================

RESULTS_CSV = "benchmarks/results/results.csv"
OUTPUT_DIR = "analysis/plots"

print("=" * 60)
print("Generating Performance Analysis Plots")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    df = pd.read_csv(RESULTS_CSV)
except FileNotFoundError:
    print(f"Error: {RESULTS_CSV} not found. Run benchmarks first.")
    exit(1)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Select a large batch size for thread scaling experiments
scaling_batch = df['BatchSize'].max()
df_scaling = df[df['BatchSize'] == scaling_batch]

# ============================================================
# 1. Execution Time vs Number of Threads
# ============================================================
plt.figure(figsize=(10, 6))
for mode in df_scaling['Mode'].unique():
    subset = df_scaling[df_scaling['Mode'] == mode]
    if mode == 'sequential':
        seq_time = subset['ExecTime'].iloc[0]
        plt.axhline(y=seq_time, color='gray', linestyle='--', label='Sequential Baseline')
    else:
        plt.plot(subset['Threads'], subset['ExecTime'], marker='o', label=mode.replace('_', ' ').title())

plt.title(f'Execution Time vs Number of Threads (Batch Size: {scaling_batch})')
plt.xlabel('Number of Threads')
plt.ylabel('Execution Time (seconds)')
plt.xticks(df_scaling['Threads'].unique())
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'execution_time_vs_threads.png'))
plt.close()

# ============================================================
# 2. Speedup vs Number of Threads & Amdahl's Law
# ============================================================
plt.figure(figsize=(10, 6))
threads = np.array(sorted(df_scaling['Threads'].unique()))

for mode in [m for m in df_scaling['Mode'].unique() if m != 'sequential']:
    subset = df_scaling[df_scaling['Mode'] == mode].sort_values('Threads')
    measured_speedup = subset['Speedup'].values
    ts = subset['Threads'].values
    
    plt.plot(ts, measured_speedup, marker='o', label=f"{mode.replace('_', ' ').title()} (Measured)")
    
    # Estimate P from the max thread count (using S = 1 / ((1-P) + P/N) -> P = (N/S - N) / (1 - N))
    if len(ts) > 1:
        N = ts[-1]
        S = measured_speedup[-1]
        if S > 1.01:
            P = (N - N/S) / (N - 1)
            # Clip P to [0, 1]
            P = max(0.0, min(1.0, P))
            amdahl_speedup = 1 / ((1 - P) + P / threads)
            plt.plot(threads, amdahl_speedup, linestyle='--', label=f"{mode.replace('_', ' ').title()} (Amdahl P={P:.2f})")

plt.plot(threads, threads, linestyle=':', color='black', label='Ideal Linear Speedup')
plt.title(f"Speedup vs Number of Threads (Batch Size: {scaling_batch})")
plt.xlabel('Number of Threads')
plt.ylabel('Speedup (T_seq / T_par)')
plt.xticks(threads)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'speedup_vs_threads.png'))
plt.close()

# ============================================================
# 3. Efficiency vs Number of Threads
# ============================================================
plt.figure(figsize=(10, 6))
for mode in [m for m in df_scaling['Mode'].unique() if m != 'sequential']:
    subset = df_scaling[df_scaling['Mode'] == mode]
    # Efficiency is usually presented as percentage 0-100%
    plt.plot(subset['Threads'], subset['Efficiency'] * 100, marker='o', label=mode.replace('_', ' ').title())

plt.title(f'Parallel Efficiency vs Number of Threads (Batch Size: {scaling_batch})')
plt.xlabel('Number of Threads')
plt.ylabel('Efficiency (%)')
plt.xticks(df_scaling['Threads'].unique())
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'efficiency_vs_threads.png'))
plt.close()

# ============================================================
# 4. Throughput vs Batch Size
# ============================================================
# Use a fixed thread count for batch scaling, e.g., max threads
max_threads = df['Threads'].max()
df_batch = df[df['Threads'] == max_threads]

plt.figure(figsize=(10, 6))
for mode in df_batch['Mode'].unique():
    subset = df_batch[df_batch['Mode'] == mode]
    plt.plot(subset['BatchSize'], subset['Throughput'], marker='o', label=mode.replace('_', ' ').title())

plt.title(f'Throughput vs Batch Size (Threads: {max_threads})')
plt.xlabel('Batch Size')
plt.ylabel('Throughput (Images / Second)')
plt.xscale('log')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'throughput_vs_batch_size.png'))
plt.close()

# ============================================================
# 5. Latency vs Batch Size
# ============================================================
plt.figure(figsize=(10, 6))
for mode in df_batch['Mode'].unique():
    subset = df_batch[df_batch['Mode'] == mode]
    plt.plot(subset['BatchSize'], subset['Latency'], marker='o', label=mode.replace('_', ' ').title())

plt.title(f'Average Latency vs Batch Size (Threads: {max_threads})')
plt.xlabel('Batch Size')
plt.ylabel('Average Latency per Image (ms)')
plt.xscale('log')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'latency_vs_batch_size.png'))
plt.close()

# ============================================================
# 6. Sequential vs Parallel Comparison
# ============================================================
plt.figure(figsize=(10, 6))
modes = df_scaling['Mode'].unique()
exec_times = [df_scaling[df_scaling['Mode'] == m]['ExecTime'].min() for m in modes] # min time (max threads)

bars = plt.bar(modes, exec_times, color=['gray', 'blue', 'orange', 'green'])
plt.title(f'Sequential vs Parallel Execution Time (Batch: {scaling_batch}, Threads: {max_threads})')
plt.xlabel('Execution Mode')
plt.ylabel('Execution Time (seconds)')

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2), ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sequential_vs_parallel.png'))
plt.close()

print(f"Successfully generated 6 plots in {OUTPUT_DIR}/")
print("=" * 60)
