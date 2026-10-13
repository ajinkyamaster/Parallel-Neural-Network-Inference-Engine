import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

OUTPUT_DIR = "analysis/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = "benchmarks/results/serial_openmp_mpi_comparison.csv"
df = pd.read_csv(CSV_PATH)

print("Generating plots from:", CSV_PATH)

# ==============================================================================
# Plot 1: Execution Time Comparison (Serial vs OpenMP vs MPI) across Batch Sizes
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

batch_sizes = [100, 1000, 10000]
x = np.arange(len(batch_sizes))
width = 0.2

# Extract times for best configurations (e.g. 12 units for parallel)
serial_times = [df[(df['Paradigm'] == 'Serial') & (df['BatchSize'] == b)]['ExecTime'].values[0] for b in batch_sizes]
omp_batch_times = [df[(df['Mode'] == 'OpenMP-Batch') & (df['Units'] == 12) & (df['BatchSize'] == b)]['ExecTime'].values[0] for b in batch_sizes]
omp_intra_times = [df[(df['Mode'] == 'OpenMP-Intra_layer') & (df['Units'] == 12) & (df['BatchSize'] == b)]['ExecTime'].values[0] for b in batch_sizes]
mpi_batch_times = [df[(df['Mode'] == 'MPI-Batch') & (df['Units'] == 12) & (df['BatchSize'] == b)]['ExecTime'].values[0] for b in batch_sizes]

rects1 = ax.bar(x - 1.5*width, serial_times, width, label='Serial Baseline (1 Core)', color='#e74c3c')
rects2 = ax.bar(x - 0.5*width, omp_batch_times, width, label='OpenMP Batch (12 Threads)', color='#3498db')
rects3 = ax.bar(x + 0.5*width, omp_intra_times, width, label='OpenMP Intra-Layer (12 Threads)', color='#9b59b6')
rects4 = ax.bar(x + 1.5*width, mpi_batch_times, width, label='MPI Batch (12 Ranks)', color='#2ecc71')

ax.set_ylabel('Execution Time (seconds, log scale)')
ax.set_yscale('log')
ax.set_title('Execution Time Comparison: Serial vs OpenMP vs MPI Across Batch Sizes')
ax.set_xticks(x)
ax.set_xticklabels([f'{b} Images' for b in batch_sizes])
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
ax.grid(True, which="both", ls="--", alpha=0.5)

# Add values above bars
def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}s' if h < 1 else f'{h:.2f}s',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=30)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

plt.tight_layout()
p1 = os.path.join(OUTPUT_DIR, 'serial_openmp_mpi_execution_time.png')
plt.savefig(p1)
plt.close()
print("Saved:", p1)

# ==============================================================================
# Plot 2: Speedup vs Number of Workers (Threads/Ranks) on Batch = 10,000
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

units = [1, 2, 4, 8, 12]

# MPI Speedup
mpi_df = df[(df['Mode'] == 'MPI-Batch') & (df['BatchSize'] == 10000)].sort_values('Units')
mpi_units = mpi_df['Units'].values
mpi_speedup = mpi_df['Speedup'].values

# OpenMP Batch Speedup
omp_df = df[(df['Mode'] == 'OpenMP-Batch') & (df['BatchSize'] == 10000)].sort_values('Units')
omp_units = [1] + list(omp_df['Units'].values)
omp_speedup = [1.0] + list(omp_df['Speedup'].values)

# OpenMP Intra-Layer Speedup
intra_df = df[(df['Mode'] == 'OpenMP-Intra_layer') & (df['BatchSize'] == 10000)].sort_values('Units')
intra_units = [1] + list(intra_df['Units'].values)
intra_speedup = [1.0] + list(intra_df['Speedup'].values)

# Ideal Linear Speedup
ax.plot(units, units, 'k--', label='Ideal Linear Speedup', linewidth=1.8, alpha=0.7)

# Curves
ax.plot(omp_units, omp_speedup, marker='o', linewidth=2.2, color='#3498db', label='OpenMP Batch Parallelism')
ax.plot(intra_units, intra_speedup, marker='s', linewidth=2.2, color='#9b59b6', label='OpenMP Intra-Layer Parallelism')
ax.plot(mpi_units, mpi_speedup, marker='^', linewidth=2.5, color='#2ecc71', label='MPI Batch Parallelism (Distributed)')

# Mark MPI Pipeline at 4 Ranks
pipe_4 = df[(df['Mode'] == 'MPI-Pipeline') & (df['BatchSize'] == 1000)]
if not pipe_4.empty:
    pipe_speedup = pipe_4['Speedup'].values[0]
    ax.scatter([4], [pipe_speedup], color='#e67e22', s=120, zorder=5, label=f'MPI Pipeline (4 Ranks: {pipe_speedup:.2f}x)')

ax.set_xlabel('Number of Parallel Execution Units (Threads / MPI Ranks)')
ax.set_ylabel('Speedup (T_serial / T_parallel)')
ax.set_title('Parallel Speedup vs Workers (Batch Size = 10,000 Images)')
ax.set_xticks(units)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
ax.grid(True, ls="--", alpha=0.6)

plt.tight_layout()
p2 = os.path.join(OUTPUT_DIR, 'serial_openmp_mpi_speedup.png')
plt.savefig(p2)
plt.close()
print("Saved:", p2)

# ==============================================================================
# Plot 3: MPI Computation Time vs Communication Overhead (Scatter/Gather)
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

mpi_scaling = df[(df['Mode'] == 'MPI-Batch') & (df['BatchSize'] == 10000)].sort_values('Units')
ranks = mpi_scaling['Units'].values
compute_times = mpi_scaling['ComputeTime'].values
comm_times = mpi_scaling['CommTime'].values

x_pos = np.arange(len(ranks))
w = 0.5

b1 = ax.bar(x_pos, compute_times, w, label='Local Forward Computation Time', color='#3498db')
b2 = ax.bar(x_pos, comm_times, w, bottom=compute_times, label='MPI Communication (Scatterv + Gatherv)', color='#e74c3c')

ax.set_xlabel('MPI Processes (Ranks)')
ax.set_ylabel('Time (seconds)')
ax.set_title('MPI Batch Breakdown: Computation vs Communication (Batch = 10,000)')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{r} Ranks' for r in ranks])
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
ax.grid(True, axis='y', ls="--", alpha=0.6)

# Annotate communication percentage
for i, (tot, comm) in enumerate(zip(mpi_scaling['ExecTime'], comm_times)):
    pct = (comm / tot) * 100.0
    ax.text(i, tot + 0.15, f'Comm: {pct:.1f}%\n({comm:.3f}s)', ha='center', fontsize=9, fontweight='bold', color='#c0392b')

plt.tight_layout()
p3 = os.path.join(OUTPUT_DIR, 'mpi_comm_vs_comp.png')
plt.savefig(p3)
plt.close()
print("Saved:", p3)

# ==============================================================================
# Plot 4: Throughput Scaling (Images per Second)
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

omp_tp = [df[(df['Mode'] == 'OpenMP-Batch') & (df['Units'] == 12) & (df['BatchSize'] == b)]['Throughput'].values[0] for b in batch_sizes]
mpi_tp = [df[(df['Mode'] == 'MPI-Batch') & (df['Units'] == 12) & (df['BatchSize'] == b)]['Throughput'].values[0] for b in batch_sizes]
serial_tp = [df[(df['Paradigm'] == 'Serial') & (df['BatchSize'] == b)]['Throughput'].values[0] for b in batch_sizes]

ax.plot(batch_sizes, serial_tp, marker='o', linewidth=2.2, color='#e74c3c', label='Serial (1 Core)')
ax.plot(batch_sizes, omp_tp, marker='s', linewidth=2.2, color='#3498db', label='OpenMP Batch (12 Threads)')
ax.plot(batch_sizes, mpi_tp, marker='^', linewidth=2.5, color='#2ecc71', label='MPI Batch (12 Ranks)')

ax.set_xscale('log')
ax.set_xlabel('Batch Size (Images, Log Scale)')
ax.set_ylabel('Throughput (Images / Second)')
ax.set_title('Inference Throughput Scaling: Serial vs OpenMP vs MPI')
ax.set_xticks(batch_sizes)
ax.set_xticklabels(['100', '1,000', '10,000'])
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
ax.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
p4 = os.path.join(OUTPUT_DIR, 'throughput_scaling.png')
plt.savefig(p4)
plt.close()
print("Saved:", p4)

print("\nAll 4 analysis plots successfully generated in analysis/plots/")
