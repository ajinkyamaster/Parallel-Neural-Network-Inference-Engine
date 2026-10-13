import subprocess
import re
import os
import pandas as pd

def run_cmd(cmd):
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.stdout

def run_benchmarks():
    os.makedirs("benchmarks/results", exist_ok=True)
    records = []
    
    batch_sizes = [100, 1000, 10000]
    ranks_list = [1, 2, 4, 8, 12]
    
    print("=" * 70)
    print("Running Consolidated Benchmarks: Serial vs OpenMP vs MPI")
    print("=" * 70)
    
    # 1. Serial Baselines
    print("\n--- 1. Serial Baseline Benchmarks ---")
    serial_times = {}
    for b in batch_sizes:
        out = run_cmd(f"./inference_engine --mode sequential --batch {b}")
        m = re.search(r"Time:\s*([\d\.]+)\s*sec", out)
        if m:
            t = float(m.group(1))
            serial_times[b] = t
            print(f"Serial Batch {b:5d}: {t:.5f} s")
            records.append({
                "Paradigm": "Serial",
                "Mode": "Sequential",
                "Units": 1,
                "BatchSize": b,
                "ExecTime": t,
                "Speedup": 1.0,
                "Throughput": b / t,
                "Efficiency": 100.0,
                "CommTime": 0.0,
                "ComputeTime": t
            })

    # 2. OpenMP Benchmarks (Batch & Intra-Layer)
    print("\n--- 2. OpenMP Benchmarks ---")
    threads_list = [2, 4, 8, 12]
    for omp_mode in ["batch", "intra_layer"]:
        for t_count in threads_list:
            for b in batch_sizes:
                out = run_cmd(f"./inference_engine --mode {omp_mode} --threads {t_count} --batch {b}")
                m = re.search(r"Time:\s*([\d\.]+)\s*sec", out)
                if m:
                    t = float(m.group(1))
                    seq_t = serial_times[b]
                    speedup = seq_t / t
                    records.append({
                        "Paradigm": "OpenMP",
                        "Mode": f"OpenMP-{omp_mode.capitalize()}",
                        "Units": t_count,
                        "BatchSize": b,
                        "ExecTime": t,
                        "Speedup": speedup,
                        "Throughput": b / t,
                        "Efficiency": (speedup / t_count) * 100.0,
                        "CommTime": 0.0,
                        "ComputeTime": t
                    })
                    print(f"OpenMP {omp_mode:11s} | Threads: {t_count:2d} | Batch: {b:5d} -> Time: {t:.4f}s | Speedup: {speedup:.2f}x")

    # 3. MPI Batch Benchmarks
    print("\n--- 3. MPI Batch Benchmarks ---")
    for r in ranks_list:
        for b in batch_sizes:
            cmd = f"mpirun --oversubscribe -np {r} ./inference_engine_mpi --mode batch --batch {b}"
            out = run_cmd(cmd)
            m_tot = re.search(r"Total Execution Time:\s*([\d\.]+)\s*s", out)
            m_comp = re.search(r"Compute Time\s*:\s*([\d\.]+)\s*s", out)
            m_comm = re.search(r"Communication Time\s*:\s*([\d\.]+)\s*s", out)
            if m_tot:
                tot_t = float(m_tot.group(1))
                comp_t = float(m_comp.group(1)) if m_comp else tot_t
                comm_t = float(m_comm.group(1)) if m_comm else 0.0
                seq_t = serial_times[b]
                speedup = seq_t / tot_t
                records.append({
                    "Paradigm": "MPI",
                    "Mode": "MPI-Batch",
                    "Units": r,
                    "BatchSize": b,
                    "ExecTime": tot_t,
                    "Speedup": speedup,
                    "Throughput": b / tot_t,
                    "Efficiency": (speedup / r) * 100.0,
                    "CommTime": comm_t,
                    "ComputeTime": comp_t
                })
                print(f"MPI Batch       | Ranks:   {r:2d} | Batch: {b:5d} -> Time: {tot_t:.4f}s | Speedup: {speedup:.2f}x | Comm: {comm_t:.4f}s")

    # 4. MPI Pipeline Benchmarks (4 Ranks)
    print("\n--- 4. MPI Pipeline Benchmarks ---")
    for b in [100, 1000]:
        cmd = f"mpirun --oversubscribe -np 4 ./inference_engine_mpi --mode pipeline --batch {b}"
        out = run_cmd(cmd)
        m_tot = re.search(r"Total Time:\s*([\d\.]+)\s*s", out)
        if m_tot:
            tot_t = float(m_tot.group(1))
            seq_t = serial_times[b]
            speedup = seq_t / tot_t
            records.append({
                "Paradigm": "MPI",
                "Mode": "MPI-Pipeline",
                "Units": 4,
                "BatchSize": b,
                "ExecTime": tot_t,
                "Speedup": speedup,
                "Throughput": b / tot_t,
                "Efficiency": (speedup / 4) * 100.0,
                "CommTime": 0.0,
                "ComputeTime": tot_t
            })
            print(f"MPI Pipeline    | Ranks:    4 | Batch: {b:5d} -> Time: {tot_t:.4f}s | Speedup: {speedup:.2f}x")

    df = pd.DataFrame(records)
    csv_path = "benchmarks/results/serial_openmp_mpi_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved consolidated benchmark results to {csv_path}")

if __name__ == "__main__":
    run_benchmarks()
