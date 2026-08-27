#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <iomanip>

#include "../inference/tensor.h"
#include "../inference/sequential/sequential.h"
#include "../inference/parallel/intra_layer.h"
#include "../inference/parallel/batch.h"
#include "../inference/pipeline/pipeline.h"

// Helper to load int32 binary files
bool load_int32_bin(const std::string& filepath, std::vector<int32_t>& data) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) return false;
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    data.resize(size / sizeof(int32_t));
    if (file.read(reinterpret_cast<char*>(data.data()), size)) return true;
    return false;
}

int main() {
    std::string weights_dir = "training/models/weights";
    std::string data_dir = "training/models/data";
    
    std::cout << "Loading test data..." << std::endl;
    Tensor all_images({10000, 1, 28, 28});
    if (!all_images.load(data_dir + "/test_images.bin")) return 1;
    
    std::vector<int32_t> all_labels;
    if (!load_int32_bin(data_dir + "/test_labels.bin", all_labels)) return 1;
    
    std::vector<int32_t> all_pt_preds;
    if (!load_int32_bin(data_dir + "/test_preds.bin", all_pt_preds)) return 1;

    std::vector<std::string> modes = {"sequential", "intra_layer", "batch", "pipeline"};
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    std::vector<size_t> batch_sizes = {100, 1000, 10000};
    
    std::string results_dir = "benchmarks/results";
    std::string cmd = "mkdir -p " + results_dir;
    system(cmd.c_str());
    
    std::ofstream csv(results_dir + "/results.csv");
    csv << "Mode,Threads,BatchSize,MatchRate,Accuracy,ExecTime,Latency,Throughput,Speedup,Efficiency\n";
    
    std::cout << "Initializing engines (loading weights)..." << std::endl;
    SequentialInference seq_engine(weights_dir);
    IntraLayerInference intra_engine(weights_dir);
    BatchInference batch_engine(weights_dir);
    PipelineInference pipe_engine(weights_dir);

    // Map to store sequential baseline times for speedup calculation
    // key: batch_size, value: execution time
    std::vector<double> seq_times(10001, 0.0);

    for (size_t batch_size : batch_sizes) {
        Tensor test_images({batch_size, 1, 28, 28});
        for (size_t i = 0; i < batch_size * 28 * 28; ++i) {
            test_images.get(i) = all_images.get(i);
        }
        
        for (const std::string& mode : modes) {
            for (int threads : thread_counts) {
                if (mode == "sequential" && threads > 1) continue; // Only run sequential once per batch size
                
                omp_set_num_threads(threads);
                
                Tensor output;
                auto start = std::chrono::high_resolution_clock::now();
                
                if (mode == "sequential") output = seq_engine.forward(test_images);
                else if (mode == "intra_layer") output = intra_engine.forward(test_images);
                else if (mode == "batch") output = batch_engine.forward(test_images);
                else if (mode == "pipeline") output = pipe_engine.forward(test_images);
                
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> exec_time = end - start;
                double t = exec_time.count();
                
                if (mode == "sequential") {
                    seq_times[batch_size] = t;
                }
                
                int match_count = 0;
                int correct_count = 0;
                for (size_t i = 0; i < output.shape[0]; ++i) {
                    float max_val = output.get2d(i, 0);
                    int max_idx = 0;
                    for (size_t c = 1; c < 10; ++c) {
                        if (output.get2d(i, c) > max_val) {
                            max_val = output.get2d(i, c);
                            max_idx = c;
                        }
                    }
                    if (max_idx == all_pt_preds[i]) match_count++;
                    if (max_idx == all_labels[i]) correct_count++;
                }
                
                double match_rate = match_count * 100.0 / batch_size;
                double accuracy = correct_count * 100.0 / batch_size;
                double latency = (t * 1000.0) / batch_size;
                double throughput = batch_size / t;
                
                double speedup = seq_times[batch_size] / t;
                double efficiency = speedup / threads;
                
                std::cout << "Run: " << mode << " | Threads: " << threads << " | Batch: " << batch_size << " | Time: " << t << "s | Speedup: " << speedup << "x" << std::endl;
                
                csv << mode << "," << threads << "," << batch_size << "," 
                    << match_rate << "," << accuracy << "," << t << "," 
                    << latency << "," << throughput << "," << speedup << "," 
                    << efficiency << "\n";
            }
        }
    }
    
    csv.close();
    std::cout << "Benchmarking complete. Results saved to " << results_dir << "/results.csv" << std::endl;
    return 0;
}
