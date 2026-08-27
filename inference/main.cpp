#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <iomanip>

#include "tensor.h"
#include "sequential/sequential.h"
#include "parallel/intra_layer.h"
#include "parallel/batch.h"
#include "pipeline/pipeline.h"

#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>

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

// Helper to read float32 binary files (for demo mode)
bool load_float32_bin(const std::string& filepath, std::vector<float>& data) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) return false;
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    data.resize(size / sizeof(float));
    if (file.read(reinterpret_cast<char*>(data.data()), size)) return true;
    return false;
}

double run_engine_get_time(const std::string& mode, const std::string& weights_dir, const Tensor& input, Tensor& output, int threads) {
    omp_set_num_threads(threads);
    auto start = std::chrono::high_resolution_clock::now();
    
    if (mode == "sequential") {
        SequentialInference engine(weights_dir);
        output = engine.forward(input);
    } else if (mode == "intra_layer") {
        IntraLayerInference engine(weights_dir);
        output = engine.forward(input);
    } else if (mode == "batch") {
        BatchInference engine(weights_dir);
        output = engine.forward(input);
    } else if (mode == "pipeline") {
        PipelineInference engine(weights_dir);
        output = engine.forward(input, ParallelMode::SEQUENTIAL);
    } else if (mode == "pipeline_nested") {
        PipelineInference engine(weights_dir);
        output = engine.forward(input, ParallelMode::INTRA_LAYER);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> exec_time = end - start;
    return exec_time.count();
}

double calc_match_rate(const Tensor& output, const std::vector<int32_t>& seq_preds) {
    int match_count = 0;
    for (size_t i = 0; i < output.shape[0]; ++i) {
        float max_val = output.get2d(i, 0);
        int max_idx = 0;
        for (size_t c = 1; c < 10; ++c) {
            if (output.get2d(i, c) > max_val) {
                max_val = output.get2d(i, c);
                max_idx = c;
            }
        }
        if (max_idx == seq_preds[i]) {
            match_count++;
        }
    }
    return match_count * 100.0 / output.shape[0];
}

int main(int argc, char* argv[]) {
    std::string mode = "sequential";
    int threads = 1;
    size_t batch_size = 1000;
    bool verify = false;
    std::string input_dir = "";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (arg == "--batch" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--verify") {
            verify = true;
        } else if (arg == "--input-dir" && i + 1 < argc) {
            input_dir = argv[++i];
        }
    }
    
    std::string weights_dir = "training/models/weights";
    std::string data_dir = "training/models/data";
    
    if (verify) {
        std::cout << "Loading test data for verification...\n";
        Tensor all_images({10000, 1, 28, 28});
        all_images.load(data_dir + "/test_images.bin");
        Tensor test_images({batch_size, 1, 28, 28});
        for (size_t i = 0; i < batch_size * 28 * 28; ++i) test_images.get(i) = all_images.get(i);
        
        std::cout << "========================================================\n";
        std::cout << " Inference Engine Verification — Batch: " << batch_size << ", Threads: " << threads << "\n";
        std::cout << "========================================================\n";
        std::cout << "Mode          Time (s)   Speedup   Accuracy Match\n";
        std::cout << "--------------------------------------------------------\n";
        
        Tensor seq_out, intra_out, batch_out, pipe_out;
        
        double seq_t = run_engine_get_time("sequential", weights_dir, test_images, seq_out, 1);
        
        std::vector<int32_t> seq_preds(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            float max_val = seq_out.get2d(i, 0);
            int max_idx = 0;
            for (size_t c = 1; c < 10; ++c) {
                if (seq_out.get2d(i, c) > max_val) {
                    max_val = seq_out.get2d(i, c);
                    max_idx = c;
                }
            }
            seq_preds[i] = max_idx;
        }
        
        std::cout << std::left << std::setw(14) << "Sequential" 
                  << std::fixed << std::setprecision(2) << std::setw(11) << seq_t
                  << std::setw(10) << "1.00x"
                  << "100.0%\n";
                  
        double intra_t = run_engine_get_time("intra_layer", weights_dir, test_images, intra_out, threads);
        double intra_match = calc_match_rate(intra_out, seq_preds);
        std::cout << std::left << std::setw(14) << "Intra-Layer" 
                  << std::fixed << std::setprecision(2) << std::setw(11) << intra_t
                  << std::setw(10) << std::to_string(seq_t / intra_t).substr(0, 4) + "x"
                  << std::fixed << std::setprecision(1) << intra_match << "%\n";
                  
        double batch_t = run_engine_get_time("batch", weights_dir, test_images, batch_out, threads);
        double batch_match = calc_match_rate(batch_out, seq_preds);
        std::cout << std::left << std::setw(14) << "Batch" 
                  << std::fixed << std::setprecision(2) << std::setw(11) << batch_t
                  << std::setw(10) << std::to_string(seq_t / batch_t).substr(0, 4) + "x"
                  << std::fixed << std::setprecision(1) << batch_match << "%\n";
                  
        double pipe_t = run_engine_get_time("pipeline", weights_dir, test_images, pipe_out, threads);
        double pipe_match = calc_match_rate(pipe_out, seq_preds);
        std::cout << std::left << std::setw(14) << "Pipeline" 
                  << std::fixed << std::setprecision(2) << std::setw(11) << pipe_t
                  << std::setw(10) << std::to_string(seq_t / pipe_t).substr(0, 4) + "x"
                  << std::fixed << std::setprecision(1) << pipe_match << "%\n";
                  
        std::cout << "========================================================\n";
        
        if (intra_match < 100.0 || batch_match < 100.0 || pipe_match < 100.0) {
            std::cerr << "\n*** ERROR: ACCURACY DIVERGENCE DETECTED! ***\n";
        }
        
        return 0;
    }
    
    if (mode == "demo") {
        std::vector<std::string> image_files;
        if (input_dir != "") {
            DIR *dir;
            struct dirent *ent;
            if ((dir = opendir(input_dir.c_str())) != NULL) {
                while ((ent = readdir(dir)) != NULL) {
                    std::string fname = ent->d_name;
                    if (fname.find(".bin") != std::string::npos) {
                        image_files.push_back(input_dir + "/" + fname);
                    }
                }
                closedir(dir);
            }
            std::sort(image_files.begin(), image_files.end());
        }
        
        if (image_files.empty()) {
            std::cerr << "No .bin files found in " << input_dir << "\n";
            std::cerr << "Format Note: Images must be 28x28 grayscale, saved as raw float32 binary (.bin) format (784 floats per file).\n";
            std::cerr << "Values should be normalized to [0, 1] if matching PyTorch ToTensor() transforms.\n";
            return 1;
        }
        
        size_t num_images = image_files.size();
        Tensor demo_images({num_images, 1, 28, 28});
        for (size_t i = 0; i < num_images; ++i) {
            std::vector<float> data;
            load_float32_bin(image_files[i], data);
            for (size_t j = 0; j < 784 && j < data.size(); ++j) {
                demo_images.get4d(i, 0, j / 28, j % 28) = data[j];
            }
        }
        
        std::cout << "========================================================\n";
        std::cout << " Demo 4a: Single Image — Sequential vs Intra-Layer\n";
        std::cout << "========================================================\n";
        std::cout << "Mode/Threads  Time (s)   Speedup\n";
        std::cout << "--------------------------------------------------------\n";
        
        Tensor single_img({1, 1, 28, 28});
        for (size_t i = 0; i < 784; ++i) single_img.get(i) = demo_images.get(i); // get first image
        
        Tensor out;
        double seq_t = run_engine_get_time("sequential", weights_dir, single_img, out, 1);
        std::cout << std::left << std::setw(14) << "Seq (1)" 
                  << std::fixed << std::setprecision(5) << std::setw(11) << seq_t
                  << "1.00x\n";
                  
        std::vector<int> t_counts = {2, 4, 8, 16};
        for (int t : t_counts) {
            double intra_t = run_engine_get_time("intra_layer", weights_dir, single_img, out, t);
            std::cout << std::left << std::setw(14) << ("Intra (" + std::to_string(t) + ")") 
                      << std::fixed << std::setprecision(5) << std::setw(11) << intra_t
                      << std::fixed << std::setprecision(2) << (seq_t / intra_t) << "x\n";
        }
        std::cout << "========================================================\n\n";
        
        std::cout << "========================================================\n";
        std::cout << " Demo 4b: Multiple Images (" << num_images << ") — Pipeline + Intra-Layer Nested\n";
        std::cout << "========================================================\n";
        std::cout << "Note: Splitting a small thread pool (e.g. 16 threads) across multiple\n";
        std::cout << "pipeline stages leaves few threads per stage, so visible speedup here\n";
        std::cout << "will likely be modest even though the approach is architecturally\n";
        std::cout << "correct and would scale better with more threads/cores.\n";
        std::cout << "--------------------------------------------------------\n";
        
        double seq_t_multi = run_engine_get_time("sequential", weights_dir, demo_images, out, 1);
        std::cout << std::left << std::setw(14) << "Seq (1)" 
                  << std::fixed << std::setprecision(5) << std::setw(11) << seq_t_multi
                  << "1.00x\n";
        
        // Enable nested parallelism
        omp_set_max_active_levels(2);
        
        for (int t : t_counts) {
            double pipe_nested_t = run_engine_get_time("pipeline_nested", weights_dir, demo_images, out, t);
            std::cout << std::left << std::setw(14) << ("Pipe+Intra(" + std::to_string(t) + ")") 
                      << std::fixed << std::setprecision(5) << std::setw(11) << pipe_nested_t
                      << std::fixed << std::setprecision(2) << (seq_t_multi / pipe_nested_t) << "x\n";
        }
        std::cout << "========================================================\n";
        
        return 0;
    }
    
    // Standard execution logic
    omp_set_num_threads(threads);
    
    std::cout << "Loading test data..." << std::endl;
    Tensor test_images({batch_size, 1, 28, 28});
    Tensor all_images({10000, 1, 28, 28});
    if (!all_images.load(data_dir + "/test_images.bin")) return 1;
    
    std::vector<int32_t> all_labels, all_pt_preds;
    if (!load_int32_bin(data_dir + "/test_labels.bin", all_labels)) return 1;
    if (!load_int32_bin(data_dir + "/test_preds.bin", all_pt_preds)) return 1;
    
    for (size_t i = 0; i < batch_size * 28 * 28; ++i) test_images.get(i) = all_images.get(i);
    
    std::cout << "Initializing " << mode << " inference engine..." << std::endl;
    Tensor output;
    double t = run_engine_get_time(mode, weights_dir, test_images, output, threads);
    
    int match_count = 0, correct_count = 0;
    for (size_t i = 0; i < output.shape[0]; ++i) {
        float max_val = output.get2d(i, 0);
        int max_idx = 0;
        for (size_t c = 1; c < 10; ++c) {
            if (output.get2d(i, c) > max_val) { max_val = output.get2d(i, c); max_idx = c; }
        }
        if (max_idx == all_pt_preds[i]) match_count++;
        if (max_idx == all_labels[i]) correct_count++;
    }
    
    double match_rate = match_count * 100.0 / output.shape[0];
    double test_acc = correct_count * 100.0 / output.shape[0];
    
    std::cout << "========================================" << std::endl;
    std::cout << " Neural Network Inference Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Mode: " << mode << " | Threads: " << threads << " | Batch: " << batch_size << std::endl;
    std::cout << "Match Rate vs PyTorch: " << match_rate << "% | Accuracy: " << test_acc << "%" << std::endl;
    std::cout << "Time: " << t << " sec | Latency: " << (t * 1000.0 / batch_size) << " ms | Throughput: " << (batch_size / t) << " im/s\n";
    std::cout << "========================================" << std::endl;
    
    return 0;
}
