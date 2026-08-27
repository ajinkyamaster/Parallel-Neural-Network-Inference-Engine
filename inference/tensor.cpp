#include "tensor.h"
#include <fstream>
#include <iostream>

Tensor::Tensor() {}

Tensor::Tensor(std::vector<size_t> shape) : shape(shape) {
    size_t total_size = 1;
    for (size_t dim : shape) {
        total_size *= dim;
    }
    data.resize(total_size, 0.0f);
}

size_t Tensor::size() const {
    return data.size();
}

bool Tensor::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open " << filepath << std::endl;
        return false;
    }
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t expected_size = data.size() * sizeof(float);
    if (file_size != (std::streamsize)expected_size) {
        std::cerr << "Size mismatch in " << filepath << ". Expected " 
                  << expected_size << " bytes but got " << file_size << std::endl;
        return false;
    }
    
    if (file.read(reinterpret_cast<char*>(data.data()), file_size)) {
        return true;
    }
    std::cerr << "Failed to read data from " << filepath << std::endl;
    return false;
}

bool Tensor::save(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open for writing: " << filepath << std::endl;
        return false;
    }
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    return true;
}
