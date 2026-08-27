#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include <cstddef>

class Tensor {
public:
    std::vector<size_t> shape;
    std::vector<float> data;

    Tensor();
    Tensor(std::vector<size_t> shape);
    
    // Load from binary file (must match expected size based on shape)
    bool load(const std::string& filepath);
    
    // Save to binary file
    bool save(const std::string& filepath) const;
    
    // Total number of elements
    size_t size() const;
    
    // 1D flat access
    float get(size_t i) const { return data[i]; }
    float& get(size_t i) { return data[i]; }
    
    // Fast inline multi-dimensional access methods
    float get2d(size_t i, size_t j) const {
        return data[i * shape[1] + j];
    }
    float& get2d(size_t i, size_t j) {
        return data[i * shape[1] + j];
    }
    
    float get3d(size_t i, size_t j, size_t k) const {
        return data[(i * shape[1] + j) * shape[2] + k];
    }
    float& get3d(size_t i, size_t j, size_t k) {
        return data[(i * shape[1] + j) * shape[2] + k];
    }

    float get4d(size_t i, size_t j, size_t k, size_t l) const {
        return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
    }
    float& get4d(size_t i, size_t j, size_t k, size_t l) {
        return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
    }
};

#endif
