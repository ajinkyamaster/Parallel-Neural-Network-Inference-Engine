#include <iostream>
#include <omp.h>
#include <algorithm>

int main() {
    int requested = 4;
    omp_set_max_active_levels(2);
    omp_set_num_threads(requested);
    
    int active_stages = 4;
    int threads_per_stage = std::max(1, requested / active_stages);
    
    #pragma omp parallel
    #pragma omp single
    {
        for (int i = 0; i < 4; ++i) {
            #pragma omp task
            {
                omp_set_num_threads(threads_per_stage);
                #pragma omp parallel
                {
                    #pragma omp critical
                    {
                        std::cout << "Outer thread " << omp_get_ancestor_thread_num(1) 
                                  << " executing task " << i 
                                  << ", Nested team size: " << omp_get_num_threads() << std::endl;
                    }
                }
            }
        }
    }
    return 0;
}
