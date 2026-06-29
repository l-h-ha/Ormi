#pragma once

namespace ormi::cuda {
    struct CudaConfig {
        int optimizer_threads = 256;
        int elementwise_threads = 256;
        int reduction_threads = 256;
    };

    CudaConfig get_config();
    void set_config(const CudaConfig& config);

    int get_optimizer_threads();
    int get_elementwise_threads();
    int get_reduction_threads();
    std::ostream& operator<<(std::ostream& os, const CudaConfig& config);
}