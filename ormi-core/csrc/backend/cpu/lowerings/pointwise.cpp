#include "backend/cpu/lowering_registry.h"
#include <cmath>

namespace ormi::cpu {
    static RegisterCPULowering reg_add(ops::OpEnum::ADD, [](std::vector<float>& r, int d, int s1, int s2) { r[d] = r[s1] + r[s2]; });
    static RegisterCPULowering reg_sub(ops::OpEnum::SUB, [](std::vector<float>& r, int d, int s1, int s2) { r[d] = r[s1] - r[s2]; });
    static RegisterCPULowering reg_mul(ops::OpEnum::MUL, [](std::vector<float>& r, int d, int s1, int s2) { r[d] = r[s1] * r[s2]; });
    static RegisterCPULowering reg_div(ops::OpEnum::DIV, [](std::vector<float>& r, int d, int s1, int s2) { r[d] = r[s1] / r[s2]; });
    static RegisterCPULowering reg_pow(ops::OpEnum::POW, [](std::vector<float>& r, int d, int s1, int s2) { r[d] = std::pow(r[s1], r[s2]); });
    
    static RegisterCPULowering reg_neg(ops::OpEnum::NEG, [](std::vector<float>& r, int d, int s1, int) { r[d] = -r[s1]; });
    static RegisterCPULowering reg_exp(ops::OpEnum::EXP, [](std::vector<float>& r, int d, int s1, int) { r[d] = std::exp(r[s1]); });
    static RegisterCPULowering reg_log(ops::OpEnum::LOG, [](std::vector<float>& r, int d, int s1, int) { r[d] = std::log(r[s1]); });
    static RegisterCPULowering reg_sin(ops::OpEnum::SIN, [](std::vector<float>& r, int d, int s1, int) { r[d] = std::sin(r[s1]); });
    static RegisterCPULowering reg_cos(ops::OpEnum::COS, [](std::vector<float>& r, int d, int s1, int) { r[d] = std::cos(r[s1]); });
}