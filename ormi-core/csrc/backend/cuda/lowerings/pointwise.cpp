#include "backend/cuda/lowering_registry.h"

namespace ormi::cuda {
    static RegisterCudaLowering reg_add(ops::OpEnum::ADD, [](const std::vector<std::string>& ops) { return ops[0] + " + " + ops[1] + ";"; });
    static RegisterCudaLowering reg_sub(ops::OpEnum::SUB, [](const std::vector<std::string>& ops) { return ops[0] + " - " + ops[1] + ";"; });
    static RegisterCudaLowering reg_mul(ops::OpEnum::MUL, [](const std::vector<std::string>& ops) { return ops[0] + " * " + ops[1] + ";"; });
    static RegisterCudaLowering reg_div(ops::OpEnum::DIV, [](const std::vector<std::string>& ops) { return ops[0] + " / " + ops[1] + ";"; });
    static RegisterCudaLowering reg_pow(ops::OpEnum::POW, [](const std::vector<std::string>& ops) { return "powf(" + ops[0] + ", " + ops[1] + ");"; });
    static RegisterCudaLowering reg_neg(ops::OpEnum::NEG, [](const std::vector<std::string>& ops) { return "-" + ops[0] + ";"; });
    static RegisterCudaLowering reg_exp(ops::OpEnum::EXP, [](const std::vector<std::string>& ops) { return "expf(" + ops[0] + ");"; });
    static RegisterCudaLowering reg_log(ops::OpEnum::LOG, [](const std::vector<std::string>& ops) { return "logf(" + ops[0] + ");"; });
    static RegisterCudaLowering reg_sin(ops::OpEnum::SIN, [](const std::vector<std::string>& ops) { return "sinf(" + ops[0] + ");"; });
    static RegisterCudaLowering reg_cos(ops::OpEnum::COS, [](const std::vector<std::string>& ops) { return "cosf(" + ops[0] + ");"; });
}