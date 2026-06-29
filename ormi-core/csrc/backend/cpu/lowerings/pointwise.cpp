#include "backend/cpu/lowering_registry.h"

namespace ormi::cpu {
    static RegisterCPULowering reg_add(ops::OpEnum::ADD, [](const std::vector<std::string>& ops) { return ops[0] + " + " + ops[1] + ";"; });
    static RegisterCPULowering reg_sub(ops::OpEnum::SUB, [](const std::vector<std::string>& ops) { return ops[0] + " - " + ops[1] + ";"; });
    static RegisterCPULowering reg_mul(ops::OpEnum::MUL, [](const std::vector<std::string>& ops) { return ops[0] + " * " + ops[1] + ";"; });
    static RegisterCPULowering reg_div(ops::OpEnum::DIV, [](const std::vector<std::string>& ops) { return ops[0] + " / " + ops[1] + ";"; });
    static RegisterCPULowering reg_pow(ops::OpEnum::POW, [](const std::vector<std::string>& ops) { return "std::pow(" + ops[0] + ", " + ops[1] + ");"; });
    
    static RegisterCPULowering reg_neg(ops::OpEnum::NEG, [](const std::vector<std::string>& ops) { return "-" + ops[0] + ";"; });
    static RegisterCPULowering reg_exp(ops::OpEnum::EXP, [](const std::vector<std::string>& ops) { return "std::exp(" + ops[0] + ");"; });
    static RegisterCPULowering reg_log(ops::OpEnum::LOG, [](const std::vector<std::string>& ops) { return "std::log(" + ops[0] + ");"; });
    static RegisterCPULowering reg_sin(ops::OpEnum::SIN, [](const std::vector<std::string>& ops) { return "std::sin(" + ops[0] + ");"; });
    static RegisterCPULowering reg_cos(ops::OpEnum::COS, [](const std::vector<std::string>& ops) { return "std::cos(" + ops[0] + ");"; });
}