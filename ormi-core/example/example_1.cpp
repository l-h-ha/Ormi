#include <iostream>
#include <memory>
#include <vector>

#include "core/node.h"
#include "core/buffer.h"
#include "core/autograd.h"
#include "core/optim/sgd.h"
#include "core/tapes/compute_tape.h"
#include "core/allocator.h"

#include "core/ops/blas/matmul.h"
#include "core/ops/pointwise/add.h"
#include "core/ops/pointwise/sub.h"
#include "core/ops/pointwise/mul.h"
#include "core/ops/reduce/sum.h"

using namespace ormi;

int main() {
    Device device = Device::cpu();
    DType dtype = DType::Float32;

    int batch = 32;
    int in_dim = 8;
    int out_dim = 1;

    // Training data
    auto X = full({batch, in_dim}, 1.0f, device, dtype, false);
    auto Y = full({batch, out_dim}, 2.0f, device, dtype, false);

    // Parameters
    auto W = full({in_dim, out_dim}, 0.01f, device, dtype, true);
    auto b = full({batch, out_dim}, 0.0f, device, dtype, true);

    W->set_name("W");
    b->set_name("b");

    // Forward pass:
    // pred = X @ W + b
    auto pred = ops::add(
        ops::matmul(X, W),
        b
    );

    // Loss:
    // loss = sum((pred - Y)^2)
    auto error = ops::sub(pred, Y);
    auto loss = ops::sum(
        ops::mul(error, error)
    );

    // Explicit gradients
    auto params = std::vector<std::shared_ptr<Node>>{W, b};
    auto grads = grad(loss, params);

    // Optimizer uses explicit param-gradient pairs
    optim::SGD optimizer({
        {W, grads[0]},
        {b, grads[1]}
    }, 1e-3f);

    // Record the static compute graph
    ComputeTape tape;
    tape.capture({
        loss,
        grads[0],
        grads[1]
    });

    // Training loop
    for (int epoch = 0; epoch <= 100; ++epoch) {
        tape.execute();
        optimizer.step();

        if (epoch % 1 == 0) {
            std::cout << "epoch=" << epoch
                      << " loss=" << to_string(loss->data)
                      << "\n";
        }
    }

    get_allocator(device)->empty_cache();

    return 0;
}