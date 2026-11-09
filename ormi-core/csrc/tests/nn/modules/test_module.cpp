#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/error.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/nn/modules/activation.h>
#include <ormi/nn/modules/module.h>
#include <ormi/nn/modules/sequential.h>

#include "tests/nn/helpers.h"

namespace {
    class Identity final : public ormi::nn::Module {
    public:
        using Module::register_module;
        using Module::register_parameter;

        std::shared_ptr<ormi::core::Tensor> forward(const std::shared_ptr<ormi::core::Tensor>& input
        ) const override {
            return input;
        }
    };

    TEST(NNModuleTest, collectsSharedParametersOnceInRegistrationOrder) {
        auto first = std::make_shared<Identity>();
        auto second = std::make_shared<Identity>();
        const auto shared = nn_test::leaf({2});
        const auto other = nn_test::leaf({3});
        first->register_parameter(shared);
        second->register_parameter(shared);
        second->register_parameter(other);
        ormi::nn::Sequential model({first, second, first});

        const auto parameters = model.parameters();
        ASSERT_EQ(parameters.size(), 2u);
        EXPECT_EQ(parameters[0], shared);
        EXPECT_EQ(parameters[1], other);
        EXPECT_EQ(model.children().size(), 3u);
    }

    TEST(NNModuleTest, ownsRegisteredParametersAndChildren) {
        auto child = std::make_shared<Identity>();
        auto parameter = nn_test::leaf({2});
        std::weak_ptr<Identity> child_lifetime = child;
        std::weak_ptr<ormi::core::Tensor> parameter_lifetime = parameter;
        {
            ormi::nn::Sequential parent({child});
            child->register_parameter(parameter);
            child.reset();
            parameter.reset();
            EXPECT_FALSE(child_lifetime.expired());
            EXPECT_FALSE(parameter_lifetime.expired());
            EXPECT_EQ(parent.parameters().size(), 1u);
        }
        EXPECT_TRUE(child_lifetime.expired());
        EXPECT_TRUE(parameter_lifetime.expired());
    }

    TEST(NNModuleTest, collectsOwnThenNestedParametersInDepthFirstOrder) {
        auto parent = std::make_shared<Identity>();
        auto first = std::make_shared<Identity>();
        auto nested = std::make_shared<Identity>();
        auto second = std::make_shared<Identity>();
        const auto parent_parameter = nn_test::leaf({1});
        const auto first_parameter = nn_test::leaf({2});
        const auto nested_parameter = nn_test::leaf({3});
        const auto second_parameter = nn_test::leaf({4});

        parent->register_parameter(parent_parameter);
        first->register_parameter(first_parameter);
        nested->register_parameter(nested_parameter);
        second->register_parameter(second_parameter);
        second->register_parameter(first_parameter);

        parent->register_module(first);
        parent->register_module(second);
        first->register_module(nested);
        second->register_module(nested);

        const std::vector<std::shared_ptr<ormi::core::Tensor>>
            expected{parent_parameter, first_parameter, nested_parameter, second_parameter};
        EXPECT_EQ(parent->parameters(), expected);
    }

    TEST(NNModuleTest, rejectsInvalidParameters) {
        Identity module;
        EXPECT_THROW(module.register_parameter(nullptr), ormi::OperationError);
        const auto input = nn_test::leaf({2});
        EXPECT_THROW(module.register_parameter(ormi::core::ops::sin(input)), ormi::OperationError);
        const auto integer = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::int32
        );
        EXPECT_THROW(module.register_parameter(integer), ormi::DTypeError);
        EXPECT_TRUE(module.parameters().empty());
    }

    TEST(NNModuleTest, rejectsOwnershipCyclesWithoutRetainingThem) {
        auto first = std::make_shared<ormi::nn::Sequential>();
        auto second = std::make_shared<ormi::nn::Sequential>();
        first->append(second);
        EXPECT_THROW(second->append(first), ormi::OperationError);
        EXPECT_THROW(first->append(first), ormi::OperationError);
        EXPECT_THROW(first->append(nullptr), ormi::OperationError);
        EXPECT_TRUE(second->children().empty());
        EXPECT_EQ(first->children().size(), 1u);
    }

    TEST(NNModuleTest, handlesNestedAndEmptySequences) {
        const auto input = nn_test::leaf({2});
        auto nested = std::make_shared<ormi::nn::Sequential>();
        ormi::nn::Sequential model({nested});
        EXPECT_EQ(model(input), input);
        EXPECT_TRUE(model.parameters().empty());
        nested->append(std::make_shared<Identity>());
        EXPECT_EQ(model(input), input);
        EXPECT_THROW(model(nullptr), ormi::OperationError);
    }

    TEST(NNModuleTest, defersAppendedLayersUntilTheNextForwardCall) {
        class AppendOnce final : public ormi::nn::Module {
        public:
            explicit AppendOnce(ormi::nn::Sequential& sequence)
                : sequence_(sequence) {}

            std::shared_ptr<ormi::core::Tensor> forward(const std::shared_ptr<ormi::core::Tensor>& input
            ) const override {
                if (!appended_) {
                    for (std::size_t index = 0; index < 32; ++index) {
                        sequence_.append(std::make_shared<ormi::nn::Sin>());
                    }
                    appended_ = true;
                }
                return input;
            }

        private:
            ormi::nn::Sequential& sequence_;
            mutable bool appended_ = false;
        };

        ormi::nn::Sequential sequence;
        sequence.append(std::make_shared<AppendOnce>(sequence));
        const auto input = nn_test::leaf({1}, 0.5);
        EXPECT_EQ(sequence(input), input);
        EXPECT_EQ(sequence.children().size(), 33u);

        const auto output = sequence(input);
        EXPECT_FALSE(output->realized());
        auto expected = 0.5;
        for (std::size_t index = 0; index < 32; ++index) {
            expected = std::sin(expected);
        }
        EXPECT_NEAR(*static_cast<const double*>(output->data()), expected, 1e-12);
    }
}
