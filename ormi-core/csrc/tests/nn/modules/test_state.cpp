#include <memory>
#include <string>

#include <gtest/gtest.h>

#include <ormi/core/error.h>
#include <ormi/nn/modules/linear.h>
#include <ormi/nn/modules/module.h>
#include <ormi/nn/modules/sequential.h>

#include "tests/nn/helpers.h"

namespace {
    class Identity final : public ormi::nn::Module {
    public:
        using Module::register_buffer;
        using Module::register_module;
        using Module::register_parameter;

        std::shared_ptr<ormi::core::Tensor> forward(const std::shared_ptr<ormi::core::Tensor>& input
        ) const override {
            return input;
        }
    };

    TEST(NNModuleStateTest, namesNestedStateAndKeepsEverySharedPath) {
        const auto weight = nn_test::leaf({2, 2});
        const auto layer = std::make_shared<ormi::nn::Linear>(weight);
        const auto layers = std::make_shared<ormi::nn::Sequential>(
            std::vector<std::shared_ptr<ormi::nn::Module>>{layer, layer}
        );
        Identity model;
        const auto buffer = nn_test::leaf({1});
        model.register_buffer(buffer, "count");
        model.register_module(layers, "layers");
        const auto state = model.state();
        ASSERT_EQ(state.size(), 3u);
        EXPECT_EQ(state.at("layers.0.weight"), weight);
        EXPECT_EQ(state.at("layers.1.weight"), weight);
        EXPECT_EQ(state.at("count"), buffer);
        EXPECT_EQ(model.parameters(), (std::vector{weight}));
        EXPECT_EQ(model.buffers(), (std::vector{buffer}));
        layers->append(std::make_shared<ormi::nn::Linear>(2, 3));
        EXPECT_EQ(model.state().at("layers.0.weight"), weight);
        EXPECT_EQ(model.state().count("layers.2.bias"), 1u);
        EXPECT_EQ(state.size(), 3u);
    }

    TEST(NNModuleStateTest, rejectsDuplicateAndAmbiguousNamesBeforeRegistration) {
        Identity model;
        const auto tensor = nn_test::leaf({1});
        const auto child = std::make_shared<Identity>();
        model.register_parameter(tensor, "weight");
        EXPECT_THROW(model.register_parameter(tensor, "weight"), ormi::OperationError);
        EXPECT_THROW(model.register_buffer(tensor, "weight"), ormi::OperationError);
        EXPECT_THROW(model.register_module(child, "weight"), ormi::OperationError);
        for (const auto& name :
             {std::string("a.b"), std::string("a/b"), std::string("a\\b"), std::string("a\0b", 3)}) {
            EXPECT_THROW(model.register_parameter(tensor, name), ormi::OperationError);
            EXPECT_THROW(model.register_buffer(tensor, name), ormi::OperationError);
            EXPECT_THROW(model.register_module(child, name), ormi::OperationError);
        }
        EXPECT_EQ(model.state().size(), 1u);
        EXPECT_TRUE(model.buffers().empty());
        EXPECT_TRUE(model.children().empty());
    }

    TEST(NNModuleStateTest, retainsLiveTensorsAndSupportsUnnamedRegistrations) {
        std::weak_ptr<ormi::core::Tensor> lifetime;
        std::map<std::string, std::shared_ptr<ormi::core::Tensor>> state;
        {
            Identity model;
            const auto tensor = nn_test::leaf({1});
            lifetime = tensor;
            model.register_parameter(tensor);
            model.register_buffer(tensor);
            model.register_module(std::make_shared<Identity>());
            state = model.state();
            EXPECT_EQ(state.at("parameter_0"), tensor);
            EXPECT_EQ(state.at("buffer_0"), tensor);
        }
        EXPECT_FALSE(lifetime.expired());
        state.clear();
        EXPECT_TRUE(lifetime.expired());
    }
}
