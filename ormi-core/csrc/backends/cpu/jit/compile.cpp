#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ormi/core/error.h>

#include "backends/cpu/jit/artifact.h"
#include "backends/cpu/jit/codegen.h"
#include "backends/cpu/jit/compile.h"
#include "backends/cpu/jit/loop.h"
#include "core/runtime/fusion/key.h"

namespace ormi::backends::cpu::jit {
    namespace {
        artifact::Kernel launch(
            const core::runtime::fusion::Expr& expression,
            const Loop& loop
        ) {
            const auto range = loop.partial && loop.strided
                ? expression.shape[*loop.split]
                : expression.elements;
            return artifact::Kernel{
                parallel_work(expression, false),
                parallel_work(expression, true),
                loop.ranged || loop.partial ? loop.unit : 0,
                loop.partial ? range : 0,
                loop.partial,
                loop.math
            };
        }

        struct Unit {
            explicit Unit(const core::runtime::fusion::Expr& value)
                : expression(&value),
                  key(value),
                  loop(plan(value)),
                  launch_plan(launch(value, loop)),
                  generated(emit(value, loop)) {}

            const core::runtime::fusion::Expr* expression;
            core::runtime::fusion::Key key;
            Loop loop;
            artifact::Kernel launch_plan;
            std::string generated;
            std::shared_ptr<Compiled> compiled;
        };

        struct Entry {
            Entry(
                const core::runtime::fusion::Expr& expression,
                std::shared_ptr<Compiled> compiled_kernel
            ) : key(expression), compiled(std::move(compiled_kernel)) {}

            core::runtime::fusion::Key key;
            std::shared_ptr<Compiled> compiled;
        };

        using Cache = std::unordered_map<std::uint64_t, std::vector<Entry>>;

        struct FrontCache {
            std::shared_ptr<const Cache> owner;
            std::array<const Entry*, 4> entries{};
        };

        std::mutex cache_mutex;
        std::atomic<std::shared_ptr<const Cache>> published{
            std::make_shared<const Cache>()
        };
        thread_local FrontCache front;

        const Entry* find_entry(
            const Cache& cache,
            const core::runtime::fusion::Expr& expression,
            std::uint64_t hash
        ) {
            const auto found = cache.find(hash);
            if (found == cache.end()) {
                return nullptr;
            }
            for (const auto& entry : found->second) {
                if (entry.key.matches(expression)) {
                    return &entry;
                }
            }
            return nullptr;
        }

        void remember(
            std::shared_ptr<const Cache> owner,
            const Entry& entry
        ) {
            if (front.owner.get() != owner.get()) {
                front.owner = std::move(owner);
                front.entries.fill(nullptr);
            }
            for (std::size_t i = front.entries.size() - 1; i > 0; --i) {
                front.entries[i] = front.entries[i - 1];
            }
            front.entries[0] = &entry;
        }

        std::shared_ptr<Compiled> find_memory(
            const core::runtime::fusion::Expr& expression
        ) {
            const auto hash = core::runtime::fusion::Key::hash(expression);
            for (std::size_t i = 0; i < front.entries.size(); ++i) {
                const auto* entry = front.entries[i];
                if (
                    entry &&
                    entry->key.hash() == hash &&
                    entry->key.matches(expression)
                ) {
                    std::swap(front.entries[0], front.entries[i]);
                    return entry->compiled;
                }
            }

            auto current = published.load(std::memory_order_acquire);
            const auto* entry = find_entry(*current, expression, hash);
            if (!entry) {
                return {};
            }
            remember(std::move(current), *entry);
            return entry->compiled;
        }

        void compile(std::span<Unit* const> units) {
            if (units.size() == 1) {
                const std::array kernels{units[0]->launch_plan};
                units[0]->compiled = artifact::get(
                    units[0]->generated,
                    kernels
                ).front();
                return;
            }

            std::vector<const core::runtime::fusion::Expr*> expressions;
            std::vector<Loop> loops;
            std::vector<artifact::Kernel> kernels;
            expressions.reserve(units.size());
            loops.reserve(units.size());
            kernels.reserve(units.size());
            for (const auto* unit : units) {
                expressions.push_back(unit->expression);
                loops.push_back(unit->loop);
                kernels.push_back(unit->launch_plan);
            }

            auto compiled = artifact::get(emit(expressions, loops), kernels);
            for (std::size_t i = 0; i < units.size(); ++i) {
                units[i]->compiled = std::move(compiled[i]);
            }
        }
    }

    std::shared_ptr<Compiled> compile(
        const core::runtime::fusion::Expr& expression
    ) {
        const core::runtime::fusion::Expr* expressions[]{&expression};
        return compile(expressions).front();
    }

    std::vector<std::shared_ptr<Compiled>> compile(
        std::span<const core::runtime::fusion::Expr* const> expressions
    ) {
        std::vector<std::shared_ptr<Compiled>> result(expressions.size());
        bool complete = true;
        for (std::size_t i = 0; i < expressions.size(); ++i) {
            if (!expressions[i]) {
                ORMI_THROW(OperationError, "Cannot compile a null fusion expression.");
            }
            result[i] = find_memory(*expressions[i]);
            complete = complete && static_cast<bool>(result[i]);
        }
        if (complete) {
            return result;
        }

        std::lock_guard lock(cache_mutex);
        auto current = published.load(std::memory_order_acquire);
        std::vector<Unit> units;
        std::vector<std::size_t> unit_for(expressions.size(), expressions.size());

        for (std::size_t i = 0; i < expressions.size(); ++i) {
            const auto hash = core::runtime::fusion::Key::hash(*expressions[i]);
            if (const auto* entry = find_entry(*current, *expressions[i], hash)) {
                result[i] = entry->compiled;
                continue;
            }

            for (std::size_t unit = 0; unit < units.size(); ++unit) {
                if (
                    units[unit].key.hash() == hash &&
                    units[unit].key.matches(*expressions[i])
                ) {
                    unit_for[i] = unit;
                    break;
                }
            }
            if (unit_for[i] == expressions.size()) {
                unit_for[i] = units.size();
                units.emplace_back(*expressions[i]);
            }
        }

        std::vector<Unit*> missing;
        for (auto& unit : units) {
            unit.compiled = artifact::find(
                unit.generated,
                unit.launch_plan
            );
            if (!unit.compiled) {
                missing.push_back(&unit);
            }
        }
        if (!missing.empty()) {
            compile(missing);
        }

        auto next = std::make_shared<Cache>(*current);
        for (const auto& unit : units) {
            (*next)[unit.key.hash()].emplace_back(
                *unit.expression,
                unit.compiled
            );
        }
        published.store(next, std::memory_order_release);

        for (std::size_t i = 0; i < expressions.size(); ++i) {
            const auto hash = core::runtime::fusion::Key::hash(*expressions[i]);
            const auto* entry = find_entry(*next, *expressions[i], hash);
            result[i] = entry->compiled;
            remember(next, *entry);
        }
        return result;
    }
}
