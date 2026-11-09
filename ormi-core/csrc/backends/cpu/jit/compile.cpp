#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ormi/core/error.h>

#include "backends/cpu/jit/artifact.h"
#include "backends/cpu/jit/cache.h"
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
                  key(value) {}

            const core::runtime::fusion::Expr* expression;
            core::runtime::fusion::Key key;
            Loop loop;
            artifact::Kernel launch_plan{};
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

        // entries points into owner. Keep that snapshot alive while another thread publishes
        // a replacement table. Replacing published does not invalidate these pointers.
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

        void compile_missing(std::span<Unit* const> units) {
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

    bool cached(const core::runtime::fusion::Expr& expression) {
        if (find_memory(expression)) {
            return true;
        }

        std::lock_guard lock(cache_mutex);
        auto current = published.load(std::memory_order_acquire);
        const core::runtime::fusion::Key key(expression);
        if (const auto* entry = find_entry(*current, expression, key.hash())) {
            remember(std::move(current), *entry);
            return true;
        }
        // Only positive scans are remembered. Every artifact lookup still validates its files.
        static std::filesystem::path populated_directory;
        const auto directory = cache::directory();
        if (populated_directory != directory) {
            if (!cache::has_entries()) {
                return false;
            }
            populated_directory = directory;
        }

        const auto loop = plan(expression);
        auto compiled = artifact::find(emit(expression, loop), launch(expression, loop));
        if (!compiled) {
            return false;
        }

        auto next = std::make_shared<Cache>(*current);
        auto& entries = (*next)[key.hash()];
        entries.emplace_back(expression, std::move(compiled));
        published.store(next, std::memory_order_release);
        remember(std::move(next), entries.back());
        return true;
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

        // Recheck the in-memory cache under the lock, then collect unique misses.
        for (std::size_t i = 0; i < expressions.size(); ++i) {
            const auto hash = core::runtime::fusion::Key::hash(*expressions[i]);
            if (const auto* entry = find_entry(*current, *expressions[i], hash)) {
                result[i] = entry->compiled;
                continue;
            }

            bool duplicate = false;
            for (const auto& unit : units) {
                if (
                    unit.key.hash() == hash &&
                    unit.key.matches(*expressions[i])
                ) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                units.emplace_back(*expressions[i]);
            }
        }

        // Planning and source generation happen only after duplicate requests are removed.
        for (auto& unit : units) {
            unit.loop = plan(*unit.expression);
            unit.launch_plan = launch(*unit.expression, unit.loop);
            unit.generated = emit(*unit.expression, unit.loop);
        }

        // Load cached libraries from disk before compiling the remaining kernels.
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
            compile_missing(missing);
        }

        // Replace the shared cache table only after every requested kernel has loaded.
        auto next = std::make_shared<Cache>(*current);
        for (const auto& unit : units) {
            (*next)[unit.key.hash()].emplace_back(
                *unit.expression,
                unit.compiled
            );
        }
        published.store(next, std::memory_order_release);

        // Return kernels in request order, including duplicates and prior memory hits.
        for (std::size_t i = 0; i < expressions.size(); ++i) {
            const auto hash = core::runtime::fusion::Key::hash(*expressions[i]);
            const auto* entry = find_entry(*next, *expressions[i], hash);
            result[i] = entry->compiled;
            remember(next, *entry);
        }
        return result;
    }
}
