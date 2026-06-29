#pragma once
#include <memory>
#include <memory_resource>

namespace ormi {
    inline std::pmr::unsynchronized_pool_resource& get_node_pool() {
        thread_local std::pmr::unsynchronized_pool_resource pool;
        return pool;
    }
    
    template<typename T>
    inline std::pmr::polymorphic_allocator<T> get_node_allocator() {
        return std::pmr::polymorphic_allocator<T>(&get_node_pool());
    }
}