#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // sizes: optional per-device budget, llama_max_devices()
    llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, const size_t * sizes = nullptr);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // VRAM lending: the banks of one device are freed and allocated again with fewer slots, so another process
    // can use the memory, and later with the old slot count again. A lend moves whole slots, taken from the group
    // that keeps the largest share of its full size first; every group keeps the slots of one decode step. The
    // slots start empty after each rebuild. dev is an index into the cache's own devices, sizes are bytes

    // the cache device of a backend device, -1 if it has no cache
    int32_t device(ggml_backend_dev_t dev) const;

    // bytes the device can lend now: the slots above the working minimum of every group, within cap, in whole slots
    size_t lendable(int32_t dev, size_t cap) const;

    // free at least bytes (rounded up to whole slots, at most lendable) on the device, the caller has synchronized it
    // returns the freed bytes, 0 if the smaller buffer could not be allocated (the cache keeps its size)
    size_t lend(int32_t dev, size_t bytes);

    // take at most cap bytes back, rounded down to whole slots, the caller has synchronized the device. A return never
    // gives back more than cap: the groups' slots differ in size, so a return that rounded up could take part of the
    // next lease back and leave that one unreturnable. returned receives the bytes taken back (0 if cap is below one
    // slot). Returns false if the larger buffer could not be allocated (the cache keeps the lent size)
    bool restore(int32_t dev, size_t cap, size_t & returned);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
