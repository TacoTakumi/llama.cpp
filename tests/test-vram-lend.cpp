// lends part of the MoE expert cache of one device while generating, then takes it back
// usage: test-vram-lend <model.gguf> <n_cpu_moe> <cache_mib,...> <lend_max_mib,...> <dev> [devices] [--cycles N] [--two-leases]
//   n_cpu_moe:    the experts of the first n layers stay in host memory
//   cache_mib:    MoE cache per model device, comma separated, 0 = none
//   lend_max_mib: lend cap per model device, comma separated, 0 = the device does not lend
//   dev:          the model device to lend from
//   devices:      comma separated backend device names, e.g. CUDA0, default: all
//   --cycles N:   N times: generate, lend, generate, decode a long prompt, return, generate (default: one short cycle)
//   --two-leases: lend 1024 and 512 MiB, return them in both orders by the MiB the API reported, lendable must come back

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

static std::vector<std::string> split(const char * arg) {
    std::vector<std::string> res;
    std::stringstream ss(arg);
    for (std::string item; std::getline(ss, item, ',');) {
        res.push_back(item);
    }
    return res;
}

static std::vector<size_t> parse_mib(const char * arg) {
    std::vector<size_t> res(llama_max_devices(), 0);
    const auto items = split(arg);
    for (size_t i = 0; i < items.size() && i < res.size(); ++i) {
        res[i] = strtoull(items[i].c_str(), nullptr, 10)*1024*1024;
    }
    return res;
}

// prints the free memory of the device, if one was named
static void print_free(ggml_backend_dev_t dev, const char * when) {
    size_t free = 0;
    size_t total = 0;
    if (dev != nullptr) {
        ggml_backend_dev_memory(dev, &free, &total);
        printf("device free %s: %zu MiB of %zu MiB\n", when, free/1024/1024, total/1024/1024);
    }
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> res(text.size() + 16);
    const int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), res.data(), (int32_t) res.size(), true, true);
    res.resize(n > 0 ? n : 0);
    return res;
}

// decodes the tokens after n_past, returns false on a decode error
static bool decode_pending(llama_context * ctx, std::vector<llama_token> & tokens, int & n_past) {
    if (n_past < (int) tokens.size() && llama_decode(ctx, llama_batch_get_one(tokens.data() + n_past, (int32_t) tokens.size() - n_past)) != 0) {
        return false;
    }
    n_past = (int) tokens.size();
    return true;
}

// generates n tokens greedily after the tokens, appends their text, returns false on a decode error
static bool generate(llama_context * ctx, const llama_vocab * vocab, std::vector<llama_token> & tokens, int & n_past, int n, std::string & text) {
    for (int i = 0; i < n; ++i) {
        if (!decode_pending(ctx, tokens, n_past)) {
            return false;
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        llama_token best = 0;
        for (int t = 1; t < llama_vocab_n_tokens(vocab); ++t) {
            best = logits[t] > logits[best] ? t : best;
        }
        char piece[64];
        const int len = llama_token_to_piece(vocab, best, piece, sizeof(piece) - 1, 0, true);
        text.append(piece, len > 0 ? len : 0);
        tokens.push_back(best);
    }
    return true;
}

// one phase of a cycle: a fresh sequence, the prompt in one batch, then n_gen tokens, timed in two parts
// right after a rebuild the slots are empty, so the first tokens carry the uploads and the rest run on a warm cache
static bool phase(llama_context * ctx, const llama_vocab * vocab, const char * name, const std::vector<llama_token> & prompt, int n_gen, ggml_backend_dev_t dev) {
    llama_memory_clear(llama_get_memory(ctx), true);
    std::vector<llama_token> tokens = prompt;
    int n_past = 0;
    std::string text;
    const int n_first = n_gen < 50 ? n_gen : 50;

    const int64_t t0 = ggml_time_us();
    bool ok = decode_pending(ctx, tokens, n_past);
    const int64_t t1 = ggml_time_us();
    ok = ok && generate(ctx, vocab, tokens, n_past, n_first, text);
    const int64_t t2 = ggml_time_us();
    ok = ok && generate(ctx, vocab, tokens, n_past, n_gen - n_first, text);
    const int64_t t3 = ggml_time_us();

    for (char & c : text) {
        c = c == '\n' ? ' ' : c;
    }
    printf("phase %-14s: %s, prompt %4zu tokens in %6.2f s, first %3d tokens %6.1f t/s, next %3d tokens %6.1f t/s, text \"%.60s\"\n",
        name, ok ? "ok" : "DECODE FAILED", prompt.size(), (t1 - t0)/1e6,
        n_first, n_first/((t2 - t1)/1e6), n_gen - n_first, n_gen - n_first > 0 ? (n_gen - n_first)/((t3 - t2)/1e6) : 0.0, text.c_str());
    print_free(dev, name);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <model.gguf> <n_cpu_moe> <cache_mib,...> <lend_max_mib,...> <dev> [devices] [--cycles N] [--two-leases]\n", argv[0]);
        return 1;
    }
    const std::vector<size_t> cache_sizes = parse_mib(argv[3]);
    const std::vector<size_t> lend_max    = parse_mib(argv[4]);
    const int32_t dev = atoi(argv[5]);
    int n_cycles = 0;
    bool two_leases = false;
    std::vector<std::string> device_names;
    for (int i = 6; i < argc; ++i) {
        if (strcmp(argv[i], "--cycles") == 0 && i + 1 < argc) {
            n_cycles = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--two-leases") == 0) {
            two_leases = true;
        } else {
            device_names = split(argv[i]);
        }
    }

    llama_backend_init();

    // the experts of the first n_cpu_moe layers stay in host memory
    std::string pattern = "blk\\.(";
    for (int il = 0; il < atoi(argv[2]); ++il) {
        pattern += (il > 0 ? "|" : "") + std::to_string(il);
    }
    pattern += ")\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
    const llama_model_tensor_buft_override overrides[] = { { pattern.c_str(), ggml_backend_cpu_buffer_type() }, { nullptr, nullptr } };

    std::vector<ggml_backend_dev_t> devices;
    for (const std::string & name : device_names) {
        devices.push_back(ggml_backend_dev_by_name(name.c_str()));
        if (devices.back() == nullptr) {
            fprintf(stderr, "unknown device %s\n", name.c_str());
            return 1;
        }
    }
    devices.push_back(nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.devices = devices.size() > 1 ? devices.data() : nullptr;
    mparams.tensor_buft_overrides = overrides;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 1024;
    cparams.n_batch = 512;
    cparams.moe_cache_split = cache_sizes.data();
    cparams.vram_lend_max = lend_max.data();
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create the context\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const std::vector<llama_token> prompt = tokenize(vocab, "The capital of France is");
    if (prompt.empty()) {
        fprintf(stderr, "failed to tokenize the prompt\n");
        return 1;
    }
    ggml_backend_dev_t dev_ptr = (size_t) dev + 1 < devices.size() ? devices[dev] : nullptr;
    bool ok = true;
    uint64_t mib = 0;
    uint64_t lent = 0;
    int32_t status = llama_vram_lendable(ctx, dev, &mib);
    printf("lendable: status %d, %llu MiB\n", status, (unsigned long long) mib);
    ok &= status == 0;

    if (two_leases) {
        // two leases, returned by the floored MiB the API reported, as a server does; the slots of the two groups differ
        // in size, so a return must not give back more than asked or the other lease can no longer be returned
        const uint64_t lendable_before = mib;
        for (int order = 0; order < 2 && ok; ++order) {
            uint64_t lent_a = 0;
            uint64_t lent_b = 0;
            status = llama_vram_lend(ctx, dev, 1024, &lent_a);
            printf("order %d: lend A 1024 MiB: status %d, lent %llu MiB\n", order, status, (unsigned long long) lent_a);
            ok &= status == 0;
            status = llama_vram_lend(ctx, dev, 512, &lent_b);
            printf("order %d: lend B 512 MiB: status %d, lent %llu MiB\n", order, status, (unsigned long long) lent_b);
            ok &= status == 0;
            for (int i = 0; i < 2; ++i) {
                const bool first_a = (i == 0) == (order == 0);
                status = llama_vram_return(ctx, dev, first_a ? lent_a : lent_b);
                printf("order %d: return %s %llu MiB: status %d\n", order, first_a ? "A" : "B", (unsigned long long) (first_a ? lent_a : lent_b), status);
                ok &= status == 0;
            }
            status = llama_vram_lendable(ctx, dev, &mib);
            printf("order %d: lendable after: status %d, %llu MiB (before %llu MiB)\n", order, status, (unsigned long long) mib, (unsigned long long) lendable_before);
            ok &= status == 0 && mib == lendable_before;
        }
    } else if (n_cycles > 0) {
        // a long prompt takes the large-batch path, which reads the cached experts through the copy views
        std::string text;
        std::vector<llama_token> prompt_long;
        while (prompt_long.size() < 500) {
            text += "The quick brown fox jumps over the lazy dog while the sun sets behind the hills. ";
            prompt_long = tokenize(vocab, text);
        }
        for (int cycle = 1; cycle <= n_cycles && ok; ++cycle) {
            printf("cycle %d\n", cycle);
            ok &= phase(ctx, vocab, "gen-before", prompt, 200, dev_ptr);

            status = llama_vram_lend(ctx, dev, 1024, &lent);
            printf("lend 1024 MiB: status %d, lent %llu MiB\n", status, (unsigned long long) lent);
            ok &= status == 0;

            ok &= phase(ctx, vocab, "gen-lent", prompt, 200, dev_ptr);
            ok &= phase(ctx, vocab, "prompt-lent", prompt_long, 5, dev_ptr);

            status = llama_vram_return(ctx, dev, lent);
            printf("return %llu MiB: status %d\n", (unsigned long long) lent, status);
            ok &= status == 0;

            ok &= phase(ctx, vocab, "gen-after", prompt, 200, dev_ptr);
        }
    } else {
        std::vector<llama_token> tokens = prompt;
        int n_past = 0;
        std::string text;

        print_free(dev_ptr, "before lend");
        status = llama_vram_lend(ctx, dev, 1024, &lent);
        printf("lend 1024 MiB: status %d, lent %llu MiB\n", status, (unsigned long long) lent);
        ok &= status == 0;
        print_free(dev_ptr, "after lend");

        ok &= generate(ctx, vocab, tokens, n_past, 50, text);
        printf("%s\n", text.c_str());
        text.clear();

        status = llama_vram_return(ctx, dev, lent);
        printf("return %llu MiB: status %d\n", (unsigned long long) lent, status);
        ok &= status == 0;
        print_free(dev_ptr, "after return");

        ok &= generate(ctx, vocab, tokens, n_past, 50, text);
        printf("%s\n", text.c_str());
    }

    status = llama_vram_lendable(ctx, dev, &mib);
    printf("lendable after: status %d, %llu MiB\n", status, (unsigned long long) mib);
    ok &= status == 0;

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
