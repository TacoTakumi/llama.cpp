// lends part of the MoE expert cache of one device while generating, then takes it back
// usage: test-vram-lend <model.gguf> <n_cpu_moe> <cache_mib,...> <lend_max_mib,...> <dev> [devices]
//   n_cpu_moe:    the experts of the first n layers stay in host memory
//   cache_mib:    MoE cache per model device, comma separated, 0 = none
//   lend_max_mib: lend cap per model device, comma separated, 0 = the device does not lend
//   dev:          the model device to lend from
//   devices:      comma separated backend device names, e.g. CUDA0, default: all

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

// generates n tokens greedily after the tokens, returns false on a decode error
static bool generate(llama_context * ctx, const llama_vocab * vocab, std::vector<llama_token> & tokens, int & n_past, int n) {
    for (int i = 0; i < n; ++i) {
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + n_past, (int32_t) tokens.size() - n_past)) != 0) {
            return false;
        }
        n_past = (int) tokens.size();
        const float * logits = llama_get_logits_ith(ctx, -1);
        llama_token best = 0;
        for (int t = 1; t < llama_vocab_n_tokens(vocab); ++t) {
            best = logits[t] > logits[best] ? t : best;
        }
        char piece[64];
        const int len = llama_token_to_piece(vocab, best, piece, sizeof(piece) - 1, 0, true);
        printf("%.*s", len > 0 ? len : 0, piece);
        tokens.push_back(best);
    }
    printf("\n");
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <model.gguf> <n_cpu_moe> <cache_mib,...> <lend_max_mib,...> <dev> [devices]\n", argv[0]);
        return 1;
    }
    const std::vector<size_t> cache_sizes = parse_mib(argv[3]);
    const std::vector<size_t> lend_max    = parse_mib(argv[4]);
    const int32_t dev = atoi(argv[5]);

    llama_backend_init();

    // the experts of the first n_cpu_moe layers stay in host memory
    std::string pattern = "blk\\.(";
    for (int il = 0; il < atoi(argv[2]); ++il) {
        pattern += (il > 0 ? "|" : "") + std::to_string(il);
    }
    pattern += ")\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
    const llama_model_tensor_buft_override overrides[] = { { pattern.c_str(), ggml_backend_cpu_buffer_type() }, { nullptr, nullptr } };

    std::vector<ggml_backend_dev_t> devices;
    for (const std::string & name : argc > 6 ? split(argv[6]) : std::vector<std::string>()) {
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
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    cparams.moe_cache_split = cache_sizes.data();
    cparams.vram_lend_max = lend_max.data();
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create the context\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const char * prompt = "The capital of France is";
    std::vector<llama_token> tokens(64);
    const int n_prompt = llama_tokenize(vocab, prompt, (int32_t) strlen(prompt), tokens.data(), (int32_t) tokens.size(), true, true);
    if (n_prompt <= 0) {
        fprintf(stderr, "failed to tokenize the prompt\n");
        return 1;
    }
    tokens.resize(n_prompt);
    int n_past = 0;

    ggml_backend_dev_t dev_ptr = (size_t) dev + 1 < devices.size() ? devices[dev] : nullptr;
    bool ok = true;
    uint64_t mib = 0;
    uint64_t lent = 0;
    int32_t status = llama_vram_lendable(ctx, dev, &mib);
    printf("lendable: status %d, %llu MiB\n", status, (unsigned long long) mib);
    ok &= status == 0;

    print_free(dev_ptr, "before lend");
    status = llama_vram_lend(ctx, dev, 1024, &lent);
    printf("lend 1024 MiB: status %d, lent %llu MiB\n", status, (unsigned long long) lent);
    ok &= status == 0;
    print_free(dev_ptr, "after lend");

    ok &= generate(ctx, vocab, tokens, n_past, 50);

    status = llama_vram_return(ctx, dev, lent);
    printf("return %llu MiB: status %d\n", (unsigned long long) lent, status);
    ok &= status == 0;
    print_free(dev_ptr, "after return");

    ok &= generate(ctx, vocab, tokens, n_past, 50);

    status = llama_vram_lendable(ctx, dev, &mib);
    printf("lendable after: status %d, %llu MiB\n", status, (unsigned long long) mib);
    ok &= status == 0;

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
