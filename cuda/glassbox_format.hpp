#pragma once
// ============================================================================
// Glassbox-100M Shared Binary Format & Layout Header (C++17 / CUDA C++17)
// Shared identically by:
//   - cuda/train_glassbox_100m.cu   (Pure CUDA C++17 GPU Trainer, zero PyTorch)
//   - cuda/train_glassbox_cpu.cpp   (Native C++17 CPU Reference Trainer)
//   - glassbox/checkpoint.py        (Python Glassbox .gbx binary format)
//   - glassbox/shards.py            (Python Glassbox .gbxtok mmap shards)
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace glassbox {

// Naturally 8-byte aligned 72-byte .gbx checkpoint header (matches Python "<4s10IffIQQ")
struct GBXHeader {
    char magic[4];         // "GBX1"
    uint32_t version;      // 1
    uint32_t dtype_code;   // 0 = float32
    uint32_t vocab_size;
    uint32_t d_model;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t d_ff;
    uint32_t context_len;
    uint32_t tie_weights;  // 1 = tied tok_emb / lm_head
    float rope_base;
    float rmsnorm_eps;
    uint32_t meta_len;     // 8-byte padded metadata JSON byte length
    uint64_t step;
    uint64_t tokens_seen;
};
static_assert(sizeof(GBXHeader) == 72, "GBXHeader must be exactly 72 bytes");

// 32-byte .gbxtok binary token shard header (matches Python "<4sIIIQQ")
struct GBXTokHeader {
    char magic[4];         // "GBXT"
    uint32_t version;      // 1
    uint32_t vocab_size;
    uint32_t dtype_bytes;  // 2 (uint16) or 4 (uint32)
    uint64_t n_tokens;
    uint64_t reserved;
};
static_assert(sizeof(GBXTokHeader) == 32, "GBXTokHeader must be exactly 32 bytes");

struct ModelConfig {
    uint32_t vocab_size = 32768;
    uint32_t d_model = 768;
    uint32_t n_layers = 12;
    uint32_t n_heads = 12;
    uint32_t n_kv_heads = 4;     // 3:1 Grouped-Query Attention (GQA)
    uint32_t d_ff = 2048;        // SwiGLU hidden width
    uint32_t context_len = 2048;
    bool tie_weights = true;
    float rope_base = 10000.0f;
    float rmsnorm_eps = 1e-6f;

    uint32_t head_dim() const { return d_model / n_heads; }
    uint32_t kv_dim() const { return n_kv_heads * head_dim(); }

    // Exact Glassbox-100M default: 100,682,496 parameters
    static ModelConfig preset_100m() {
        ModelConfig c;
        c.vocab_size = 32768;
        c.d_model = 768;
        c.n_layers = 12;
        c.n_heads = 12;
        c.n_kv_heads = 4;
        c.d_ff = 2048;
        c.context_len = 2048;
        c.tie_weights = true;
        return c;
    }

    static ModelConfig preset_micro() {
        ModelConfig c;
        c.vocab_size = 600;
        c.d_model = 96;
        c.n_layers = 4;
        c.n_heads = 6;
        c.n_kv_heads = 2;
        c.d_ff = 256;
        c.context_len = 128;
        c.tie_weights = true;
        return c;
    }
};

struct LayerOffsets {
    size_t norm1;    // [D]
    size_t w_q;      // [D, D]
    size_t w_k;      // [kv_dim, D]
    size_t w_v;      // [kv_dim, D]
    size_t w_o;      // [D, D]
    size_t norm2;    // [D]
    size_t w_gate;   // [d_ff, D]
    size_t w_up;     // [d_ff, D]
    size_t w_down;   // [D, d_ff]
};

// Canonical contiguous flat parameter layout matching glassbox/checkpoint.py
struct ParamLayout {
    size_t tok_emb = 0;
    std::vector<LayerOffsets> layers;
    size_t final_norm = 0;
    size_t lm_head = 0;
    size_t total_params = 0;

    static ParamLayout build(const ModelConfig& c) {
        ParamLayout l;
        size_t off = 0;
        const size_t V = c.vocab_size;
        const size_t D = c.d_model;
        const size_t KV = c.kv_dim();
        const size_t FF = c.d_ff;

        l.tok_emb = off; off += V * D;
        l.layers.resize(c.n_layers);
        for (uint32_t i = 0; i < c.n_layers; ++i) {
            auto& lo = l.layers[i];
            lo.norm1  = off; off += D;
            lo.w_q    = off; off += D * D;
            lo.w_k    = off; off += KV * D;
            lo.w_v    = off; off += KV * D;
            lo.w_o    = off; off += D * D;
            lo.norm2  = off; off += D;
            lo.w_gate = off; off += FF * D;
            lo.w_up   = off; off += FF * D;
            lo.w_down = off; off += D * FF;
        }
        l.final_norm = off; off += D;
        if (!c.tie_weights) {
            l.lm_head = off; off += V * D;
        } else {
            l.lm_head = l.tok_emb;
        }
        l.total_params = off;
        return l;
    }
};

// Deterministic SplitMix64 + Box-Muller initialization matching Glassbox GPT-2/Llama scaling
inline uint64_t splitmix64(uint64_t& st) {
    uint64_t z = (st += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

inline void init_normal(float* dst, size_t n, float std_dev, uint64_t& rng_state) {
    constexpr double TWO_PI = 6.28318530717958647692;
    constexpr double INV_53 = 1.0 / 9007199254740992.0;
    size_t i = 0;
    while (i < n) {
        double u1 = static_cast<double>(splitmix64(rng_state) >> 11) * INV_53;
        double u2 = static_cast<double>(splitmix64(rng_state) >> 11) * INV_53;
        if (u1 < 1e-15) u1 = 1e-15;
        const double mag = static_cast<double>(std_dev) * std::sqrt(-2.0 * std::log(u1));
        const double ang = TWO_PI * u2;
        dst[i++] = static_cast<float>(mag * std::cos(ang));
        if (i < n) dst[i++] = static_cast<float>(mag * std::sin(ang));
    }
}

inline void init_model_weights(float* weights, const ModelConfig& c, const ParamLayout& lay, uint64_t seed) {
    uint64_t st = seed ^ 0x243f6a8885a308d3ULL;
    const float in_std = 0.02f;
    const float resid_std = 0.02f / std::sqrt(2.0f * static_cast<float>(c.n_layers));
    const size_t D = c.d_model, KV = c.kv_dim(), FF = c.d_ff;

    init_normal(weights + lay.tok_emb, static_cast<size_t>(c.vocab_size) * D, in_std, st);
    for (uint32_t l = 0; l < c.n_layers; ++l) {
        const auto& lo = lay.layers[l];
        std::fill_n(weights + lo.norm1, D, 1.0f);
        init_normal(weights + lo.w_q, D * D, in_std, st);
        init_normal(weights + lo.w_k, KV * D, in_std, st);
        init_normal(weights + lo.w_v, KV * D, in_std, st);
        init_normal(weights + lo.w_o, D * D, resid_std, st);
        std::fill_n(weights + lo.norm2, D, 1.0f);
        init_normal(weights + lo.w_gate, FF * D, in_std, st);
        init_normal(weights + lo.w_up, FF * D, in_std, st);
        init_normal(weights + lo.w_down, D * FF, resid_std, st);
    }
    std::fill_n(weights + lay.final_norm, D, 1.0f);
    if (!c.tie_weights) {
        init_normal(weights + lay.lm_head, static_cast<size_t>(c.vocab_size) * D, in_std, st);
    }
}

// Save/Load exact .gbx binary checkpoint (readable directly by glassbox/checkpoint.py)
inline void save_gbx(const std::string& path, const ModelConfig& c, const ParamLayout& lay,
                     const float* host_weights, uint64_t step, uint64_t tokens_seen,
                     const std::string& meta_json = "{}") {
    std::string meta = meta_json;
    while (meta.size() % 8 != 0) meta.push_back('\0');

    GBXHeader hdr{};
    std::memcpy(hdr.magic, "GBX1", 4);
    hdr.version = 1;
    hdr.dtype_code = 0;
    hdr.vocab_size = c.vocab_size;
    hdr.d_model = c.d_model;
    hdr.n_layers = c.n_layers;
    hdr.n_heads = c.n_heads;
    hdr.n_kv_heads = c.n_kv_heads;
    hdr.d_ff = c.d_ff;
    hdr.context_len = c.context_len;
    hdr.tie_weights = c.tie_weights ? 1 : 0;
    hdr.rope_base = c.rope_base;
    hdr.rmsnorm_eps = c.rmsnorm_eps;
    hdr.meta_len = static_cast<uint32_t>(meta.size());
    hdr.step = step;
    hdr.tokens_seen = tokens_seen;

    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) throw std::runtime_error("Cannot open .gbx for writing: " + path);
    std::fwrite(&hdr, sizeof(GBXHeader), 1, fp);
    if (!meta.empty()) std::fwrite(meta.data(), 1, meta.size(), fp);
    size_t written = std::fwrite(host_weights, sizeof(float), lay.total_params, fp);
    std::fclose(fp);
    if (written != lay.total_params) {
        throw std::runtime_error("Short write saving .gbx checkpoint: " + path);
    }
}

inline GBXHeader load_gbx(const std::string& path, ModelConfig& c, ParamLayout& lay,
                          std::vector<float>& host_weights) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open .gbx for reading: " + path);
    GBXHeader hdr{};
    if (std::fread(&hdr, sizeof(GBXHeader), 1, fp) != 1 || std::memcmp(hdr.magic, "GBX1", 4) != 0) {
        std::fclose(fp);
        throw std::runtime_error("Invalid .gbx header in: " + path);
    }
    if (hdr.meta_len > 0) std::fseek(fp, static_cast<long>(hdr.meta_len), SEEK_CUR);
    c.vocab_size = hdr.vocab_size;
    c.d_model = hdr.d_model;
    c.n_layers = hdr.n_layers;
    c.n_heads = hdr.n_heads;
    c.n_kv_heads = hdr.n_kv_heads;
    c.d_ff = hdr.d_ff;
    c.context_len = hdr.context_len;
    c.tie_weights = (hdr.tie_weights != 0);
    c.rope_base = hdr.rope_base;
    c.rmsnorm_eps = hdr.rmsnorm_eps;

    lay = ParamLayout::build(c);
    host_weights.resize(lay.total_params);
    size_t nread = std::fread(host_weights.data(), sizeof(float), lay.total_params, fp);
    std::fclose(fp);
    if (nread != lay.total_params) {
        throw std::runtime_error("Truncated .gbx weights in: " + path);
    }
    return hdr;
}

// Zero-RAM POSIX mmap .gbxtok Shard Streamer (scales seamlessly from 100M to 10B+ tokens)
class ShardStream {
public:
    struct Shard {
        std::string path;
        int fd = -1;
        void* map_ptr = nullptr;
        size_t file_bytes = 0;
        uint32_t dtype_bytes = 2;
        uint64_t n_tokens = 0;
    };

    uint64_t total_tokens = 0;
    uint32_t vocab_size = 0;

    explicit ShardStream(const std::string& path_or_dir, const std::string& prefix = "") {
        std::vector<std::string> files;
        struct stat st{};
        if (stat(path_or_dir.c_str(), &st) != 0) {
            throw std::runtime_error("Shard path does not exist: " + path_or_dir);
        }
        if (S_ISDIR(st.st_mode)) {
            DIR* d = opendir(path_or_dir.c_str());
            if (!d) throw std::runtime_error("Failed to open shard dir: " + path_or_dir);
            while (struct dirent* ent = readdir(d)) {
                std::string name(ent->d_name);
                if (name.size() > 7 && name.substr(name.size() - 7) == ".gbxtok") {
                    if (prefix.empty() || name.rfind(prefix, 0) == 0) {
                        files.push_back(path_or_dir + "/" + name);
                    }
                }
            }
            closedir(d);
            std::sort(files.begin(), files.end());
        } else {
            files.push_back(path_or_dir);
        }
        if (files.empty()) {
            throw std::runtime_error("No .gbxtok shard files found in: " + path_or_dir);
        }

        for (const auto& f : files) {
            Shard s;
            s.path = f;
            s.fd = open(f.c_str(), O_RDONLY);
            if (s.fd < 0) throw std::runtime_error("Failed to open shard: " + f);
            struct stat fst{};
            fstat(s.fd, &fst);
            s.file_bytes = static_cast<size_t>(fst.st_size);
            s.map_ptr = mmap(nullptr, s.file_bytes, PROT_READ, MAP_PRIVATE, s.fd, 0);
            if (s.map_ptr == MAP_FAILED) {
                close(s.fd);
                throw std::runtime_error("mmap failed on shard: " + f);
            }
            madvise(s.map_ptr, s.file_bytes, MADV_SEQUENTIAL);
            const auto* hdr = reinterpret_cast<const GBXTokHeader*>(s.map_ptr);
            if (std::memcmp(hdr->magic, "GBXT", 4) != 0) {
                throw std::runtime_error("Invalid .gbxtok magic in: " + f);
            }
            s.dtype_bytes = hdr->dtype_bytes;
            s.n_tokens = hdr->n_tokens;
            if (vocab_size == 0) vocab_size = hdr->vocab_size;
            total_tokens += s.n_tokens;
            shards_.push_back(s);
        }
    }

    ~ShardStream() {
        for (auto& s : shards_) {
            if (s.map_ptr && s.map_ptr != MAP_FAILED) munmap(s.map_ptr, s.file_bytes);
            if (s.fd >= 0) close(s.fd);
        }
    }

    size_t num_shards() const { return shards_.size(); }

    void reset() {
        cur_shard_ = 0;
        cur_pos_ = 0;
    }

    // Fill [B, T] inputs and [B, T] shifted targets (zero extra allocations)
    void next_batch(int B, int T, int32_t* x_out, int32_t* y_out) {
        for (int b = 0; b < B; ++b) {
            const Shard* s = &shards_[cur_shard_];
            if (cur_pos_ + static_cast<uint64_t>(T) + 1ULL > s->n_tokens) {
                cur_shard_ = (cur_shard_ + 1) % shards_.size();
                cur_pos_ = 0;
                s = &shards_[cur_shard_];
            }
            const uint8_t* data = reinterpret_cast<const uint8_t*>(s->map_ptr) + sizeof(GBXTokHeader);
            int32_t* xb = x_out + static_cast<size_t>(b) * T;
            int32_t* yb = y_out + static_cast<size_t>(b) * T;
            if (s->dtype_bytes == 2) {
                const uint16_t* toks = reinterpret_cast<const uint16_t*>(data) + cur_pos_;
                for (int t = 0; t < T; ++t) {
                    xb[t] = static_cast<int32_t>(toks[t]);
                    yb[t] = static_cast<int32_t>(toks[t + 1]);
                }
            } else {
                const uint32_t* toks = reinterpret_cast<const uint32_t*>(data) + cur_pos_;
                for (int t = 0; t < T; ++t) {
                    xb[t] = static_cast<int32_t>(toks[t]);
                    yb[t] = static_cast<int32_t>(toks[t + 1]);
                }
            }
            cur_pos_ += static_cast<uint64_t>(T);
        }
    }

private:
    std::vector<Shard> shards_;
    size_t cur_shard_ = 0;
    uint64_t cur_pos_ = 0;
};

}  // namespace glassbox
