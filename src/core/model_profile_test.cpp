// src/core/model_profile_test.cpp - the profile table and the geometry a file declares for itself
// (synthetic GGUF metadata: no model, no GPU).
//
//   1. a qwen35moe file (RegesCore-1.0-35's numbers, copied from its BF16 shards' metadata on
//      2026-10-08) is accepted by its profile, and read_geometry reproduces those numbers - including the
//      two GDN widths the kernels size themselves from, which the file states only as inner_size;
//   2. the same file is still refused by the default profile, and the error names both architectures;
//   3. a qwen4exp file behaves exactly as before: accepted by the default profile, and the fields its
//      file does not state (the gated residual, the indexer) stay compiled-in rather than zeroed;
//   4. a value that disagrees is refused with the key named, and an architecture with no profile is
//      refused rather than run.
#include "strata/core/model_profile.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-72s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

template <class T> void put(std::vector<uint8_t>& b, T v) {
    const auto n = b.size();
    b.resize(n + sizeof(T));
    std::memcpy(b.data() + n, &v, sizeof(T));
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

// GGUF v3 with metadata only (no tensors): a value made only of digits is written as u32 (type 4),
// anything else as a string (type 8).
std::filesystem::path write_meta(const std::vector<std::pair<std::string, std::string>>& kv) {
    std::vector<uint8_t> b;
    put<uint32_t>(b, 0x46554747u);   // "GGUF"
    put<uint32_t>(b, 3);
    put<uint64_t>(b, 0);             // n_tensors
    put<uint64_t>(b, kv.size());     // n_kv
    for (const auto& entry : kv) {
        const std::string& key = entry.first;
        const std::string& val = entry.second;
        put_str(b, key);
        const bool numeric = !val.empty() && val.find_first_not_of("0123456789") == std::string::npos;
        if (numeric) {
            put<uint32_t>(b, 4);     // U32
            put<uint32_t>(b, (uint32_t) std::stoul(val));
        } else {
            put<uint32_t>(b, 8);     // STRING
            put_str(b, val);
        }
    }
    b.resize((b.size() + 31) / 32 * 32 + 32, 0);   // a real file has a data section; the reader refuses one past EOF
    const auto path = std::filesystem::temp_directory_path() / "strata_model_profile_test.gguf";
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), (std::streamsize)b.size());
    return path;
}

// RegesCore-1.0-35 / Ornith-1.0-35B, exactly as its own BF16 shard declares them.
std::vector<std::pair<std::string, std::string>> qwen35moe_meta() {
    return {
        {"general.architecture", "qwen35moe"},
        {"qwen35moe.block_count", "40"},
        {"qwen35moe.embedding_length", "2048"},
        {"qwen35moe.full_attention_interval", "4"},
        {"qwen35moe.expert_count", "256"},
        {"qwen35moe.expert_used_count", "8"},
        {"qwen35moe.expert_feed_forward_length", "512"},
        {"qwen35moe.attention.head_count", "16"},
        {"qwen35moe.attention.head_count_kv", "2"},
        {"qwen35moe.attention.key_length", "256"},
        {"qwen35moe.ssm.state_size", "128"},
        {"qwen35moe.ssm.group_count", "16"},
        {"qwen35moe.ssm.inner_size", "4096"},
        {"qwen35moe.ssm.conv_kernel", "4"},
    };
}
}  // namespace

int main() {
    using namespace strata::core;
    std::printf("model_profile_test\n");

    {   // 1. the RegesCore file: accepted by its profile, and its geometry read back
        const auto path = write_meta(qwen35moe_meta());
        std::string err;
        ModelGeometry g;
        try {
            strata::GgufFile f(path.string());
            err = strata::check_architecture(f, strata::qwen35moe_profile());
            if (err.empty()) err = read_geometry(f, g);
        } catch (const std::exception& e) { err = e.what(); }
        std::filesystem::remove(path);
        check(err.empty(), "qwen35moe: the guard and read_geometry both accept the file");
        if (!err.empty()) std::printf("  (%s)\n", err.c_str());
        check(g.n_layers == 40 && g.n_embd == 2048 && g.qsa_interval == 4,
              "  40 layers, 2048 wide, every 4th layer full attention");
        check(g.n_expert == 256 && g.n_ff == 512, "  256 experts, 512 wide");
        check(g.n_head == 16 && g.n_head_kv == 2 && g.head_dim == 256, "  16 heads, 2 KV, head_dim 256");
        check(g.ssm_state_size == 128 && g.ssm_k_heads == 16 && g.ssm_v_heads == 32 && g.ssm_d_conv == 4,
              "  GDN: state 128, 16 key heads, 32 value heads (derived), conv 4");
        check(g.ssm_conv_channels == 8192 && g.ssm_value_dim == 4096,
              "  derived widths match the file's attn_qkv [2048,8192] and attn_gate [2048,4096]");
        check(!g.gated_residual && !g.qsa_indexer && g.hc == 0 && g.hc_lr == 0,
              "  no gated residual and no indexer: flags off, their widths zero");
        check(g.n_qsa_layers() == 10 && g.n_gdn_layers() == 30,
              "  the layer split is 10 full / 30 linear, as the file's tensors are");
    }

    {   // 2. the same file under the default profile: refused, and the error names both
        const auto path = write_meta(qwen35moe_meta());
        std::string err;
        try {
            strata::GgufFile f(path.string());
            err = strata::check_architecture(f);
        } catch (const std::exception& e) { err = e.what(); }
        std::filesystem::remove(path);
        check(!err.empty() && err.find("qwen35moe") != std::string::npos &&
              err.find("qwen4exp") != std::string::npos,
              "qwen4exp default: the same file is refused, naming both architectures");
    }

    {   // 3. a qwen4exp file: unchanged behaviour, and unstated fields stay compiled-in
        const std::vector<std::pair<std::string, std::string>> kv = {
            {"general.architecture", "qwen4exp"},
            {"qwen4exp.block_count", "48"},
            {"qwen4exp.embedding_length", "2560"},
            {"qwen4exp.expert_count", "512"},
            {"qwen4exp.expert_used_count", "10"},
            {"qwen4exp.attention.head_count", "24"},
            {"qwen4exp.attention.head_count_kv", "2"},
        };
        const auto path = write_meta(kv);
        std::string err;
        ModelGeometry g;
        try {
            strata::GgufFile f(path.string());
            err = strata::check_architecture(f);
            if (err.empty()) err = read_geometry(f, g);
        } catch (const std::exception& e) { err = e.what(); }
        std::filesystem::remove(path);
        check(err.empty(), "qwen4exp: the default guard and read_geometry accept it");
        check(g.n_layers == 48 && g.n_embd == 2560 && g.n_expert == 512 && g.n_head == 24,
              "  what its file states is read");
        check(g.gated_residual && g.qsa_indexer && g.hc == 4 && g.hc_lr == 320 && g.n_ff == 640,
              "  what its file does not state stays compiled-in, not zeroed");
    }

    {   // 4. a value that disagrees, and an architecture with no profile
        auto kv = qwen35moe_meta();
        for (auto& entry : kv) if (entry.first == "qwen35moe.block_count") entry.second = "41";
        const auto path = write_meta(kv);
        std::string err;
        try {
            strata::GgufFile f(path.string());
            err = strata::check_architecture(f, strata::qwen35moe_profile());
        } catch (const std::exception& e) { err = e.what(); }
        std::filesystem::remove(path);
        check(!err.empty() && err.find("block_count") != std::string::npos,
              "a file declaring 41 layers is refused with the key named");

        const auto path2 = write_meta({{"general.architecture", "llama"}, {"llama.block_count", "32"}});
        std::string err2;
        ModelGeometry g2;
        try {
            strata::GgufFile f(path2.string());
            err2 = read_geometry(f, g2);
        } catch (const std::exception& e) { err2 = e.what(); }
        std::filesystem::remove(path2);
        check(profile_for("llama") == nullptr && !err2.empty(), "an architecture with no profile is refused, not run");
    }

    std::printf(g_fail ? "model_profile_test: %d FAILED\n" : "model_profile_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
