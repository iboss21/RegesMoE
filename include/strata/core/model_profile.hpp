// include/strata/core/model_profile.hpp - the geometry a model file declares for itself.
//
// `ModelGeometry`'s defaults are the compiled-in ones, and the guard in `artifact/gguf_reader.hpp` refuses
// a file that disagrees with them.  This reads the same numbers out of the file instead, so the rest of the
// engine can ask "what model is this" rather than assuming, and it records which optional parts of the
// graph the profile carries - the two flags `core/layout.hpp` uses to tell "this model has no indexer"
// from "this pack is corrupt".
//
// A field is taken from the file when the file states it under one of the names below, and left at the
// profile's compiled-in value when it does not: the guard has already checked the six it requires, and
// inventing a number a kernel depends on is the one thing this header must not do.  `qwen35moe` states all
// of them (measured from RegesCore-1.0-35's BF16 shards, 2026-10-08); `qwen4exp`'s files state the six the
// guard checks, so its other numbers stay compiled-in, exactly as today.
//
// Header-only for the reason the reader is: no state to hide, and no new symbol for the CUDA target to link.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/layout.hpp"

#include <string>
#include <vector>

namespace strata::core {

/// The profile for an architecture name; null when this engine has none for it.
inline const ModelProfile* profile_for(const std::string& arch) {
    static const ModelProfile k_qwen4exp = ModelProfile{};
    static const ModelProfile k_qwen35moe = qwen35moe_profile();
    if (arch == "qwen4exp") return &k_qwen4exp;
    if (arch == "qwen35moe") return &k_qwen35moe;
    return nullptr;
}

/// Fills `out` with the geometry `g` declares.  Returns a non-empty error, naming the key, only when a
/// value the file does state is not a number.
inline std::string read_geometry(const GgufFile& g, ModelGeometry& out) {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return "missing general.architecture";
    const ModelProfile* p = profile_for(arch->s);
    if (!p) return "architecture is '" + arch->s + "', this engine has no profile for it";
    out = ModelGeometry{};
    out.gated_residual = p->gated_residual;
    out.qsa_indexer = p->qsa_indexer;

    // Each field lists the names a file may state it under.  `ssm_v_heads` is derived rather than read:
    // both architectures state the GDN value count as inner_size / state_size.
    struct Field {
        int64_t* into;
        std::vector<std::string> keys;
    };
    const std::string k = arch->s + ".";
    const Field fields[] = {
        {&out.n_layers, {k + "block_count"}},
        {&out.n_embd, {k + "embedding_length"}},
        {&out.qsa_interval, {k + "full_attention_interval"}},
        {&out.ssm_state_size, {k + "ssm.state_size"}},
        {&out.ssm_k_heads, {k + "ssm.group_count"}},
        {&out.ssm_d_conv, {k + "ssm.conv_kernel"}},
        {&out.n_head, {k + "attention.head_count"}},
        {&out.n_head_kv, {k + "attention.head_count_kv"}},
        {&out.head_dim, {k + "attention.key_length", k + "attention.head_dim"}},
        {&out.n_expert, {k + "expert_count"}},
        {&out.n_ff, {k + "expert_feed_forward_length"}},
    };
    for (const Field& f : fields) {
        for (const std::string& key : f.keys) {
            const MetaValue* v = g.get(key);
            if (!v) continue;
            if (!v->is_num()) return std::string(key) + " is not a number";
            *f.into = (int64_t) v->num();
            break;
        }
    }
    if (const MetaValue* inner = g.get(k + "ssm.inner_size"); inner && inner->is_num() && out.ssm_state_size)
        out.ssm_v_heads = (int64_t) inner->num() / out.ssm_state_size;

    // The two derived widths every GDN kernel sizes itself from.  Computed here so a caller cannot get a
    // geometry whose tensors and its arithmetic disagree.
    if (out.ssm_k_heads && out.ssm_v_heads && out.ssm_state_size) {
        out.ssm_conv_channels = 2 * out.ssm_state_size * out.ssm_k_heads + out.ssm_state_size * out.ssm_v_heads;
        out.ssm_value_dim = out.ssm_state_size * out.ssm_v_heads;
    }
    if (!out.qsa_indexer) {
        out.idx_q_heads = 0;
        out.idx_key_dim = 0;
    }
    if (!out.gated_residual) {
        out.hc = 0;
        out.hc_lr = 0;
    }
    return {};
}

}  // namespace strata::core
