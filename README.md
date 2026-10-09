# RegesMoE — running RegesCore on a normal PC

**RegesMoE** runs [RegesCore](https://huggingface.co/iBossonline/RegesCore-1.0-35) — a 35B mixture-of-experts
model for systems engineering, security work and agentic coding — on one NVIDIA or AMD graphics card plus
system RAM, on Windows or Linux. One installer, one command, then a local OpenAI- and Anthropic-compatible API
and a web app.

RegesMoE is **RegesAi**'s distribution of the engine. The engine underneath is Strata (MIT, © Niko1221 and the
Strata contributors) — see [LICENSE](LICENSE) and [Attribution](#attribution).

| | |
|---|---|
| Developer | Davidio / Dev — [davidio.dev](https://davidio.dev) |
| Company | LIKE A KING INC. — [likeakinginc.com](https://likeakinginc.com) |
| Projects | [regescore.pro](https://regescore.pro) · [regescode.com](https://regescode.com) · [animatedwebs.com](https://animatedwebs.com) |
| Source | [github.com/iboss21/RegesMoE](https://github.com/iboss21/RegesMoE) |
| Model | [iBossonline/RegesCore-1.0-35](https://huggingface.co/iBossonline/RegesCore-1.0-35) |
| Licence | MIT (engine) · MIT (model) |

---

## Install

```
git clone https://github.com/iboss21/RegesMoE.git
cd RegesMoE
START-HERE.bat          # Windows
./setup.sh              # Linux / macOS
```

Setup checks the PC, asks which model and context to use, downloads the model from Hugging Face at a **pinned
revision**, verifies it by size **and SHA-256**, builds the engine if there is no ready-made one, writes the
start script, and verifies the server answers. Everything after that is `START-HERE.bat`.

On a machine with a ready-made engine download, no compiler is needed.

---

## The measured numbers, and where they come from

Read out of the two BF16 shards' own GGUF headers (733 tensors, revision `50a61ad1`, on 2026-10-08) — **not**
from the model card:

| | |
|---|---|
| Architecture | `qwen35moe` — a fine-tune of `deepreinforce-ai/Ornith-1.0-35B` |
| Layers | 40, every 4th full attention (10 full / 30 linear) |
| Width | **2048** |
| Experts | 256 per layer, **8 activated**, 512 intermediate |
| Attention | 16 heads, 2 KV, head_dim 256 |
| GDN | state 128, 16 key heads, 32 value heads (derived), conv 4 |
| Context | 262,144 trained |
| **Non-expert side** | **4.95 GB** |
| **Experts (BF16)** | **64.4 GB** of the 69.4 GB file |

Two consequences decide the whole design:

**The dense side fits a small card.** 4.95 GB in BF16 leaves room on a 12 GB card for the KV cache and a
useful slice of the expert cache. RegesCore suits a GPU/RAM/SSD split *better* than the model Strata was
written for.

**The wall is bandwidth, not compute.** 8 experts × 40 layers × 6.29 MB is **2.01 GB of expert reads per
token** at BF16. Projected from that, per token:

| quant | experts | per token | at 40 GB/s | at 90 GB/s |
|---|---|---|---|---|
| BF16 | 69.4 GB | 2.01 GB | 20 tok/s | 45 tok/s |
| Q8_0 | 36.9 GB | 1.07 GB | 37 | 84 |
| Q4_K_M | 22.1 GB | 0.64 GB | 62 | 140 |

*(projected from the measured byte counts and the bandwidths above; not measured on a GPU — this engine has
not yet been run on RegesCore, see [What does not work yet](#what-does-not-work-yet))*

**Q8_0 is the recommended size, not BF16.** For a 35B MoE, Q8_0 is within noise of BF16 and costs 1.9× less
bandwidth; on a RAM-streaming path that is roughly 2× the tokens/s. BF16 is the *reference* — the thing the
others are compared against — and the right choice on a machine with ~80 GB of RAM, not on a 12 GB card.

---

## What is in this repo

```
START-HERE.bat / setup.sh    the installer
setup.py                    the installer itself: PC check, download, verify, configure
CMakeLists.txt              the engine build
src/                        the engine (C++ / CUDA / HIP)
  core/                      loader, layer graph, expert placement, geometry
  kernels/                   the GPU and CPU kernels
include/strata/             engine headers
serve/                     the server + web app (Python)
  server.py                  OpenAI- and Anthropic-compatible API
  web/                       the chat app, settings drawer and API monitor
tools/                     the packer, the installer helpers, the tests
docs/                      documentation
```

## The API

Both surfaces are served on `127.0.0.1` by default:

- **OpenAI** — `http://127.0.0.1:8080/v1/chat/completions`, `/v1/responses`
- **Anthropic** — `http://127.0.0.1:8080/v1/messages`

The web app is at `http://127.0.0.1:8080/`, with a live monitor (`/monitor`) showing throughput, token counts,
per-layer timings and VRAM/RAM/SSD placement.

> The server is local. Anyone who can reach the port can use it. Keep it on `127.0.0.1` or set `--api-key`.

---

## Rebranding the GGUFs

`tools/reges_rebrand.py` rewrites a GGUF's **identity** metadata — `general.name`, `general.author`,
`general.description`, `general.source.*` — and copies the tensor data through **byte for byte**.

```
python tools/reges_rebrand.py show     <file.gguf>
python tools/reges_rebrand.py check    <file.gguf>
python tools/reges_rebrand.py rewrite  <file.gguf> --out branded.gguf
python tools/reges_rebrand.py selftest          # proves the guard, no model needed
```

**It refuses to rename `general.architecture`**, and that refusal is the point. That string is not a label, it
is a dispatch key:

- llama.cpp maps it to a C++ model class — an unknown string is a hard load failure;
- LM Studio, Ollama, koboldcpp and text-generation-webui all branch on it;
- six metadata keys are **namespaced** under it (`qwen35moe.block_count`, `qwen35moe.expert_count`, …), so
  renaming it without renaming every key under the prefix yields a file that loads as a model of zero layers;
- this engine's own `profile_for(arch)` returns null for an unknown name and refuses the file.

Renaming it would produce a 69 GB file that no tool on earth can load. The guard also refuses any change to
the architecture-namespaced keys and to the vocabulary (`tokenizer.ggml.*` — a tokenizer is data, not
branding; rewriting it changes token ids and silently corrupts every prompt). The self-test proves the
weights come out byte-identical and that all three dangerous rewrites are refused.

---

## What does not work yet

Stated plainly, because the rest of this document would otherwise be misleading.

**RegesCore cannot be run yet.** The engine's expert and GDN kernels are compiled for one exact geometry:
`src/kernels/cpu/expert.hpp` fixes `H = 2560` and `FF = 640`, and `expert_source.cpp` refuses anything else
with *"the expert kernel is compiled for a 2560-wide activation"*. RegesCore is **2048** wide with a 512
intermediate. It loads, then refuses at the first token.

What exists so far:

- `include/strata/core/model_profile.hpp` reads the geometry out of the model's own GGUF instead of assuming
  it, with a `qwen35moe` profile carrying RegesCore's measured numbers (`src/core/model_profile_test.cpp`,
  14 checks, synthetic metadata, no GPU).
- `src/core/layout.cpp` knows which tensors a layer may omit: RegesCore has **no** gated residual (`hc_*`),
  **no** QSA indexer (`indexer.*`) and **no** n-gram PLE table, and instead carries plain
  `attn_norm.weight` / `post_attention_norm.weight`, both `[2048]` F32 across all 40 layers.
- The installer knows the family, its pinned revision, its files' sizes and SHA-256s, and the engine version
  it needs (`tools/test_setup_reges.py`, 22 checks).

What does not, in order:

1. **2048-wide kernels** — templated on `H`/`FF` rather than `constexpr 2560`, with the PLE / gated-residual /
   indexer paths dropped for this profile. *Needs a CUDA toolkit to build and verify.*
2. **Expert placement** across GPU / RAM / SSD for the 64.4 GB arena with a RAM budget — the same
   `--resident-budget-gib` technique Strata uses for Unsloth's file, which `RC35-BF16` already requests.
3. **Speculative decoding is off.** RegesCore publishes no MTP tensors — not in the GGUF (733 tensors, none
   `mtp.*`), and not in the base checkpoint either. Strata gets a large share of its speed from the MTP draft
   layer, so "fast" here means without speculation until those weights are exported.

Also not offered yet, and deliberately:

- **RegesCore 1.0 9B** — pinned, but it is a **dense** `qwen35` model, a different graph, and this engine has
  no `qwen35` profile. Listing it would offer a download that stops at the first token.
- **The K- and i-quants** RegesCore publishes beyond the four listed (`Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q3_K_M`,
  `Q2_K_XL`, `IQ4_XS`, `MXFP4_MOE`, …). This engine serves exactly two K-quant files today (Unsloth's
  `UD-Q4_K_XL` and `UD-IQ4_XS`); listing more would widen what the installer accepts from *every* family and
  start handing out files nothing can run.

There is **no RegesCore 397B**. The 397B is `ornith-ai/Ornith-1.0-397B`, the *base* model RegesCore-1.0-35 is
fine-tuned from; it ships safetensors only, and no GGUF of it exists.

---

## Testing

Setup's own tests run without a GPU, a model or a download:

```
python -m unittest tools.test_setup_reges      # this family: pins, sizes, the collision guard
python -m unittest serve.test_compliance       # the web app's first-run gate
python tools/reges_rebrand.py selftest         # the GGUF rebrand's safety guard
```

The engine's C++ tests are built by CMake (`-DSTRATA_BUILD_TESTS=ON`) and `ctest` runs them; the GPU tests
need a card and a CUDA toolkit.

## Attribution

The engine in `src/`, `include/` and `serve/` is Strata — MIT, © 2026 Niko1221 and the Strata contributors.
RegesMoE is a distribution of it plus the RegesCore profiles, the installer entries and the rebranding tool.
The model weights are not ours: they are [RegesCore-1.0-35](https://huggingface.co/iBossonline/RegesCore-1.0-35),
a fine-tune of `deepreinforce-ai/Ornith-1.0-35B`, MIT, © its authors.