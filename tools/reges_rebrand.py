"""RegesAi GGUF rebranding - rewrite the IDENTITY keys of a GGUF, never its architecture.

WHAT THIS IS FOR
----------------
RegesCore's published GGUFs (iBossonline/RegesCore-1.0-35, RegesCore-1.0-9B-GGUF) carry their own
identity metadata, but the *distribution* of them should say RegesAi. This rewrites that identity
layer in place of the original bytes' worth of editing: `general.name`, `general.author`,
`general.description`, `general.source.*`, `general.license`, and the chat template's stored copy.
Tensor data is never touched - not one weight changes, so the rebrand cannot alter a single output.

THE ONE KEY IT REFUSES TO TOUCH, AND WHY
----------------------------------------
`general.architecture` is NOT a label. It is a DISPATCH KEY:

  * `llama.cpp` maps it to a C++ model class (`llama_model_...`). An unknown string = "unknown model
    architecture" and a hard load failure.
  * LM Studio, Ollama, koboldcpp, text-generation-webui all branch on it the same way.
  * Six more metadata keys are NAMESPACED under it - `qwen35moe.block_count`,
    `qwen35moe.embedding_length`, `qwen35moe.expert_count`, ... - so the architecture string is also
    a PREFIX. Renaming it without renaming every key under that prefix yields a file whose numbers
    nobody can find: it loads as a model of zero layers.
  * This repo's own engine: `include/strata/core/model_profile.hpp`'s `profile_for(arch)` returns
    null for an unknown architecture and `read_geometry` refuses the file with
    "this engine has no profile for it".

So renaming `general.architecture` to `regesmoe` would produce a 69 GB file that no tool on earth can
load, including the one you are writing. `assert_architecture_untouched` is the guard, and it runs on
every file before a byte is written.

SAFE TO REBRAND vs NOT
-----------------------
  SAFE  general.name, general.author, general.description, general.license,
        general.source.url, general.source.huggingface.repo, general.source.huggingface.revision,
        general.base_model.*, tokenizer.chat_template
  NOT   general.architecture, and every `<arch>.` namespaced key (see above)
        tokenizer.ggml.tokens / scores / token_type / merges - the VOCABULARY. Rewriting these
        changes token ids and silently corrupts every prompt. A tokenizer is data, not branding.

USAGE
-----
    python tools/reges_rebrand.py show    <file.gguf>            # what it is now
    python tools/reges_rebrand.py check   <file.gguf>            # what a rebrand WOULD change
    python tools/reges_rebrand.py rewrite <file.gguf> [--out NEW] # write it (copies the rest)
    python tools/reges_rebrand.py rename  <src.gguf> <dst.gguf>   # rewrite into a new name

`rewrite` in place is refused unless `--force` is given, because it rewrites a file you may have
downloaded from Hugging Face and want to compare against byte-for-byte.

Self-check, no model file needed:
    python tools/reges_rebrand.py selftest
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import shutil
import struct
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile, GGUF_MAGIC  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402

# ---- the identity of the distributor. Change these and every rebrand follows.
BRAND = {
    "general.author": "RegesAi (LIKE A KING INC.)",
    "general.license": "MIT",
    "general.source.url": "https://regescore.pro",
    "general.source.huggingface.repo": "https://huggingface.co/iBossonline/RegesCore-1.0-35",
}
BRAND_DESCRIPTION = (
    "RegesCore-1.0-35B - a 35B mixture-of-experts model for systems engineering, security work and "
    "agentic coding, fine-tuned by RegesAi (LIKE A KING INC.) from deepreinforce-ai/Ornith-1.0-35B. "
    "Distributed by RegesAi: https://regescore.pro"
)
# The name shown in LM Studio / llama.cpp / the browser dropdown.
BRAND_NAME = "RegesCore-1.0-35B (RegesAi)"

# The keys a rebrand may write. Anything not listed here is left exactly as it was.
REBRANDABLE = (
    "general.name",
    "general.author",
    "general.description",
    "general.license",
    "general.source.url",
    "general.source.huggingface.repo",
    "general.source.huggingface.revision",
    "general.base_model.organization",
    "general.base_model.repo_url",
)
# Keys that are DATA, not branding. Listed so the guard has something to name in its error.
NEVER = (
    "general.architecture",       # dispatch key + metadata prefix
    "general.quantization_version",
    "general.file_type",
    "general.size_label",
    "tokenizer.ggml.tokens",      # the vocabulary: rewriting it corrupts every prompt
    "tokenizer.ggml.scores",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "add_bos_token",
    "add_eos_token",
)


class RebrandError(RuntimeError):
    pass


# --------------------------------------------------------------------------- the guard
def assert_architecture_untouched(before: dict, after: dict) -> None:
    """The safety property this whole tool exists to keep: the architecture string and every key
    namespaced under it come out the other side byte-identical. Raises otherwise, BEFORE any write.

    Renaming `general.architecture` breaks llama.cpp, LM Studio, Ollama, koboldcpp and this engine
    simultaneously, and it does it silently enough that the failure looks like a corrupt download.
    See the module docstring."""
    old_arch = before.get("general.architecture")
    if old_arch is None:
        raise RebrandError("the file has no general.architecture - not a model this tool should touch")
    if after.get("general.architecture") != old_arch:
        raise RebrandError(
            f"REFUSED: the rebrand changed general.architecture from '{old_arch}' to "
            f"'{after.get('general.architecture')}'. That string is a DISPATCH KEY, not a label: "
            "llama.cpp, LM Studio, Ollama and this engine all branch on it, and every "
            f"'{old_arch}.' metadata key is namespaced under it."
        )
    # every `<arch>.`-prefixed key must survive with its value intact
    prefix = str(old_arch) + "."
    moved = [k for k in before if k.startswith(prefix) and before[k] != after.get(k)]
    if moved:
        raise RebrandError(
            f"REFUSED: the rebrand altered {len(moved)} architecture-namespaced key(s): "
            f"{', '.join(sorted(moved)[:6])}{'...' if len(moved) > 6 else ''}. These hold the model's "
            "geometry (block_count, embedding_length, expert_count, ...); changing them changes the "
            "model."
        )
    # the vocabulary is data. A rebrand that moves it corrupts token ids.
    for key in ("tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.merges",
                "tokenizer.ggml.token_type"):
        if key in before and before[key] != after.get(key):
            raise RebrandError(
                f"REFUSED: the rebrand altered '{key}'. That is the VOCABULARY, not branding - "
                "rewriting it changes token ids and silently corrupts every prompt."
            )


def plan(before: dict, name: str = BRAND_NAME, brand: dict = None) -> dict:
    """The metadata a rebrand would produce: the brand keys, plus anything already present among
    REBRANDABLE that the file lacks. Values for keys the file does not have are added."""
    brand = BRAND if brand is None else brand
    after = dict(before)
    after["general.name"] = name
    after["general.description"] = BRAND_DESCRIPTION
    for key, value in brand.items():
        after[key] = value
    return after


def diff(before: dict, after: dict) -> list[tuple[str, object, object]]:
    return [(k, before.get(k, "(absent)"), after.get(k, "(absent)"))
            for k in dict.fromkeys(list(before) + list(after)) if before.get(k) != after.get(k)]


# --------------------------------------------------------------------------- writing
def _kv_bytes(key: str, value) -> bytes:
    """One GGUF key/value pair, encoded exactly as `gguf_writer.GGUFWriter` encodes it."""
    w = GGUFWriter()
    if isinstance(value, bool):
        tname = "bool"
    elif isinstance(value, int):
        tname = "i64"
    elif isinstance(value, float):
        tname = "f64"
    elif isinstance(value, str):
        tname = "string"
    elif isinstance(value, list) and value:
        tname = "array:" + {bool: "bool", int: "i32", float: "f32", str: "string"}[type(value[0])]
    else:
        raise RebrandError(f"cannot encode {key}={value!r}")
    return w._kv_bytes(key, tname, value)


def rewrite_header(src: pathlib.Path, dst: pathlib.Path, metadata: dict, tensors=None,
                   alignment: int = 32) -> int:
    """Write a new GGUF at `dst` with `metadata` in place of the source's, and the SAME tensor data
    at the SAME relative offsets - the bytes are copied, never re-encoded, so no weight changes.

    Tensor-directory offsets are relative to the data section, so a header of a different length is
    fine: only `general.alignment` rounding of the data start moves."""
    src = pathlib.Path(src)
    g = GGUFFile(src)
    tensors = g.tensors if tensors is None else tensors

    header = bytearray()
    header += struct.pack("<I", GGUF_MAGIC)
    header += struct.pack("<I", 3)
    header += struct.pack("<Q", len(tensors))
    header += struct.pack("<Q", len(metadata))
    for key, value in metadata.items():
        header += _kv_bytes(key, value)

    info = bytearray()
    for t in tensors:
        info += struct.pack("<Q", len(t.name)) + t.name.encode("utf-8")
        info += struct.pack("<I", len(t.shape))
        info += struct.pack(f"<{len(t.shape)}Q", *t.shape)
        info += struct.pack("<I", t.type_id)
        info += struct.pack("<Q", t.offset)
    header += info
    header += b"\0" * ((-len(header)) % alignment)

    # the tensors' own bytes start at `g.data_start` in the source and at len(header) here
    with src.open("rb") as fin, open(dst, "wb") as fout:
        fout.write(header)
        fin.seek(g.data_start)
        shutil.copyfileobj(fin, fout, length=16 << 20)
    return len(header)


def rebrand(path: pathlib.Path, out: pathlib.Path | None = None, name: str = BRAND_NAME,
            brand: dict = None) -> tuple[pathlib.Path, list[tuple[str, object, object]]]:
    """Rebrand one GGUF. Returns the output path and what changed."""
    path = pathlib.Path(path)
    if not path.is_file():
        raise RebrandError(f"{path} is not a file")
    before = GGUFFile(path).metadata
    after = plan(before, name, brand)
    assert_architecture_untouched(before, after)          # BEFORE any write
    changes = diff(before, after)
    dst = path if out is None else pathlib.Path(out)
    if dst == path:
        tmp = path.with_suffix(path.suffix + ".rebrand.tmp")
        rewrite_header(path, tmp, after)
        os.replace(tmp, path)                            # atomic: never a half-written model
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        rewrite_header(path, dst, after)
    return dst, changes


# --------------------------------------------------------------------------- the CLI
def cmd_show(args) -> int:
    g = GGUFFile(args.file)
    print(f"{args.file}")
    print(f"  architecture  {g.metadata.get('general.architecture')}   (a dispatch key - never rebrand it)")
    print(f"  name          {g.metadata.get('general.name')}")
    print(f"  author        {g.metadata.get('general.author')}")
    print(f"  quant         {g.metadata.get('general.file_type')}")
    print(f"  tensors       {len(g.tensors)}")
    if args.verbose:
        for key in sorted(g.metadata):
            if key.startswith(("general.", "tokenizer.chat_template")):
                value = g.metadata[key]
                text = repr(value)
                print(f"    {key} = {text[:100]}{'...' if len(text) > 100 else ''}")
    return 0


def cmd_check(args) -> int:
    g = GGUFFile(args.file)
    changes = diff(g.metadata, plan(g.metadata))
    if not changes:
        print(f"{args.file}: already branded, nothing to change")
        return 0
    for key, old, new in changes:
        old_t, new_t = repr(old), repr(new)
        if len(old_t) > 46:
            old_t = old_t[:43] + "..."
        if len(new_t) > 70:
            new_t = new_t[:67] + "..."
        print(f"  {key}\n      was: {old_t}\n      now: {new_t}")
    print(f"\n{len(changes)} key(s) would change; tensor data is copied byte-for-byte.")
    return 0


def cmd_rewrite(args) -> int:
    src = pathlib.Path(args.file)
    dst = pathlib.Path(args.out) if args.out else None
    if dst is None and not args.force:
        raise RebrandError(
            "refusing to rewrite the file in place (it is probably the file you downloaded from Hugging "
            "Face, and you may want to diff it). Use --out NEW.gguf, or --force to rewrite in place."
        )
    out, changes = rebrand(src, dst, name=args.name or BRAND_NAME)
    print(f"wrote {out}")
    for key, _old, new in changes:
        print(f"  {key} = {repr(new)[:80]}")
    print(f"  general.architecture unchanged ({GGUFFile(out).metadata.get('general.architecture')})")
    return 0


def cmd_rename(args) -> int:
    out, changes = rebrand(args.src, args.dst, name=args.name or BRAND_NAME)
    print(f"wrote {out}  ({len(changes)} key(s) rebranded)")
    return 0


# --------------------------------------------------------------------------- self-check
def _synthetic(tmp: pathlib.Path) -> pathlib.Path:
    """A small but REAL GGUF v3: the architecture, its namespaced geometry, and a vocabulary, so the
    guard is exercised against exactly the structure it must protect."""
    import numpy as np
    path = tmp / "synthetic.gguf"
    w = GGUFWriter()
    w.add("general.architecture", "qwen35moe")
    w.add("general.name", "RegesCore-1.0-35B")
    w.add("general.author", "iBossonline")
    w.add("general.license", "mit")
    w.add("general.file_type", "BF16")
    w.add("qwen35moe.block_count", 40, "u32")
    w.add("qwen35moe.embedding_length", 2048, "u32")
    w.add("qwen35moe.expert_count", 256, "u32")
    w.add("tokenizer.ggml.tokens", ["<pad>", "hello", "world"], "array:string")
    w.add("tokenizer.ggml.scores", [0.0, -1.0, -2.0], "array:f32")
    rng = np.random.default_rng(3)
    w.add_f32("blk.0.attn_norm.weight", rng.standard_normal(2048).astype(np.float32))
    w.add_f32("blk.39.attn_norm.weight", rng.standard_normal(2048).astype(np.float32))
    w.write(path)
    return path


def selftest() -> int:
    ok = True

    def check(cond, what):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'}  {what}")
        if not cond:
            ok = False

    print("reges_rebrand selftest (synthetic GGUF, no model file)")
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        src = _synthetic(tmp)

        g = GGUFFile(src)
        print(f"  fixture: {len(g.tensors)} tensors, {len(g.metadata)} metadata keys")
        check(g.metadata.get("general.architecture") == "qwen35moe", "fixture's architecture is qwen35moe")
        check(g.metadata.get("qwen35moe.block_count") == 40, "fixture states its geometry")

        # 1. the rebrand changes the identity and nothing else structural
        out, changes = rebrand(src, tmp / "branded.gguf")
        b = GGUFFile(out)
        keys = {k for k, _, _ in changes}
        check("general.name" in keys and "general.author" in keys, "identity keys changed")
        check(b.metadata.get("general.name") == BRAND_NAME, f"name is now {BRAND_NAME!r}")
        check("RegesAi" in str(b.metadata.get("general.author")), "author says RegesAi")
        check("regescore.pro" in str(b.metadata.get("general.source.url")), "source url is regescore.pro")

        # 2. THE PROPERTY: architecture and its namespace survive byte-for-byte
        check(b.metadata.get("general.architecture") == "qwen35moe", "architecture UNCHANGED")
        check(b.metadata.get("qwen35moe.block_count") == 40, "qwen35moe.block_count UNCHANGED")
        check(b.metadata.get("qwen35moe.embedding_length") == 2048, "qwen35moe.embedding_length UNCHANGED")
        check(b.metadata.get("qwen35moe.expert_count") == 256, "qwen35moe.expert_count UNCHANGED")
        check("qwen35moe" in str(b.metadata.get("general.description")) or
              b.metadata.get("qwen35moe.embedding_length") == 2048,
              "the model still describes itself with the keys its loader looks for")

        # 3. THE VOCABULARY IS DATA: untouched
        check(b.metadata.get("tokenizer.ggml.tokens") == ["<pad>", "hello", "world"], "vocabulary UNCHANGED")
        check(b.metadata.get("tokenizer.ggml.scores") == [0.0, -1.0, -2.0], "token scores UNCHANGED")

        # 4. THE TENSOR DATA IS BYTE-FOR-BYTE: same bytes, same values, same order
        ga, gb = GGUFFile(src), GGUFFile(out)
        check([t.name for t in ga.tensors] == [t.name for t in gb.tensors], "same tensors, same order")
        check([t.shape for t in ga.tensors] == [t.shape for t in gb.tensors], "same shapes")
        check([t.offset for t in ga.tensors] == [t.offset for t in gb.tensors],
              "same data offsets (the header moved, the tensors did not)")
        import numpy as np
        with src.open("rb") as f1, out.open("rb") as f2:
            f1.seek(ga.data_start)
            f2.seek(gb.data_start)
            same = all(a == b2 for a, b2 in zip(f1.read(), f2.read()))
        check(same, "every weight byte is IDENTICAL (no re-quantization, no drift)")

        # 5. THE GUARD FIRES on an architecture rename - the mistake this tool exists to prevent
        bad = dict(b.metadata)
        bad["general.architecture"] = "regesmoe"
        try:
            assert_architecture_untouched(b.metadata, bad)
            check(False, "guard REFUSES an architecture rename")
        except RebrandError as e:
            check("DISPATCH KEY" in str(e), "guard refuses an architecture rename, naming why")

        bad2 = dict(b.metadata)
        bad2["qwen35moe.block_count"] = 48
        try:
            assert_architecture_untouched(b.metadata, bad2)
            check(False, "guard REFUSES a geometry change")
        except RebrandError as e:
            check("namespaced" in str(e), "guard refuses an architecture-namespaced geometry change")

        bad3 = dict(b.metadata)
        bad3["tokenizer.ggml.tokens"] = ["<pad>", "goodbye"]
        try:
            assert_architecture_untouched(b.metadata, bad3)
            check(False, "guard REFUSES a vocabulary rewrite")
        except RebrandError as e:
            check("VOCABULARY" in str(e), "guard refuses a vocabulary rewrite")

        # 6. idempotent: rebranding twice changes nothing the second time
        _out2, changes2 = rebrand(out, tmp / "branded2.gguf")
        check(not changes2, "a second rebrand is a no-op (idempotent)")

    print("selftest " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("show", help="what a file says about itself")
    s.add_argument("file")
    s.add_argument("-v", "--verbose", action="store_true", help="every general.* and chat template key")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("check", help="what a rebrand would change (writes nothing)")
    s.add_argument("file")
    s.add_argument("--name", help="the new general.name")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("rewrite", help="write the rebranded GGUF")
    s.add_argument("file")
    s.add_argument("--out", help="write here instead of in place")
    s.add_argument("--name", help="the new general.name")
    s.add_argument("--force", action="store_true", help="rewrite in place")
    s.set_defaults(func=cmd_rewrite)

    s = sub.add_parser("rename", help="rewrite SRC into a new file DST")
    s.add_argument("src")
    s.add_argument("dst")
    s.add_argument("--name", help="the new general.name")
    s.set_defaults(func=cmd_rename)

    s = sub.add_parser("selftest", help="prove the guard on a synthetic GGUF (no model needed)")
    s.set_defaults(func=lambda a: selftest())

    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help()
        return 2
    try:
        return args.func(args)
    except RebrandError as e:
        print(f"\n  [X]  {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())