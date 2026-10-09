"""Tests for setup.py's RegesCore choices (the RegesAi families).

What is pinned here, and why each one is a test rather than a comment:

  * THE HUB'S OWN BYTES.  Every file's exact size and SHA-256 at the pinned revision, so an install on any
    machine gets the same weights.  These came from the Hub's LFS pointers on 2026-10-09; `download_gb` is
    cross-checked against them, because a download_gb that drifts from reality is a disk-space lie.
  * **NO SIZE NAME MAY COLLIDE.**  `MODELS` is a dict keyed by size name.  A RegesCore entry called "IQ3_S"
    SILENTLY REPLACED the original model's own IQ3_S - its 83.6 GB best-quality size simply vanished from the
    installer, and the only symptom was an unrelated test failing with "Qwen has no IQ3_S".  Every RegesCore
    size is namespaced `RC35-`, and `test_no_size_shadows_an_existing_one` is the test that keeps it that way.
  * THE ENGINE GATE.  RegesCore needs an engine carrying the qwen35moe profile (0.2.0); an older engine is
    refused BEFORE the 69 GB download, the same rule UD-Q4_K_XL has.
  * THE PINNED REVISIONS, so two machines on two days install the same files.

Mocked throughout - no GPU, no downloads, no model files, nothing written outside a temp folder.

    python -m unittest tools.test_setup_reges
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

R35 = "50a61ad1dce5285a24f4b73c937ab0cae6398bab"
R9 = "92d3db691d6a1a7173179826c76fc2973fac9e03"

# name -> (bytes, sha256), exactly as https://huggingface.co/api/models/iBossonline/RegesCore-1.0-35?blobs=true
# reported them on 2026-10-09.
REGES_SHARDS = {
    "RegesCore-1.0-35B-BF16-00001-of-00002.gguf":
        (49781577216, "6d4f8b08fbee72eff293b2741d9b1460070e1ec4597d859b0d9999fca388265e"),
    "RegesCore-1.0-35B-BF16-00002-of-00002.gguf":
        (19595060416, "390fdcd2b4bc18d0ce8c430fc8612654f1f0ca36e3898a03d691ed0bd5b0aa86"),
    "RegesCore-1.0-35B-Q8_0.gguf":
        (36903139744, "8f94f093f907065e182924225ef2182a7fc35cae3822a3eab29cc5a8ee02fe4f"),
    "RegesCore-1.0-35B-IQ3_S.gguf":
        (14984321440, "7ddd07b77936f8b98054f79290bcf134f846ca6be6cb5d7e33c03ece1e40ff3b"),
    "RegesCore-1.0-35B-IQ2_XXS.gguf":
        (11496839584, "f57a9d2799d2b19933ef7dc1348e55c0148b933afcd79b11538d4467f13d3101"),
}


class Pins(unittest.TestCase):
    def test_the_revision_is_pinned(self):
        self.assertEqual(setup.HF_REVISIONS["iBossonline/RegesCore-1.0-35"], R35)
        # the 9B's pin is kept even though it is not a family yet, so adding one is a single line
        self.assertEqual(setup.HF_REVISIONS["iBossonline/RegesCore-1.0-9B-GGUF"], R9)
        fam = setup.FAMILIES["reges"]
        self.assertEqual(fam["hf"],
                         f"https://huggingface.co/iBossonline/RegesCore-1.0-35/resolve/{R35}/")

    def test_every_sha256_is_a_hash(self):
        for name, (size, sha) in REGES_SHARDS.items():
            self.assertRegex(sha, r"^[0-9a-f]{64}$", name)
            self.assertGreater(size, 0, name)


class FileNames(unittest.TestCase):
    """The names an install actually asks the Hub for. A wrong one is a 404 after a long wait."""

    def test_each_size_names_the_published_file(self):
        fam = setup.FAMILIES["reges"]
        for m in self.sizes():
            names = [setup.model_file(fam, m, i) for i in range(1, setup.model_shards(fam, m) + 1)]
            self.assertIn(names[0], REGES_SHARDS, f"{m} -> {names[0]} is not a published RegesCore file")

    def test_bf16_is_the_two_published_shards(self):
        fam = setup.FAMILIES["reges"]
        names = [setup.model_file(fam, "RC35-BF16", i) for i in (1, 2)]
        self.assertEqual(names, list(REGES_SHARDS)[:2])
        self.assertEqual(setup.model_shards(fam, "RC35-BF16"), 2)

    def test_the_single_file_sizes_are_one_shard(self):
        fam = setup.FAMILIES["reges"]
        for m in ("RC35-Q8_0", "RC35-IQ3_S", "RC35-IQ2_XXS"):
            self.assertEqual(setup.model_shards(fam, m), 1, m)

    @staticmethod
    def sizes():
        return [m for m in setup.MODELS if "reges" in setup.MODELS[m].get("families", ())]


class Sizes(unittest.TestCase):
    def test_download_gb_matches_the_hubs_own_bytes(self):
        """A download_gb that drifts from the real file is a disk-space promise setup cannot keep."""
        fam = setup.FAMILIES["reges"]
        for m in FileNames.sizes():
            total = sum(REGES_SHARDS[setup.model_file(fam, m, i)][0]
                        for i in range(1, setup.model_shards(fam, m) + 1))
            self.assertAlmostEqual(setup.MODELS[m]["download_gb"], total / 1e9, places=1,
                                   msg=f"{m}: setup says {setup.MODELS[m]['download_gb']} GB, "
                                       f"the Hub's files are {total/1e9:.2f} GB")

    def test_the_expert_arena_is_computed_not_guessed(self):
        """40 layers x 256 experts x 3 x 2048 x 512 = 32,212,254,720 expert parameters.  At BF16's 2 bytes
        that is 64.4 GB - which is what the BF16 file's own headers measured (64.4 of 69.4 GB is experts,
        4.95 GB is everything else).  So the arithmetic behind every other arena_gb is checked, not asserted."""
        params = 40 * 256 * 3 * 2048 * 512
        self.assertEqual(params, 32_212_254_720)
        self.assertAlmostEqual(params * 2 / 1e9, setup.MODELS["RC35-BF16"]["arena_gb"], places=1)
        # and the file is that plus a ~5 GB dense side
        self.assertAlmostEqual(setup.MODELS["RC35-BF16"]["download_gb"],
                               setup.MODELS["RC35-BF16"]["arena_gb"] + 4.95, places=0)

    def test_bigger_quant_has_a_smaller_arena(self):
        d = setup.MODELS
        self.assertGreater(d["RC35-BF16"]["arena_gb"], d["RC35-Q8_0"]["arena_gb"])
        self.assertGreater(d["RC35-Q8_0"]["arena_gb"], d["RC35-IQ3_S"]["arena_gb"])
        self.assertGreater(d["RC35-IQ3_S"]["arena_gb"], d["RC35-IQ2_XXS"]["arena_gb"])

    def test_ram_clears_the_arena_by_a_stated_margin(self):
        for m in FileNames.sizes():
            arena, ram = setup.MODELS[m]["arena_gb"], setup.MODELS[m]["ram_gb"]
            self.assertGreaterEqual(ram, arena, f"{m}: ram_gb must leave room beside the experts")
            if not setup.MODELS[m].get("budget"):
                self.assertGreater(ram - arena, 5, f"{m}: too little beside the experts (LOW_RAM_HEADROOM_GB is 10)")


class Collisions(unittest.TestCase):
    """THE BUG THIS FILE EXISTS BESIDE. `MODELS` is keyed by name; a RegesCore size called "IQ3_S" replaced
    the original model's IQ3_S outright, deleting a shipped size. Python dicts do not complain about a
    duplicate key - the later entry just wins, silently."""

    def test_no_size_shadows_an_existing_one(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        body = src[src.index("MODELS = {"):src.index("\n# The experimental Unsloth file's four shards")]
        # every key assigned inside MODELS, with its line, so a duplicate can be named
        seen: dict[str, int] = {}
        for i, line in enumerate(body.splitlines(), start=src[:src.index("MODELS = {")].count("\n") + 1):
            m = re.match(r'\s{4}"([^"]+)":\s*\{', line)
            if not m:
                continue
            key = m.group(1)
            self.assertNotIn(key, seen,
                             f'"{key}" is assigned twice in MODELS (lines {seen.get(key)} and {i}): the second '
                             f"silently replaces the first")
            seen[key] = i

    def test_the_original_models_are_all_still_there(self):
        """The concrete damage, asserted directly: the sizes the original installer shipped."""
        for m, ram in (("Q2_0", 48), ("IQ2_XS", 48), ("IQ3_XXS", 60), ("IQ3_S", 62), ("IQ1_M", 32),
                       ("UD-Q4_K_XL", 48), ("UD-IQ4_XS", 48)):
            self.assertIn(m, setup.MODELS, f"{m} was removed from the installer")
            self.assertEqual(setup.MODELS[m]["ram_gb"], ram, m)

    def test_reges_sizes_are_namespaced(self):
        for m in FileNames.sizes():
            self.assertTrue(m.startswith("RC35-"), f"{m} is not namespaced RC35-, so it can collide")

    def test_gguf_unsupported_is_unchanged_in_meaning(self):
        """Adding a family must not widen what setup accepts elsewhere: a bare K-quant from anyone is still
        refused, and the published files are still recognised."""
        self.assertEqual(setup.gguf_unsupported("model-Q4_K_M.gguf"), "Q4_K_M")
        self.assertEqual(setup.gguf_unsupported("Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"),
                         "UD-IQ3_XXS")
        self.assertIsNone(setup.gguf_unsupported("RegesCore-1.0-35B-Q8_0.gguf"))
        self.assertIsNone(setup.gguf_unsupported("RegesCore-1.0-35B-IQ3_S.gguf"))
        self.assertIsNone(setup.gguf_unsupported("mmproj-BF16.gguf"))

    def test_the_new_files_resolve_to_their_family(self):
        self.assertEqual(setup.gguf_choice("RegesCore-1.0-35B-Q8_0.gguf"), ("reges", "RC35-Q8_0"))
        self.assertEqual(setup.gguf_choice("RegesCore-1.0-35B-IQ2_XXS.gguf"), ("reges", "RC35-IQ2_XXS"))
        self.assertIsNone(setup.gguf_choice("someone-elses-Q8_0.gguf"))


class EngineGate(unittest.TestCase):
    """RegesCore needs the engine with the qwen35moe profile. Checked before the download, like UD-Q4_K_XL."""

    def test_the_gate_is_a_version(self):
        self.assertEqual(setup.REGES_ENGINE, (0, 2, 0))
        for m in FileNames.sizes():
            self.assertEqual(setup.MODELS[m]["engine"], setup.REGES_ENGINE, m)

    def test_the_gate_is_above_the_current_engine(self):
        """If REGES_ENGINE were <= the engine this checkout builds, the gate would never fire and an old
        engine would be accepted for a model it cannot load."""
        self.assertGreater(setup.REGES_ENGINE, setup.MIN_ENGINE,
                           "REGES_ENGINE must be above MIN_ENGINE or the check is dead code")


class Recommendations(unittest.TestCase):
    def test_q8_0_is_the_recommended_size(self):
        """The measured reason: Q8_0 is within noise of BF16 on this model and costs 1.9x less bandwidth, and
        on a RAM/SSD-streaming path bandwidth is the wall. BF16 is the reference, not the recommendation."""
        rec = [m for m in FileNames.sizes() if setup.MODELS[m].get("recommended")]
        self.assertEqual(rec, ["RC35-Q8_0"])
        self.assertTrue(setup.MODELS["RC35-BF16"].get("reference"), "BF16 is the reference")
        self.assertLess(setup.MODELS["RC35-Q8_0"]["download_gb"],
                        setup.MODELS["RC35-BF16"]["download_gb"])

    def test_bf16_needs_a_ram_budget(self):
        """64.4 GB of experts do not fit a 64 GB PC, so BF16 is the one size that takes --resident-budget-gib -
        exactly like Unsloth's file. Everything smaller holds its arena in RAM."""
        self.assertTrue(setup.MODELS["RC35-BF16"].get("budget"))
        self.assertGreater(setup.MODELS["RC35-BF16"]["arena_gb"], 48)
        for m in FileNames.sizes():
            if m != "RC35-BF16":
                self.assertFalse(setup.MODELS[m].get("budget"), f"{m} does not need a RAM budget")


class RamFloorIsPerFamily(unittest.TestCase):
    """The --check gate asks the CHOSEN family's smallest size. It used to ask the smallest size that existed
    anywhere, which - once a second family shipped a small model - waved through PCs that cannot run any of
    the original's models."""

    def test_the_original_families_keep_their_floors(self):
        floors = {f: min((setup.MODELS[m]["ram_gb"] for m in setup.MODELS
                          if f in setup.MODELS[m].get("families", ("qwen", "swift"))), default=None)
                  for f in ("qwen", "swift", "coder", "unsloth")}
        self.assertEqual(floors["coder"], 32)      # IQ1_M - what the old global min() returned
        self.assertEqual(floors["unsloth"], 48)

    def test_reges_is_the_new_family(self):
        self.assertIn("reges", setup.FAMILIES)
        self.assertNotEqual(setup.FAMILIES["reges"]["name"], "qwen3.8-flash-next")
        self.assertIn("RegesAi", setup.FAMILIES["reges"]["by"])
        self.assertEqual(list(setup.FAMILIES)[0], "qwen", "reges must not become the default choice")


class Branding(unittest.TestCase):
    def test_the_family_carries_the_attribution(self):
        fam = setup.FAMILIES["reges"]
        self.assertIn("LIKE A KING INC.", fam["by"])
        self.assertIn("regescore.pro", fam["license"])

    def test_the_image_encoder_is_the_publishers_own(self):
        """Not Qwen's mmproj: RegesCore ships its own, and pointing at Qwen's would load a projector that
        does not match these weights."""
        fam = setup.FAMILIES["reges"]
        self.assertEqual(fam["mmproj"], "mmproj-BF16.gguf")
        self.assertIn("iBossonline/RegesCore-1.0-35", fam["mmproj_hf"])
        self.assertNotIn("Qwen3.8", fam["mmproj"])


if __name__ == "__main__":
    unittest.main()