"""Tests for scdiag.param_align — automatic state-dict key alignment."""

import torch

from scdiag.param_align import (
    AlignConfig,
    AlignReport,
    _build_shape_tree,
    _tokenize,
    _walk_tree,
    align_state_dicts,
)

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


class TestTokenize:

  def test_single_token(self):
    assert _tokenize("weight") == ("weight",)

  def test_multi_token(self):
    assert _tokenize("a.b.c.weight") == ("weight", "c", "b", "a")

  def test_empty_string(self):
    assert _tokenize("") == ("",)


# ---------------------------------------------------------------------------
# Shape tree construction
# ---------------------------------------------------------------------------


class TestBuildShapeTree:

  def test_basic_structure(self):
    old = {
        "model.block1.fc.weight": torch.zeros(10, 5),
        "model.block1.fc.bias": torch.zeros(10),
    }
    tree = _build_shape_tree(old)
    assert (10, 5) in tree
    assert (10,) in tree

  def test_leaf_keys_stored(self):
    old = {"a.b.weight": torch.zeros(3, 3)}
    tree = _build_shape_tree(old)
    node = tree[(3, 3)]
    node = node.children["weight"]
    node = node.children["b"]
    node = node.children["a"]
    assert node.keys == ["a.b.weight"]

  def test_same_shape_different_names(self):
    old = {
        "a.weight": torch.zeros(4, 4),
        "b.weight": torch.zeros(4, 4),
    }
    tree = _build_shape_tree(old)
    weight_node = tree[(4, 4)].children["weight"]
    a_node = weight_node.children["a"]
    b_node = weight_node.children["b"]
    assert a_node.keys == ["a.weight"]
    assert b_node.keys == ["b.weight"]


# ---------------------------------------------------------------------------
# Walk tree
# ---------------------------------------------------------------------------


class TestWalkTree:

  def test_exact_match(self):
    old = {"a.b.c.weight": torch.zeros(3, 3)}
    tree = _build_shape_tree(old)
    new_tokens = _tokenize("a.b.c.weight")
    matches = _walk_tree(tree, (3, 3), new_tokens, max_prefix_drop=2)
    keys = [k for k, s in matches]
    assert "a.b.c.weight" in keys

  def test_prefix_drop(self):
    old = {"src.layer1.weight": torch.zeros(5,)}
    tree = _build_shape_tree(old)
    new_tokens = _tokenize("dst.layer1.weight")
    matches = _walk_tree(tree, (5,), new_tokens, max_prefix_drop=1)
    keys = [k for k, s in matches]
    assert "src.layer1.weight" in keys

  def test_no_match_different_shape(self):
    old = {"a.weight": torch.zeros(3,)}
    tree = _build_shape_tree(old)
    new_tokens = _tokenize("a.weight")
    matches = _walk_tree(tree, (5,), new_tokens, max_prefix_drop=2)
    assert matches == []

  def test_score_longer_leaf_match(self):
    """When multiple old keys share a shape, the one matching more leaf-side
    tokens should score higher."""
    old = {
        "blocks.0.attn.weight": torch.zeros(8, 8),
        "blocks.0.mlp.weight": torch.zeros(8, 8),
    }
    tree = _build_shape_tree(old)
    new_tokens = _tokenize("blocks.0.attn.weight")
    matches = _walk_tree(tree, (8, 8), new_tokens, max_prefix_drop=2)
    scores = {k: s for k, s in matches}
    assert scores["blocks.0.attn.weight"] > scores["blocks.0.mlp.weight"]

  def test_layer_expansion(self):
    """Old has 4 layers, new has 8. Layer 3 in new should match layer 3
    in old exactly, while layers 4-7 should have lower (or zero) scores."""
    old = {f"enc.layers.{i}.weight": torch.zeros(4, 4) for i in range(4)}
    tree = _build_shape_tree(old)
    # New layer 3 should match old layer 3.
    new_tokens = _tokenize("dec.layers.3.weight")
    matches = _walk_tree(tree, (4, 4), new_tokens, max_prefix_drop=1)
    scores = {k: s for k, s in matches}
    assert scores.get("enc.layers.3.weight", 0) == 3  # weight, 3, layers
    # New layer 7 — no old key with 7.
    new_tokens_7 = _tokenize("dec.layers.7.weight")
    matches_7 = _walk_tree(tree, (4, 4), new_tokens_7, max_prefix_drop=1)
    for k, s in matches_7:
      if k == "enc.layers.3.weight":
        # Score should be 2 (weight, layers — 7≠3 at the layers level).
        assert s == 2


# ---------------------------------------------------------------------------
# align_state_dicts — full integration
# ---------------------------------------------------------------------------


class TestAlignStateDicts:

  def test_identical_keys(self):
    sd = {
        "a.b.weight": torch.zeros(5, 5),
        "a.b.bias": torch.zeros(5),
    }
    report = align_state_dicts(sd, sd)
    assert report.mapping == {"a.b.weight": "a.b.weight", "a.b.bias": "a.b.bias"}
    assert report.ok

  def test_prefix_only_change(self):
    old = {
        "model.encoder.layer1.weight": torch.zeros(3, 3),
        "model.encoder.layer1.bias": torch.zeros(3),
    }
    new = {
        "vision.encoder.layer1.weight": torch.zeros(3, 3),
        "vision.encoder.layer1.bias": torch.zeros(3),
    }
    report = align_state_dicts(old, new)
    assert report.mapping["vision.encoder.layer1.weight"] == (
        "model.encoder.layer1.weight")
    assert report.mapping["vision.encoder.layer1.bias"] == ("model.encoder.layer1.bias")
    assert report.ok

  def test_prefix_only_change_short_keys(self):
    """Short keys with a prefix difference need min_token_match <= 2."""
    old = {
        "model.layer1.weight": torch.zeros(3, 3),
        "model.layer1.bias": torch.zeros(3),
    }
    new = {
        "encoder.layer1.weight": torch.zeros(3, 3),
        "encoder.layer1.bias": torch.zeros(3),
    }
    report = align_state_dicts(old, new, AlignConfig(min_token_match=2))
    assert report.mapping["encoder.layer1.weight"] == "model.layer1.weight"
    assert report.mapping["encoder.layer1.bias"] == "model.layer1.bias"
    assert report.ok

  def test_shape_mismatch_no_match(self):
    old = {"a.weight": torch.zeros(5, 5)}
    new = {"a.weight": torch.zeros(8, 8)}
    report = align_state_dicts(old, new)
    assert "a.weight" in report.unmatched_new

  def test_layer_expansion_no_false_conflict(self):
    """When new has more layers than old, extra layers should be unmatched
    without generating false 1→M conflicts."""
    old = {f"layers.{i}.weight": torch.zeros(4, 4) for i in range(4)}
    new = {f"layers.{i}.weight": torch.zeros(4, 4) for i in range(8)}
    report = align_state_dicts(old, new)
    # Layers 0-3 should match.
    for i in range(4):
      assert report.mapping[f"layers.{i}.weight"] == f"layers.{i}.weight"
    # Layers 4-7 should be unmatched (no old key with index 4-7).
    for i in range(4, 8):
      assert f"layers.{i}.weight" in report.unmatched_new
    # No 1→M conflicts.
    assert not report.divergent

  def test_layer_expansion_longer_match_wins(self):
    """The new key with a longer leaf-side match should win."""
    old = {
        "A.B.3.C.D": torch.zeros(2, 2),
    }
    new = {
        "X.Y.3.C.D": torch.zeros(2, 2),
        "X.Y.4.C.D": torch.zeros(2, 2),
    }
    # Use max_prefix_drop=3 to allow both new keys to match despite
    # having 3 root-side mismatches (4≠3, Y≠B, X≠A).
    report = align_state_dicts(old, new,
                               AlignConfig(min_token_match=1, max_prefix_drop=3))
    assert report.mapping["X.Y.3.C.D"] == "A.B.3.C.D"
    # X.Y.4.C.D should also match A.B.3.C.D (C and D match, 2 tokens).
    assert report.mapping["X.Y.4.C.D"] == "A.B.3.C.D"
    # That's a 1→M case.
    assert report.divergent

  def test_layer_expansion_high_threshold(self):
    """With min_token_match=3, only layer 3 matches exactly."""
    old = {
        "A.B.3.C.D": torch.zeros(2, 2),
    }
    new = {
        "X.Y.3.C.D": torch.zeros(2, 2),
        "X.Y.4.C.D": torch.zeros(2, 2),
    }
    report = align_state_dicts(old, new, AlignConfig(min_token_match=3))
    assert report.mapping["X.Y.3.C.D"] == "A.B.3.C.D"
    assert "X.Y.4.C.D" in report.unmatched_new
    assert not report.divergent

  def test_unused_old_keys(self):
    old = {
        "a.weight": torch.zeros(3, 3),
        "b.weight": torch.zeros(3, 3),
    }
    new = {
        "a.weight": torch.zeros(3, 3),
    }
    report = align_state_dicts(old, new, AlignConfig(min_token_match=1))
    assert report.mapping["a.weight"] == "a.weight"
    assert "b.weight" in report.unused_old

  def test_empty_state_dicts(self):
    report = align_state_dicts({}, {})
    assert report.ok
    assert report.mapping == {}

  def test_old_empty_new_has_keys(self):
    new = {"a.weight": torch.zeros(3, 3)}
    report = align_state_dicts({}, new)
    assert "a.weight" in report.unmatched_new
    assert not report.ok

  def test_new_empty_old_has_keys(self):
    old = {"a.weight": torch.zeros(3, 3)}
    report = align_state_dicts(old, {})
    assert "a.weight" in report.unused_old
    assert not report.ok

  def test_many_to_one_old_conflict(self):
    """Two new keys with identical shape and very similar names both map
    to the same old key — should flag as divergent."""
    old = {"layer.0.weight": torch.zeros(4, 4)}
    new = {
        "a.0.weight": torch.zeros(4, 4),
        "b.0.weight": torch.zeros(4, 4),
    }
    report = align_state_dicts(old, new, AlignConfig(min_token_match=2))
    assert report.divergent  # 1→M
    assert len(report.divergent) == 1
    assert report.divergent[0][0] == "layer.0.weight"

  def test_mixed_shapes(self):
    old = {
        "block.conv.weight": torch.zeros(16, 8, 3, 3),
        "block.norm.weight": torch.zeros(16),
        "head.fc.weight": torch.zeros(10, 16),
    }
    new = {
        "encoder.conv.weight": torch.zeros(16, 8, 3, 3),
        "encoder.norm.weight": torch.zeros(16),
        "classifier.fc.weight": torch.zeros(10, 16),
    }
    report = align_state_dicts(old, new, AlignConfig(min_token_match=2))
    assert report.mapping["encoder.conv.weight"] == "block.conv.weight"
    assert report.mapping["encoder.norm.weight"] == "block.norm.weight"
    assert report.mapping["classifier.fc.weight"] == "head.fc.weight"
    assert report.ok

  def test_vit_style(self):
    """Simulate typical ViT key renaming between pretrain and finetune."""
    old = {}
    new = {}
    for i in range(12):
      old[f"backbone.blocks.{i}.attn.qkv.weight"] = torch.zeros(24, 24)
      old[f"backbone.blocks.{i}.attn.proj.weight"] = torch.zeros(24, 24)
      old[f"backbone.blocks.{i}.mlp.fc1.weight"] = torch.zeros(48, 24)
      old[f"backbone.blocks.{i}.mlp.fc2.weight"] = torch.zeros(24, 48)
      old[f"backbone.blocks.{i}.norm1.weight"] = torch.zeros(24)
      old[f"backbone.blocks.{i}.norm2.weight"] = torch.zeros(24)
      new[f"model.blocks.{i}.attn.qkv.weight"] = torch.zeros(24, 24)
      new[f"model.blocks.{i}.attn.proj.weight"] = torch.zeros(24, 24)
      new[f"model.blocks.{i}.mlp.fc1.weight"] = torch.zeros(48, 24)
      new[f"model.blocks.{i}.mlp.fc2.weight"] = torch.zeros(24, 48)
      new[f"model.blocks.{i}.norm1.weight"] = torch.zeros(24)
      new[f"model.blocks.{i}.norm2.weight"] = torch.zeros(24)
    # New has extra classification head.
    new["model.head.weight"] = torch.zeros(7, 24)
    new["model.head.bias"] = torch.zeros(7)

    report = align_state_dicts(old, new)
    # All backbone params should match.
    for i in range(12):
      for suffix in [
          "attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight",
          "norm1.weight", "norm2.weight"
      ]:
        new_key = f"model.blocks.{i}.{suffix}"
        old_key = f"backbone.blocks.{i}.{suffix}"
        assert new_key in report.mapping, f"{new_key} not mapped"
        assert report.mapping[new_key] == old_key
    # Head params should be unmatched.
    assert "model.head.weight" in report.unmatched_new
    assert "model.head.bias" in report.unmatched_new

  def test_conv_stem_layer_expansion(self):
    """ConvViT stem: old has 3 conv blocks, new has 4 (different channel
    progression). Higher layers in new that share the same channel shapes
    should not blindly match old layers."""
    old = {
        "stem.blocks.0.conv.weight": torch.zeros(64, 3, 3, 3),
        "stem.blocks.0.norm.weight": torch.zeros(64),
        "stem.blocks.1.conv.weight": torch.zeros(128, 64, 3, 3),
        "stem.blocks.1.norm.weight": torch.zeros(128),
        "stem.blocks.2.conv.weight": torch.zeros(256, 128, 3, 3),
        "stem.blocks.2.norm.weight": torch.zeros(256),
    }
    new = {
        "encoder.stem.blocks.0.conv.weight": torch.zeros(64, 3, 3, 3),
        "encoder.stem.blocks.0.norm.weight": torch.zeros(64),
        "encoder.stem.blocks.1.conv.weight": torch.zeros(128, 64, 3, 3),
        "encoder.stem.blocks.1.norm.weight": torch.zeros(128),
        "encoder.stem.blocks.2.conv.weight": torch.zeros(256, 128, 3, 3),
        "encoder.stem.blocks.2.norm.weight": torch.zeros(256),
        # Block 3 is new — different shape (512 channels).
        "encoder.stem.blocks.3.conv.weight": torch.zeros(512, 256, 3, 3),
        "encoder.stem.blocks.3.norm.weight": torch.zeros(512),
    }
    report = align_state_dicts(old, new, AlignConfig(min_token_match=2))
    # Blocks 0-2 should map correctly.
    for i in range(3):
      for suffix in ["conv.weight", "norm.weight"]:
        new_key = f"encoder.stem.blocks.{i}.{suffix}"
        old_key = f"stem.blocks.{i}.{suffix}"
        assert report.mapping[new_key] == old_key, (
            f"{new_key} -> {report.mapping.get(new_key)}, expected {old_key}")
    # Block 3 is unmatched (no old key has shape (512, 256, 3, 3)).
    assert "encoder.stem.blocks.3.conv.weight" in report.unmatched_new
    assert "encoder.stem.blocks.3.norm.weight" in report.unmatched_new
    assert not report.divergent


# ---------------------------------------------------------------------------
# AlignReport
# ---------------------------------------------------------------------------


class TestAlignReport:

  def _make(self, mapping=None, unmatched_new=None, unused_old=None, divergent=None):
    r = AlignReport()
    r.mapping = mapping or {}
    r.unmatched_new = unmatched_new or []
    r.unused_old = unused_old or []
    r.divergent = divergent or []
    return r

  def test_ok_when_perfect(self):
    report = self._make(mapping={"a": "a"})
    assert report.ok

  def test_not_ok_when_unmatched(self):
    report = self._make(mapping={"a": "a"}, unmatched_new=["b"])
    assert not report.ok

  def test_not_ok_when_unused(self):
    report = self._make(mapping={"a": "a"}, unused_old=["b"])
    assert not report.ok

  def test_not_ok_when_divergent(self):
    report = self._make(mapping={"a": "a", "b": "c"}, divergent=[("c", ["a", "b"])])
    assert not report.ok

  def test_empty_report_ok(self):
    report = AlignReport()
    assert report.ok


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

  def test_single_token_keys(self):
    old = {"weight": torch.zeros(3, 3)}
    new = {"bias": torch.zeros(3, 3)}
    report = align_state_dicts(old, new, AlignConfig(min_token_match=0))
    # Shape matches but names differ; with min_token_match=0 it still
    # requires prefix_drop check. Single-token mismatch = 1 prefix drop.
    assert report.mapping.get("bias") == "weight"

  def test_deeply_nested_keys(self):
    old = {"a.b.c.d.e.f.g.weight": torch.zeros(2, 2)}
    new = {"x.y.c.d.e.f.g.weight": torch.zeros(2, 2)}
    report = align_state_dicts(old, new,
                               AlignConfig(min_token_match=5, max_prefix_drop=2))
    assert report.mapping["x.y.c.d.e.f.g.weight"] == "a.b.c.d.e.f.g.weight"
    assert report.ok

  def test_tensor_shape_preserved(self):
    """Ensure we are not confused by batch dims or other ambiguities."""
    old = {
        "a.weight": torch.zeros(16, 8, 3, 3),
        "b.weight": torch.zeros(8, 16, 3, 3),
    }
    new = {
        "enc.a.weight": torch.zeros(16, 8, 3, 3),
        "enc.b.weight": torch.zeros(8, 16, 3, 3),
    }
    report = align_state_dicts(old, new)
    assert report.mapping["enc.a.weight"] == "a.weight"
    assert report.mapping["enc.b.weight"] == "b.weight"
    assert report.ok

  def test_param_rename_not_needed(self):
    """The whole point — common renaming should just work."""
    old = {"vision_model.encoder.layer.0.weight": torch.zeros(5, 5)}
    new = {"vision_model.transformer.blocks.0.weight": torch.zeros(5, 5)}
    report = align_state_dicts(old, new)
    assert report.mapping["vision_model.transformer.blocks.0.weight"] == (
        "vision_model.encoder.layer.0.weight")
    assert report.ok

  def test_realistic_convvit_pretrain_to_finetune(self):
    """Real-world ConvViT + SimMIM pretrain → standalone fine-tune.

    Pre-training wraps ConvViTForClassification inside SimMIM, adding an
    ``encoder.`` prefix and SimMIM-specific params (mask_token, decoder).
    Fine-tuning uses ConvViTForClassification directly (``model.`` prefix)
    with a fresh classification head (``head.``).
    """
    D, H = 24, 48
    old_sd = {}
    new_sd = {}

    # --- Pretrain checkpoint (SimMIM wrapping ConvViTForClassification) ---
    # ConvPatchEmbedding stem.
    for i in range(4):
      old_sd[f"encoder.model.patch_embed.blocks.{i}.conv1.weight"] = (
          torch.zeros(D, D, 3, 3))
      old_sd[f"encoder.model.patch_embed.blocks.{i}.bn1.weight"] = (
          torch.zeros(D))
      old_sd[f"encoder.model.patch_embed.blocks.{i}.conv2.weight"] = (
          torch.zeros(D, D, 3, 3))
      old_sd[f"encoder.model.patch_embed.blocks.{i}.bn2.weight"] = (
          torch.zeros(D))
      old_sd[f"encoder.model.patch_embed.blocks.{i}.skip_proj.weight"] = (
          torch.zeros(D, D, 1, 1))
      old_sd[f"encoder.model.patch_embed.blocks.{i}.skip_bn.weight"] = (
          torch.zeros(D))
    # Transformer blocks.
    for i in range(6):
      old_sd[f"encoder.model.blocks.{i}.ln1.weight"] = torch.zeros(D)
      old_sd[f"encoder.model.blocks.{i}.attn.in_proj_weight"] = (
          torch.zeros(3 * D, D))
      old_sd[f"encoder.model.blocks.{i}.attn.out_proj.weight"] = (
          torch.zeros(D, D))
      old_sd[f"encoder.model.blocks.{i}.ln2.weight"] = torch.zeros(D)
      old_sd[f"encoder.model.blocks.{i}.mlp.w12.weight"] = (
          torch.zeros(2 * H, D))
      old_sd[f"encoder.model.blocks.{i}.mlp.w3.weight"] = (
          torch.zeros(D, H))
    # Final norm.
    old_sd["encoder.model.ln.weight"] = torch.zeros(D)
    # SimMIM-specific (should be unused).
    old_sd["encoder.head.weight"] = torch.zeros(D, D)
    old_sd["mask_token"] = torch.zeros(1, 1, D)
    old_sd["decoder.weight"] = torch.zeros(D, D, 1, 1)
    old_sd["decoder.bias"] = torch.zeros(D)

    # --- Fine-tune target (standalone ConvViTForClassification) ---
    # ConvPatchEmbedding stem (same architecture).
    for i in range(4):
      new_sd[f"model.patch_embed.blocks.{i}.conv1.weight"] = (
          torch.zeros(D, D, 3, 3))
      new_sd[f"model.patch_embed.blocks.{i}.bn1.weight"] = (
          torch.zeros(D))
      new_sd[f"model.patch_embed.blocks.{i}.conv2.weight"] = (
          torch.zeros(D, D, 3, 3))
      new_sd[f"model.patch_embed.blocks.{i}.bn2.weight"] = (
          torch.zeros(D))
      new_sd[f"model.patch_embed.blocks.{i}.skip_proj.weight"] = (
          torch.zeros(D, D, 1, 1))
      new_sd[f"model.patch_embed.blocks.{i}.skip_bn.weight"] = (
          torch.zeros(D))
    # Transformer blocks.
    for i in range(6):
      new_sd[f"model.blocks.{i}.ln1.weight"] = torch.zeros(D)
      new_sd[f"model.blocks.{i}.attn.in_proj_weight"] = (
          torch.zeros(3 * D, D))
      new_sd[f"model.blocks.{i}.attn.out_proj.weight"] = (
          torch.zeros(D, D))
      new_sd[f"model.blocks.{i}.ln2.weight"] = torch.zeros(D)
      new_sd[f"model.blocks.{i}.mlp.w12.weight"] = (
          torch.zeros(2 * H, D))
      new_sd[f"model.blocks.{i}.mlp.w3.weight"] = (
          torch.zeros(D, H))
    # Final norm.
    new_sd["model.ln.weight"] = torch.zeros(D)
    # New classification head (unmatched).
    new_sd["model.head.weight"] = torch.zeros(7, D)
    new_sd["model.head.bias"] = torch.zeros(7)

    report = align_state_dicts(old_sd, new_sd)

    # All backbone params should match (encoder.model.* → model.*).
    for i in range(4):
      for suffix in ["conv1.weight", "bn1.weight", "conv2.weight",
                     "bn2.weight", "skip_proj.weight", "skip_bn.weight"]:
        new_k = f"model.patch_embed.blocks.{i}.{suffix}"
        old_k = f"encoder.model.patch_embed.blocks.{i}.{suffix}"
        assert report.mapping[new_k] == old_k, f"{new_k} → {report.mapping.get(new_k)}"

    for i in range(6):
      for suffix in ["ln1.weight", "attn.in_proj_weight",
                     "attn.out_proj.weight", "ln2.weight",
                     "mlp.w12.weight", "mlp.w3.weight"]:
        new_k = f"model.blocks.{i}.{suffix}"
        old_k = f"encoder.model.blocks.{i}.{suffix}"
        assert report.mapping[new_k] == old_k, f"{new_k} → {report.mapping.get(new_k)}"

    assert report.mapping["model.ln.weight"] == "encoder.model.ln.weight"

    # SimMIM-specific params should be unused.
    assert "encoder.head.weight" in report.unused_old
    assert "mask_token" in report.unused_old
    assert "decoder.weight" in report.unused_old
    assert "decoder.bias" in report.unused_old

    # New classification head should be unmatched.
    assert "model.head.weight" in report.unmatched_new
    assert "model.head.bias" in report.unmatched_new

    assert not report.divergent
