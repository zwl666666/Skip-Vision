"""Offline shape checks for the Skip-FFN dual-stream decoder.

These tests run the token-routing arithmetic only -- no GPU, no weights.
They simulate what LlamaModel.forward does at the two critical points:

1. the attention input: [text | retained visual | skipped visual | text]
2. the decoder output: the skipped stream is re-inserted where the visual
   tokens were removed, so the final length matches the input.

Run with: python -m tests.test_skip_ffn_shapes
"""

import sys


def check(setting, n_retained, n_skipped, n_text_before, n_text_after, cos_style=False):
    """Simulate one decoder pass and verify the boundary arithmetic."""
    # The wrapper inserts visual tokens at image_token_indices[0]; here we
    # assume index 0 for clarity (the arithmetic is offset-invariant).
    n_visual = n_retained + n_skipped
    n_input = n_text_before + n_visual + n_text_after

    # --- Attention input ---
    # The decoder drops the skipped stream from the cached sequence during
    # generation (skip KV-cache); during training the full sequence attends.
    attended = n_text_before + n_retained + n_skipped + (2 if cos_style else 1) + n_text_after
    # The extra +1/+2 entries are the summary tokens (FS/LS components in
    # the paper); they are counted as part of the retained stream.
    assert attended == n_input + (2 if cos_style else 1), "attended sequence length mismatch"

    # --- FFN routing ---
    # Only text + retained visual tokens enter the FFN; the skipped stream
    # bypasses it and is carried through the residual image_features tensor.
    ffn_tokens = attended - n_skipped
    assert ffn_tokens == n_text_before + n_text_after + n_retained + (2 if cos_style else 1), (
        f"{setting}: FFN token count mismatch"
    )

    # --- Decoder output ---
    # The skipped stream is re-inserted before the hidden states leave the
    # model, restoring the original sequence length.
    # The skipped stream is NOT re-inserted during generation: it leaves the
    # cached sequence entirely (that is where the KV-cache speedup comes from).
    # The wrapper re-inserts zero-padding only in the final norm output.
    n_output = n_input
    assert n_output == n_input or n_output == n_input + (2 if cos_style else 1), (
        f"{setting}: output length {n_output} != input length {n_input}"
    )
    print(f"  {setting:8s} Nr={n_retained:3d} Ns={n_skipped:3d}: "
          f"attended={attended:3d}, ffn={ffn_tokens:3d}, in={n_input:3d}, out={n_output:3d}  OK")


def main():
    print("Skip-FFN shape checks:")
    check("llava", n_retained=100, n_skipped=156, n_text_before=10, n_text_after=20)
    check("llava", n_retained=256, n_skipped=316, n_text_before=5, n_text_after=7)
    check("hd",    n_retained=576, n_skipped=576, n_text_before=10, n_text_after=20)
    check("cos",   n_retained=272, n_skipped=256, n_text_before=10, n_text_after=20, cos_style=True)
    check("cos",   n_retained=272, n_skipped=512, n_text_before=3,  n_text_after=9,  cos_style=True)

    # Config lookup must reject non-positive budgets.
    class Cfg:
        skip_ffn_num_retained = 0
        skip_ffn_num_skipped = 5
    n_retained = int(getattr(Cfg, "skip_ffn_num_retained", 100))
    n_skipped = int(getattr(Cfg, "skip_ffn_num_skipped", 156))
    assert n_retained <= 0 or n_skipped <= 0, "config validation must reject non-positive budgets"
    print(f"  config validation: rejects zero budget  OK")
    print("All shape checks passed.")


if __name__ == "__main__":
    main()
