"""AI box ComfyUI node (idea #131): SageAttention 3 where it really applies, Comfy Kitchen's int8 kernel for the rest.

ComfyUI registers SageAttention 3 as 'sage3', but no node selects it, and on Qwen-Image 2.1 it is slower as wired:
every masked call (the text and reference segments) falls back to plain PyTorch. 'sage3_int8' sends a call to
SageAttention 3 only when it can take it (no mask, fp16/bf16 on CUDA, > 1024 tokens, head dim < 256) and everything
else to Comfy Kitchen's int8 kernel, which takes masks (the live attention before this node). Benched in
docs/qwen-speed-bench.md (round 2): with the tiny VAE and 20 steps, -20% against 25 steps of int8.

Installed by a link from /opt/comfyui/ComfyUI/custom_nodes/aibox_attention.py to this file; SageAttention 3 is the
sageattn3 wheel in /opt/comfyui/wheels (build notes in the doc). Without sageattn3 (a rebuilt venv, a non-Blackwell
card) or without Comfy Kitchen's int8 kernel, the node passes the model through unchanged, so the workflow keeps
drawing with whatever attention came before it."""
import logging
import torch
import comfy.ldm.modules.attention as attention


def sage3_int8(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    args = dict(mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape, **kwargs)
    ok = mask is None and q.device.type == 'cuda' and q.dtype in (torch.float16, torch.bfloat16)
    if ok:
        if skip_reshape:   # (batch, heads, tokens, dim)
            n, d = q.shape[2], q.shape[3]
            ok = q.shape[1] == heads
        else:              # (batch, tokens, heads * dim)
            n, inner = q.shape[1], q.shape[2]
            ok = inner % heads == 0
            d = inner // heads if ok else 0
        ok = ok and d < 256 and n > 1024   # where ComfyUI's own sage3 path would use the kernel, not its fallback
    if ok:
        return attention.attention3_sage(q, k, v, heads, **args)
    return attention.attention_comfy_kitchen_int8(q, k, v, heads, **args)


AVAILABLE = (getattr(attention, 'SAGE_ATTENTION3_IS_AVAILABLE', False)
             and hasattr(attention, 'attention3_sage') and hasattr(attention, 'attention_comfy_kitchen_int8'))
if AVAILABLE:
    attention.register_attention_function('sage3_int8', sage3_int8)


class AIBoxAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "attention": (["sage3_int8"], {"default": "sage3_int8"})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/patch"
    DESCRIPTION = "SageAttention 3 where it applies, Comfy Kitchen int8 for masked calls (AI box, idea #131)"

    def patch(self, model, attention):
        fn = globals()[attention] if AVAILABLE else None
        if fn is None:
            logging.warning("aibox_attention: %s unavailable here (sageattn3 or Comfy Kitchen int8 missing): "
                            "the model keeps the attention it came with", attention)
            return (model,)
        m = model.clone()
        m.set_model_optimized_attention(fn)
        return (m,)


NODE_CLASS_MAPPINGS = {"AIBoxAttention": AIBoxAttention}
NODE_DISPLAY_NAME_MAPPINGS = {"AIBoxAttention": "AI box attention (SageAttention 3 + int8)"}
