# aibox-vast-worker

Worker code for a private image service's overflow on [Vast.ai serverless](https://docs.vast.ai/guides/serverless/architecture).
Vast clones this repo onto each worker; nothing here is secret.

- `onstart.sh`: the template's start script. `AIBOX_ROLE=images` installs ComfyUI (pinned commit, torch 2.15 cu134,
  SageAttention 3), the `AIBoxAttention` node and Qwen-Image 2.1 NVFP4 (`BennyDaBall/Qwen-Image-2.1-NVFP4`) and starts
  `drawserver.py`; `AIBOX_ROLE=captions` starts llama-server with `unsloth/flux-3-image-klein-pup-GGUF`.
  Then it runs Vast's `start_server.sh`, which runs `worker.py`.
- `worker.py`: the PyWorker config (routes `/draw` and `/v1/chat/completions`, benchmark, readiness lines).
- `drawserver.py`: one ComfyUI workflow per `/draw` call, reference images and results as base64.
- Release `v1` holds the prebuilt `sageattn3` wheel (sm_120, cp312, torch 2.15) and the `taeqi2_1` tiny VAE
  (madebyollin) files.

Needs an RTX 50-series (Blackwell) card for NVFP4.
