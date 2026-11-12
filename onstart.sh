#!/bin/bash
# Set up and start one AI box cloud worker on a Vast.ai serverless instance, then hand over to Vast's PyWorker.
#   AIBOX_ROLE=images    ComfyUI (pinned commit, torch 2.15 cu134, SageAttention 3, Qwen-Image 2.1 NVFP4) + drawserver.py
#   AIBOX_ROLE=captions  llama-server with FLUX 3 Klein PUP (UD-Q4_K_XL + mmproj F16), AIBOX_CAPTION_SLOTS at once
# Safe to run again on a restarted worker: finished steps leave a marker in /workspace/aibox/done.
set -uo pipefail
ROLE=${AIBOX_ROLE:-images}
REPO=${AIBOX_REPO:-https://github.com/xrsxj/aibox-vast-worker}
ASSETS=${AIBOX_ASSETS:-$REPO/releases/download/v1}
HERE=$(cd "$(dirname "$0")" && pwd)
W=/workspace/aibox
DONE=$W/done
mkdir -p $W $DONE /var/log/aibox
LOG=/var/log/aibox/server.log        # the PyWorker tails this (AIBOX_READY / AIBOX_FATAL)
touch $LOG
exec >>/var/log/aibox/onstart.log 2>&1
echo "=== $(date -u +%FT%TZ) onstart role=$ROLE"
fatal() { echo "AIBOX_FATAL $*" | tee -a $LOG; exit 1; }
info() { echo "AIBOX_INFO $*" | tee -a $LOG; }
step() { [ -f "$DONE/$1" ]; }
mark() { touch "$DONE/$1"; }

# plain resumable downloads (a host's network refused hf's own client: "tlsv1 alert no application protocol")
# Hugging Face first, then its public mirror (some hosts' DNS sends huggingface.co elsewhere)
get() {
  local urls=("$1"); [[ $1 == https://huggingface.co/* ]] && urls+=("https://hf-mirror.com/${1#https://huggingface.co/}")
  for i in 1 2 3; do for u in "${urls[@]}"; do
    curl -fL --retry 2 --retry-delay 5 --connect-timeout 20 -C - -o "$2.part" "$u" && mv "$2.part" "$2" && return 0
  done; sleep 5; done; return 1
}
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh || fatal "uv install failed"
fi
export PATH=$HOME/.local/bin:$PATH

start_pyworker() {   # in the background from the start: it tails $LOG, so a failed setup (AIBOX_FATAL) reaches Vast
  export PYWORKER_REPO=$REPO BACKEND=aibox MODEL_LOG=$LOG
  if ! pgrep -f "python3 -m worker" >/dev/null; then
    (curl -fsSL https://raw.githubusercontent.com/vast-ai/pyworker/main/start_server.sh | bash) &
  fi
}
start_pyworker

if [ "$ROLE" = images ]; then
  V=/workspace/venv; C=/workspace/ComfyUI; M=/workspace/models
  if ! step venv; then
    uv venv --python 3.12 --python-preference only-managed $V || fatal "venv failed"
    uv pip install --python $V/bin/python --index-url https://download.pytorch.org/whl/cu134 \
      torch==2.15.0+cu134 torchvision==0.29.2+cu134 || fatal "torch install failed"
    uv pip install --python $V/bin/python torchaudio==2.11.1+cu134 --index-url https://download.pytorch.org/whl/cu134 \
      || info "torchaudio skipped"
    uv pip install --python $V/bin/python -r $HERE/requirements-comfy.txt || fatal "requirements failed"
    curl -fsSL -o /tmp/sageattn3-1.0.0-cp312-cp312-linux_x86_64.whl $ASSETS/sageattn3-1.0.0-cp312-cp312-linux_x86_64.whl \
      && uv pip install --python $V/bin/python /tmp/sageattn3-1.0.0-cp312-cp312-linux_x86_64.whl || fatal "sageattn3 failed"
    mark venv
  fi
  if ! step comfy; then
    rm -rf $C
    git clone --filter=blob:none https://github.com/comfyanonymous/ComfyUI $C && git -C $C checkout -q 423cbd20511becb1f16122ab518e92e58bfa03d9 \
      || fatal "ComfyUI clone failed"
    cp $HERE/custom_nodes/aibox_attention.py $C/custom_nodes/
    mark comfy
  fi
  if ! step models; then
    mkdir -p $M
    HF=https://huggingface.co/BennyDaBall/Qwen-Image-2.1-NVFP4/resolve/main
    for f in diffusion_models/qwen_image_2.1_nvfp4.safetensors text_encoders/qwen3vl_8b_nvfp4.safetensors \
             vae/qwen_image_2.1_vae_bf16.safetensors; do
      mkdir -p $M/$(dirname $f)
      get $HF/$f $M/$f || fatal "model download failed: $f"
    done
    for f in taeqi2_1_decoder.safetensors taeqi2_1_encoder.safetensors; do
      get $ASSETS/$f $C/models/vae_approx/$f || fatal "$f download failed"
    done
    ln -sf $M/diffusion_models/qwen_image_2.1_nvfp4.safetensors $C/models/diffusion_models/
    ln -sf $M/text_encoders/qwen3vl_8b_nvfp4.safetensors $C/models/text_encoders/
    ln -sf $M/vae/qwen_image_2.1_vae_bf16.safetensors $C/models/vae/
    mark models
  fi
  info "setup done; starting ComfyUI"
  (cd $C && nohup $V/bin/python main.py --listen 127.0.0.1 --port 18188 --disable-auto-launch \
      --use-pytorch-cross-attention --fast fp8_matrix_mult >>/var/log/aibox/comfyui.log 2>&1 &)
  AIBOX_COMFY_DIR=$C nohup $V/bin/python $HERE/drawserver.py >>$LOG 2>&1 &
else
  M=/workspace/pup
  SLOTS=${AIBOX_CAPTION_SLOTS:-8}
  LS=$(command -v llama-server || ls /opt/llama.cpp/build/bin/llama-server /opt/llama.cpp/llama-server /app/llama-server \
       /usr/local/bin/llama-server 2>/dev/null | head -1)
  [ -n "$LS" ] || LS=$(find / -xdev -name llama-server -type f -perm -u+x 2>/dev/null | head -1)
  [ -n "$LS" ] || fatal "no llama-server in this image"
  if ! step pup; then
    mkdir -p $M
    HF=https://huggingface.co/unsloth/flux-3-image-klein-pup-GGUF/resolve/main
    for f in flux-3-image-klein-pup-UD-Q4_K_XL.gguf mmproj-F16.gguf; do
      get $HF/$f $M/$f || fatal "PUP download failed: $f"
    done
    mark pup
  fi
  info "setup done; starting llama-server ($LS, $SLOTS slots)"
  nohup $LS --host 127.0.0.1 --port 18400 --no-webui -m $M/flux-3-image-klein-pup-UD-Q4_K_XL.gguf --mmproj $M/mmproj-F16.gguf \
    --alias flux3-pup --jinja --reasoning off -ngl 99 -c $((SLOTS * 16384)) --parallel $SLOTS -fa on -ctk q8_0 -ctv q8_0 \
    >>/var/log/aibox/llama.log 2>&1 &
  ( for i in $(seq 1800); do
      curl -fs -m 5 http://127.0.0.1:18400/health >/dev/null && { echo "AIBOX_READY llama-server answers" >>$LOG; exit 0; }
      sleep 1
    done; echo "AIBOX_FATAL llama-server never answered" >>$LOG ) &
fi
wait
