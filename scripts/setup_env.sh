#!/bin/bash
# Set up conda environment(s) for FoSA.
# Each foundation model pins its own torch / xformers / CUDA build, so each one gets its own
# conda environment. The official code of each model is cloned into third_party/ at a pinned
# commit and installed following its own instructions; FoSA is then installed into the same env.
#
# Usage:
#   bash scripts/setup_env.sh unidepth
#   bash scripts/setup_env.sh vda
#   bash scripts/setup_env.sh dvd
#   bash scripts/setup_env.sh da3
#   bash scripts/setup_env.sh all
set -e

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
THIRD_PARTY="$PROJ_ROOT/third_party"

# Official repositories and pinned commits.
UNIDEPTH_REPO="https://github.com/lpiccinelli-eth/UniDepth.git"
# UniDepth: bebc4b2 is the commit the paper's backbone outputs come from. Later commits rename
# the model (UniDepthV2old) and change its attention kernels, which moves the bf16 output.
UNIDEPTH_COMMIT="bebc4b2fdda5c223693c390739db76897aee42e5"
VDA_REPO="https://github.com/DepthAnything/Video-Depth-Anything.git"
VDA_COMMIT="4f5ae23172ba60fd7bc11ef671cca678842c7072"
DVD_REPO="https://github.com/EnVision-Research/DVD.git"
DVD_COMMIT="62f52d16faa2cac10e31eb7815ece6e61d5449ec"
DA3_REPO="https://github.com/ByteDance-Seed/Depth-Anything-3.git"
DA3_COMMIT="3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"

# Checkpoints that are not downloaded automatically on first use, pinned to the Hugging Face
# revisions whose files match the ones used in the paper (SHA-256 checked).
# (UniDepth V2 and Depth Anything 3 load their weights from Hugging Face at run time.)
VDA_CKPT_URL="https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/607fcdbd454b95c3bd39abbd3054142869a527d3/metric_video_depth_anything_vitl.pth"
DVD_HF_REPO="FayeHongfeiZhang/DVD"
DVD_HF_REVISION="769534dec2d77f8da3667719568fbe57fe69cd9b"   # model.safetensors = DVD v1.0
WAN_HF_REPO="Wan-AI/Wan2.1-T2V-1.3B"
WAN_HF_REVISION="37ec512624d61f7aa208f7ea8140a131f93afc9a"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Install Anaconda or Miniconda first: https://docs.anaconda.com/miniconda/"
    exit 1
fi
eval "$(conda shell.bash hook)"

# clone_repo <url> <commit> <dir>
clone_repo() {
    local url="$1" commit="$2" dir="$3"
    if [ ! -d "$dir/.git" ]; then
        git clone --filter=blob:none "$url" "$dir"
    fi
    git -C "$dir" fetch --quiet origin "$commit" || true
    git -C "$dir" checkout --quiet "$commit"
}

env_exists() {
    conda env list | awk '{print $1}' | grep -qx "$1"
}

setup_env() {
    local name="$1"
    local env_name="fosa-${name}"

    if env_exists "$env_name"; then
        echo "[$name] Environment already exists: $env_name"
        echo "[$name] To recreate, remove it first: conda env remove -n $env_name"
        return 0
    fi

    mkdir -p "$THIRD_PARTY"
    echo "[$name] Creating environment..."

    case "$name" in
        unidepth)
            # UniDepth V2 at bebc4b2 in the paper's environment: Python 3.10, torch 2.5.1 (CUDA 12.4),
            # xformers 0.0.28.post3, and Pillow 11.3.0 (JPEG decoding). UniDepth's own requirements
            # pin torch 2.2.0, whose bf16 output differs from the paper's by about 0.25%, so its
            # other requirements are installed without the torch / xformers / triton / numpy /
            # Pillow pins.
            local dir="$THIRD_PARTY/UniDepth"
            clone_repo "$UNIDEPTH_REPO" "$UNIDEPTH_COMMIT" "$dir"
            conda create -y -n "$env_name" python=3.10
            conda activate "$env_name"
            pip install torch==2.5.1 torchvision==0.20.1 xformers==0.0.28.post3 pillow==11.3.0 numpy==2.0.1
            pip install -r <(grep -viE '^(torch|torchvision|torchaudio|xformers|triton|numpy|pillow)\b' \
                                 "$dir/requirements.txt")
            pip install -e "$dir" --no-deps
            ;;
        vda)
            # Video Depth Anything: requirements.txt pins torch 2.1.1 / xformers 0.0.23.
            # It is not a pip package; FoSA imports it from third_party/Video-Depth-Anything.
            local dir="$THIRD_PARTY/Video-Depth-Anything"
            clone_repo "$VDA_REPO" "$VDA_COMMIT" "$dir"
            conda create -y -n "$env_name" python=3.10
            conda activate "$env_name"
            pip install -r "$dir/requirements.txt"
            mkdir -p "$dir/checkpoints"
            wget -c -P "$dir/checkpoints" "$VDA_CKPT_URL"
            ;;
        dvd)
            # DVD: official instructions (Python 3.10, pip install -e .). DVD v1.0 weights into
            # ckpt/ (next to the tracked ckpt/model_config.yaml), and the Wan2.1-T2V-1.3B DiT, VAE
            # and tokenizer into models/, where DVD's loader looks for them.
            local dir="$THIRD_PARTY/DVD"
            clone_repo "$DVD_REPO" "$DVD_COMMIT" "$dir"
            conda create -y -n "$env_name" python=3.10
            conda activate "$env_name"
            pip install -e "$dir"
            python -c "from huggingface_hub import snapshot_download; \
snapshot_download('$DVD_HF_REPO', revision='$DVD_HF_REVISION', allow_patterns=['model.safetensors'], \
                  local_dir='$dir/ckpt'); \
snapshot_download('$WAN_HF_REPO', revision='$WAN_HF_REVISION', \
                  allow_patterns=['diffusion_pytorch_model*.safetensors', 'Wan2.1_VAE.pth', 'google/umt5-xxl/*'], \
                  local_dir='$dir/models/Wan-AI/Wan2.1-T2V-1.3B')"
            ;;
        da3)
            # Depth Anything 3: official instructions (xformers + torch>=2, pip install -e .).
            # The paper used Python 3.11 (torch 2.11.0+cu128, xformers 0.0.35).
            local dir="$THIRD_PARTY/Depth-Anything-3"
            clone_repo "$DA3_REPO" "$DA3_COMMIT" "$dir"
            conda create -y -n "$env_name" python=3.11
            conda activate "$env_name"
            pip install xformers "torch>=2" torchvision
            pip install -e "$dir"
            ;;
    esac

    # FoSA's own dependencies (numpy, pillow, h5py, pyyaml, tqdm); torch comes from the model above
    pip install -e "$PROJ_ROOT"

    conda deactivate
    echo "[$name] Environment ready: $env_name"
    echo "[$name] Activate with: conda activate $env_name"
}

# --- Main ---
MODELS=("$@")

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Usage: bash scripts/setup_env.sh {unidepth|vda|dvd|da3|all}"
    exit 1
fi

if [ "${MODELS[0]}" = "all" ]; then
    MODELS=(unidepth vda dvd da3)
fi

for model in "${MODELS[@]}"; do
    case "$model" in
        unidepth|vda|dvd|da3)
            setup_env "$model"
            ;;
        *)
            echo "Unknown model: $model (expected unidepth, vda, dvd, da3, or all)"
            exit 1
            ;;
    esac
done
