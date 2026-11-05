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
UNIDEPTH_COMMIT="8d8cfe4c7ee15297099983607febf0d4f32eb3d6"
VDA_REPO="https://github.com/DepthAnything/Video-Depth-Anything.git"
VDA_COMMIT="4f5ae23172ba60fd7bc11ef671cca678842c7072"
DVD_REPO="https://github.com/EnVision-Research/DVD.git"
DVD_COMMIT="62f52d16faa2cac10e31eb7815ece6e61d5449ec"
DA3_REPO="https://github.com/ByteDance-Seed/Depth-Anything-3.git"
DA3_COMMIT="3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"

# Checkpoints that are not downloaded automatically on first use.
# (UniDepth V2 and Depth Anything 3 load their weights from Hugging Face at run time.)
VDA_CKPT_URL="https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth"
DVD_HF_REPO="FayeHongfeiZhang/DVD"

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
            # UniDepth V2: official conda instructions (Python 3.11, CUDA 11.8+ wheels)
            local dir="$THIRD_PARTY/UniDepth"
            clone_repo "$UNIDEPTH_REPO" "$UNIDEPTH_COMMIT" "$dir"
            conda create -y -n "$env_name" python=3.11
            conda activate "$env_name"
            pip install -e "$dir" --extra-index-url https://download.pytorch.org/whl/cu118
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
            # DVD: official instructions (Python 3.10, pip install -e .), weights into ckpt/
            local dir="$THIRD_PARTY/DVD"
            clone_repo "$DVD_REPO" "$DVD_COMMIT" "$dir"
            conda create -y -n "$env_name" python=3.10
            conda activate "$env_name"
            pip install -e "$dir"
            python -c "from huggingface_hub import snapshot_download; \
snapshot_download('$DVD_HF_REPO', revision='main', local_dir='$dir/ckpt')"
            ;;
        da3)
            # Depth Anything 3: official instructions (xformers + torch>=2, pip install -e .)
            local dir="$THIRD_PARTY/Depth-Anything-3"
            clone_repo "$DA3_REPO" "$DA3_COMMIT" "$dir"
            conda create -y -n "$env_name" python=3.10
            conda activate "$env_name"
            pip install xformers "torch>=2" torchvision
            pip install -e "$dir"
            ;;
    esac

    # FoSA itself (numpy-level dependencies only; torch comes from the model above)
    if [ -f "$PROJ_ROOT/pyproject.toml" ]; then
        pip install -e "$PROJ_ROOT"
    else
        echo "[$name] FoSA package not found yet (code release in progress); skipped."
    fi

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
