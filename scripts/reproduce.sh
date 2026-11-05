#!/usr/bin/env bash
# Reproduce the FoSA rows of the paper's tables with one foundation model.
#
#   bash scripts/reproduce.sh <model> [main] [sparse]
#     model:  unidepth | vda | dvd | da3      (run inside the conda env fosa-<model>)
#     main:   the five benchmarks (Tables I-V)
#     sparse: the sparser measurements (Table VI, UniDepth V2 only)
#     default: main, plus sparse for unidepth
#
# DATA_ROOT must hold the datasets as described in docs/REPRODUCE.md. With CACHE_ROOT set,
# the backbone outputs are cached under $CACHE_ROOT/<model>_<dataset>/ and reused. Results go
# to ${OUTPUT:-outputs}/<time>_<model>_<setting>/eval_results.txt (last line: averages).
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
model=$1
case $model in unidepth|vda|dvd|da3) ;; *) sed -n '2,13p' "${BASH_SOURCE[0]}"; exit 1;; esac
shift
: "${DATA_ROOT:?set DATA_ROOT to the dataset folder (see docs/REPRODUCE.md)}"
parts=("$@")
[[ ${#parts[@]} -eq 0 ]] && parts=(main) && [[ $model == unidepth ]] && parts+=(sparse)

declare -A DIR=([nyu_video]=NYUv2 [void]=VOID/void_500 [void150]=VOID/void_150
                [kitti_video]=KITTI-DC [dydtof]=DyDToF [ddad_vid]=DDAD)

run() {  # run <setting> <dataset dir key> [main.py args]
  local name=$1 key=$2; shift 2
  local ds=${key/void150/void} cache=()
  [[ -n $CACHE_ROOT ]] && cache=(--base-cache "$CACHE_ROOT/${model}_${key}")
  python main.py --config "config/$model.yaml" --dataset "$ds" --dataset-path "$DATA_ROOT/${DIR[$key]}" \
      --output "${OUTPUT:-outputs}" --name "${model}_${name}" "${cache[@]}" "$@"
}

for part in "${parts[@]}"; do
  case $part in
    main)
      for ds in nyu_video void kitti_video dydtof ddad_vid; do
        run "$ds" "$ds"
      done ;;
    sparse)
      [[ $model == unidepth ]] || { echo "Table VI uses UniDepth V2 only"; exit 1; }
      run kitti_video_8lines   kitti_video --num-lidar-lines 8
      run dydtof_50points      dydtof      --num-sparse-samples 50
      run nyu_video_50points   nyu_video   --num-sparse-samples 50
      run void_150             void150
      run ddad_vid_10pct_hints ddad_vid    --sparse-keep-ratio 0.1 ;;
    *) echo "unknown part $part (main | sparse)"; exit 1 ;;
  esac
done
