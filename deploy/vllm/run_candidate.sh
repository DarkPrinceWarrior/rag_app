#!/usr/bin/env bash
# Запуск одного квалификационного профиля на альтернативном loopback-порту.
# Никогда не останавливает production сам и не содержит профиля MinerU/Paddle.
set -euo pipefail

profile="${1:-}"
main_dir="${VLLM_MAIN_CANDIDATE_DIR:-/root/services/vllm-main-0.24.0}"
# Все профили используют один и тот же pinned vLLM wheel. По умолчанию visual
# переиспользует immutable main-env, чтобы не дублировать ~9 ГБ CUDA-зависимостей.
# Отдельное окружение остаётся доступно явным override для будущих конфликтов ABI.
visual_dir="${VLLM_VISUAL_CANDIDATE_DIR:-$main_dir}"

case "$profile" in
  qwen35)
    production_unit=vllm-qwen35.service; venv="$main_dir/.venv"; gpu=3; numa=1
    args=(serve /root/models/Qwen3.5-35B-A3B-GPTQ-Int4 --served-model-name qwen3.5-35b-a3b --quantization gptq_marlin --dtype bfloat16 --kv-cache-dtype fp8 --host 127.0.0.1 --port 18006 --gpu-memory-utilization 0.9 --max-model-len 16384 --trust-remote-code)
    ;;
  hymt2)
    production_unit=vllm-hymt2.service; venv="$main_dir/.venv"; gpu=1; numa=0
    args=(serve /root/models/Hy-MT2-7B --served-model-name hy-mt2-7b --host 127.0.0.1 --port 18005 --gpu-memory-utilization 0.5 --max-model-len 8192 --max-num-seqs 64)
    ;;
  embedding)
    production_unit=vllm-embedding.service; venv="$main_dir/.venv"; gpu=4; numa=1
    args=(serve /root/models/Qwen3-Embedding-8B --served-model-name qwen3-embedding-8b --runner pooling --host 127.0.0.1 --port 18002 --gpu-memory-utilization 0.45 --max-model-len 8192 --enforce-eager --dtype float16)
    ;;
  reranker)
    production_unit=vllm-reranker.service; venv="$main_dir/.venv"; gpu=4; numa=1
    args=(serve /root/models/Qwen3-Reranker-4B --served-model-name qwen3-reranker-4b --runner pooling --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' --host 127.0.0.1 --port 18003 --gpu-memory-utilization 0.35 --max-model-len 8192)
    ;;
  visual-embedding)
    production_unit=vllm-visual-embedding.service; venv="$visual_dir/.venv"; gpu=2; numa=0
    args=(serve /root/models/Qwen3-VL-Embedding-8B --served-model-name qwen3-vl-embedding-8b --runner pooling --host 127.0.0.1 --port 18007 --gpu-memory-utilization 0.6 --max-model-len 16384 --dtype float16 --enforce-eager)
    ;;
  dots-mocr)
    production_unit=dots-mocr.service; venv="$visual_dir/.venv"; gpu=0; numa=0
    args=(serve /root/models/DotsMOCR --served-model-name model --host 127.0.0.1 --port 18120 --trust-remote-code --gpu-memory-utilization 0.33 --max-model-len 24576 --chat-template-content-format string)
    ;;
  *) echo "usage: $0 {qwen35|hymt2|embedding|reranker|visual-embedding|dots-mocr}" >&2; exit 2 ;;
esac

if systemctl is-active --quiet "$production_unit"; then
  echo "отказ: сначала в согласованное окно остановите $production_unit" >&2
  exit 1
fi
test -x "$venv/bin/vllm"
export CUDA_VISIBLE_DEVICES="$gpu" NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_HOME=/usr/local/cuda
export PATH="$venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
exec /usr/bin/numactl --cpunodebind="$numa" --membind="$numa" "$venv/bin/vllm" "${args[@]}"
