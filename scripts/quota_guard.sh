#!/bin/bash
# quota_guard.sh — Kaggle GPU 配额守卫
# 用法: ./scripts/quota_guard.sh [最低剩余小时数，默认 6.0]
# 输出: 退出码 0 = 配额充足, 1 = 余量不足
# 依赖: .venv/bin/kaggle CLI (access_token 认证)

MIN_HOURS="${1:-6.0}"
KAGGLE="$(dirname "$0")/../.venv/bin/kaggle"

QUOTA_OUTPUT=$("$KAGGLE" quota 2>/dev/null)
if [ -z "$QUOTA_OUTPUT" ]; then
    echo "❌ kaggle quota 无输出（网络或凭据问题）"
    exit 1
fi

GPU_REMAINING=$(echo "$QUOTA_OUTPUT" | grep "^GPU" | awk '{gsub(/h/,"",$3); print $3}')
if [ -z "$GPU_REMAINING" ]; then
    echo "❌ 无法解析 GPU 剩余配额"
    exit 1
fi

GPU_TOTAL=$(echo "$QUOTA_OUTPUT" | grep "^GPU" | awk '{gsub(/h/,"",$4); print $4}')
GPU_USED=$(echo "$QUOTA_OUTPUT" | grep "^GPU" | awk '{gsub(/h/,"",$2); print $2}')
REFRESH=$(echo "$QUOTA_OUTPUT" | grep "^GPU" | awk '{print $5}')

echo "GPU 配额: 已用 ${GPU_USED}h / ${GPU_TOTAL}h, 剩余 ${GPU_REMAINING}h, 重置 ${REFRESH}"

# 浮点比较
ENOUGH=$(echo "$GPU_REMAINING >= $MIN_HOURS" | bc -l 2>/dev/null)
if [ "$ENOUGH" = "1" ]; then
    echo "✅ GPU 余量 ≥ ${MIN_HOURS}h，可以推送 GPU kernel"
    exit 0
else
    echo "⚠️ GPU 余量 < ${MIN_HOURS}h，冻结 GPU 推送，降级 CPU"
    exit 1
fi
