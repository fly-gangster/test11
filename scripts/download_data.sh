set -euo pipefail
SRC="${1:-${DENSE_DATA_SRC:-/home/dfg/workspace/1DenseFusion/datasets}}"
DEST_DIR="$(pwd)/datasets"
DEST_LINK="$DEST_DIR/full"

MODE="link"
if [[ "${2:-}" == "--copy" ]]; then
  MODE="copy"
fi

echo "[INFO] 源数据目录: $SRC"
echo "[INFO] 目标目录:   $DEST_LINK"
echo "[INFO] 模式:       $MODE (可用 --copy 进行拷贝)"

if [[ ! -d "$SRC" ]]; then
  echo "[ERROR] 找不到源数据目录：$SRC"
  echo "        请传入正确路径，如：bash scripts/download_data.sh /path/to/datasets"
  exit 1
fi

mkdir -p "$DEST_DIR"

if [[ "$MODE" == "copy" ]]; then
  # 复制（耗时且占空间，但不依赖软链接）
  rsync -av --progress "$SRC"/ "$DEST_LINK"/
else
  # 软链接（推荐，快且不占空间）
  ln -sfn "$SRC" "$DEST_LINK"
fi

echo "[OK] 数据已准备完成：$DEST_LINK"
ls -al "$DEST_DIR"
