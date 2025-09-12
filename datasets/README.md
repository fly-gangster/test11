此目录不包含完整数据集（体积大/可能有授权限制）。
请运行 scripts/download_data.sh 来准备本地数据：
- 默认会把你本机已有的数据目录软链接到 datasets/full
- 也可以传入自定义路径，或用 --copy 复制而不是软链
用法：
  bash scripts/download_data.sh                     # 使用默认路径
  bash scripts/download_data.sh /path/to/datasets   # 指定源路径
  bash scripts/download_data.sh /path/to/datasets --copy  # 强制复制
