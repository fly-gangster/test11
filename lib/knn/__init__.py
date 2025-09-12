"""
lib.knn  ——  轻量 KNN 封装 (CUDA)
优先 knn_cuda，其次 torch_cluster.knn；输出 (B, k, M)
"""
import torch
from torch.autograd import Function

# ----------------- 选择后端 -----------------
_BACKEND = None
try:
    import knn_cuda                      # pip install knn_cuda==0.2   （可选）
    _BACKEND = "knn_cuda"
except ImportError:
    try:
        from torch_cluster import knn as tc_knn  # pip install torch_cluster
        _BACKEND = "torch_cluster"
    except ImportError as e:
        raise ImportError(
            "未检测到 knn_cuda / torch_cluster，"
            "请先 `pip install torch_cluster==1.6.3 "
            "-f https://data.pyg.org/whl/torch-1.6.0+cu102.html`"
        ) from e

# ----------------- Autograd Function -----------------
class _KNN(Function):
    @staticmethod
    def forward(ctx, ref, query, k):
        """
        ref   : (B,C,N)  float32
        query : (B,C,M)  float32
        k     : int
        return: idx (B,k,M) int64
        """
        ref   = ref.contiguous().float().cuda()
        query = query.contiguous().float().cuda()

        if _BACKEND == "knn_cuda":
            idx, _ = knn_cuda.knn(ref, query, k)          # (B,M,k)
            idx = idx.permute(0, 2, 1).contiguous()       # → (B,k,M)

        else:  # torch_cluster
            B, _, M = query.shape
            idx_out = torch.empty(B, k, M, dtype=torch.long, device=query.device)
            ref_t, query_t = ref.transpose(1, 2), query.transpose(1, 2)  # (B,N,C)/(B,M,C)

            for b in range(B):
                row, col = tc_knn(ref_t[b], query_t[b], k, num_workers=0)  # row→ref idx
                sort_idx = torch.argsort(col)            # 让同一 query idx 连续
                row_sorted = row[sort_idx].view(M, k)     # (M,k)
                idx_out[b] = row_sorted.t().contiguous()  # (k,M)
            idx = idx_out

        return idx

# ----------------- 对外接口 -----------------
class KNearestNeighbor:
    def __init__(self, k): self.k = k
    def __call__(self, ref, query): return _KNN.apply(ref, query, self.k)

