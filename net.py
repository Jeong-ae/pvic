import torch
import torch.nn as nn
# from torchvision.models.resnet import model_urls
from net_util import BasicBlock, Bottleneck
import torch.nn.functional as F
from typing import List, Tuple
import math

class InteractivenessHead(nn.Module):
    """Light MLP that predicts ρ ∈ (0,1) for each pair query."""

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)
        # Make the head slightly *conservative* at start
        nn.init.constant_(self.mlp[-1].bias, -1.5)

    def forward(self, q: torch.Tensor) -> torch.Tensor:  # (N_pair, dim)
        return self.mlp(q).squeeze(-1)   # (N_pair,)


class GroupFC(nn.Module):
    """
    N_pair 의 query 마다 K (=num_groups) 개의 작은 weight 그룹을 재사용해
    C(=117) verb 로짓을 뽑아내는 헤드. 파라미터 수가 크게 줄고,
    희귀 class 가 weight 공유 혜택을 받습니다.
    """
    def __init__(self, q_dim: int, num_classes: int, num_groups: int = 32):
        super().__init__()
        self.num_groups = num_groups
        self.num_classes = num_classes

        g = math.ceil(num_classes / num_groups)     # group 당 라벨 수
        # shape: (K, q_dim, g)
        self.weight = nn.Parameter(torch.randn(num_groups, q_dim, g) * 0.02)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # q: (B*N, D)
        K, D, g = self.weight.shape
        # (B*N, K, g)
        out = torch.einsum('nd,kdg->nkg', q, self.weight)
        out = out.reshape(q.size(0), K * g)[:, :self.num_classes]
        return out

class LabelAttentionHead(nn.Module):
    """
    Multi-Head Label-Attention for multi-label HOI classification.
    query     : (B*N_pair, D)
    logits out: (B*N_pair, C=117)
    """
    def __init__(
        self,
        q_dim: int,                # D
        num_classes: int,          # C
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        ff_hidden: int = None,
        drop_path_prob: float = 0.0,
        label_dropout: float = 0.0,
        init_scale: float = 0.02,
    ):
        super().__init__()
        assert q_dim % num_heads == 0, "q_dim must be divisible by num_heads"
        self.h = num_heads
        self.dh = q_dim // num_heads
        self.num_classes = num_classes
        self.temperature = math.sqrt(self.dh)

        # learnable label embeddings  (C, D)
        self.label_embed = nn.Parameter(torch.randn(num_classes, q_dim) * init_scale)

        # Optional Q/K/V projections (weight tying with label_embed is possible)
        self.q_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.k_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.v_proj = nn.Linear(q_dim, q_dim, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)

        # Output projection & residual-FFN
        self.o_proj = nn.Linear(q_dim, q_dim, bias=False)

        ff_hidden = q_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(q_dim, ff_hidden),
            nn.GELU(),
            nn.Dropout(attn_dropout),
            nn.Linear(ff_hidden, q_dim),
        )

        # classification layer (shared for all heads)
        self.cls_layer = nn.Linear(q_dim, num_classes, bias=True)

        # stochastic depth
        self.drop_path_prob = drop_path_prob

        # label dropout prob
        self.label_dropout = label_dropout

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls_layer.weight, std=0.02)
        nn.init.constant_(self.cls_layer.bias, 0.)

    # --------------------------
    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: (M, D)   where M = B*N_pair
        return logits: (M, C)
        """
        M, D = q.shape
        C = self.num_classes

        # label dropout: randomly mask some labels per batch (training only)
        if self.training and self.label_dropout > 0:
            keep_mask = torch.rand(C, device=q.device) > self.label_dropout
            label_embed = self.label_embed[keep_mask]            # (C', D)
            pad_needed   = C - keep_mask.sum()                   # keep shape for logits
        else:
            label_embed = self.label_embed
            keep_mask   = None

        # 1. Q, K, V
        q_proj = self.q_proj(q)                                  # (M, D)
        k_proj = self.k_proj(label_embed)                        # (C', D)
        v_proj = self.v_proj(label_embed)                        # (C', D)

        # 2. reshape for multi-head   (M, h, dh)
        qh = q_proj.view(M, self.h, self.dh)
        kh = k_proj.view(-1, self.h, self.dh).transpose(0, 1)    # (h, C', dh)
        vh = v_proj.view(-1, self.h, self.dh).transpose(0, 1)    # (h, C', dh)

        # 3. Scaled dot-product attention: each head independent
        # attn_scores: (M, h, C')
        attn_scores = (qh.unsqueeze(2) * kh.unsqueeze(0)).sum(-1) / self.temperature
        attn_probs  = F.softmax(attn_scores, dim=-1)
        attn_probs  = self.attn_dropout(attn_probs)

        # 4. context: (M, h, dh)
        context = torch.einsum('mhc,hcd->mhd', attn_probs, vh)

        context = context.reshape(M, D)                          # concat heads
        context = self.o_proj(context)

        # 5. Residual + FFN + optional DropPath
        if self.drop_path_prob > 0 and self.training:
            drop_mask = torch.rand(M, 1, device=q.device) > self.drop_path_prob
            context = context * drop_mask.float()

        y = q + context
        y = y + self.ffn(y)

        # 6. per-label logits
        logits_partial = self.cls_layer(y)                       # (M, C)

        # if some labels were dropped, pad zero logits there
        if keep_mask is not None:
            logits = logits_partial.new_full((M, C), -1e4)      # large negative → sigmoid≈0
            logits[:, keep_mask] = logits_partial
        else:
            logits = logits_partial

        return logits
    


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y, weight =None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        loss = self.targets * torch.log(xs_pos.clamp(min=self.eps))
        loss =  loss+ self.anti_targets * torch.log(xs_neg.clamp(min=self.eps))
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.set_grad_enabled(not self.disable_torch_grad_focal_loss):
                pt = xs_pos * self.targets + xs_neg * self.anti_targets
                asymmetric_w = torch.pow(1 - pt, self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            loss *= asymmetric_w

        # 클래스 가중치 적용
        if weight is not None:
            loss *= weight.to(x.device)  # (batch, num_classes) × (num_classes,)


        return -loss.sum()