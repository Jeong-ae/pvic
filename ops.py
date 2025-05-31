"""
Opearations

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""

import math
import torch
import torchvision.ops.boxes as box_ops

from torch import Tensor
from typing import Optional, List, Tuple

def compute_sinusoidal_pe(pos_tensor: Tensor, temperature: float = 10000.) -> Tensor:
    """
    Compute positional embeddings for points or bounding boxes

    Parameters:
    -----------
    pos_tensor: Tensor
        Coordinates of 2d points (x, y) normalised to (0, 1). The shape is (n_q, bs, 2).
    temperature: float, Default: 10000.
        The temperature parameter in sinusoidal functions.

    Returns:
    --------
    pos: Tensor
        Sinusoidal positional embeddings of shape (n_q, bs, 256).
    """
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos

def prepare_region_proposals(
    results, hidden_states, image_sizes,
    box_score_thresh, human_idx,
    min_instances, max_instances
):
    region_props = []
    for res, hs, sz in zip(results, hidden_states, image_sizes):
        sc, lb, bx = res.values() ##sc: 예측된 객체의 confidence score / lb: 예측된 객체의 class label / bx: 예측된 객체의 bounding box (x1,y1,x2,y2)

        keep = box_ops.batched_nms(bx, sc, lb, 0.5) #겹치는 박스 제거 (동일 클래서 IoU 0.5이상이면 낮은 score박스 제거)
        sc = sc[keep].view(-1)
        lb = lb[keep].view(-1)
        bx = bx[keep].view(-1, 4)
        hs = hs[keep].view(-1, 256)

        # Clamp boxes to image
        bx[:, :2].clamp_(min=0)
        bx[:, 2].clamp_(max=sz[1])
        bx[:, 3].clamp_(max=sz[0])

        keep = torch.nonzero(sc >= box_score_thresh).squeeze(1)

        is_human = lb == human_idx #label이 human_idx (0)인것만 사람으로 분류 / 나머지 객체로 간주
        hum = torch.nonzero(is_human).squeeze(1)
        obj = torch.nonzero(is_human == 0).squeeze(1)
        n_human = is_human[keep].sum(); n_object = len(keep) - n_human
        # Keep the number of human and object instances in a specified interval
        if n_human < min_instances:
            keep_h = sc[hum].argsort(descending=True)[:min_instances]
            keep_h = hum[keep_h]
        elif n_human > max_instances:
            keep_h = sc[hum].argsort(descending=True)[:max_instances]
            keep_h = hum[keep_h]
        else:
            keep_h = torch.nonzero(is_human[keep]).squeeze(1)
            keep_h = keep[keep_h]

        if n_object < min_instances:
            keep_o = sc[obj].argsort(descending=True)[:min_instances]
            keep_o = obj[keep_o]
        elif n_object > max_instances:
            keep_o = sc[obj].argsort(descending=True)[:max_instances]
            keep_o = obj[keep_o]
        else:
            keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
            keep_o = keep[keep_o]

        keep = torch.cat([keep_h, keep_o])

        region_props.append(dict(
            boxes=bx[keep], #선택된 객체들의 위치
            scores=sc[keep], #confidence score
            labels=lb[keep], #class label
            hidden_states=hs[keep] #transformer에서 추출된 해당박스에 대응되는 feature vector(256 차원)
        ))

    return region_props

def associate_with_ground_truth(boxes, paired_inds, targets, num_classes, thresh=0.5):
    labels = []
    for bx, p_inds, target in zip(boxes, paired_inds, targets):
        is_match = torch.zeros(len(p_inds), num_classes, device=bx.device)

        bx_h, bx_o = bx[p_inds].unbind(1)
        gt_bx_h = recover_boxes(target["boxes_h"], target["size"])
        gt_bx_o = recover_boxes(target["boxes_o"], target["size"])

        x, y = torch.nonzero(torch.min(
            box_ops.box_iou(bx_h, gt_bx_h),
            box_ops.box_iou(bx_o, gt_bx_o)
        ) >= thresh).unbind(1)
        is_match[x, target["labels"][y]] = 1

        labels.append(is_match)
    return torch.cat(labels)

def recover_boxes(boxes, size):
    boxes = box_cxcywh_to_xyxy(boxes)
    h, w = size
    scale_fct = torch.stack([w, h, w, h])
    boxes = boxes * scale_fct
    return boxes

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def pad_queries(queries):
    b = len(queries)
    k = queries[0].shape[1]
    ns = [len(q) for q in queries]
    device = queries[0].device
    dtype = queries[0].dtype

    padded_queries = torch.zeros(b, max(ns), k, device=device, dtype=dtype)
    q_padding_mask = torch.zeros(b, max(ns), device=device, dtype=torch.bool)
    for i, n in enumerate(ns):
        padded_queries[i, :n] = queries[i]
        q_padding_mask[i, n:] = True
    return padded_queries, q_padding_mask

def make_pair_verb_mask(
        prior_i,
    pair_obj_labels: torch.Tensor,      # (N,)
    obj_to_verbs: dict,                 # object_id -> list[int]
    num_verbs: int = 117,
    device=None
):
    """
    Returns
        mask : (N, num_verbs)  float32, 가능한 verb 위치 1
    """
    if device is None:
        device = pair_obj_labels.device
    N = pair_obj_labels.size(0)

    # ② object 단위로 가능한 verb 위치를 1.0 으로 덮어쓰기
    for obj_id in pair_obj_labels.unique().tolist():
        verb_ids = obj_to_verbs.get(obj_id, [])
        if not verb_ids:
            continue
        rows = (pair_obj_labels == obj_id).nonzero(as_tuple=False).flatten()   # (k,)
        cols = torch.tensor(verb_ids, device=device)                           # (m,)
        prior_i[rows.unsqueeze(1), cols] = 1.0                                 # overwrite

    return prior_i    

def compute_prior_scores(
    x: Tensor, y: Tensor,
    scores: Tensor, labels: Tensor,
    num_classes: int, is_training: bool,
    obj_cls_to_tgt_cls: list, 
    sim_interaction: Tensor,
    hoi_to_verb: list, hoi_to_object: list,
    alpha: float = 0.5, 
) -> Tensor:
    device = scores.device
    P = len(x)
    # ------------------------------------------------------------------
    # 1) object×verb CLIP 가중치 행렬 W[obj, verb] 생성
    # ------------------------------------------------------------------
    
    #####initialize mask#####
    M = torch.zeros(80, num_classes)
    initialize_reranking = [obj_cls_to_tgt_cls[obj.item()]
        for obj in labels[y].unique()]
    unique_objects_id = labels[y].unique()
    row_idx = torch.tensor([
                r for r, c_list in zip(unique_objects_id, initialize_reranking) for _ in c_list])
    col_idx = torch.tensor([c for c_list in initialize_reranking for c in c_list])
    M[row_idx, col_idx] = 0.1

    ####sim value #####
    obj_id = hoi_to_object[sim_interaction.topk(20).indices.detach().cpu()]
    verb_id = hoi_to_verb[sim_interaction.topk(20).indices.detach().cpu()]
    use_sim =sim_interaction.topk(20).values.float().detach().cpu()
    sim_z = (use_sim - use_sim.mean()) / (use_sim.std(unbiased=False) + 1e-6)
    sim_prop = sim_z.sigmoid()
    M[obj_id, verb_id] = sim_prop  
    M = M.to(device) 

    
    # W = torch.zeros(max(hoi_to_object) + 1, num_classes, device=device)

    # for intr_id, sim in enumerate(sim_interaction):     # 600회 루프
    #     o_id = hoi_to_object[intr_id]
    #     v_ids = hoi_to_verb[intr_id]
    #     if not isinstance(v_ids, (list, tuple)):
    #         v_ids = [v_ids]
    #     for v in v_ids:
    #         W[o_id, v] = torch.maximum(W[o_id, v], sim)

    #     # 2) object-별로 가능한 verb 부분만 정규화 (0~1)
    # for o_id, v_list in enumerate(obj_cls_to_tgt_cls):
    #     if not v_list:                     # 어떤 object는 verb set이 비어 있을 수 있음
    #         continue
    #     w_slice = W[o_id, v_list]
    #     if w_slice.max() > 0:
    #         # min-max → 0~1 스케일
    #         W[o_id, v_list] = (w_slice - w_slice.min()) / (w_slice.max() - w_slice.min() + 1e-6)

    # 1) detector 기반 prior 초기화
    prior_h = torch.zeros(P, num_classes, device=device)
    prior_o = torch.zeros_like(prior_h)

    s_h, s_o = scores[x], scores[y]

    # Map object class index to target class index
    # Object class index to target class index is a one-to-many mapping
    target_cls_idx = [obj_cls_to_tgt_cls[obj.item()]
        for obj in labels[y]]
    # Duplicate box pair indices for each target class
    pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
    # Flatten mapped target indices
    flat_target_idx = [t for tar in target_cls_idx for t in tar]

    prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
    prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

    # ------------------------------------------------------------------
    # 4) object-verb CLIP weight 적용 → prior_o 갱신
    # ------------------------------------------------------------------
    obj_ids_pair = labels[y]                 # (P,)
    clip_weight  = M[obj_ids_pair]           # (P, 117)  object별 verb weight

    # prior_o = s_o * ( α + (1-α)·clip_weight )
    prior_h *= alpha + (1.0 - alpha) * clip_weight
    prior_o *= alpha + (1.0 - alpha) * clip_weight

    return torch.stack([prior_h, prior_o], dim=1)


def compute_spatial_encodings(
    boxes_1: List[Tensor], boxes_2: List[Tensor],
    shapes: List[Tuple[int, int]], eps: float = 1e-10
) -> Tensor:
    """
    Parameters:
    -----------
    boxes_1: List[Tensor]
        First set of bounding boxes (M, 4)
    boxes_1: List[Tensor]
        Second set of bounding boxes (M, 4)
    shapes: List[Tuple[int, int]]
        Image shapes, heights followed by widths
    eps: float
        A small constant used for numerical stability

    Returns:
    --------
    Tensor
        Computed spatial encodings between the boxes (N, 36)
    """
    features = []
    for b1, b2, shape in zip(boxes_1, boxes_2, shapes):
        h, w = shape

        c1_x = (b1[:, 0] + b1[:, 2]) / 2; c1_y = (b1[:, 1] + b1[:, 3]) / 2
        c2_x = (b2[:, 0] + b2[:, 2]) / 2; c2_y = (b2[:, 1] + b2[:, 3]) / 2

        b1_w = b1[:, 2] - b1[:, 0]; b1_h = b1[:, 3] - b1[:, 1]
        b2_w = b2[:, 2] - b2[:, 0]; b2_h = b2[:, 3] - b2[:, 1]

        d_x = torch.abs(c2_x - c1_x) / (b1_w + eps)
        d_y = torch.abs(c2_y - c1_y) / (b1_h + eps)

        iou = torch.diag(box_ops.box_iou(b1, b2))

        # Construct spatial encoding
        f = torch.stack([
            # Relative position of box centre
            c1_x / w, c1_y / h, c2_x / w, c2_y / h,
            # Relative box width and height
            b1_w / w, b1_h / h, b2_w / w, b2_h / h,
            # Relative box area
            b1_w * b1_h / (h * w), b2_w * b2_h / (h * w),
            b2_w * b2_h / (b1_w * b1_h + eps),
            # Box aspect ratio
            b1_w / (b1_h + eps), b2_w / (b2_h + eps),
            # Intersection over union
            iou,
            # Relative distance and direction of the object w.r.t. the person
            (c2_x > c1_x).float() * d_x,
            (c2_x < c1_x).float() * d_x,
            (c2_y > c1_y).float() * d_y,
            (c2_y < c1_y).float() * d_y,
        ], 1)

        features.append(
            torch.cat([f, torch.log(f + eps)], 1)
        )
    return torch.cat(features)

def binary_focal_loss_with_logits(
    x: Tensor, y: Tensor,
    alpha: float = 0.5,
    gamma: float = 2.0,
    reduction: str = 'mean',
    eps: float = 1e-6
) -> Tensor:
    """
    Focal loss by Lin et al.
    https://arxiv.org/pdf/1708.02002.pdf

    L = - |1-y-alpha| * |y-x|^{gamma} * log(|1-y-x|)

    Parameters:
    -----------
    x: Tensor[N, K]
        Post-normalisation scores
    y: Tensor[N, K]
        Binary labels
    alpha: float
        Hyper-parameter that balances between postive and negative examples
    gamma: float
        Hyper-paramter suppresses well-classified examples
    reduction: str
        Reduction methods
    eps: float
        A small constant to avoid NaN values from 'PowBackward'

    Returns:
    --------
    loss: Tensor
        Computed loss tensor
    """
    loss = (1 - y - alpha).abs() * ((y-torch.sigmoid(x)).abs() + eps) ** gamma * \
        torch.nn.functional.binary_cross_entropy_with_logits(
            x, y, reduction='none'
        )
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError("Unsupported reduction method {}".format(reduction))
