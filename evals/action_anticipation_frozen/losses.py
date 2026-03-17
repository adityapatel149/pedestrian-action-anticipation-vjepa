import torch.nn.functional as F
import torch

def softmax_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 1.0,
    alpha: torch.Tensor | None = None,   # shape [C], e.g. tensor([0.25, 0.75])
    reduction: str = "mean",
):
    """
    Softmax focal loss for multi-class classification.

    inputs:  [B, C] logits
    targets: [B] int64 class indices in [0, C-1]
    alpha:   optional per-class weights tensor of shape [C]
    """
    ce = F.cross_entropy(inputs, targets, reduction="none")  # [B]
    pt = torch.exp(-ce)  # [B], pt = p(true_class)
    loss = (1.0 - pt).pow(gamma) * ce  # [B]

    if alpha is not None:
        alpha = alpha.to(device=loss.device, dtype=loss.dtype)
        loss = loss * alpha[targets]  # [B]

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")




def sigmoid_focal_loss(
    inputs,
    targets,
    alpha=0.25,
    gamma=2.0,
    reduction="sum",
    detach=False,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    :param Tensor inputs: Prediction logits for each sample [B x K]
    :param Tensor targets: Class label for each sample [B] (long tensor)
    :param float alpha: Weight in range (0,1) to balance pos vs neg samples.
    :param float gamma: Exponent of modulating factor (1-p_t) to balance easy vs hard samples.
    :param str reduction: 'mean' | 'sum'
    """
    B, K = inputs.size()  # [batch_size, class logits]

    # convert to one-hot targets
    targets = F.one_hot(targets, K).float()  # [B, K]

    p = F.sigmoid(inputs)

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    if detach:
        loss = loss.detach()

    return loss

