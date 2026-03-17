import importlib
import logging

import torch
import torch.nn as nn

from src.models.attentive_pooler import AttentivePooler

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class AttentiveClassifier(nn.Module):
    def __init__(
        self,
        cross_classes: dict,
        action_classes: dict,
        intersection_classes: dict,
        signalized_classes: dict,
        embed_dim: int,
        num_heads: int,
        depth: int,
        use_activation_checkpointing: bool,
    ):
        super().__init__()
        self.num_cross_classes = len(cross_classes)
        num_action_classes = len(action_classes)
        num_intersection_classes = len(intersection_classes)
        num_signalized_classes = len(signalized_classes)

        # Bounding Box Encoder, (x1,y1,x2,y2) -> same dim as encoder features
        self.bbox_encoder = nn.Linear(4, embed_dim)

        # Optional but recommended: a learned "type" embedding so bbox tokens are
        # distinguishable from video tokens.
        self.bbox_type = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pooler = AttentivePooler(
            num_queries=4,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            use_activation_checkpointing=use_activation_checkpointing,
        )

        self.cross_classifier = nn.Linear(embed_dim, self.num_cross_classes, bias=True)
        self.action_classifier = nn.Linear(embed_dim, num_action_classes, bias=True)
        self.intersection_classifier = nn.Linear(embed_dim, num_intersection_classes, bias=True)
        self.signalized_classifier = nn.Linear(embed_dim, num_signalized_classes, bias=True)

    def forward(self, x, bboxes=None):
        if torch.isnan(x).any():
            print("Nan detected at output of encoder")
            raise RuntimeError("NaN detected at output of encoder")

        if bboxes is not None:
            # bboxes: [B, T, 4]
            bbox_embed = self.bbox_encoder(bboxes)      # [B, T, D]
            bbox_embed = bbox_embed + self.bbox_type    # tag as bbox tokens
            x = torch.cat([x, bbox_embed], dim=1)       # [B, N+T, D]

        # Temporal attention pooling via learned queries
        # With num_queries=4, output is [B, 4, D]
        x = self.pooler(x)

        x_cross, x_action, x_intersection, x_signalized = (
            x[:, 0, :],
            x[:, 1, :],
            x[:, 2, :],
            x[:, 3, :],
        )

        return dict(
            cross=self.cross_classifier(x_cross),
            action=self.action_classifier(x_action),
            intersection=self.intersection_classifier(x_intersection),
            signalized=self.signalized_classifier(x_signalized),
        )


def init_module(
    module_name,
    device,
    frames_per_clip,
    frames_per_second,
    resolution,
    checkpoint,
    model_kwargs,
    wrapper_kwargs,
):
    """
    Build (frozen) model and initialize from pretrained checkpoint

    API requirements for "model" module:
      1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: (Tensor) Video clip (shape=[batch_size x num_channels x num_frames x height x width])
        :param anticipation_time: (Tensor) Seconds into the future to predict for each sample in batch
            (shape=[batch_size])
        :returns: (Tensor) Representations of future frames (shape=[batch_size x num_output_tokens x feature_dim])

      2) Needs to have a public attribute called 'embed_dim' (int) describing its
         output feature dimension.
    """
    model = (
        importlib.import_module(f"{module_name}")
        .init_module(
            frames_per_clip=frames_per_clip,
            frames_per_second=frames_per_second,
            resolution=resolution,
            checkpoint=checkpoint,
            model_kwargs=model_kwargs,
            wrapper_kwargs=wrapper_kwargs,
        )
        .to(device)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(model)
    return model


def init_classifier(
    embed_dim: int,
    num_heads: int,
    num_blocks: int,
    device: torch.device,
    num_classifiers: int,
    action_classes: dict,
    cross_classes: dict,
    intersection_classes: dict,
    signalized_classes: dict,
):
    classifiers = [
        AttentiveClassifier(
            cross_classes=cross_classes,
            action_classes=action_classes,
            intersection_classes=intersection_classes,
            signalized_classes=signalized_classes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=num_blocks,
            use_activation_checkpointing=True,
        ).to(device)
        for _ in range(num_classifiers)
    ]
    print(classifiers[0])
    return classifiers
