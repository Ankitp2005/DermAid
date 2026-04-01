import torch
import torch.nn as nn
import torchvision.models as models


class DermAidModel(nn.Module):
    """
    DermAid Model: MobileNetV3-Large backbone with three parallel heads:
    1. 7-class condition classifier
    2. 3-class severity tier classifier
    3. Confidence estimator
    """

    def __init__(self, num_classes=7, dropout_rate=0.3):
        super(DermAidModel, self).__init__()

        # Load MobileNetV3-Large pretrained on ImageNet
        self.backbone = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        )

        # Get the number of features from backbone
        # MobileNetV3-Large classifier input: 960 features
        backbone_out_features = self.backbone.classifier[0].in_features  # 960

        # REMOVE the original classifier — we replace it entirely
        self.backbone.classifier = nn.Identity()

        # ── THREE PARALLEL BRANCHES (as promised in PPT) ──────────────────

        # Branch 1: 7-class condition classifier
        self.condition_head = nn.Sequential(
            nn.Linear(backbone_out_features, 512),
            nn.Hardswish(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),  # 7 outputs
        )

        # Branch 2: 3-class severity tier
        # nv+df+bkl → Low Risk (0)
        # vasc+akiec → Refer Soon (1)
        # bcc+mel → Refer Immediately (2)
        self.severity_head = nn.Sequential(
            nn.Linear(backbone_out_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.Hardswish(),
            nn.Linear(128, 3),  # 3 severity tiers
        )

        # Branch 3: Confidence estimator (for uncertainty-aware output)
        self.confidence_head = nn.Sequential(
            nn.Linear(backbone_out_features, 128),
            nn.Hardswish(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # confidence score 0-1
        )

    def forward(self, x):
        # Extract features from MobileNetV3 backbone
        features = self.backbone(x)

        # Three parallel outputs
        condition_logits = self.condition_head(features)
        severity_logits = self.severity_head(features)
        confidence = self.confidence_head(features)

        return condition_logits, severity_logits, confidence

    def predict(self, x):
        """Full inference pipeline — returns human-readable output."""
        self.eval()
        with torch.no_grad():
            cond_logits, sev_logits, conf = self.forward(x)

            cond_probs = torch.softmax(cond_logits, dim=1)
            sev_probs = torch.softmax(sev_logits, dim=1)

            cond_class = torch.argmax(cond_probs, dim=1)
            sev_class = torch.argmax(sev_probs, dim=1)

        return {
            "condition_class": cond_class.item(),
            "condition_probs": cond_probs.squeeze().tolist(),
            "severity_class": sev_class.item(),
            "severity_probs": sev_probs.squeeze().tolist(),
            "confidence": conf.item(),
        }
