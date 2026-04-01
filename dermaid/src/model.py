import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

import config

class DermAidModel(nn.Module):
    def __init__(self):
        super(DermAidModel, self).__init__()
        
        # Load the MobileNetV3 Large backbone
        self.backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        
        # Replace the backbone classifier to extract 960-dim pooled features
        self.backbone.classifier = nn.Identity()
        
        # --- PARALLEL HEADS ---
        
        # Condition head (7-class classification)
        self.condition_head = nn.Sequential(
            nn.Linear(960, 512),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(256, 7)
        )
        
        # Severity head (3-class risk assessment)
        self.severity_head = nn.Sequential(
            nn.Linear(960, 256),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.Hardswish(),
            nn.Linear(128, 3)
        )
        
        # Confidence head (1 output for binary confidence scoring)
        self.confidence_head = nn.Sequential(
            nn.Linear(960, 128),
            nn.Hardswish(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        condition_logits = self.condition_head(features)
        severity_logits = self.severity_head(features)
        confidence = self.confidence_head(features)
        return condition_logits, severity_logits, confidence

    @torch.no_grad()
    def predict(self, x):
        """
        Runs inference and applies softmax to logits. Returns a dictionary of results 
        for the first image in the batch.
        """
        was_training = self.training
        self.eval()
        
        if x.dim() == 3:
            # Add batch dimension if single image tensor is passed
            x = x.unsqueeze(0)
            
        condition_logits, severity_logits, confidence = self.forward(x)
        
        # Extract for the first element in batch
        cond_probs = torch.softmax(condition_logits[0], dim=0).cpu().numpy().tolist()
        cond_class = int(torch.argmax(condition_logits[0]).item())
        
        sev_probs = torch.softmax(severity_logits[0], dim=0).cpu().numpy().tolist()
        sev_class = int(torch.argmax(severity_logits[0]).item())
        
        # For a single sigmoid output
        conf_val = float(confidence[0].item())
        
        if was_training:
            self.train()
            
        return {
            'condition_class': cond_class,
            'condition_probs': cond_probs,
            'severity_class': sev_class,
            'severity_probs': sev_probs,
            'confidence': conf_val
        }

    def freeze_backbone(self):
        """Freezes all layers of the backbone so they are not updated during training."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes all layers of the backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_feature_extractor(self):
        """Returns the modified backbone for feature extraction purposes (e.g. SMOTE)."""
        return self.backbone
