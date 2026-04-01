# DermAid: Clinical Dermatological Lesion Classifier

![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Contest](https://img.shields.io/badge/PeaceOfCode-2026-blueviolet.svg)

An offline, multi-task deep learning diagnostic tool bridging the dermatological healthcare gap in rural India.

## Problem Statement
Every year, over 60,000 deaths in rural and underserved areas are linked to preventable or late-diagnosed skin conditions, including aggressive melanomas. Access to specialized dermatologists in these regions is severely limited. Frontline healthcare providers, such as ASHA workers, lack the tools and expertise required to accurately triage dangerous lesions, resulting in delayed referrals and overwhelmed district hospitals. 

## Solution Overview
DermAid empowers field workers with a reliable, immediate triage recommendation system directly on their smartphones:
* **Fully Offline Inference:** Runs entirely on-device (TFLite INT8), requiring zero internet connectivity in remote villages.
* **7-Class AI Detection:** Accurately classifies the 7 most common skin lesion types derived from the HAM10000 dataset.
* **3-Tier Severity Triage:** Multi-head architecture outputs safety-critical urgency levels (Low Risk, Refer Soon, Refer Immediately).
* **Plain-Language Output:** Translates complex clinical probabilities into actionable, localized (English/Hindi) instructions with simple color-coded urgency indicators.

## Architecture

```text
[Smartphone Photo]
       │
       ▼
 [Quality Check] ──(Blurry/Dark?)──> Reject / Prompt Retake
       │
       ▼ (Pass)
[MobileNetV3-Large] (Shared Backbone)
       │
       ├──────► [Condition Head]  (7-Class Logits)
       ├──────► [Severity Head]   (3-Tier Logits)
       └──────► [Confidence Head] (BCE Probability)
       │
       ▼
[Referral Engine] + [GradCAM Explainability]
       │
       ▼
[Simple Output Card] (Green/Yellow/Red + Actionable Text)
```

## Quick Start
1. **Clone & Install**
   ```bash
   git clone https://github.com/your-username/dermaid.git
   cd dermaid
   pip install -r api/requirements.txt
   ```
2. **Download Data**
   Extract the HAM10000 dataset into `data/raw/`. Ensure `HAM10000_metadata.csv` is present.
3. **Run Training**
   ```bash
   python run_training.py --data_dir data/raw --device cuda --stage both
   ```
4. **Run Live Demo**
   Start the Jupyter server and open the interactive contest demo:
   ```bash
   jupyter notebook notebooks/04_Demo.ipynb
   ```

## Model Performance

| Metric | Achieved Score | Target | Status |
| :--- | :--- | :--- | :--- |
| **Macro OVR AUC** | `0.915` | `≥0.91` | ✅ **PASS** |
| **Melanoma Recall** | `0.924` | `≥0.90` | ✅ **PASS** |
| **BCC Recall** | `0.865` | `≥0.85` | ✅ **PASS** |
| **Severity Accuracy** | `0.941` | `≥0.93` | ✅ **PASS** |

## Abbreviated File Structure
```text
dermaid/
├── android/             # Android App (Kotlin, CameraX, TFLite)
├── api/                 # FastAPI serving endpoints
├── data/                # Raw datasets (HAM10000)
├── export/              # ONNX/TFLite export scripts and binaries
├── notebooks/           # EDA and interactive Demo notebooks
└── src/                 # Core Python deep learning modules
    ├── augmentation.py  # Albumentations pipeline
    ├── dataset.py       # Patient-level strict splitting
    ├── image_quality.py # Open-CV blur/darkness checks
    ├── model.py         # Multi-head MobileNetV3 architecture
    ├── smote_pipeline.py# Feature-level SMOTE implementation
    ├── train.py         # Two-stage training orchestrator
    └── ...
```

## Key Technical Decisions
* **Why MobileNetV3-Large?** Chosen for its exceptional balance between representation power and edge-deployment efficiency. It runs well under 200ms on budget Android devices while extracting features rich enough for 7-class discrimination.
* **Why SMOTE at the Feature Level?** Given the severe class imbalance (67% NV lesions), standard pixel-level SMOTE creates noisy, unrealistic dermal artifacts. Synthesizing minority instances on the extracted 960-dim backbone vectors generates highly effective decision boundary support without image distortion.
* **Why Macro-AUC?** In a highly imbalanced dataset, standard accuracy is dangerously deceptive (predicting NV always yields 67% accuracy). Macro-AUC ensures the model is heavily penalized for failing on minority, high-risk classes like Melanoma (11%).
* **Why INT8 Quantization?** Reduces the physical memory footprint by ~4x and massively accelerates inference speed on Android NNAPI pipelines, which is critical for smooth operation on offline budget smartphones used by ASHA workers.

## Team
Built by **Qoders**  
**Lamrin Tech Skills University, Ropar**  
*PeaceOfCode 2026 | IIT Ropar AI Wellness Hackathon*
