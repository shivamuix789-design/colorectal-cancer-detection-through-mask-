Polyp SAM 2 — Zero-Shot Polyp Segmentation for Colorectal Cancer Detection
![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange) ![SAM2](https://img.shields.io/badge/SAM-2.1-green) ![License](https://img.shields.io/badge/License-MIT-yellow)
A zero-shot polyp segmentation pipeline built on top of Meta AI's Segment Anything Model 2.1 (SAM 2.1). This project evaluates and applies SAM 2.1 to colonoscopy images for early colorectal cancer detection — achieving mDice 0.8882 and mIoU 0.8302 on a 1612-image dataset without any fine-tuning.
---
Results
Method	mDice	mIoU
Ours (SAM 2.1, zero-shot)	0.8882	0.8302
SAM 2 (paper, BBox prompt)	0.939	0.885
UNet baseline	0.818	0.746
SFA	0.700	0.607
> Zero-shot means no training or fine-tuning on polyp data — the model generalises directly from its pretrained weights.
---
How It Works
SAM 2.1 segments polyps using a bounding box prompt automatically extracted from the ground truth mask. The pipeline:
Loads a colonoscopy image
Extracts a tight bounding box from the GT mask
Feeds the image + box into SAM 2.1
SAM 2.1 predicts the polyp segmentation mask
Dice and IoU scores are computed against the GT mask
---
Architecture
SAM 2.1 consists of four main components:
Hiera Image Encoder — Hierarchical Vision Transformer that extracts multi-scale features from the colonoscopy image
Prompt Encoder — Encodes bounding box or point prompts into embeddings
Mask Decoder — Two-way transformer that fuses image features and prompt embeddings to predict the segmentation mask
Memory Module — Stores past frame context for video segmentation (used in video colonoscopy)
---
Dataset
This project was evaluated on a combined dataset of:
Kvasir-SEG — 1000 polyp images from the Vestre Viken Health Trust, Norway
CVC-ClinicDB — 612 images from colonoscopy examination videos
Custom collected data — Additional colonoscopy images
Total: 1612 images with corresponding ground truth masks.
---
Requirements
```
torch>=2.1.0
torchvision>=0.16.0
opencv-python-headless>=4.8.0
albumentations>=1.3.0
numpy
matplotlib
tqdm
scikit-learn
Pillow<12.0
```
---
Installation
Step 1 — Clone this repo and SAM 2:
```bash
git clone https://github.com/yourusername/polyp-sam2.git
cd polyp-sam2

git clone https://github.com/facebookresearch/segment-anything-2.git
pip install -e segment-anything-2
```
Step 2 — Install dependencies:
```bash
pip install opencv-python-headless albumentations tqdm scikit-learn "Pillow<12.0"
```
Step 3 — Download SAM 2.1 weights:
```bash
mkdir checkpoints
wget -O checkpoints/sam2.1_hiera_base_plus.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```
---
Usage
Load the model
```python
import sys, torch
sys.path.insert(0, './segment-anything-2')

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

sam2_model = build_sam2(
    'configs/sam2.1/sam2.1_hiera_b+.yaml',
    'checkpoints/sam2.1_hiera_base_plus.pt',
    device=DEVICE
)
predictor = SAM2ImagePredictor(sam2_model)
```
Run prediction on a single image
```python
import cv2, numpy as np

image   = cv2.imread('image.jpg')
image   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gt_mask = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)
gt_mask = (gt_mask > 127).astype(np.uint8)

# Extract bounding box from GT mask
rows = np.any(gt_mask, axis=1)
cols = np.any(gt_mask, axis=0)
y1, y2 = np.where(rows)[0][[0, -1]]
x1, x2 = np.where(cols)[0][[0, -1]]
box = np.array([[x1, y1, x2, y2]])

# Predict
predictor.set_image(image)
masks, scores, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=box,
    multimask_output=True
)

best_mask = masks[np.argmax(scores)].astype(bool)
```
Evaluate on full dataset
```python
import os
from tqdm import tqdm

image_folder = 'data/images'
mask_folder  = 'data/masks'

images = sorted(os.listdir(image_folder))
masks  = sorted(os.listdir(mask_folder))

dice_scores, iou_scores = [], []

for img_name, mask_name in tqdm(zip(images, masks), total=len(images)):
    image   = cv2.imread(os.path.join(image_folder, img_name))
    image   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt_mask = cv2.imread(os.path.join(mask_folder, mask_name), cv2.IMREAD_GRAYSCALE)
    gt_mask = (gt_mask > 127).astype(np.uint8)

    if gt_mask.sum() == 0:
        continue

    rows = np.any(gt_mask, axis=1)
    cols = np.any(gt_mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    box = np.array([[x1, y1, x2, y2]])

    predictor.set_image(image)
    masks_pred, scores, _ = predictor.predict(
        point_coords=None, point_labels=None,
        box=box, multimask_output=True
    )
    best_mask = masks_pred[np.argmax(scores)].astype(np.uint8)

    inter = (best_mask * gt_mask).sum()
    dice  = (2 * inter) / (best_mask.sum() + gt_mask.sum() + 1e-6)
    union = (best_mask + gt_mask).clip(0, 1).sum()
    iou   = inter / (union + 1e-6)

    dice_scores.append(dice)
    iou_scores.append(iou)

print(f"mDice : {np.mean(dice_scores):.4f}")
print(f"mIoU  : {np.mean(iou_scores):.4f}")
```
---
Project Structure
```
polyp-sam2/
├── checkpoints/
│   └── sam2.1_hiera_base_plus.pt
├── data/
│   ├── images/
│   └── masks/
├── segment-anything-2/       # SAM 2 cloned from Meta AI
├── predict.py                # Single image prediction
├── evaluate.py               # Full dataset evaluation
├── visualize.py              # Overlay visualization
└── README.md
```
---
Google Colab
A ready-to-run Colab notebook is available — no local setup needed.
> **Runtime:** T4 GPU (free Colab) is sufficient  
> **Time:** ~27 minutes to evaluate 1612 images
Key Colab notes:
Re-run the install cell every session (Colab resets on disconnect)
Save checkpoints to Google Drive, not `/content/`
Use `Pillow<12.0` to avoid import errors with SAM 2
---
Evaluation Metrics
mDice (Mean Dice Score) — measures overlap between predicted and ground truth mask. Range 0–1, higher is better.
mIoU (Mean Intersection over Union) — measures the ratio of intersection to union of predicted and GT mask. Range 0–1, higher is better.
---
References
SAM 2: Segment Anything in Images and Videos — Ravi et al., Meta AI Research, 2024
Kvasir-SEG Dataset — Jha et al., 2020
CVC-ClinicDB Dataset — Bernal et al., 2015
Polyp SAM 2 Paper — Mansoori et al., 2024
---
License
This project is licensed under the MIT License. SAM 2 is licensed under Apache 2.0 by Meta AI Research.
