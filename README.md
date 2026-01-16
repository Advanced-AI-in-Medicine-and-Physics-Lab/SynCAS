

# SynCAS: Synthetic-data-driven Coronary Artery Segmentation

Official implementation of the paper:
**"Coronary Artery Segmentation in Non-Contrast Cardiac CT using Synthetic Data-driven Anatomy-informed Contrastive Learning"**

|**[Paper Link](https://iopscience.iop.org/article/10.1088/1361-6560/ae387c)**|
**[Model Weights](https://drive.google.com/drive/folders/13FleJ8FCO_gZtZ-qPzGlqJHhB-HlO_u9?usp=drive_link)**|

---

##  Project Structure

```
SynCAS/
├── dataloading/           # Data loading utilities
├── loss/                  # Custom loss functions
├── models/                # Model architectures
├── config.json            # Training config file
├── data_engine.m          # MATLAB-based synthetic data generator
├── inference.py           # Inference script (includes normalization)
├── predictor.py           # Model prediction wrapper
├── sampling.py            # Sampling and contrastive learning utils
├── training.py            # Model training script
├── LICENSE
└── README.md              # This file
```

---

##  Installation

```bash
git clone https://github.com/JinkuiH/SynCAS
cd SynCAS
conda create -n syncas python=3.9
conda activate syncas
pip install -r requirements.txt
```


---

##  Inference

1. **Download Pretrained Weights**
   Download both versions (with and without fine-tuning) from the [Google drive](https://drive.google.com/drive/folders/13FleJ8FCO_gZtZ-qPzGlqJHhB-HlO_u9?usp=drive_link), and place them in the `weights/` directory.

2. **Prepare Your Data**
   Ensure your NCCT scan is in `.nii.gz` format.

3. **Run Inference**
   Modify paths in `inference.py` as needed:

   ```bash
   python inference.py
   ```

    **Note:** Custom NCCT normalization is handled in `inference.py` (around **line 105**).

---

##  Synthetic Dataset Generation

Our synthetic dataset pipeline consists of three main steps:

1. **Download [ImageCAS](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT) (CCTA) Dataset**
   This acts as anatomical templates.

2. **Extract Anatomical Structures**
   Use [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) on ImageCAS to segment structures (e.g., myocardium, aorta, heart).

3. **Generate Synthetic NCCT Volumes**
   Run the MATLAB script:

   ```matlab
   data_engine.m
   ```

   Each execution produces one synthetic volume per template. You can run multiple times to expand the dataset.

---

## Training

1. **Preprocess Data**
   We follow the nnU-Net data preprocess pipeline and format, but use **custom normalization**, as seen in `inference.py (line 105)`.

2. **Set Configurations**
   Open `config.json` and set:

   * Training data path
   * Output directory
   * Training hyperparameter etc.

3. **Run Training**

   ```bash
   python training.py
   ```

---

##  Implementation Highlights
* **Synthetic Data Engine**

   See `data_engine.m` for details on synthesizing NCCT volumes from anatomical templates.

* **Anatomy-informed Contrastive Learning**

  See `contrastive_loss_branchAware` in `training.py` for how voxel-level pseudo-negatives are generated using anatomical priors.
---

##  Citation

If this work helps your research, please cite:

```bibtex
@article{SynCAS2026,
	author={Hao, Jinkui and He, Xiaoyi and Durak, Gorkem and Aktas, Halil Ertugrul and Bagci, Ulas and Shah, Nilay S and Zhou, Bo},
	title={Coronary artery segmentation in non-contrast cardiac CT using anatomy-informed contrastive learning and synthetic data},
	journal={Physics in Medicine & Biology},
	year={2026},
}
```

---

##  Resources
* [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
* [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
* [ImageCAS Dataset](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT)

---
