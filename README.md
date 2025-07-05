<div align="center">     

# VICI: VLM-Instructed Cross-view Image-localisation 
<p align="middle">
 <a href="https://zxh009123.github.io/">Xiaohan Zhang*</a>
 <a href="https://tavisshore.co.uk/">Tavis Shore*</a>
 <a href="">Chen Chen</a> <br>
 <a href="https://cvssp.org/Personal/OscarMendez/index.html">Oscar Mendez</a>
 <a href="https://personalpages.surrey.ac.uk/s.hadfield/biography.html">Simon Hadfield</a>
 <a href="https://www.uvm.edu/cems/cs/profile/safwan-wshah">Safwan Wshah</a>
</p>
<p align="middle">
 <a href="https://www.wshahaigroup.com/">Vermont Artificial Intelligence Laboratory (VaiL)</a>
 <a href="https://www.surrey.ac.uk/centre-vision-speech-signal-processing">Centre for Vision, Speech, and Signal Processing (CVSSP)</a> <br>
 <a href="https://www.ucf.edu/">University of Central Florida</a>
 <a href="https://locusrobotics.com/">Locus Robotics</a>
</p>

![vici_diagram](https://github.com/user-attachments/assets/3fe71cbe-2850-4c00-88ac-60b60317bab5)

</div>

## 📓 Description 

### 🧬 Feature Extractors
|  Backbone  | Params (M) | FLOPs (G) | Dims |  R@1  |  R@5  |  R@10 |
|:----------:|:----------:|:---------:|:----:|:-----:|:-----:|:-----:|
| ConvNeXt-T |     28     |    4.5    |  768 |  1.36 |  4.34 |  7.95 |
| ConvNeXt-B |     89     |    15.4   | 1024 |  3.14 |  8.14 | 13.22 |
|    ViT-B   |     86     |    17.6   |  768 |  3.30 |  8.92 | 13.96 |
|    ViT-L   |     307    |    60.6   | 1024 |  9.62 | 23.42 | 32.73 |
|  DINOv2-B  |     86     |    152    |  768 | 17.37 | 36.14 | 46.96 |
|  DINOv2-L  |     304    |    507    | 1024 | 27.49 | 51.96 | 63.13 |

### 🧰 Vision-Language Models
|          VLM          | R@1   | R@5   | R@10  |
|:---------------------:|-------|-------|-------|
|   Without Re-ranking  | 27.49 | 51.96 | 63.13 |
| Gemini 2.5 Flash Lite | 23.54 | 48.39 | 63.13 |
|    Gemini 2.5 Flash   | 30.21 | 53.04 | 63.13 |

### 🛸 Drone Augmentation
| $P$ |  R@1  |  R@5  |  R@10 |
|:---:|:-----:|:-----:|:-----:|
|  0  | 24.47 | 48.16 | 60.99 |
| 0.1 | 26.98 | 51.34 | 61.92 |
| 0.3 | 27.49 | 51.96 | 63.13 |
| 0.5 | 24.89 | 52.03 | 62.66 |

### 🎯 Ablation study and baseline comparison.
|           Configuration           | R@1   | R@5   | R@10  |
|:---------------------------------:|-------|-------|-------|
|  U1652~\cite{zheng2020university} | 1.20  | -     | -     |
| LPN w/o drone~\cite{wang2021each} | 0.74  | -     | -     |
|  LPN w/ drone~\cite{wang2021each} | 0.81  | -     | -     |
|              DINOv2-L             | 24.66 | 48.00 | 59.02 |
|            + Drone Data           | 27.49 | 51.96 | 63.13 |
|        + VLM Re-rank (Ours)       | 30.21 | 53.04 | 63.13 |

## 📊 Evaluation
### 🐍 Environment Setup
```bash
conda env create -n ENV -f requirements.yaml && conda activate ENV
```

### 🐍 Stage 1 - Image Retrieval


### 🐍 Stage 2 - VLM Re-ranking


## 📗 Related Works
### 🕺 [PEnG: Pose-Enhanced Geo-Localisation ](https://github.com/tavisshore/PEnG)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![arxiv](https://img.shields.io/badge/cs.LG-2411.15742-b31b1b?style=flat&logo=arxiv&logoColor=red)](https://arxiv.org/abs/2411.15742)
[![Conference](http://img.shields.io/badge/RA--L-2025-4b44ce.svg)]()
[![Project Page](http://img.shields.io/badge/Project-Page-green)](https://tavisshore.co.uk/peng/)
[![GitHub](https://img.shields.io/badge/GitHub-PEnG-%23121011.svg?logo=github&logoColor=white)](https://github.com/tavisshore/peng)

### 🕺 [GeoDTR+: Toward generic cross-view geolocalization via geometric disentanglement](https://github.com/tavisshore/PEnG)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![arxiv](https://img.shields.io/badge/cs.LG-2308.09624-b31b1b?style=flat&logo=arxiv&logoColor=red)](https://arxiv.org/pdf/2308.09624)
[![Conference](http://img.shields.io/badge/TPAMI-2023-4b44ce.svg)]()
[![Project Page](http://img.shields.io/badge/Project-Page-green)](https://zxh009123.github.io/publication/zhang-geodtr-2023/)
[![GitHub](https://img.shields.io/badge/GitLab-GeoDTR+-%23121011.svg?logo=gitlab)](https://gitlab.com/vail-uvm/geodtr_plus)




## ⭐ Star History
 <a href="https://star-history.com/#tavisshore/VICI&Date" align="middle">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=tavisshore/VICI&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=tavisshore/VICI&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=tavisshore/VICI&type=Date" />
 </picture>
</a>


