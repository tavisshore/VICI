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

## 🧬 Feature Extractors
|  Backbone  | Params (M) | FLOPs (G) | Dims |  R@1  |  R@5  |  R@10 |
|:----------:|:----------:|:---------:|:----:|:-----:|:-----:|:-----:|
| ConvNeXt-T |     28     |    4.5    |  768 |  1.36 |  4.34 |  7.95 |
| ConvNeXt-B |     89     |    15.4   | 1024 |  3.14 |  8.14 | 13.22 |
|    ViT-B   |     86     |    17.6   |  768 |  3.30 |  8.92 | 13.96 |
|    ViT-L   |     307    |    60.6   | 1024 |  9.62 | 23.42 | 32.73 |
|  DINOv2-B  |     86     |    152    |  768 | 17.37 | 36.14 | 46.96 |
|  DINOv2-L  |     304    |    507    | 1024 | 27.49 | 51.96 | 63.13 |

## 🧰 Vision-Language Models

## 📊 Evaluation
### 🐍 Environment Setup
```bash
conda env create -f requirements.yaml && conda activate spagbol
```

### 🐍 Stage 1 - Image Retrieval


### 🐍 Stage 2 - VLM Re-ranking




