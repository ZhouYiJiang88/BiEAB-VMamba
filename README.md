# BiEAB-VMamba
This is the official code repository for "BiEAB-VMamba: Vision Mamba with Bidirectional Scanning and Enhanced Attention Block 
for Medical Image Segmentation“

## Abstract
Medical image segmentation is a key technology for assisting clinical diagnosis and treatment.
Among existing methods, CNNs are limited by their local receptive fields, while Transformers suffer from
quadratic computational complexity. Despite the linear complexity of State Space Models (SSMs), 
the intrinsic unidirectional scanning mechanism in conventional Mamba architectures imposes a ’sequential bias’ 
that disrupts the inherent two-dimensional spatial topology and translation invariance of medical images. 
Such structural fragmentation severely limits the model’s capability to capture isotropic contextual dependencies, 
rendering it insensitive to ambiguous lesion boundaries and fine-grained morphological features. 
This paper proposes a Vision Mamba with Bidirectional Scanning and Enhanced Attention Block (BiEAB-VMamba) to address these challenges.
First, to overcome the limitations of the unidirectional scanning in SSMs, a bidirectional selective scanning mechanism is designed,
which enhances bidirectional longrange dependency modeling by processing forward and reverse scanning paths in parallel. Second, 
a lightweight Enhanced Attention Block (EAB) is introduced, employing a serial fusion strategy of channel and spatial attention 
to improve local feature perception without significantly increasing computational overhead. Experiments on the ISIC-2017, ISIC-2018, 
and PH2 datasets demonstrate the excellent performance of the BiEAB-VMamba model in medical image segmentation tasks.


## 0. Main Environments
```bash
conda create -n vmunet python=3.8
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```


## 1. Prepare the dataset

### ISIC datasets
- The ISIC17 and ISIC18 datasets, divided into a 7:3 ratio, can be found here
 [GoogleDrive](https://drive.google.com/file/d/1XM10fmAXndVLtXWOt5G0puYSQyI2veWy/view?usp=sharing)}. 

- After downloading the datasets, you are supposed to put them into
'./data/isic17/' and './data/isic18/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)

- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png
     
        
### PH2

- For the PH2, you could download them from {[Baidu](https://pan.baidu.com/s/1QGIdKRLFQDWWglXVDnuBvQ pwd=9wb6)}.

- After downloading the datasets, you are supposed to put them into './data/Synapse/', and the file format reference is as follows.

- './data/PH2/'
  - train
    - images
      - .bmp
    - masks
      - .bmp
  - val
    - images
      - .bmp
    - masks
      - .bmp

## 2. Prepare the pre_trained weights

- The weights of the pre-trained BiEAB-VMamba could be downloaded from [GoogleDrive](https://drive.google.com/drive/folders/1tZGs1YFHiDrMa-MjYY8ZoEnCyy7m7Gaj?usp=sharing). After that, the pre-trained weights should be stored in './pre/'.


## 3. Train the BiEAB-VMamba
```bash
cd BiEAB-VMamba
python train.py  # Train and test BiEAB-VMamba on the ISIC17 、 ISIC18 dataset or PH2.

```
