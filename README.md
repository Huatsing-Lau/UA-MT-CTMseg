<!-- #region -->
# UA-MT-CTMseg: Semi-supervised multi-class segmentation of CTM image utilized UA-MT network


### Introduction

This repository is a modified version of the MICCAI 2019 paper '[Uncertainty-aware Self-ensembling Model for Semi-supervised 3D Left Atrium Segmentation](https://arxiv.org/abs/1907.07034)'. 

by [Lequan Yu](http://yulequan.github.io), [Shujun Wang](https://emmaw8.github.io/), [Xiaomeng Li](https://xmengli999.github.io/), [Chi-Wing Fu](http://www.cse.cuhk.edu.hk/~cwfu/), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/). 
<!-- #endregion -->

<!-- #region -->
### Installation
This repository is based on PyTorch 0.4.1.

### Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/Huatsing-Lau/UA-MT-CTMseg
   cd UA-MT-CTMseg
   ```
2. Put the data in `data`.
   
3. Train the model:
 
   ```shell
   cd code
   python train_LA_meanteacher_certainty_unlabel.py --gpu 0
   ```

## Citation

If UA-MT-CTMseg is useful for your research, please consider citing:

    

### Questions

Please contact 'liuhuaqing@zju.edu.cn'

<!-- #endregion -->
