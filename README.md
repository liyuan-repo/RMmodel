# A Robust Multisource Remote Sensing Image Matching Method Utilizing Attention and Feature Enhancement Against Noise Interference




## Abstract 
Image matching is a fundamental and critical task of multisource remote sensing image applications. However, remote sensing images are susceptible to various noises. Accordingly, how to effectively achieve accurate matching in noise images is a challenging problem. To solve this issue, we propose a robust multisource remote sensing image matching method utilizing attention and feature enhancement against noise interference. In the first stage, we combine deep convolution with the attention mechanism of transformer to perform dense feature extraction, constructing feature descriptors with higher discriminability and robustness. Subsequently, we employ a coarse-to-fine matching strategy to achieve dense matches. In the second stage, we introduce an outlier removal network based on a binary classification mechanism, which can establish effective and geometrically consistent correspondences between images; through weighting for each correspondence, inliers vs. outliers classification are performed, as well as removing outliers from dense matches. Ultimately, we can accomplish more efficient and accurate matches. To validate the performance of the proposed method, we conduct experiments using multisource remote sensing image datasets for comparison with other state-of-the-art methods under different scenarios, including noise-free, additive random noise, and periodic stripe noise. Comparative results indicate that the proposed method has a more well-balanced performance and robustness. The proposed method contributes a valuable reference for solving the difficult problem of noise image matching. Our code will be released at https://github.com/liyuan-repo/RMmodel.


## Installation 

conda env create -f environment.yaml

conda activate RMmodel   

## Download the datasets

### 1) Training Dataset:
#### MegaDepth and the dataset indices   

- https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz
- https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf (the dataset indices)

After the download is complete, unzip the MegaDepth dataset in the /path/to/RMmodel/data/megadepth/train/phoenix/S6/zl548/MegaDepth_v1; 
and unzip the dataset indices in the /path/to/RMmodel/data/megadepth/index.



### 2) Testing Dataset:
- https://github.com/liyuan-repo/Testing-Dataset


## Data Preparation

```cd /data```

```python process_megadepth.py```


## Training
```cd /scripts/reproduce_train```

``` shell trainer.sh ```

      
## Acknowledgement
Our work are based on [LoFTR](https://github.com/zju3dv/LoFTR) and [Order-Aware Network](https://github.com/zjhthu/OANet), and we use their code. If you use the part of code related to data generation, testing and evaluation, you should cite these paper and follow their license.


## Citation
If you find this project useful, please cite:
```
@article{,
  title={A Robust Multisource Remote Sensing Image Matching Method Utilizing Attention and Feature Enhancement Against Noise Interference},
  author={Yuan Li, Dapeng Wu, Yaping Cui, Peng He, Yuan Zhang and Ruyan Wang},
  journal={Arxiv},
  year={2024}
}
```
