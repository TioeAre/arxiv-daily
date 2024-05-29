<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#EffoVPR:-Effective-Foundation-Model-Utilization-for-Visual-Place-Recognition>EffoVPR: Effective Foundation Model Utilization for Visual Place Recognition</a></li>
        <li><a href=#AdapNet:-Adaptive-Noise-Based-Network-for-Low-Quality-Image-Retrieval>AdapNet: Adaptive Noise-Based Network for Low-Quality Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Self-supervised-Pre-training-for-Transferable-Multi-modal-Perception>Self-supervised Pre-training for Transferable Multi-modal Perception</a></li>
        <li><a href=#A-Refined-3D-Gaussian-Representation-for-High-Quality-Dynamic-Scene-Reconstruction>A Refined 3D Gaussian Representation for High-Quality Dynamic Scene Reconstruction</a></li>
        <li><a href=#HFGS:-4D-Gaussian-Splatting-with-Emphasis-on-Spatial-and-Temporal-High-Frequency-Components-for-Endoscopic-Scene-Reconstruction>HFGS: 4D Gaussian Splatting with Emphasis on Spatial and Temporal High-Frequency Components for Endoscopic Scene Reconstruction</a></li>
        <li><a href=#Mani-GS:-Gaussian-Splatting-Manipulation-with-Triangular-Mesh>Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [EffoVPR: Effective Foundation Model Utilization for Visual Place Recognition](http://arxiv.org/abs/2405.18065)  
Issar Tzachor, Boaz Lerner, Matan Levy, Michael Green, Tal Berkovitz Shalev, Gavriel Habib, Dvir Samuel, Noam Korngut Zailer, Or Shimshi, Nir Darshan, Rami Ben-Ari  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The task of Visual Place Recognition (VPR) is to predict the location of a query image from a database of geo-tagged images. Recent studies in VPR have highlighted the significant advantage of employing pre-trained foundation models like DINOv2 for the VPR task. However, these models are often deemed inadequate for VPR without further fine-tuning on task-specific data. In this paper, we propose a simple yet powerful approach to better exploit the potential of a foundation model for VPR. We first demonstrate that features extracted from self-attention layers can serve as a powerful re-ranker for VPR. Utilizing these features in a zero-shot manner, our method surpasses previous zero-shot methods and achieves competitive results compared to supervised methods across multiple datasets. Subsequently, we demonstrate that a single-stage method leveraging internal ViT layers for pooling can generate global features that achieve state-of-the-art results, even when reduced to a dimensionality as low as 128D. Nevertheless, incorporating our local foundation features for re-ranking, expands this gap. Our approach further demonstrates remarkable robustness and generalization, achieving state-of-the-art results, with a significant gap, in challenging scenarios, involving occlusion, day-night variations, and seasonal changes.  
  </ol>  
</details>  
  
### [AdapNet: Adaptive Noise-Based Network for Low-Quality Image Retrieval](http://arxiv.org/abs/2405.17718)  
Sihe Zhang, Qingdong He, Jinlong Peng, Yuxi Li, Zhengkai Jiang, Jiafu Wu, Mingmin Chi, Yabiao Wang, Chengjie Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image retrieval aims to identify visually similar images within a database using a given query image. Traditional methods typically employ both global and local features extracted from images for matching, and may also apply re-ranking techniques to enhance accuracy. However, these methods often fail to account for the noise present in query images, which can stem from natural or human-induced factors, thereby negatively impacting retrieval performance. To mitigate this issue, we introduce a novel setting for low-quality image retrieval, and propose an Adaptive Noise-Based Network (AdapNet) to learn robust abstract representations. Specifically, we devise a quality compensation block trained to compensate for various low-quality factors in input images. Besides, we introduce an innovative adaptive noise-based loss function, which dynamically adjusts its focus on the gradient in accordance with image quality, thereby augmenting the learning of unknown noisy samples during training and enhancing intra-class compactness. To assess the performance, we construct two datasets with low-quality queries, which is built by applying various types of noise on clean query images on the standard Revisited Oxford and Revisited Paris datasets. Comprehensive experimental results illustrate that AdapNet surpasses state-of-the-art methods on the Noise Revisited Oxford and Noise Revisited Paris benchmarks, while maintaining competitive performance on high-quality datasets. The code and constructed datasets will be made available.  
  </ol>  
</details>  
  
  



## NeRF  

### [Self-supervised Pre-training for Transferable Multi-modal Perception](http://arxiv.org/abs/2405.17942)  
Xiaohao Xu, Tianyi Zhang, Jinrong Yang, Matthew Johnson-Roberson, Xiaonan Huang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In autonomous driving, multi-modal perception models leveraging inputs from multiple sensors exhibit strong robustness in degraded environments. However, these models face challenges in efficiently and effectively transferring learned representations across different modalities and tasks. This paper presents NeRF-Supervised Masked Auto Encoder (NS-MAE), a self-supervised pre-training paradigm for transferable multi-modal representation learning. NS-MAE is designed to provide pre-trained model initializations for efficient and high-performance fine-tuning. Our approach uses masked multi-modal reconstruction in neural radiance fields (NeRF), training the model to reconstruct missing or corrupted input data across multiple modalities. Specifically, multi-modal embeddings are extracted from corrupted LiDAR point clouds and images, conditioned on specific view directions and locations. These embeddings are then rendered into projected multi-modal feature maps using neural rendering techniques. The original multi-modal signals serve as reconstruction targets for the rendered feature maps, facilitating self-supervised representation learning. Extensive experiments demonstrate the promising transferability of NS-MAE representations across diverse multi-modal and single-modal perception models. This transferability is evaluated on various 3D perception downstream tasks, such as 3D object detection and BEV map segmentation, using different amounts of fine-tuning labeled data. Our code will be released to support the community.  
  </ol>  
</details>  
**comments**: 8 pages. arXiv admin note: substantial text overlap with
  arXiv:2311.13750  
  
### [A Refined 3D Gaussian Representation for High-Quality Dynamic Scene Reconstruction](http://arxiv.org/abs/2405.17891)  
Bin Zhang, Bi Zeng, Zexin Peng  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In recent years, Neural Radiance Fields (NeRF) has revolutionized three-dimensional (3D) reconstruction with its implicit representation. Building upon NeRF, 3D Gaussian Splatting (3D-GS) has departed from the implicit representation of neural networks and instead directly represents scenes as point clouds with Gaussian-shaped distributions. While this shift has notably elevated the rendering quality and speed of radiance fields but inevitably led to a significant increase in memory usage. Additionally, effectively rendering dynamic scenes in 3D-GS has emerged as a pressing challenge. To address these concerns, this paper purposes a refined 3D Gaussian representation for high-quality dynamic scene reconstruction. Firstly, we use a deformable multi-layer perceptron (MLP) network to capture the dynamic offset of Gaussian points and express the color features of points through hash encoding and a tiny MLP to reduce storage requirements. Subsequently, we introduce a learnable denoising mask coupled with denoising loss to eliminate noise points from the scene, thereby further compressing 3D Gaussian model. Finally, motion noise of points is mitigated through static constraints and motion consistency constraints. Experimental results demonstrate that our method surpasses existing approaches in rendering quality and speed, while significantly reducing the memory usage associated with 3D-GS, making it highly suitable for various tasks such as novel view synthesis, and dynamic mapping.  
  </ol>  
</details>  
  
### [HFGS: 4D Gaussian Splatting with Emphasis on Spatial and Temporal High-Frequency Components for Endoscopic Scene Reconstruction](http://arxiv.org/abs/2405.17872)  
Haoyu Zhao, Xingyue Zhao, Lingting Zhu, Weixi Zheng, Yongchao Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Robot-assisted minimally invasive surgery benefits from enhancing dynamic scene reconstruction, as it improves surgical outcomes. While Neural Radiance Fields (NeRF) have been effective in scene reconstruction, their slow inference speeds and lengthy training durations limit their applicability. To overcome these limitations, 3D Gaussian Splatting (3D-GS) based methods have emerged as a recent trend, offering rapid inference capabilities and superior 3D quality. However, these methods still struggle with under-reconstruction in both static and dynamic scenes. In this paper, we propose HFGS, a novel approach for deformable endoscopic reconstruction that addresses these challenges from spatial and temporal frequency perspectives. Our approach incorporates deformation fields to better handle dynamic scenes and introduces Spatial High-Frequency Emphasis Reconstruction (SHF) to minimize discrepancies in spatial frequency spectra between the rendered image and its ground truth. Additionally, we introduce Temporal High-Frequency Emphasis Reconstruction (THF) to enhance dynamic awareness in neural rendering by leveraging flow priors, focusing optimization on motion-intensive parts. Extensive experiments on two widely used benchmarks demonstrate that HFGS achieves superior rendering quality. Our code will be available.  
  </ol>  
</details>  
**comments**: 13 pages, 4 figures  
  
### [Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh](http://arxiv.org/abs/2405.17811)  
Xiangjun Gao, Xiaoyu Li, Yiyu Zhuang, Qi Zhang, Wenbo Hu, Chaopeng Zhang, Yao Yao, Ying Shan, Long Quan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural 3D representations such as Neural Radiance Fields (NeRF), excel at producing photo-realistic rendering results but lack the flexibility for manipulation and editing which is crucial for content creation. Previous works have attempted to address this issue by deforming a NeRF in canonical space or manipulating the radiance field based on an explicit mesh. However, manipulating NeRF is not highly controllable and requires a long training and inference time. With the emergence of 3D Gaussian Splatting (3DGS), extremely high-fidelity novel view synthesis can be achieved using an explicit point-based 3D representation with much faster training and rendering speed. However, there is still a lack of effective means to manipulate 3DGS freely while maintaining rendering quality. In this work, we aim to tackle the challenge of achieving manipulable photo-realistic rendering. We propose to utilize a triangular mesh to manipulate 3DGS directly with self-adaptation. This approach reduces the need to design various algorithms for different types of Gaussian manipulation. By utilizing a triangle shape-aware Gaussian binding and adapting method, we can achieve 3DGS manipulation and preserve high-fidelity rendering after manipulation. Our approach is capable of handling large deformations, local manipulations, and soft body simulations while keeping high-quality rendering. Furthermore, we demonstrate that our method is also effective with inaccurate meshes extracted from 3DGS. Experiments conducted demonstrate the effectiveness of our method and its superiority over baseline approaches.  
  </ol>  
</details>  
**comments**: Project page here: https://gaoxiangjun.github.io/mani_gs/  
  
  



