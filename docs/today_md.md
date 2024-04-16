<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Increasing-SLAM-Pose-Accuracy-by-Ground-to-Satellite-Image-Registration>Increasing SLAM Pose Accuracy by Ground-to-Satellite Image Registration</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#LetsGo:-Large-Scale-Garage-Modeling-and-Rendering-via-LiDAR-Assisted-Gaussian-Primitives>LetsGo: Large-Scale Garage Modeling and Rendering via LiDAR-Assisted Gaussian Primitives</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#CREST:-Cross-modal-Resonance-through-Evidential-Deep-Learning-for-Enhanced-Zero-Shot-Learning>CREST: Cross-modal Resonance through Evidential Deep Learning for Enhanced Zero-Shot Learning</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#XoFTR:-Cross-modal-Feature-Matching-Transformer>XoFTR: Cross-modal Feature Matching Transformer</a></li>
        <li><a href=#DeDoDe-v2:-Analyzing-and-Improving-the-DeDoDe-Keypoint-Detector>DeDoDe v2: Analyzing and Improving the DeDoDe Keypoint Detector</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#DeferredGS:-Decoupled-and-Editable-Gaussian-Splatting-with-Deferred-Shading>DeferredGS: Decoupled and Editable Gaussian Splatting with Deferred Shading</a></li>
        <li><a href=#VRS-NeRF:-Visual-Relocalization-with-Sparse-Neural-Radiance-Field>VRS-NeRF: Visual Relocalization with Sparse Neural Radiance Field</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Increasing SLAM Pose Accuracy by Ground-to-Satellite Image Registration](http://arxiv.org/abs/2404.09169)  
Yanhao Zhang, Yujiao Shi, Shan Wang, Ankit Vora, Akhil Perincherry, Yongbo Chen, Hongdong Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Vision-based localization for autonomous driving has been of great interest among researchers. When a pre-built 3D map is not available, the techniques of visual simultaneous localization and mapping (SLAM) are typically adopted. Due to error accumulation, visual SLAM (vSLAM) usually suffers from long-term drift. This paper proposes a framework to increase the localization accuracy by fusing the vSLAM with a deep-learning-based ground-to-satellite (G2S) image registration method. In this framework, a coarse (spatial correlation bound check) to fine (visual odometry consistency check) method is designed to select the valid G2S prediction. The selected prediction is then fused with the SLAM measurement by solving a scaled pose graph problem. To further increase the localization accuracy, we provide an iterative trajectory fusion pipeline. The proposed framework is evaluated on two well-known autonomous driving datasets, and the results demonstrate the accuracy and robustness in terms of vehicle localization.  
  </ol>  
</details>  
**comments**: 7 pages, 6 figures, to be published in 2024 International Conference
  on Robotics and Automation (ICRA)  
  
  



## SFM  

### [LetsGo: Large-Scale Garage Modeling and Rendering via LiDAR-Assisted Gaussian Primitives](http://arxiv.org/abs/2404.09748)  
Jiadi Cui, Junming Cao, Yuhui Zhong, Liao Wang, Fuqiang Zhao, Penghao Wang, Yifan Chen, Zhipeng He, Lan Xu, Yujiao Shi, Yingliang Zhang, Jingyi Yu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Large garages are ubiquitous yet intricate scenes in our daily lives, posing challenges characterized by monotonous colors, repetitive patterns, reflective surfaces, and transparent vehicle glass. Conventional Structure from Motion (SfM) methods for camera pose estimation and 3D reconstruction fail in these environments due to poor correspondence construction. To address these challenges, this paper introduces LetsGo, a LiDAR-assisted Gaussian splatting approach for large-scale garage modeling and rendering. We develop a handheld scanner, Polar, equipped with IMU, LiDAR, and a fisheye camera, to facilitate accurate LiDAR and image data scanning. With this Polar device, we present a GarageWorld dataset consisting of five expansive garage scenes with diverse geometric structures and will release the dataset to the community for further research. We demonstrate that the collected LiDAR point cloud by the Polar device enhances a suite of 3D Gaussian splatting algorithms for garage scene modeling and rendering. We also propose a novel depth regularizer for 3D Gaussian splatting algorithm training, effectively eliminating floating artifacts in rendered images, and a lightweight Level of Detail (LOD) Gaussian renderer for real-time viewing on web-based devices. Additionally, we explore a hybrid representation that combines the advantages of traditional mesh in depicting simple geometry and colors (e.g., walls and the ground) with modern 3D Gaussian representations capturing complex details and high-frequency textures. This strategy achieves an optimal balance between memory performance and rendering quality. Experimental results on our dataset, along with ScanNet++ and KITTI-360, demonstrate the superiority of our method in rendering quality and resource efficiency.  
  </ol>  
</details>  
**comments**: Project Page: https://jdtsui.github.io/letsgo/  
  
  



## Visual Localization  

### [CREST: Cross-modal Resonance through Evidential Deep Learning for Enhanced Zero-Shot Learning](http://arxiv.org/abs/2404.09640)  
[[code](https://github.com/JethroJames/CREST)]  
Haojian Huang, Xiaozhen Qiao, Zhuo Chen, Haodong Chen, Bingyu Li, Zhe Sun, Mulin Chen, Xuelong Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Zero-shot learning (ZSL) enables the recognition of novel classes by leveraging semantic knowledge transfer from known to unknown categories. This knowledge, typically encapsulated in attribute descriptions, aids in identifying class-specific visual features, thus facilitating visual-semantic alignment and improving ZSL performance. However, real-world challenges such as distribution imbalances and attribute co-occurrence among instances often hinder the discernment of local variances in images, a problem exacerbated by the scarcity of fine-grained, region-specific attribute annotations. Moreover, the variability in visual presentation within categories can also skew attribute-category associations. In response, we propose a bidirectional cross-modal ZSL approach CREST. It begins by extracting representations for attribute and visual localization and employs Evidential Deep Learning (EDL) to measure underlying epistemic uncertainty, thereby enhancing the model's resilience against hard negatives. CREST incorporates dual learning pathways, focusing on both visual-category and attribute-category alignments, to ensure robust correlation between latent and observable spaces. Moreover, we introduce an uncertainty-informed cross-modal fusion technique to refine visual-attribute inference. Extensive experiments demonstrate our model's effectiveness and unique explainability across multiple datasets. Our code and data are available at: Comments: Ongoing work; 10 pages, 2 Tables, 9 Figures; Repo is available at https://github.com/JethroJames/CREST.  
  </ol>  
</details>  
**comments**: Ongoing work; 10 pages, 2 Tables, 9 Figures; Repo is available at
  https://github.com/JethroJames/CREST  
  
  



## Image Matching  

### [XoFTR: Cross-modal Feature Matching Transformer](http://arxiv.org/abs/2404.09692)  
Önder Tuzcuoğlu, Aybora Köksal, Buğra Sofu, Sinan Kalkan, A. Aydın Alatan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce, XoFTR, a cross-modal cross-view method for local feature matching between thermal infrared (TIR) and visible images. Unlike visible images, TIR images are less susceptible to adverse lighting and weather conditions but present difficulties in matching due to significant texture and intensity differences. Current hand-crafted and learning-based methods for visible-TIR matching fall short in handling viewpoint, scale, and texture diversities. To address this, XoFTR incorporates masked image modeling pre-training and fine-tuning with pseudo-thermal image augmentation to handle the modality differences. Additionally, we introduce a refined matching pipeline that adjusts for scale discrepancies and enhances match reliability through sub-pixel level refinement. To validate our approach, we collect a comprehensive visible-thermal dataset, and show that our method outperforms existing methods on many benchmarks.  
  </ol>  
</details>  
**comments**: CVPR Image Matching Workshop, 2024. 12 pages, 7 figures, 5 tables.
  Codes and dataset are available at https://github.com/OnderT/XoFTR  
  
### [DeDoDe v2: Analyzing and Improving the DeDoDe Keypoint Detector](http://arxiv.org/abs/2404.08928)  
[[code](https://github.com/parskatt/dedode)]  
Johan Edstedt, Georg Bökman, Zhenjun Zhao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we analyze and improve into the recently proposed DeDoDe keypoint detector. We focus our analysis on some key issues. First, we find that DeDoDe keypoints tend to cluster together, which we fix by performing non-max suppression on the target distribution of the detector during training. Second, we address issues related to data augmentation. In particular, the DeDoDe detector is sensitive to large rotations. We fix this by including 90-degree rotations as well as horizontal flips. Finally, the decoupled nature of the DeDoDe detector makes evaluation of downstream usefulness problematic. We fix this by matching the keypoints with a pretrained dense matcher (RoMa) and evaluating two-view pose estimates. We find that the original long training is detrimental to performance, and therefore propose a much shorter training schedule. We integrate all these improvements into our proposed detector DeDoDe v2 and evaluate it with the original DeDoDe descriptor on the MegaDepth-1500 and IMC2022 benchmarks. Our proposed detector significantly increases pose estimation results, notably from 75.9 to 78.3 mAA on the IMC2022 challenge. Code and weights are available at https://github.com/Parskatt/DeDoDe  
  </ol>  
</details>  
**comments**: Accepted to Sixth Workshop on Image Matching - CVPRW 2024  
  
  



## NeRF  

### [DeferredGS: Decoupled and Editable Gaussian Splatting with Deferred Shading](http://arxiv.org/abs/2404.09412)  
Tong Wu, Jia-Mu Sun, Yu-Kun Lai, Yuewen Ma, Leif Kobbelt, Lin Gao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Reconstructing and editing 3D objects and scenes both play crucial roles in computer graphics and computer vision. Neural radiance fields (NeRFs) can achieve realistic reconstruction and editing results but suffer from inefficiency in rendering. Gaussian splatting significantly accelerates rendering by rasterizing Gaussian ellipsoids. However, Gaussian splatting utilizes a single Spherical Harmonic (SH) function to model both texture and lighting, limiting independent editing capabilities of these components. Recently, attempts have been made to decouple texture and lighting with the Gaussian splatting representation but may fail to produce plausible geometry and decomposition results on reflective scenes. Additionally, the forward shading technique they employ introduces noticeable blending artifacts during relighting, as the geometry attributes of Gaussians are optimized under the original illumination and may not be suitable for novel lighting conditions. To address these issues, we introduce DeferredGS, a method for decoupling and editing the Gaussian splatting representation using deferred shading. To achieve successful decoupling, we model the illumination with a learnable environment map and define additional attributes such as texture parameters and normal direction on Gaussians, where the normal is distilled from a jointly trained signed distance function. More importantly, we apply deferred shading, resulting in more realistic relighting effects compared to previous methods. Both qualitative and quantitative experiments demonstrate the superior performance of DeferredGS in novel view synthesis and editing tasks.  
  </ol>  
</details>  
  
### [VRS-NeRF: Visual Relocalization with Sparse Neural Radiance Field](http://arxiv.org/abs/2404.09271)  
[[code](https://github.com/feixue94/vrs-nerf)]  
Fei Xue, Ignas Budvytis, Daniel Olmeda Reino, Roberto Cipolla  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual relocalization is a key technique to autonomous driving, robotics, and virtual/augmented reality. After decades of explorations, absolute pose regression (APR), scene coordinate regression (SCR), and hierarchical methods (HMs) have become the most popular frameworks. However, in spite of high efficiency, APRs and SCRs have limited accuracy especially in large-scale outdoor scenes; HMs are accurate but need to store a large number of 2D descriptors for matching, resulting in poor efficiency. In this paper, we propose an efficient and accurate framework, called VRS-NeRF, for visual relocalization with sparse neural radiance field. Precisely, we introduce an explicit geometric map (EGM) for 3D map representation and an implicit learning map (ILM) for sparse patches rendering. In this localization process, EGP provides priors of spare 2D points and ILM utilizes these sparse points to render patches with sparse NeRFs for matching. This allows us to discard a large number of 2D descriptors so as to reduce the map size. Moreover, rendering patches only for useful points rather than all pixels in the whole image reduces the rendering time significantly. This framework inherits the accuracy of HMs and discards their low efficiency. Experiments on 7Scenes, CambridgeLandmarks, and Aachen datasets show that our method gives much better accuracy than APRs and SCRs, and close performance to HMs but is much more efficient.  
  </ol>  
</details>  
**comments**: source code https://github.com/feixue94/vrs-nerf  
  
  



