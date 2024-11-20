<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#SPARS3R:-Semantic-Prior-Alignment-and-Regularization-for-Sparse-3D-Reconstruction>SPARS3R: Semantic Prior Alignment and Regularization for Sparse 3D Reconstruction</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#IoT-Based-3D-Pose-Estimation-and-Motion-Optimization-for-Athletes:-Application-of-C3D-and-OpenPose>IoT-Based 3D Pose Estimation and Motion Optimization for Athletes: Application of C3D and OpenPose</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#SCIGS:-3D-Gaussians-Splatting-from-a-Snapshot-Compressive-Image>SCIGS: 3D Gaussians Splatting from a Snapshot Compressive Image</a></li>
        <li><a href=#GaussianPretrain:-A-Simple-Unified-3D-Gaussian-Representation-for-Visual-Pre-training-in-Autonomous-Driving>GaussianPretrain: A Simple Unified 3D Gaussian Representation for Visual Pre-training in Autonomous Driving</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [SPARS3R: Semantic Prior Alignment and Regularization for Sparse 3D Reconstruction](http://arxiv.org/abs/2411.12592)  
Yutao Tang, Yuxiang Guo, Deming Li, Cheng Peng  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent efforts in Gaussian-Splat-based Novel View Synthesis can achieve photorealistic rendering; however, such capability is limited in sparse-view scenarios due to sparse initialization and over-fitting floaters. Recent progress in depth estimation and alignment can provide dense point cloud with few views; however, the resulting pose accuracy is suboptimal. In this work, we present SPARS3R, which combines the advantages of accurate pose estimation from Structure-from-Motion and dense point cloud from depth estimation. To this end, SPARS3R first performs a Global Fusion Alignment process that maps a prior dense point cloud to a sparse point cloud from Structure-from-Motion based on triangulated correspondences. RANSAC is applied during this process to distinguish inliers and outliers. SPARS3R then performs a second, Semantic Outlier Alignment step, which extracts semantically coherent regions around the outliers and performs local alignment in these regions. Along with several improvements in the evaluation process, we demonstrate that SPARS3R can achieve photorealistic rendering with sparse images and significantly outperforms existing approaches.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [IoT-Based 3D Pose Estimation and Motion Optimization for Athletes: Application of C3D and OpenPose](http://arxiv.org/abs/2411.12676)  
Fei Ren, Chao Ren, Tianyi Lyu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This study proposes the IoT-Enhanced Pose Optimization Network (IE-PONet) for high-precision 3D pose estimation and motion optimization of track and field athletes. IE-PONet integrates C3D for spatiotemporal feature extraction, OpenPose for real-time keypoint detection, and Bayesian optimization for hyperparameter tuning. Experimental results on NTURGB+D and FineGYM datasets demonstrate superior performance, with AP\(^p50\) scores of 90.5 and 91.0, and mAP scores of 74.3 and 74.0, respectively. Ablation studies confirm the essential roles of each module in enhancing model accuracy. IE-PONet provides a robust tool for athletic performance analysis and optimization, offering precise technical insights for training and injury prevention. Future work will focus on further model optimization, multimodal data integration, and developing real-time feedback mechanisms to enhance practical applications.  
  </ol>  
</details>  
**comments**: 17 pages  
  
  



## NeRF  

### [SCIGS: 3D Gaussians Splatting from a Snapshot Compressive Image](http://arxiv.org/abs/2411.12471)  
Zixu Wang, Hao Yang, Yu Guo, Fei Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Snapshot Compressive Imaging (SCI) offers a possibility for capturing information in high-speed dynamic scenes, requiring efficient reconstruction method to recover scene information. Despite promising results, current deep learning-based and NeRF-based reconstruction methods face challenges: 1) deep learning-based reconstruction methods struggle to maintain 3D structural consistency within scenes, and 2) NeRF-based reconstruction methods still face limitations in handling dynamic scenes. To address these challenges, we propose SCIGS, a variant of 3DGS, and develop a primitive-level transformation network that utilizes camera pose stamps and Gaussian primitive coordinates as embedding vectors. This approach resolves the necessity of camera pose in vanilla 3DGS and enhances multi-view 3D structural consistency in dynamic scenes by utilizing transformed primitives. Additionally, a high-frequency filter is introduced to eliminate the artifacts generated during the transformation. The proposed SCIGS is the first to reconstruct a 3D explicit scene from a single compressed image, extending its application to dynamic 3D scenes. Experiments on both static and dynamic scenes demonstrate that SCIGS not only enhances SCI decoding but also outperforms current state-of-the-art methods in reconstructing dynamic 3D scenes from a single compressed image. The code will be made available upon publication.  
  </ol>  
</details>  
  
### [GaussianPretrain: A Simple Unified 3D Gaussian Representation for Visual Pre-training in Autonomous Driving](http://arxiv.org/abs/2411.12452)  
Shaoqing Xu, Fang Li, Shengyin Jiang, Ziying Song, Li Liu, Zhi-xin Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Self-supervised learning has made substantial strides in image processing, while visual pre-training for autonomous driving is still in its infancy. Existing methods often focus on learning geometric scene information while neglecting texture or treating both aspects separately, hindering comprehensive scene understanding. In this context, we are excited to introduce GaussianPretrain, a novel pre-training paradigm that achieves a holistic understanding of the scene by uniformly integrating geometric and texture representations. Conceptualizing 3D Gaussian anchors as volumetric LiDAR points, our method learns a deepened understanding of scenes to enhance pre-training performance with detailed spatial structure and texture, achieving that 40.6% faster than NeRF-based method UniPAD with 70% GPU memory only. We demonstrate the effectiveness of GaussianPretrain across multiple 3D perception tasks, showing significant performance improvements, such as a 7.05% increase in NDS for 3D object detection, boosts mAP by 1.9% in HD map construction and 0.8% improvement on Occupancy prediction. These significant gains highlight GaussianPretrain's theoretical innovation and strong practical potential, promoting visual pre-training development for autonomous driving. Source code will be available at https://github.com/Public-BOTs/GaussianPretrain  
  </ol>  
</details>  
**comments**: 10 pages, 5 figures  
  
  



