<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#SLAM-in-the-Dark:-Self-Supervised-Learning-of-Pose,-Depth-and-Loop-Closure-from-Thermal-Images>SLAM in the Dark: Self-Supervised Learning of Pose, Depth and Loop-Closure from Thermal Images</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#On-the-Importance-of-Text-Preprocessing-for-Multimodal-Representation-Learning-and-Pathology-Report-Generation>On the Importance of Text Preprocessing for Multimodal Representation Learning and Pathology Report Generation</a></li>
        <li><a href=#BEV-LIO(LC):-BEV-Image-Assisted-LiDAR-Inertial-Odometry-with-Loop-Closure>BEV-LIO(LC): BEV Image Assisted LiDAR-Inertial Odometry with Loop Closure</a></li>
        <li><a href=#SLAM-in-the-Dark:-Self-Supervised-Learning-of-Pose,-Depth-and-Loop-Closure-from-Thermal-Images>SLAM in the Dark: Self-Supervised Learning of Pose, Depth and Loop-Closure from Thermal Images</a></li>
        <li><a href=#A-Comprehensive-Survey-on-Composed-Image-Retrieval>A Comprehensive Survey on Composed Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Rewards-based-image-analysis-in-microscopy>Rewards-based image analysis in microscopy</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#BEV-LIO(LC):-BEV-Image-Assisted-LiDAR-Inertial-Odometry-with-Loop-Closure>BEV-LIO(LC): BEV Image Assisted LiDAR-Inertial Odometry with Loop Closure</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Does-3D-Gaussian-Splatting-Need-Accurate-Volumetric-Rendering?>Does 3D Gaussian Splatting Need Accurate Volumetric Rendering?</a></li>
        <li><a href=#The-NeRF-Signature:-Codebook-Aided-Watermarking-for-Neural-Radiance-Fields>The NeRF Signature: Codebook-Aided Watermarking for Neural Radiance Fields</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [SLAM in the Dark: Self-Supervised Learning of Pose, Depth and Loop-Closure from Thermal Images](http://arxiv.org/abs/2502.18932)  
Yangfan Xu, Qu Hao, Lilian Zhang, Jun Mao, Xiaofeng He, Wenqi Wu, Changhao Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual SLAM is essential for mobile robots, drone navigation, and VR/AR, but traditional RGB camera systems struggle in low-light conditions, driving interest in thermal SLAM, which excels in such environments. However, thermal imaging faces challenges like low contrast, high noise, and limited large-scale annotated datasets, restricting the use of deep learning in outdoor scenarios. We present DarkSLAM, a noval deep learning-based monocular thermal SLAM system designed for large-scale localization and reconstruction in complex lighting conditions.Our approach incorporates the Efficient Channel Attention (ECA) mechanism in visual odometry and the Selective Kernel Attention (SKA) mechanism in depth estimation to enhance pose accuracy and mitigate thermal depth degradation. Additionally, the system includes thermal depth-based loop closure detection and pose optimization, ensuring robust performance in low-texture thermal scenes. Extensive outdoor experiments demonstrate that DarkSLAM significantly outperforms existing methods like SC-Sfm-Learner and Shin et al., delivering precise localization and 3D dense mapping even in challenging nighttime environments.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [On the Importance of Text Preprocessing for Multimodal Representation Learning and Pathology Report Generation](http://arxiv.org/abs/2502.19285)  
Ruben T. Lucassen, Tijn van de Luijtgaarden, Sander P. J. Moonemans, Gerben E. Breimer, Willeke A. M. Blokx, Mitko Veta  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Vision-language models in pathology enable multimodal case retrieval and automated report generation. Many of the models developed so far, however, have been trained on pathology reports that include information which cannot be inferred from paired whole slide images (e.g., patient history), potentially leading to hallucinated sentences in generated reports. To this end, we investigate how the selection of information from pathology reports for vision-language modeling affects the quality of the multimodal representations and generated reports. More concretely, we compare a model trained on full reports against a model trained on preprocessed reports that only include sentences describing the cell and tissue appearances based on the H&E-stained slides. For the experiments, we built upon the BLIP-2 framework and used a cutaneous melanocytic lesion dataset of 42,433 H&E-stained whole slide images and 19,636 corresponding pathology reports. Model performance was assessed using image-to-text and text-to-image retrieval, as well as qualitative evaluation of the generated reports by an expert pathologist. Our results demonstrate that text preprocessing prevents hallucination in report generation. Despite the improvement in the quality of the generated reports, training the vision-language model on full reports showed better cross-modal retrieval performance.  
  </ol>  
</details>  
**comments**: 11 pages, 1 figure  
  
### [BEV-LIO(LC): BEV Image Assisted LiDAR-Inertial Odometry with Loop Closure](http://arxiv.org/abs/2502.19242)  
Haoxin Cai, Shenghai Yuan, Xinyi Li, Junfeng Guo, Jianqi Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This work introduces BEV-LIO(LC), a novel LiDAR-Inertial Odometry (LIO) framework that combines Bird's Eye View (BEV) image representations of LiDAR data with geometry-based point cloud registration and incorporates loop closure (LC) through BEV image features. By normalizing point density, we project LiDAR point clouds into BEV images, thereby enabling efficient feature extraction and matching. A lightweight convolutional neural network (CNN) based feature extractor is employed to extract distinctive local and global descriptors from the BEV images. Local descriptors are used to match BEV images with FAST keypoints for reprojection error construction, while global descriptors facilitate loop closure detection. Reprojection error minimization is then integrated with point-to-plane registration within an iterated Extended Kalman Filter (iEKF). In the back-end, global descriptors are used to create a KD-tree-indexed keyframe database for accurate loop closure detection. When a loop closure is detected, Random Sample Consensus (RANSAC) computes a coarse transform from BEV image matching, which serves as the initial estimate for Iterative Closest Point (ICP). The refined transform is subsequently incorporated into a factor graph along with odometry factors, improving the global consistency of localization. Extensive experiments conducted in various scenarios with different LiDAR types demonstrate that BEV-LIO(LC) outperforms state-of-the-art methods, achieving competitive localization accuracy. Our code, video and supplementary materials can be found at https://github.com/HxCa1/BEV-LIO-LC.  
  </ol>  
</details>  
  
### [SLAM in the Dark: Self-Supervised Learning of Pose, Depth and Loop-Closure from Thermal Images](http://arxiv.org/abs/2502.18932)  
Yangfan Xu, Qu Hao, Lilian Zhang, Jun Mao, Xiaofeng He, Wenqi Wu, Changhao Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual SLAM is essential for mobile robots, drone navigation, and VR/AR, but traditional RGB camera systems struggle in low-light conditions, driving interest in thermal SLAM, which excels in such environments. However, thermal imaging faces challenges like low contrast, high noise, and limited large-scale annotated datasets, restricting the use of deep learning in outdoor scenarios. We present DarkSLAM, a noval deep learning-based monocular thermal SLAM system designed for large-scale localization and reconstruction in complex lighting conditions.Our approach incorporates the Efficient Channel Attention (ECA) mechanism in visual odometry and the Selective Kernel Attention (SKA) mechanism in depth estimation to enhance pose accuracy and mitigate thermal depth degradation. Additionally, the system includes thermal depth-based loop closure detection and pose optimization, ensuring robust performance in low-texture thermal scenes. Extensive outdoor experiments demonstrate that DarkSLAM significantly outperforms existing methods like SC-Sfm-Learner and Shin et al., delivering precise localization and 3D dense mapping even in challenging nighttime environments.  
  </ol>  
</details>  
  
### [A Comprehensive Survey on Composed Image Retrieval](http://arxiv.org/abs/2502.18495)  
Xuemeng Song, Haoqiang Lin, Haokun Wen, Bohan Hou, Mingzhu Xu, Liqiang Nie  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed Image Retrieval (CIR) is an emerging yet challenging task that allows users to search for target images using a multimodal query, comprising a reference image and a modification text specifying the user's desired changes to the reference image. Given its significant academic and practical value, CIR has become a rapidly growing area of interest in the computer vision and machine learning communities, particularly with the advances in deep learning. To the best of our knowledge, there is currently no comprehensive review of CIR to provide a timely overview of this field. Therefore, we synthesize insights from over 120 publications in top conferences and journals, including ACM TOIS, SIGIR, and CVPR In particular, we systematically categorize existing supervised CIR and zero-shot CIR models using a fine-grained taxonomy. For a comprehensive review, we also briefly discuss approaches for tasks closely related to CIR, such as attribute-based CIR and dialog-based CIR. Additionally, we summarize benchmark datasets for evaluation and analyze existing supervised and zero-shot CIR methods by comparing experimental results across multiple datasets. Furthermore, we present promising future directions in this field, offering practical insights for researchers interested in further exploration.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Rewards-based image analysis in microscopy](http://arxiv.org/abs/2502.18522)  
Kamyar Barakati, Yu Liu, Utkarsh Pratiush, Boris N. Slautin, Sergei V. Kalinin  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Analyzing imaging and hyperspectral data is crucial across scientific fields, including biology, medicine, chemistry, and physics. The primary goal is to transform high-resolution or high-dimensional data into an interpretable format to generate actionable insights, aiding decision-making and advancing knowledge. Currently, this task relies on complex, human-designed workflows comprising iterative steps such as denoising, spatial sampling, keypoint detection, feature generation, clustering, dimensionality reduction, and physics-based deconvolutions. The introduction of machine learning over the past decade has accelerated tasks like image segmentation and object detection via supervised learning, and dimensionality reduction via unsupervised methods. However, both classical and NN-based approaches still require human input, whether for hyperparameter tuning, data labeling, or both. The growing use of automated imaging tools, from atomically resolved imaging to biological applications, demands unsupervised methods that optimize data representation for human decision-making or autonomous experimentation. Here, we discuss advances in reward-based workflows, which adopt expert decision-making principles and demonstrate strong transfer learning across diverse tasks. We represent image analysis as a decision-making process over possible operations and identify desiderata and their mappings to classical decision-making frameworks. Reward-driven workflows enable a shift from supervised, black-box models sensitive to distribution shifts to explainable, unsupervised, and robust optimization in image analysis. They can function as wrappers over classical and DCNN-based methods, making them applicable to both unsupervised and supervised workflows (e.g., classification, regression for structure-property mapping) across imaging and hyperspectral data.  
  </ol>  
</details>  
**comments**: 38 pages, 11 figures  
  
  



## Image Matching  

### [BEV-LIO(LC): BEV Image Assisted LiDAR-Inertial Odometry with Loop Closure](http://arxiv.org/abs/2502.19242)  
Haoxin Cai, Shenghai Yuan, Xinyi Li, Junfeng Guo, Jianqi Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This work introduces BEV-LIO(LC), a novel LiDAR-Inertial Odometry (LIO) framework that combines Bird's Eye View (BEV) image representations of LiDAR data with geometry-based point cloud registration and incorporates loop closure (LC) through BEV image features. By normalizing point density, we project LiDAR point clouds into BEV images, thereby enabling efficient feature extraction and matching. A lightweight convolutional neural network (CNN) based feature extractor is employed to extract distinctive local and global descriptors from the BEV images. Local descriptors are used to match BEV images with FAST keypoints for reprojection error construction, while global descriptors facilitate loop closure detection. Reprojection error minimization is then integrated with point-to-plane registration within an iterated Extended Kalman Filter (iEKF). In the back-end, global descriptors are used to create a KD-tree-indexed keyframe database for accurate loop closure detection. When a loop closure is detected, Random Sample Consensus (RANSAC) computes a coarse transform from BEV image matching, which serves as the initial estimate for Iterative Closest Point (ICP). The refined transform is subsequently incorporated into a factor graph along with odometry factors, improving the global consistency of localization. Extensive experiments conducted in various scenarios with different LiDAR types demonstrate that BEV-LIO(LC) outperforms state-of-the-art methods, achieving competitive localization accuracy. Our code, video and supplementary materials can be found at https://github.com/HxCa1/BEV-LIO-LC.  
  </ol>  
</details>  
  
  



## NeRF  

### [Does 3D Gaussian Splatting Need Accurate Volumetric Rendering?](http://arxiv.org/abs/2502.19318)  
Adam Celarek, George Kopanas, George Drettakis, Michael Wimmer, Bernhard Kerbl  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Since its introduction, 3D Gaussian Splatting (3DGS) has become an important reference method for learning 3D representations of a captured scene, allowing real-time novel-view synthesis with high visual quality and fast training times. Neural Radiance Fields (NeRFs), which preceded 3DGS, are based on a principled ray-marching approach for volumetric rendering. In contrast, while sharing a similar image formation model with NeRF, 3DGS uses a hybrid rendering solution that builds on the strengths of volume rendering and primitive rasterization. A crucial benefit of 3DGS is its performance, achieved through a set of approximations, in many cases with respect to volumetric rendering theory. A naturally arising question is whether replacing these approximations with more principled volumetric rendering solutions can improve the quality of 3DGS. In this paper, we present an in-depth analysis of the various approximations and assumptions used by the original 3DGS solution. We demonstrate that, while more accurate volumetric rendering can help for low numbers of primitives, the power of efficient optimization and the large number of Gaussians allows 3DGS to outperform volumetric rendering despite its approximations.  
  </ol>  
</details>  
**comments**: To be published in Eurogrpahics 2025, code:
  https://github.com/cg-tuwien/does_3d_gaussian_splatting_need_accurate_volumetric_rendering  
  
### [The NeRF Signature: Codebook-Aided Watermarking for Neural Radiance Fields](http://arxiv.org/abs/2502.19125)  
Ziyuan Luo, Anderson Rocha, Boxin Shi, Qing Guo, Haoliang Li, Renjie Wan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) have been gaining attention as a significant form of 3D content representation. With the proliferation of NeRF-based creations, the need for copyright protection has emerged as a critical issue. Although some approaches have been proposed to embed digital watermarks into NeRF, they often neglect essential model-level considerations and incur substantial time overheads, resulting in reduced imperceptibility and robustness, along with user inconvenience. In this paper, we extend the previous criteria for image watermarking to the model level and propose NeRF Signature, a novel watermarking method for NeRF. We employ a Codebook-aided Signature Embedding (CSE) that does not alter the model structure, thereby maintaining imperceptibility and enhancing robustness at the model level. Furthermore, after optimization, any desired signatures can be embedded through the CSE, and no fine-tuning is required when NeRF owners want to use new binary signatures. Then, we introduce a joint pose-patch encryption watermarking strategy to hide signatures into patches rendered from a specific viewpoint for higher robustness. In addition, we explore a Complexity-Aware Key Selection (CAKS) scheme to embed signatures in high visual complexity patches to enhance imperceptibility. The experimental results demonstrate that our method outperforms other baseline methods in terms of imperceptibility and robustness. The source code is available at: https://github.com/luo-ziyuan/NeRF_Signature.  
  </ol>  
</details>  
**comments**: 16 pages, accepted by TPAMI  
  
  



