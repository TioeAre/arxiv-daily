<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Dense-SfM:-Structure-from-Motion-with-Dense-Consistent-Matching>Dense-SfM: Structure from Motion with Dense Consistent Matching</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Visual-Localization-via-Semantic-Structures-in-Autonomous-Photovoltaic-Power-Plant-Inspection>Visual Localization via Semantic Structures in Autonomous Photovoltaic Power Plant Inspection</a></li>
        <li><a href=#Revisiting-CLIP:-Efficient-Alignment-of-3D-MRI-and-Tabular-Data-using-Domain-Specific-Foundation-Models>Revisiting CLIP: Efficient Alignment of 3D MRI and Tabular Data using Domain-Specific Foundation Models</a></li>
        <li><a href=#Triplet-Synthesis-For-Enhancing-Composed-Image-Retrieval-via-Counterfactual-Image-Generation>Triplet Synthesis For Enhancing Composed Image Retrieval via Counterfactual Image Generation</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Dense-SfM:-Structure-from-Motion-with-Dense-Consistent-Matching>Dense-SfM: Structure from Motion with Dense Consistent Matching</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#SyncAnimation:-A-Real-Time-End-to-End-Framework-for-Audio-Driven-Human-Pose-and-Talking-Head-Animation>SyncAnimation: A Real-Time End-to-End Framework for Audio-Driven Human Pose and Talking Head Animation</a></li>
        <li><a href=#GS-LiDAR:-Generating-Realistic-LiDAR-Point-Clouds-with-Panoramic-Gaussian-Splatting>GS-LiDAR: Generating Realistic LiDAR Point Clouds with Panoramic Gaussian Splatting</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Dense-SfM: Structure from Motion with Dense Consistent Matching](http://arxiv.org/abs/2501.14277)  
JongMin Lee, Sungjoo Yoo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present Dense-SfM, a novel Structure from Motion (SfM) framework designed for dense and accurate 3D reconstruction from multi-view images. Sparse keypoint matching, which traditional SfM methods often rely on, limits both accuracy and point density, especially in texture-less areas. Dense-SfM addresses this limitation by integrating dense matching with a Gaussian Splatting (GS) based track extension which gives more consistent, longer feature tracks. To further improve reconstruction accuracy, Dense-SfM is equipped with a multi-view kernelized matching module leveraging transformer and Gaussian Process architectures, for robust track refinement across multi-views. Evaluations on the ETH3D and Texture-Poor SfM datasets show that Dense-SfM offers significant improvements in accuracy and density over state-of-the-art methods.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Visual Localization via Semantic Structures in Autonomous Photovoltaic Power Plant Inspection](http://arxiv.org/abs/2501.14587)  
Viktor Kozák, Karel Košnar, Jan Chudoba, Miroslav Kulich, Libor Přeučil  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Inspection systems utilizing unmanned aerial vehicles (UAVs) equipped with thermal cameras are increasingly popular for the maintenance of photovoltaic (PV) power plants. However, automation of the inspection task is a challenging problem as it requires precise navigation to capture images from optimal distances and viewing angles.   This paper presents a novel localization pipeline that directly integrates PV module detection with UAV navigation, allowing precise positioning during inspection. Detections are used to identify the power plant structures in the image and associate these with the power plant model. We define visually recognizable anchor points for the initial association and use object tracking to discern global associations. We present three distinct methods for visual segmentation of PV modules based on traditional computer vision, deep learning, and their fusion, and we evaluate their performance in relation to the proposed localization pipeline.   The presented methods were verified and evaluated using custom aerial inspection data sets, demonstrating their robustness and applicability for real-time navigation. Additionally, we evaluate the influence of the power plant model's precision on the localization methods.  
  </ol>  
</details>  
**comments**: 47 pages, 22 figures  
  
### [Revisiting CLIP: Efficient Alignment of 3D MRI and Tabular Data using Domain-Specific Foundation Models](http://arxiv.org/abs/2501.14051)  
[[code](https://github.com/jakekrogh/3d-clip-for-brain-mri)]  
Jakob Krogh Petersen, Valdemar Licht, Mads Nielsen, Asbjørn Munk  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Multi-modal models require aligned, shared embedding spaces. However, common CLIP-based approaches need large amounts of samples and do not natively support 3D or tabular data, both of which are crucial in the medical domain. To address these issues, we revisit CLIP-style alignment by training a domain-specific 3D foundation model as an image encoder and demonstrate that modality alignment is feasible with only 62 MRI scans. Our approach is enabled by a simple embedding accumulation strategy required for training in 3D, which scales the amount of negative pairs across batches in order to stabilize training. We perform a thorough evaluation of various design choices, including the choice of backbone and loss functions, and evaluate the proposed methodology on zero-shot classification and image-retrieval tasks. While zero-shot image-retrieval remains challenging, zero-shot classification results demonstrate that the proposed approach can meaningfully align the representations of 3D MRI with tabular data.  
  </ol>  
</details>  
**comments**: 10 pages, 2 figures. To be published in ISBI 2025  
  
### [Triplet Synthesis For Enhancing Composed Image Retrieval via Counterfactual Image Generation](http://arxiv.org/abs/2501.13968)  
Kenta Uesugi, Naoki Saito, Keisuke Maeda, Takahiro Ogawa, Miki Haseyama  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed Image Retrieval (CIR) provides an effective way to manage and access large-scale visual data. Construction of the CIR model utilizes triplets that consist of a reference image, modification text describing desired changes, and a target image that reflects these changes. For effectively training CIR models, extensive manual annotation to construct high-quality training datasets, which can be time-consuming and labor-intensive, is required. To deal with this problem, this paper proposes a novel triplet synthesis method by leveraging counterfactual image generation. By controlling visual feature modifications via counterfactual image generation, our approach automatically generates diverse training triplets without any manual intervention. This approach facilitates the creation of larger and more expressive datasets, leading to the improvement of CIR model's performance.  
  </ol>  
</details>  
**comments**: 4 pages, 4 figures  
  
  



## Image Matching  

### [Dense-SfM: Structure from Motion with Dense Consistent Matching](http://arxiv.org/abs/2501.14277)  
JongMin Lee, Sungjoo Yoo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present Dense-SfM, a novel Structure from Motion (SfM) framework designed for dense and accurate 3D reconstruction from multi-view images. Sparse keypoint matching, which traditional SfM methods often rely on, limits both accuracy and point density, especially in texture-less areas. Dense-SfM addresses this limitation by integrating dense matching with a Gaussian Splatting (GS) based track extension which gives more consistent, longer feature tracks. To further improve reconstruction accuracy, Dense-SfM is equipped with a multi-view kernelized matching module leveraging transformer and Gaussian Process architectures, for robust track refinement across multi-views. Evaluations on the ETH3D and Texture-Poor SfM datasets show that Dense-SfM offers significant improvements in accuracy and density over state-of-the-art methods.  
  </ol>  
</details>  
  
  



## NeRF  

### [SyncAnimation: A Real-Time End-to-End Framework for Audio-Driven Human Pose and Talking Head Animation](http://arxiv.org/abs/2501.14646)  
Yujian Liu, Shidang Xu, Jing Guo, Dingbin Wang, Zairan Wang, Xianfeng Tan, Xiaoli Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Generating talking avatar driven by audio remains a significant challenge. Existing methods typically require high computational costs and often lack sufficient facial detail and realism, making them unsuitable for applications that demand high real-time performance and visual quality. Additionally, while some methods can synchronize lip movement, they still face issues with consistency between facial expressions and upper body movement, particularly during silent periods. In this paper, we introduce SyncAnimation, the first NeRF-based method that achieves audio-driven, stable, and real-time generation of speaking avatar by combining generalized audio-to-pose matching and audio-to-expression synchronization. By integrating AudioPose Syncer and AudioEmotion Syncer, SyncAnimation achieves high-precision poses and expression generation, progressively producing audio-synchronized upper body, head, and lip shapes. Furthermore, the High-Synchronization Human Renderer ensures seamless integration of the head and upper body, and achieves audio-sync lip. The project page can be found at https://syncanimation.github.io  
  </ol>  
</details>  
**comments**: 11 pages, 7 figures  
  
### [GS-LiDAR: Generating Realistic LiDAR Point Clouds with Panoramic Gaussian Splatting](http://arxiv.org/abs/2501.13971)  
[[code](https://github.com/fudan-zvg/gs-lidar)]  
Junzhe Jiang, Chun Gu, Yurui Chen, Li Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    LiDAR novel view synthesis (NVS) has emerged as a novel task within LiDAR simulation, offering valuable simulated point cloud data from novel viewpoints to aid in autonomous driving systems. However, existing LiDAR NVS methods typically rely on neural radiance fields (NeRF) as their 3D representation, which incurs significant computational costs in both training and rendering. Moreover, NeRF and its variants are designed for symmetrical scenes, making them ill-suited for driving scenarios. To address these challenges, we propose GS-LiDAR, a novel framework for generating realistic LiDAR point clouds with panoramic Gaussian splatting. Our approach employs 2D Gaussian primitives with periodic vibration properties, allowing for precise geometric reconstruction of both static and dynamic elements in driving scenarios. We further introduce a novel panoramic rendering technique with explicit ray-splat intersection, guided by panoramic LiDAR supervision. By incorporating intensity and ray-drop spherical harmonic (SH) coefficients into the Gaussian primitives, we enhance the realism of the rendered point clouds. Extensive experiments on KITTI-360 and nuScenes demonstrate the superiority of our method in terms of quantitative metrics, visual quality, as well as training and rendering efficiency.  
  </ol>  
</details>  
  
  



