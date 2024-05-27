<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#ETA-INIT:-Enhancing-the-Translation-Accuracy-for-Stereo-Visual-Inertial-SLAM-Initialization>ETA-INIT: Enhancing the Translation Accuracy for Stereo Visual-Inertial SLAM Initialization</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Composed-Image-Retrieval-for-Remote-Sensing>Composed Image Retrieval for Remote Sensing</a></li>
        <li><a href=#Self-distilled-Dynamic-Fusion-Network-for-Language-based-Fashion-Retrieval>Self-distilled Dynamic Fusion Network for Language-based Fashion Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Neural-Elevation-Models-for-Terrain-Mapping-and-Path-Planning>Neural Elevation Models for Terrain Mapping and Path Planning</a></li>
        <li><a href=#HDR-GS:-Efficient-High-Dynamic-Range-Novel-View-Synthesis-at-1000x-Speed-via-Gaussian-Splatting>HDR-GS: Efficient High Dynamic Range Novel View Synthesis at 1000x Speed via Gaussian Splatting</a></li>
        <li><a href=#GS-Hider:-Hiding-Messages-into-3D-Gaussian-Splatting>GS-Hider: Hiding Messages into 3D Gaussian Splatting</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [ETA-INIT: Enhancing the Translation Accuracy for Stereo Visual-Inertial SLAM Initialization](http://arxiv.org/abs/2405.15082)  
Han Song, Zhongche Qu, Zhi Zhang, Zihan Ye, Cong Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    As the current initialization method in the state-of-the-art Stereo Visual-Inertial SLAM framework, ORB-SLAM3 has limitations. Its success depends on the performance of the pure stereo SLAM system and is based on the underlying assumption that pure visual SLAM can accurately estimate the camera trajectory, which is essential for inertial parameter estimation. Meanwhile, the further improved initialization method for ORB-SLAM3, known as Stereo-NEC, is time-consuming due to applying keypoint tracking to estimate gyroscope bias with normal epipolar constraints. To address the limitations of previous methods, this paper proposes a method aimed at enhancing translation accuracy during the initialization stage. The fundamental concept of our method is to improve the translation estimate with a 3 Degree-of-Freedom (DoF) Bundle Adjustment (BA), independently, while the rotation estimate is fixed, instead of using ORB-SLAM3's 6-DoF BA. Additionally, the rotation estimate will be updated by considering IMU measurements and gyroscope bias, unlike ORB-SLAM3's rotation, which is directly obtained from stereo visual odometry and may yield inferior results when operating in challenging scenarios. We also conduct extensive evaluations on the public benchmark, the EuRoC dataset, demonstrating that our method excels in accuracy.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Composed Image Retrieval for Remote Sensing](http://arxiv.org/abs/2405.15587)  
Bill Psomas, Ioannis Kakogeorgiou, Nikos Efthymiadis, Giorgos Tolias, Ondrej Chum, Yannis Avrithis, Konstantinos Karantzalos  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This work introduces composed image retrieval to remote sensing. It allows to query a large image archive by image examples alternated by a textual description, enriching the descriptive power over unimodal queries, either visual or textual. Various attributes can be modified by the textual part, such as shape, color, or context. A novel method fusing image-to-image and text-to-image similarity is introduced. We demonstrate that a vision-language model possesses sufficient descriptive power and no further learning step or training data are necessary. We present a new evaluation benchmark focused on color, context, density, existence, quantity, and shape modifications. Our work not only sets the state-of-the-art for this task, but also serves as a foundational step in addressing a gap in the field of remote sensing image retrieval. Code at: https://github.com/billpsomas/rscir  
  </ol>  
</details>  
  
### [Self-distilled Dynamic Fusion Network for Language-based Fashion Retrieval](http://arxiv.org/abs/2405.15451)  
Yiming Wu, Hangfei Li, Fangfang Wang, Yilong Zhang, Ronghua Liang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In the domain of language-based fashion image retrieval, pinpointing the desired fashion item using both a reference image and its accompanying textual description is an intriguing challenge. Existing approaches lean heavily on static fusion techniques, intertwining image and text. Despite their commendable advancements, these approaches are still limited by a deficiency in flexibility. In response, we propose a Self-distilled Dynamic Fusion Network to compose the multi-granularity features dynamically by considering the consistency of routing path and modality-specific information simultaneously. Two new modules are included in our proposed method: (1) Dynamic Fusion Network with Modality Specific Routers. The dynamic network enables a flexible determination of the routing for each reference image and modification text, taking into account their distinct semantics and distributions. (2) Self Path Distillation Loss. A stable path decision for queries benefits the optimization of feature extraction as well as routing, and we approach this by progressively refine the path decision with previous path information. Extensive experiments demonstrate the effectiveness of our proposed model compared to existing methods.  
  </ol>  
</details>  
**comments**: ICASSP 2024  
  
  



## NeRF  

### [Neural Elevation Models for Terrain Mapping and Path Planning](http://arxiv.org/abs/2405.15227)  
Adam Dai, Shubh Gupta, Grace Gao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This work introduces Neural Elevations Models (NEMos), which adapt Neural Radiance Fields to a 2.5D continuous and differentiable terrain model. In contrast to traditional terrain representations such as digital elevation models, NEMos can be readily generated from imagery, a low-cost data source, and provide a lightweight representation of terrain through an implicit continuous and differentiable height field. We propose a novel method for jointly training a height field and radiance field within a NeRF framework, leveraging quantile regression. Additionally, we introduce a path planning algorithm that performs gradient-based optimization of a continuous cost function for minimizing distance, slope changes, and control effort, enabled by differentiability of the height field. We perform experiments on simulated and real-world terrain imagery, demonstrating NEMos ability to generate high-quality reconstructions and produce smoother paths compared to discrete path planning methods. Future work will explore the incorporation of features and semantics into the height field, creating a generalized terrain model.  
  </ol>  
</details>  
  
### [HDR-GS: Efficient High Dynamic Range Novel View Synthesis at 1000x Speed via Gaussian Splatting](http://arxiv.org/abs/2405.15125)  
Yuanhao Cai, Zihao Xiao, Yixun Liang, Yulun Zhang, Xiaokang Yang, Yaoyao Liu, Alan Yuille  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    High dynamic range (HDR) novel view synthesis (NVS) aims to create photorealistic images from novel viewpoints using HDR imaging techniques. The rendered HDR images capture a wider range of brightness levels containing more details of the scene than normal low dynamic range (LDR) images. Existing HDR NVS methods are mainly based on NeRF. They suffer from long training time and slow inference speed. In this paper, we propose a new framework, High Dynamic Range Gaussian Splatting (HDR-GS), which can efficiently render novel HDR views and reconstruct LDR images with a user input exposure time. Specifically, we design a Dual Dynamic Range (DDR) Gaussian point cloud model that uses spherical harmonics to fit HDR color and employs an MLP-based tone-mapper to render LDR color. The HDR and LDR colors are then fed into two Parallel Differentiable Rasterization (PDR) processes to reconstruct HDR and LDR views. To establish the data foundation for the research of 3D Gaussian splatting-based methods in HDR NVS, we recalibrate the camera parameters and compute the initial positions for Gaussian point clouds. Experiments demonstrate that our HDR-GS surpasses the state-of-the-art NeRF-based method by 3.84 and 1.91 dB on LDR and HDR NVS while enjoying 1000x inference speed and only requiring 6.3% training time.  
  </ol>  
</details>  
**comments**: The first 3D Gaussian Splatting-based method for HDR imaging  
  
### [GS-Hider: Hiding Messages into 3D Gaussian Splatting](http://arxiv.org/abs/2405.15118)  
Xuanyu Zhang, Jiarui Meng, Runyi Li, Zhipei Xu, Yongbing Zhang, Jian Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D Gaussian Splatting (3DGS) has already become the emerging research focus in the fields of 3D scene reconstruction and novel view synthesis. Given that training a 3DGS requires a significant amount of time and computational cost, it is crucial to protect the copyright, integrity, and privacy of such 3D assets. Steganography, as a crucial technique for encrypted transmission and copyright protection, has been extensively studied. However, it still lacks profound exploration targeted at 3DGS. Unlike its predecessor NeRF, 3DGS possesses two distinct features: 1) explicit 3D representation; and 2) real-time rendering speeds. These characteristics result in the 3DGS point cloud files being public and transparent, with each Gaussian point having a clear physical significance. Therefore, ensuring the security and fidelity of the original 3D scene while embedding information into the 3DGS point cloud files is an extremely challenging task. To solve the above-mentioned issue, we first propose a steganography framework for 3DGS, dubbed GS-Hider, which can embed 3D scenes and images into original GS point clouds in an invisible manner and accurately extract the hidden messages. Specifically, we design a coupled secured feature attribute to replace the original 3DGS's spherical harmonics coefficients and then use a scene decoder and a message decoder to disentangle the original RGB scene and the hidden message. Extensive experiments demonstrated that the proposed GS-Hider can effectively conceal multimodal messages without compromising rendering quality and possesses exceptional security, robustness, capacity, and flexibility. Our project is available at: https://xuanyuzhang21.github.io/project/gshider.  
  </ol>  
</details>  
**comments**: 3DGS steganography  
  
  



