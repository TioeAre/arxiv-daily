<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#SPOT:-Point-Cloud-Based-Stereo-Visual-Place-Recognition-for-Similar-and-Opposing-Viewpoints>SPOT: Point Cloud Based Stereo Visual Place Recognition for Similar and Opposing Viewpoints</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#SPOT:-Point-Cloud-Based-Stereo-Visual-Place-Recognition-for-Similar-and-Opposing-Viewpoints>SPOT: Point Cloud Based Stereo Visual Place Recognition for Similar and Opposing Viewpoints</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#MeshLRM:-Large-Reconstruction-Model-for-High-Quality-Mesh>MeshLRM: Large Reconstruction Model for High-Quality Mesh</a></li>
        <li><a href=#AG-NeRF:-Attention-guided-Neural-Radiance-Fields-for-Multi-height-Large-scale-Outdoor-Scene-Rendering>AG-NeRF: Attention-guided Neural Radiance Fields for Multi-height Large-scale Outdoor Scene Rendering</a></li>
        <li><a href=#Cicero:-Addressing-Algorithmic-and-Architectural-Bottlenecks-in-Neural-Rendering-by-Radiance-Warping-and-Memory-Optimizations>Cicero: Addressing Algorithmic and Architectural Bottlenecks in Neural Rendering by Radiance Warping and Memory Optimizations</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [SPOT: Point Cloud Based Stereo Visual Place Recognition for Similar and Opposing Viewpoints](http://arxiv.org/abs/2404.12339)  
Spencer Carmichael, Rahul Agrawal, Ram Vasudevan, Katherine A. Skinner  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recognizing places from an opposing viewpoint during a return trip is a common experience for human drivers. However, the analogous robotics capability, visual place recognition (VPR) with limited field of view cameras under 180 degree rotations, has proven to be challenging to achieve. To address this problem, this paper presents Same Place Opposing Trajectory (SPOT), a technique for opposing viewpoint VPR that relies exclusively on structure estimated through stereo visual odometry (VO). The method extends recent advances in lidar descriptors and utilizes a novel double (similar and opposing) distance matrix sequence matching method. We evaluate SPOT on a publicly available dataset with 6.7-7.6 km routes driven in similar and opposing directions under various lighting conditions. The proposed algorithm demonstrates remarkable improvement over the state-of-the-art, achieving up to 91.7% recall at 100% precision in opposing viewpoint cases, while requiring less storage than all baselines tested and running faster than all but one. Moreover, the proposed method assumes no a priori knowledge of whether the viewpoint is similar or opposing, and also demonstrates competitive performance in similar viewpoint cases.  
  </ol>  
</details>  
**comments**: Accepted to ICRA 2024, project website:
  https://umautobots.github.io/spot  
  
  



## Visual Localization  

### [SPOT: Point Cloud Based Stereo Visual Place Recognition for Similar and Opposing Viewpoints](http://arxiv.org/abs/2404.12339)  
Spencer Carmichael, Rahul Agrawal, Ram Vasudevan, Katherine A. Skinner  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recognizing places from an opposing viewpoint during a return trip is a common experience for human drivers. However, the analogous robotics capability, visual place recognition (VPR) with limited field of view cameras under 180 degree rotations, has proven to be challenging to achieve. To address this problem, this paper presents Same Place Opposing Trajectory (SPOT), a technique for opposing viewpoint VPR that relies exclusively on structure estimated through stereo visual odometry (VO). The method extends recent advances in lidar descriptors and utilizes a novel double (similar and opposing) distance matrix sequence matching method. We evaluate SPOT on a publicly available dataset with 6.7-7.6 km routes driven in similar and opposing directions under various lighting conditions. The proposed algorithm demonstrates remarkable improvement over the state-of-the-art, achieving up to 91.7% recall at 100% precision in opposing viewpoint cases, while requiring less storage than all baselines tested and running faster than all but one. Moreover, the proposed method assumes no a priori knowledge of whether the viewpoint is similar or opposing, and also demonstrates competitive performance in similar viewpoint cases.  
  </ol>  
</details>  
**comments**: Accepted to ICRA 2024, project website:
  https://umautobots.github.io/spot  
  
  



## NeRF  

### [MeshLRM: Large Reconstruction Model for High-Quality Mesh](http://arxiv.org/abs/2404.12385)  
Xinyue Wei, Kai Zhang, Sai Bi, Hao Tan, Fujun Luan, Valentin Deschaintre, Kalyan Sunkavalli, Hao Su, Zexiang Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose MeshLRM, a novel LRM-based approach that can reconstruct a high-quality mesh from merely four input images in less than one second. Different from previous large reconstruction models (LRMs) that focus on NeRF-based reconstruction, MeshLRM incorporates differentiable mesh extraction and rendering within the LRM framework. This allows for end-to-end mesh reconstruction by fine-tuning a pre-trained NeRF LRM with mesh rendering. Moreover, we improve the LRM architecture by simplifying several complex designs in previous LRMs. MeshLRM's NeRF initialization is sequentially trained with low- and high-resolution images; this new LRM training strategy enables significantly faster convergence and thereby leads to better quality with less compute. Our approach achieves state-of-the-art mesh reconstruction from sparse-view inputs and also allows for many downstream applications, including text-to-3D and single-image-to-3D generation. Project page: https://sarahweiii.github.io/meshlrm/  
  </ol>  
</details>  
  
### [AG-NeRF: Attention-guided Neural Radiance Fields for Multi-height Large-scale Outdoor Scene Rendering](http://arxiv.org/abs/2404.11897)  
Jingfeng Guo, Xiaohan Zhang, Baozhu Zhao, Qi Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Existing neural radiance fields (NeRF)-based novel view synthesis methods for large-scale outdoor scenes are mainly built on a single altitude. Moreover, they often require a priori camera shooting height and scene scope, leading to inefficient and impractical applications when camera altitude changes. In this work, we propose an end-to-end framework, termed AG-NeRF, and seek to reduce the training cost of building good reconstructions by synthesizing free-viewpoint images based on varying altitudes of scenes. Specifically, to tackle the detail variation problem from low altitude (drone-level) to high altitude (satellite-level), a source image selection method and an attention-based feature fusion approach are developed to extract and fuse the most relevant features of target view from multi-height images for high-fidelity rendering. Extensive experiments demonstrate that AG-NeRF achieves SOTA performance on 56 Leonard and Transamerica benchmarks and only requires a half hour of training time to reach the competitive PSNR as compared to the latest BungeeNeRF.  
  </ol>  
</details>  
  
### [Cicero: Addressing Algorithmic and Architectural Bottlenecks in Neural Rendering by Radiance Warping and Memory Optimizations](http://arxiv.org/abs/2404.11852)  
Yu Feng, Zihan Liu, Jingwen Leng, Minyi Guo, Yuhao Zhu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Field (NeRF) is widely seen as an alternative to traditional physically-based rendering. However, NeRF has not yet seen its adoption in resource-limited mobile systems such as Virtual and Augmented Reality (VR/AR), because it is simply extremely slow. On a mobile Volta GPU, even the state-of-the-art NeRF models generally execute only at 0.8 FPS. We show that the main performance bottlenecks are both algorithmic and architectural. We introduce, CICERO, to tame both forms of inefficiencies. We first introduce two algorithms, one fundamentally reduces the amount of work any NeRF model has to execute, and the other eliminates irregular DRAM accesses. We then describe an on-chip data layout strategy that eliminates SRAM bank conflicts. A pure software implementation of CICERO offers an 8.0x speed-up and 7.9x energy saving over a mobile Volta GPU. When compared to a baseline with a dedicated DNN accelerator, our speed-up and energy reduction increase to 28.2x and 37.8x, respectively - all with minimal quality loss (less than 1.0 dB peak signal-to-noise ratio reduction).  
  </ol>  
</details>  
  
  



