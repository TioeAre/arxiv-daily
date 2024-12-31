<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#MambaVO:-Deep-Visual-Odometry-Based-on-Sequential-Matching-Refinement-and-Training-Smoothing>MambaVO: Deep Visual Odometry Based on Sequential Matching Refinement and Training Smoothing</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#KeyGS:-A-Keyframe-Centric-Gaussian-Splatting-Method-for-Monocular-Image-Sequences>KeyGS: A Keyframe-Centric Gaussian Splatting Method for Monocular Image Sequences</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#GSplatLoc:-Ultra-Precise-Camera-Localization-via-3D-Gaussian-Splatting>GSplatLoc: Ultra-Precise Camera Localization via 3D Gaussian Splatting</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Towards-Real-Time-2D-Mapping:-Harnessing-Drones,-AI,-and-Computer-Vision-for-Advanced-Insights>Towards Real-Time 2D Mapping: Harnessing Drones, AI, and Computer Vision for Advanced Insights</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Bringing-Objects-to-Life:-4D-generation-from-3D-objects>Bringing Objects to Life: 4D generation from 3D objects</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [MambaVO: Deep Visual Odometry Based on Sequential Matching Refinement and Training Smoothing](http://arxiv.org/abs/2412.20082)  
Shuo Wang, Wanting Li, Yongcai Wang, Zhaoxin Fan, Zhe Huang, Xudong Cai, Jian Zhao, Deying Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Deep visual odometry has demonstrated great advancements by learning-to-optimize technology. This approach heavily relies on the visual matching across frames. However, ambiguous matching in challenging scenarios leads to significant errors in geometric modeling and bundle adjustment optimization, which undermines the accuracy and robustness of pose estimation. To address this challenge, this paper proposes MambaVO, which conducts robust initialization, Mamba-based sequential matching refinement, and smoothed training to enhance the matching quality and improve the pose estimation in deep visual odometry. Specifically, when a new frame is received, it is matched with the closest keyframe in the maintained Point-Frame Graph (PFG) via the semi-dense based Geometric Initialization Module (GIM). Then the initialized PFG is processed by a proposed Geometric Mamba Module (GMM), which exploits the matching features to refine the overall inter-frame pixel-to-pixel matching. The refined PFG is finally processed by deep BA to optimize the poses and the map. To deal with the gradient variance, a Trending-Aware Penalty (TAP) is proposed to smooth training by balancing the pose loss and the matching loss to enhance convergence and stability. A loop closure module is finally applied to enable MambaVO++. On public benchmarks, MambaVO and MambaVO++ demonstrate SOTA accuracy performance, while ensuring real-time running performance with low GPU memory requirement. Codes will be publicly available.  
  </ol>  
</details>  
  
  



## SFM  

### [KeyGS: A Keyframe-Centric Gaussian Splatting Method for Monocular Image Sequences](http://arxiv.org/abs/2412.20767)  
Keng-Wei Chang, Zi-Ming Wang, Shang-Hong Lai  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Reconstructing high-quality 3D models from sparse 2D images has garnered significant attention in computer vision. Recently, 3D Gaussian Splatting (3DGS) has gained prominence due to its explicit representation with efficient training speed and real-time rendering capabilities. However, existing methods still heavily depend on accurate camera poses for reconstruction. Although some recent approaches attempt to train 3DGS models without the Structure-from-Motion (SfM) preprocessing from monocular video datasets, these methods suffer from prolonged training times, making them impractical for many applications.   In this paper, we present an efficient framework that operates without any depth or matching model. Our approach initially uses SfM to quickly obtain rough camera poses within seconds, and then refines these poses by leveraging the dense representation in 3DGS. This framework effectively addresses the issue of long training times. Additionally, we integrate the densification process with joint refinement and propose a coarse-to-fine frequency-aware densification to reconstruct different levels of details. This approach prevents camera pose estimation from being trapped in local minima or drifting due to high-frequency signals. Our method significantly reduces training time from hours to minutes while achieving more accurate novel view synthesis and camera pose estimation compared to previous methods.  
  </ol>  
</details>  
**comments**: AAAI 2025  
  
  



## Visual Localization  

### [GSplatLoc: Ultra-Precise Camera Localization via 3D Gaussian Splatting](http://arxiv.org/abs/2412.20056)  
[[code](https://github.com/atticuszeller/gsplatloc)]  
Atticus J. Zeller  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present GSplatLoc, a camera localization method that leverages the differentiable rendering capabilities of 3D Gaussian splatting for ultra-precise pose estimation. By formulating pose estimation as a gradient-based optimization problem that minimizes discrepancies between rendered depth maps from a pre-existing 3D Gaussian scene and observed depth images, GSplatLoc achieves translational errors within 0.01 cm and near-zero rotational errors on the Replica dataset - significantly outperforming existing methods. Evaluations on the Replica and TUM RGB-D datasets demonstrate the method's robustness in challenging indoor environments with complex camera motions. GSplatLoc sets a new benchmark for localization in dense mapping, with important implications for applications requiring accurate real-time localization, such as robotics and augmented reality.  
  </ol>  
</details>  
**comments**: 11 pages, 2 figures. Code available at
  https://github.com/AtticusZeller/GsplatLoc  
  
  



## Image Matching  

### [Towards Real-Time 2D Mapping: Harnessing Drones, AI, and Computer Vision for Advanced Insights](http://arxiv.org/abs/2412.20210)  
Bharath Kumar Agnur  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Real-time 2D mapping is a vital tool in aerospace and defense, where accurate and timely geographic data is essential for operations like surveillance, reconnaissance, and target tracking. This project introduces a cutting-edge mapping system that integrates drone imagery with machine learning and computer vision to address challenges in processing speed, accuracy, and adaptability to diverse terrains. By automating feature detection, image matching, and stitching, the system generates seamless, high-resolution maps with minimal delay, providing strategic advantages in defense operations.   Implemented in Python, the system leverages OpenCV for image processing, NumPy for efficient computations, and Concurrent.futures for parallel processing. ORB (Oriented FAST and Rotated BRIEF) handles feature detection, while FLANN (Fast Library for Approximate Nearest Neighbors) ensures precise keypoint matching. Homography transformations align overlapping images, creating distortion-free maps in real time. This automated approach eliminates manual intervention, enabling live updates critical in dynamic environments. Designed for adaptability, the system performs well under varying light conditions and rugged terrains, making it highly effective in aerospace and defense scenarios. Testing demonstrates significant improvements in speed and accuracy compared to traditional methods, enhancing situational awareness and decision-making. This scalable solution leverages advanced technologies to deliver reliable, actionable data for mission-critical operations.  
  </ol>  
</details>  
**comments**: 7 pages, 7 figures, 1 table  
  
  



## NeRF  

### [Bringing Objects to Life: 4D generation from 3D objects](http://arxiv.org/abs/2412.20422)  
Ohad Rahamim, Ori Malca, Dvir Samuel, Gal Chechik  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in generative modeling now enable the creation of 4D content (moving 3D objects) controlled with text prompts. 4D generation has large potential in applications like virtual worlds, media, and gaming, but existing methods provide limited control over the appearance and geometry of generated content. In this work, we introduce a method for animating user-provided 3D objects by conditioning on textual prompts to guide 4D generation, enabling custom animations while maintaining the identity of the original object. We first convert a 3D mesh into a ``static" 4D Neural Radiance Field (NeRF) that preserves the visual attributes of the input object. Then, we animate the object using an Image-to-Video diffusion model driven by text. To improve motion realism, we introduce an incremental viewpoint selection protocol for sampling perspectives to promote lifelike movement and a masked Score Distillation Sampling (SDS) loss, which leverages attention maps to focus optimization on relevant regions. We evaluate our model in terms of temporal coherence, prompt adherence, and visual fidelity and find that our method outperforms baselines that are based on other approaches, achieving up to threefold improvements in identity preservation measured using LPIPS scores, and effectively balancing visual quality with dynamic content.  
  </ol>  
</details>  
  
  



