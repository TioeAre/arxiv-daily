<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#MP-SfM:-Monocular-Surface-Priors-for-Robust-Structure-from-Motion>MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#VISUALCENT:-Visual-Human-Analysis-using-Dynamic-Centroid-Representation>VISUALCENT: Visual Human Analysis using Dynamic Centroid Representation</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Mitigating-Modality-Bias-in-Multi-modal-Entity-Alignment-from-a-Causal-Perspective>Mitigating Modality Bias in Multi-modal Entity Alignment from a Causal Perspective</a></li>
        <li><a href=#Dynamic-Arthroscopic-Navigation-System-for-Anterior-Cruciate-Ligament-Reconstruction-Based-on-Multi-level-Memory-Architecture>Dynamic Arthroscopic Navigation System for Anterior Cruciate Ligament Reconstruction Based on Multi-level Memory Architecture</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Joint-Optimization-of-Neural-Radiance-Fields-and-Continuous-Camera-Motion-from-a-Monocular-Video>Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video</a></li>
        <li><a href=#Beyond-Physical-Reach:-Comparing-Head--and-Cane-Mounted-Cameras-for-Last-Mile-Navigation-by-Blind-Users>Beyond Physical Reach: Comparing Head- and Cane-Mounted Cameras for Last-Mile Navigation by Blind Users</a></li>
        <li><a href=#IM-Portrait:-Learning-3D-aware-Video-Diffusion-for-PhotorealisticTalking-Heads-from-Monocular-Videos>IM-Portrait: Learning 3D-aware Video Diffusion for PhotorealisticTalking Heads from Monocular Videos</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion](http://arxiv.org/abs/2504.20040)  
Zador Pataki, Paul-Edouard Sarlin, Johannes L. Schönberger, Marc Pollefeys  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    While Structure-from-Motion (SfM) has seen much progress over the years, state-of-the-art systems are prone to failure when facing extreme viewpoint changes in low-overlap, low-parallax or high-symmetry scenarios. Because capturing images that avoid these pitfalls is challenging, this severely limits the wider use of SfM, especially by non-expert users. We overcome these limitations by augmenting the classical SfM paradigm with monocular depth and normal priors inferred by deep neural networks. Thanks to a tight integration of monocular and multi-view constraints, our approach significantly outperforms existing ones under extreme viewpoint changes, while maintaining strong performance in standard conditions. We also show that monocular priors can help reject faulty associations due to symmetries, which is a long-standing problem for SfM. This makes our approach the first capable of reliably reconstructing challenging indoor environments from few images. Through principled uncertainty propagation, it is robust to errors in the priors, can handle priors inferred by different models with little tuning, and will thus easily benefit from future progress in monocular depth and normal estimation. Our code is publicly available at https://github.com/cvg/mpsfm.  
  </ol>  
</details>  
**comments**: CVPR 2025  
  
  



## Keypoint Detection  

### [VISUALCENT: Visual Human Analysis using Dynamic Centroid Representation](http://arxiv.org/abs/2504.19032)  
Niaz Ahmad, Youngmoon Lee, Guanghui Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce VISUALCENT, a unified human pose and instance segmentation framework to address generalizability and scalability limitations to multi person visual human analysis. VISUALCENT leverages centroid based bottom up keypoint detection paradigm and uses Keypoint Heatmap incorporating Disk Representation and KeyCentroid to identify the optimal keypoint coordinates. For the unified segmentation task, an explicit keypoint is defined as a dynamic centroid called MaskCentroid to swiftly cluster pixels to specific human instance during rapid changes in human body movement or significantly occluded environment. Experimental results on COCO and OCHuman datasets demonstrate VISUALCENTs accuracy and real time performance advantages, outperforming existing methods in mAP scores and execution frame rate per second. The implementation is available on the project page.  
  </ol>  
</details>  
  
  



## Image Matching  

### [Mitigating Modality Bias in Multi-modal Entity Alignment from a Causal Perspective](http://arxiv.org/abs/2504.19458)  
Taoyu Su, Jiawei Sheng, Duohe Ma, Xiaodong Li, Juwei Yue, Mengxiao Song, Yingkai Tang, Tingwen Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Multi-Modal Entity Alignment (MMEA) aims to retrieve equivalent entities from different Multi-Modal Knowledge Graphs (MMKGs), a critical information retrieval task. Existing studies have explored various fusion paradigms and consistency constraints to improve the alignment of equivalent entities, while overlooking that the visual modality may not always contribute positively. Empirically, entities with low-similarity images usually generate unsatisfactory performance, highlighting the limitation of overly relying on visual features. We believe the model can be biased toward the visual modality, leading to a shortcut image-matching task. To address this, we propose a counterfactual debiasing framework for MMEA, termed CDMEA, which investigates visual modality bias from a causal perspective. Our approach aims to leverage both visual and graph modalities to enhance MMEA while suppressing the direct causal effect of the visual modality on model predictions. By estimating the Total Effect (TE) of both modalities and excluding the Natural Direct Effect (NDE) of the visual modality, we ensure that the model predicts based on the Total Indirect Effect (TIE), effectively utilizing both modalities and reducing visual modality bias. Extensive experiments on 9 benchmark datasets show that CDMEA outperforms 14 state-of-the-art methods, especially in low-similarity, high-noise, and low-resource data scenarios.  
  </ol>  
</details>  
**comments**: Accepted by SIGIR 2025, 11 pages, 10 figures, 4 tables,  
  
### [Dynamic Arthroscopic Navigation System for Anterior Cruciate Ligament Reconstruction Based on Multi-level Memory Architecture](http://arxiv.org/abs/2504.19398)  
Shuo Wang, Weili Shi, Shuai Yang, Jiahao Cui, Qinwei Guo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper presents a dynamic arthroscopic navigation system based on multi-level memory architecture for anterior cruciate ligament (ACL) reconstruction surgery. The system extends our previously proposed markerless navigation method from static image matching to dynamic video sequence tracking. By integrating the Atkinson-Shiffrin memory model's three-level architecture (sensory memory, working memory, and long-term memory), our system maintains continuous tracking of the femoral condyle throughout the surgical procedure, providing stable navigation support even in complex situations involving viewpoint changes, instrument occlusion, and tissue deformation. Unlike existing methods, our system operates in real-time on standard arthroscopic equipment without requiring additional tracking hardware, achieving 25.3 FPS with a latency of only 39.5 ms, representing a 3.5-fold improvement over our previous static system. For extended sequences (1000 frames), the dynamic system maintained an error of 5.3 plus-minus 1.5 pixels, compared to the static system's 12.6 plus-minus 3.7 pixels - an improvement of approximately 45 percent. For medium-length sequences (500 frames) and short sequences (100 frames), the system achieved approximately 35 percent and 19 percent accuracy improvements, respectively. Experimental results demonstrate the system overcomes limitations of traditional static matching methods, providing new technical support for improving surgical precision in ACL reconstruction.  
  </ol>  
</details>  
**comments**: 28 pages, 13 figures  
  
  



## NeRF  

### [Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video](http://arxiv.org/abs/2504.19819)  
Hoang Chuong Nguyen, Wei Mao, Jose M. Alvarez, Miaomiao Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) has demonstrated its superior capability to represent 3D geometry but require accurately precomputed camera poses during training. To mitigate this requirement, existing methods jointly optimize camera poses and NeRF often relying on good pose initialisation or depth priors. However, these approaches struggle in challenging scenarios, such as large rotations, as they map each camera to a world coordinate system. We propose a novel method that eliminates prior dependencies by modeling continuous camera motions as time-dependent angular velocity and velocity. Relative motions between cameras are learned first via velocity integration, while camera poses can be obtained by aggregating such relative motions up to a world coordinate system defined at a single time step within the video. Specifically, accurate continuous camera movements are learned through a time-dependent NeRF, which captures local scene geometry and motion by training from neighboring frames for each time step. The learned motions enable fine-tuning the NeRF to represent the full scene geometry. Experiments on Co3D and Scannet show our approach achieves superior camera pose and depth estimation and comparable novel-view synthesis performance compared to state-of-the-art methods. Our code is available at https://github.com/HoangChuongNguyen/cope-nerf.  
  </ol>  
</details>  
  
### [Beyond Physical Reach: Comparing Head- and Cane-Mounted Cameras for Last-Mile Navigation by Blind Users](http://arxiv.org/abs/2504.19345)  
Apurv Varshney, Lucas Nadolskis, Tobias Höllerer, Michael Beyeler  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Blind individuals face persistent challenges in last-mile navigation, including locating entrances, identifying obstacles, and navigating complex or cluttered spaces. Although wearable cameras are increasingly used in assistive systems, there has been no systematic, vantage-focused comparison to guide their design. This paper addresses that gap through a two-part investigation. First, we surveyed ten experienced blind cane users, uncovering navigation strategies, pain points, and technology preferences. Participants stressed the importance of multi-sensory integration, destination-focused travel, and assistive tools that complement (rather than replace) the cane's tactile utility. Second, we conducted controlled data collection with a blind participant navigating five real-world environments using synchronized head- and cane-mounted cameras, isolating vantage placement as the primary variable. To assess how each vantage supports spatial perception, we evaluated SLAM performance (for localization and mapping) and NeRF-based 3D reconstruction (for downstream scene understanding). Head-mounted sensors delivered superior localization accuracy, while cane-mounted views offered broader ground-level coverage and richer environmental reconstructions. A combined (head+cane) configuration consistently outperformed both. These results highlight the complementary strengths of different sensor placements and offer actionable guidance for developing hybrid navigation aids that are perceptive, robust, and user-aligned.  
  </ol>  
</details>  
  
### [IM-Portrait: Learning 3D-aware Video Diffusion for PhotorealisticTalking Heads from Monocular Videos](http://arxiv.org/abs/2504.19165)  
Yuan Li, Ziqian Bai, Feitong Tan, Zhaopeng Cui, Sean Fanello, Yinda Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose a novel 3D-aware diffusion-based method for generating photorealistic talking head videos directly from a single identity image and explicit control signals (e.g., expressions). Our method generates Multiplane Images (MPIs) that ensure geometric consistency, making them ideal for immersive viewing experiences like binocular videos for VR headsets. Unlike existing methods that often require a separate stage or joint optimization to reconstruct a 3D representation (such as NeRF or 3D Gaussians), our approach directly generates the final output through a single denoising process, eliminating the need for post-processing steps to render novel views efficiently. To effectively learn from monocular videos, we introduce a training mechanism that reconstructs the output MPI randomly in either the target or the reference camera space. This approach enables the model to simultaneously learn sharp image details and underlying 3D information. Extensive experiments demonstrate the effectiveness of our method, which achieves competitive avatar quality and novel-view rendering capabilities, even without explicit 3D reconstruction or high-quality multi-view training data.  
  </ol>  
</details>  
**comments**: CVPR2025; project page:
  https://y-u-a-n-l-i.github.io/projects/IM-Portrait/  
  
  



