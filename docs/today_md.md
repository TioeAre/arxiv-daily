<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Revisit-Anything:-Visual-Place-Recognition-via-Image-Segment-Retrieval>Revisit Anything: Visual Place Recognition via Image Segment Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#SKT:-Integrating-State-Aware-Keypoint-Trajectories-with-Vision-Language-Models-for-Robotic-Garment-Manipulation>SKT: Integrating State-Aware Keypoint Trajectories with Vision-Language Models for Robotic Garment Manipulation</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#LightAvatar:-Efficient-Head-Avatar-as-Dynamic-Neural-Light-Field>LightAvatar: Efficient Head Avatar as Dynamic Neural Light Field</a></li>
        <li><a href=#Deblur-e-NeRF:-NeRF-from-Motion-Blurred-Events-under-High-speed-or-Low-light-Conditions>Deblur e-NeRF: NeRF from Motion-Blurred Events under High-speed or Low-light Conditions</a></li>
        <li><a href=#Neural-Implicit-Representation-for-Highly-Dynamic-LiDAR-Mapping-and-Odometry>Neural Implicit Representation for Highly Dynamic LiDAR Mapping and Odometry</a></li>
        <li><a href=#TFS-NeRF:-Template-Free-NeRF-for-Semantic-3D-Reconstruction-of-Dynamic-Scene>TFS-NeRF: Template-Free NeRF for Semantic 3D Reconstruction of Dynamic Scene</a></li>
        <li><a href=#SeaSplat:-Representing-Underwater-Scenes-with-3D-Gaussian-Splatting-and-a-Physically-Grounded-Image-Formation-Model>SeaSplat: Representing Underwater Scenes with 3D Gaussian Splatting and a Physically Grounded Image Formation Model</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Revisit Anything: Visual Place Recognition via Image Segment Retrieval](http://arxiv.org/abs/2409.18049)  
[[code](https://github.com/anyloc/revisit-anything)]  
Kartik Garg, Sai Shubodh Puligilla, Shishir Kolathaya, Madhava Krishna, Sourav Garg  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Accurately recognizing a revisited place is crucial for embodied agents to localize and navigate. This requires visual representations to be distinct, despite strong variations in camera viewpoint and scene appearance. Existing visual place recognition pipelines encode the "whole" image and search for matches. This poses a fundamental challenge in matching two images of the same place captured from different camera viewpoints: "the similarity of what overlaps can be dominated by the dissimilarity of what does not overlap". We address this by encoding and searching for "image segments" instead of the whole images. We propose to use open-set image segmentation to decompose an image into `meaningful' entities (i.e., things and stuff). This enables us to create a novel image representation as a collection of multiple overlapping subgraphs connecting a segment with its neighboring segments, dubbed SuperSegment. Furthermore, to efficiently encode these SuperSegments into compact vector representations, we propose a novel factorized representation of feature aggregation. We show that retrieving these partial representations leads to significantly higher recognition recall than the typical whole image based retrieval. Our segments-based approach, dubbed SegVLAD, sets a new state-of-the-art in place recognition on a diverse selection of benchmark datasets, while being applicable to both generic and task-specialized image encoders. Finally, we demonstrate the potential of our method to ``revisit anything'' by evaluating our method on an object instance retrieval task, which bridges the two disparate areas of research: visual place recognition and object-goal navigation, through their common aim of recognizing goal objects specific to a place. Source code: https://github.com/AnyLoc/Revisit-Anything.  
  </ol>  
</details>  
**comments**: Presented at ECCV 2024; Includes supplementary; 29 pages; 8 figures  
  
  



## Keypoint Detection  

### [SKT: Integrating State-Aware Keypoint Trajectories with Vision-Language Models for Robotic Garment Manipulation](http://arxiv.org/abs/2409.18082)  
Xin Li, Siyuan Huang, Qiaojun Yu, Zhengkai Jiang, Ce Hao, Yimeng Zhu, Hongsheng Li, Peng Gao, Cewu Lu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Automating garment manipulation poses a significant challenge for assistive robotics due to the diverse and deformable nature of garments. Traditional approaches typically require separate models for each garment type, which limits scalability and adaptability. In contrast, this paper presents a unified approach using vision-language models (VLMs) to improve keypoint prediction across various garment categories. By interpreting both visual and semantic information, our model enables robots to manage different garment states with a single model. We created a large-scale synthetic dataset using advanced simulation techniques, allowing scalable training without extensive real-world data. Experimental results indicate that the VLM-based method significantly enhances keypoint detection accuracy and task success rates, providing a more flexible and general solution for robotic garment manipulation. In addition, this research also underscores the potential of VLMs to unify various garment manipulation tasks within a single framework, paving the way for broader applications in home automation and assistive robotics for future.  
  </ol>  
</details>  
  
  



## NeRF  

### [LightAvatar: Efficient Head Avatar as Dynamic Neural Light Field](http://arxiv.org/abs/2409.18057)  
[[code](https://github.com/mingsun-tse/lightavatar-tensorflow)]  
Huan Wang, Feitong Tan, Ziqian Bai, Yinda Zhang, Shichen Liu, Qiangeng Xu, Menglei Chai, Anish Prabhu, Rohit Pandey, Sean Fanello, Zeng Huang, Yun Fu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent works have shown that neural radiance fields (NeRFs) on top of parametric models have reached SOTA quality to build photorealistic head avatars from a monocular video. However, one major limitation of the NeRF-based avatars is the slow rendering speed due to the dense point sampling of NeRF, preventing them from broader utility on resource-constrained devices. We introduce LightAvatar, the first head avatar model based on neural light fields (NeLFs). LightAvatar renders an image from 3DMM parameters and a camera pose via a single network forward pass, without using mesh or volume rendering. The proposed approach, while being conceptually appealing, poses a significant challenge towards real-time efficiency and training stability. To resolve them, we introduce dedicated network designs to obtain proper representations for the NeLF model and maintain a low FLOPs budget. Meanwhile, we tap into a distillation-based training strategy that uses a pretrained avatar model as teacher to synthesize abundant pseudo data for training. A warping field network is introduced to correct the fitting error in the real data so that the model can learn better. Extensive experiments suggest that our method can achieve new SOTA image quality quantitatively or qualitatively, while being significantly faster than the counterparts, reporting 174.1 FPS (512x512 resolution) on a consumer-grade GPU (RTX3090) with no customized optimization.  
  </ol>  
</details>  
**comments**: Appear in ECCV'24 CADL Workshop. Code:
  https://github.com/MingSun-Tse/LightAvatar-TensorFlow  
  
### [Deblur e-NeRF: NeRF from Motion-Blurred Events under High-speed or Low-light Conditions](http://arxiv.org/abs/2409.17988)  
Weng Fei Low, Gim Hee Lee  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The stark contrast in the design philosophy of an event camera makes it particularly ideal for operating under high-speed, high dynamic range and low-light conditions, where standard cameras underperform. Nonetheless, event cameras still suffer from some amount of motion blur, especially under these challenging conditions, in contrary to what most think. This is attributed to the limited bandwidth of the event sensor pixel, which is mostly proportional to the light intensity. Thus, to ensure that event cameras can truly excel in such conditions where it has an edge over standard cameras, it is crucial to account for event motion blur in downstream applications, especially reconstruction. However, none of the recent works on reconstructing Neural Radiance Fields (NeRFs) from events, nor event simulators, have considered the full effects of event motion blur. To this end, we propose, Deblur e-NeRF, a novel method to directly and effectively reconstruct blur-minimal NeRFs from motion-blurred events generated under high-speed motion or low-light conditions. The core component of this work is a physically-accurate pixel bandwidth model proposed to account for event motion blur under arbitrary speed and lighting conditions. We also introduce a novel threshold-normalized total variation loss to improve the regularization of large textureless patches. Experiments on real and novel realistically simulated sequences verify our effectiveness. Our code, event simulator and synthetic event dataset will be open-sourced.  
  </ol>  
</details>  
**comments**: Accepted to ECCV 2024. Project website is accessible at
  https://wengflow.github.io/deblur-e-nerf. arXiv admin note: text overlap with
  arXiv:2006.07722 by other authors  
  
### [Neural Implicit Representation for Highly Dynamic LiDAR Mapping and Odometry](http://arxiv.org/abs/2409.17729)  
Qi Zhang, He Wang, Ru Li, Wenbin Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in Simultaneous Localization and Mapping (SLAM) have increasingly highlighted the robustness of LiDAR-based techniques. At the same time, Neural Radiance Fields (NeRF) have introduced new possibilities for 3D scene reconstruction, exemplified by SLAM systems. Among these, NeRF-LOAM has shown notable performance in NeRF-based SLAM applications. However, despite its strengths, these systems often encounter difficulties in dynamic outdoor environments due to their inherent static assumptions. To address these limitations, this paper proposes a novel method designed to improve reconstruction in highly dynamic outdoor scenes. Based on NeRF-LOAM, the proposed approach consists of two primary components. First, we separate the scene into static background and dynamic foreground. By identifying and excluding dynamic elements from the mapping process, this segmentation enables the creation of a dense 3D map that accurately represents the static background only. The second component extends the octree structure to support multi-resolution representation. This extension not only enhances reconstruction quality but also aids in the removal of dynamic objects identified by the first module. Additionally, Fourier feature encoding is applied to the sampled points, capturing high-frequency information and leading to more complete reconstruction results. Evaluations on various datasets demonstrate that our method achieves more competitive results compared to current state-of-the-art approaches.  
  </ol>  
</details>  
  
### [TFS-NeRF: Template-Free NeRF for Semantic 3D Reconstruction of Dynamic Scene](http://arxiv.org/abs/2409.17459)  
Sandika Biswas, Qianyi Wu, Biplab Banerjee, Hamid Rezatofighi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Despite advancements in Neural Implicit models for 3D surface reconstruction, handling dynamic environments with arbitrary rigid, non-rigid, or deformable entities remains challenging. Many template-based methods are entity-specific, focusing on humans, while generic reconstruction methods adaptable to such dynamic scenes often require additional inputs like depth or optical flow or rely on pre-trained image features for reasonable outcomes. These methods typically use latent codes to capture frame-by-frame deformations. In contrast, some template-free methods bypass these requirements and adopt traditional LBS (Linear Blend Skinning) weights for a detailed representation of deformable object motions, although they involve complex optimizations leading to lengthy training times. To this end, as a remedy, this paper introduces TFS-NeRF, a template-free 3D semantic NeRF for dynamic scenes captured from sparse or single-view RGB videos, featuring interactions among various entities and more time-efficient than other LBS-based approaches. Our framework uses an Invertible Neural Network (INN) for LBS prediction, simplifying the training process. By disentangling the motions of multiple entities and optimizing per-entity skinning weights, our method efficiently generates accurate, semantically separable geometries. Extensive experiments demonstrate that our approach produces high-quality reconstructions of both deformable and non-deformable objects in complex interactions, with improved training efficiency compared to existing methods.  
  </ol>  
</details>  
**comments**: Accepted in NeuRIPS 2024  
  
### [SeaSplat: Representing Underwater Scenes with 3D Gaussian Splatting and a Physically Grounded Image Formation Model](http://arxiv.org/abs/2409.17345)  
Daniel Yang, John J. Leonard, Yogesh Girdhar  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce SeaSplat, a method to enable real-time rendering of underwater scenes leveraging recent advances in 3D radiance fields. Underwater scenes are challenging visual environments, as rendering through a medium such as water introduces both range and color dependent effects on image capture. We constrain 3D Gaussian Splatting (3DGS), a recent advance in radiance fields enabling rapid training and real-time rendering of full 3D scenes, with a physically grounded underwater image formation model. Applying SeaSplat to the real-world scenes from SeaThru-NeRF dataset, a scene collected by an underwater vehicle in the US Virgin Islands, and simulation-degraded real-world scenes, not only do we see increased quantitative performance on rendering novel viewpoints from the scene with the medium present, but are also able to recover the underlying true color of the scene and restore renders to be without the presence of the intervening medium. We show that the underwater image formation helps learn scene structure, with better depth maps, as well as show that our improvements maintain the significant computational improvements afforded by leveraging a 3D Gaussian representation.  
  </ol>  
</details>  
**comments**: Project page here: https://seasplat.github.io  
  
  



