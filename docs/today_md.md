<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#MPVO:-Motion-Prior-based-Visual-Odometry-for-PointGoal-Navigation>MPVO: Motion-Prior based Visual Odometry for PointGoal Navigation</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#The-Impact-of-Semi-Supervised-Learning-on-Line-Segment-Detection>The Impact of Semi-Supervised Learning on Line Segment Detection</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Planar-Reflection-Aware-Neural-Radiance-Fields>Planar Reflection-Aware Neural Radiance Fields</a></li>
        <li><a href=#GANESH:-Generalizable-NeRF-for-Lensless-Imaging>GANESH: Generalizable NeRF for Lensless Imaging</a></li>
        <li><a href=#SuperQ-GRASP:-Superquadrics-based-Grasp-Pose-Estimation-on-Larger-Objects-for-Mobile-Manipulation>SuperQ-GRASP: Superquadrics-based Grasp Pose Estimation on Larger Objects for Mobile-Manipulation</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [MPVO: Motion-Prior based Visual Odometry for PointGoal Navigation](http://arxiv.org/abs/2411.04796)  
Sayan Paul, Ruddra dev Roychoudhury, Brojeshwar Bhowmick  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual odometry (VO) is essential for enabling accurate point-goal navigation of embodied agents in indoor environments where GPS and compass sensors are unreliable and inaccurate. However, traditional VO methods face challenges in wide-baseline scenarios, where fast robot motions and low frames per second (FPS) during inference hinder their performance, leading to drift and catastrophic failures in point-goal navigation. Recent deep-learned VO methods show robust performance but suffer from sample inefficiency during training; hence, they require huge datasets and compute resources. So, we propose a robust and sample-efficient VO pipeline based on motion priors available while an agent is navigating an environment. It consists of a training-free action-prior based geometric VO module that estimates a coarse relative pose which is further consumed as a motion prior by a deep-learned VO model, which finally produces a fine relative pose to be used by the navigation policy. This strategy helps our pipeline achieve up to 2x sample efficiency during training and demonstrates superior accuracy and robustness in point-goal navigation tasks compared to state-of-the-art VO method(s). Realistic indoor environments of the Gibson dataset is used in the AI-Habitat simulator to evaluate the proposed approach using navigation metrics (like success/SPL) and pose metrics (like RPE/ATE). We hope this method further opens a direction of work where motion priors from various sources can be utilized to improve VO estimates and achieve better results in embodied navigation tasks.  
  </ol>  
</details>  
**comments**: Accepted in 50SFM Workshop of the 18th European Conference on
  Computer Vision (ECCV) 2024  
  
  



## Image Matching  

### [The Impact of Semi-Supervised Learning on Line Segment Detection](http://arxiv.org/abs/2411.04596)  
[[code](https://github.com/jo6815en/semi-lines)]  
Johanna Engman, Karl Åström, Magnus Oskarsson  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper we present a method for line segment detection in images, based on a semi-supervised framework. Leveraging the use of a consistency loss based on differently augmented and perturbed unlabeled images with a small amount of labeled data, we show comparable results to fully supervised methods. This opens up application scenarios where annotation is difficult or expensive, and for domain specific adaptation of models. We are specifically interested in real-time and online applications, and investigate small and efficient learning backbones. Our method is to our knowledge the first to target line detection using modern state-of-the-art methodologies for semi-supervised learning. We test the method on both standard benchmarks and domain specific scenarios for forestry applications, showing the tractability of the proposed method.  
  </ol>  
</details>  
**comments**: 9 pages, 6 figures, 7 tables  
  
  



## NeRF  

### [Planar Reflection-Aware Neural Radiance Fields](http://arxiv.org/abs/2411.04984)  
Chen Gao, Yipeng Wang, Changil Kim, Jia-Bin Huang, Johannes Kopf  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) have demonstrated exceptional capabilities in reconstructing complex scenes with high fidelity. However, NeRF's view dependency can only handle low-frequency reflections. It falls short when handling complex planar reflections, often interpreting them as erroneous scene geometries and leading to duplicated and inaccurate scene representations. To address this challenge, we introduce a reflection-aware NeRF that jointly models planar reflectors, such as windows, and explicitly casts reflected rays to capture the source of the high-frequency reflections. We query a single radiance field to render the primary color and the source of the reflection. We propose a sparse edge regularization to help utilize the true sources of reflections for rendering planar reflections rather than creating a duplicate along the primary ray at the same depth. As a result, we obtain accurate scene geometry. Rendering along the primary ray results in a clean, reflection-free view, while explicitly rendering along the reflected ray allows us to reconstruct highly detailed reflections. Our extensive quantitative and qualitative evaluations of real-world datasets demonstrate our method's enhanced performance in accurately handling reflections.  
  </ol>  
</details>  
  
### [GANESH: Generalizable NeRF for Lensless Imaging](http://arxiv.org/abs/2411.04810)  
Rakesh Raj Madavan, Akshat Kaimal, Badhrinarayanan K V, Vinayak Gupta, Rohit Choudhary, Chandrakala Shanmuganathan, Kaushik Mitra  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Lensless imaging offers a significant opportunity to develop ultra-compact cameras by removing the conventional bulky lens system. However, without a focusing element, the sensor's output is no longer a direct image but a complex multiplexed scene representation. Traditional methods have attempted to address this challenge by employing learnable inversions and refinement models, but these methods are primarily designed for 2D reconstruction and do not generalize well to 3D reconstruction. We introduce GANESH, a novel framework designed to enable simultaneous refinement and novel view synthesis from multi-view lensless images. Unlike existing methods that require scene-specific training, our approach supports on-the-fly inference without retraining on each scene. Moreover, our framework allows us to tune our model to specific scenes, enhancing the rendering and refinement quality. To facilitate research in this area, we also present the first multi-view lensless dataset, LenslessScenes. Extensive experiments demonstrate that our method outperforms current approaches in reconstruction accuracy and refinement quality. Code and video results are available at https://rakesh-123-cryp.github.io/Rakesh.github.io/  
  </ol>  
</details>  
  
### [SuperQ-GRASP: Superquadrics-based Grasp Pose Estimation on Larger Objects for Mobile-Manipulation](http://arxiv.org/abs/2411.04386)  
Xun Tu, Karthik Desingh  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Grasp planning and estimation have been a longstanding research problem in robotics, with two main approaches to find graspable poses on the objects: 1) geometric approach, which relies on 3D models of objects and the gripper to estimate valid grasp poses, and 2) data-driven, learning-based approach, with models trained to identify grasp poses from raw sensor observations. The latter assumes comprehensive geometric coverage during the training phase. However, the data-driven approach is typically biased toward tabletop scenarios and struggle to generalize to out-of-distribution scenarios with larger objects (e.g. chair). Additionally, raw sensor data (e.g. RGB-D data) from a single view of these larger objects is often incomplete and necessitates additional observations. In this paper, we take a geometric approach, leveraging advancements in object modeling (e.g. NeRF) to build an implicit model by taking RGB images from views around the target object. This model enables the extraction of explicit mesh model while also capturing the visual appearance from novel viewpoints that is useful for perception tasks like object detection and pose estimation. We further decompose the NeRF-reconstructed 3D mesh into superquadrics (SQs) -- parametric geometric primitives, each mapped to a set of precomputed grasp poses, allowing grasp composition on the target object based on these primitives. Our proposed pipeline overcomes the problems: a) noisy depth and incomplete view of the object, with a modeling step, and b) generalization to objects of any size. For more qualitative results, refer to the supplementary video and webpage https://bit.ly/3ZrOanU  
  </ol>  
</details>  
**comments**: 8 pages, 7 figures, submitted to ICRA 2025 for review  
  
  



