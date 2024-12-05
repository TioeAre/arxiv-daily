<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#MCVO:-A-Generic-Visual-Odometry-for-Arbitrarily-Arranged-Multi-Cameras>MCVO: A Generic Visual Odometry for Arbitrarily Arranged Multi-Cameras</a></li>
        <li><a href=#An-indoor-DSO-based-ceiling-vision-odometry-system-for-indoor-industrial-environments>An indoor DSO-based ceiling-vision odometry system for indoor industrial environments</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Distillation-of-Diffusion-Features-for-Semantic-Correspondence>Distillation of Diffusion Features for Semantic Correspondence</a></li>
        <li><a href=#Composed-Image-Retrieval-for-Training-Free-Domain-Conversion>Composed Image Retrieval for Training-Free Domain Conversion</a></li>
        <li><a href=#A-Minimalistic-3D-Self-Organized-UAV-Flocking-Approach-for-Desert-Exploration>A Minimalistic 3D Self-Organized UAV Flocking Approach for Desert Exploration</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Measure-Anything:-Real-time,-Multi-stage-Vision-based-Dimensional-Measurement-using-Segment-Anything>Measure Anything: Real-time, Multi-stage Vision-based Dimensional Measurement using Segment Anything</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Appearance-Matching-Adapter-for-Exemplar-based-Semantic-Image-Synthesis>Appearance Matching Adapter for Exemplar-based Semantic Image Synthesis</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#NeRF-and-Gaussian-Splatting-SLAM-in-the-Wild>NeRF and Gaussian Splatting SLAM in the Wild</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [MCVO: A Generic Visual Odometry for Arbitrarily Arranged Multi-Cameras](http://arxiv.org/abs/2412.03146)  
Huai Yu, Junhao Wang, Yao He, Wen Yang, Gui-Song Xia  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Making multi-camera visual SLAM systems easier to set up and more robust to the environment is always one of the focuses of vision robots. Existing monocular and binocular vision SLAM systems have narrow FoV and are fragile in textureless environments with degenerated accuracy and limited robustness. Thus multi-camera SLAM systems are gaining attention because they can provide redundancy for texture degeneration with wide FoV. However, current multi-camera SLAM systems face massive data processing pressure and elaborately designed camera configurations, leading to estimation failures for arbitrarily arranged multi-camera systems. To address these problems, we propose a generic visual odometry for arbitrarily arranged multi-cameras, which can achieve metric-scale state estimation with high flexibility in the cameras' arrangement. Specifically, we first design a learning-based feature extraction and tracking framework to shift the pressure of CPU processing of multiple video streams. Then we use the rigid constraints between cameras to estimate the metric scale poses for robust SLAM system initialization. Finally, we fuse the features of the multi-cameras in the SLAM back-end to achieve robust pose estimation and online scale optimization. Additionally, multi-camera features help improve the loop detection for pose graph optimization. Experiments on KITTI-360 and MultiCamData datasets validate the robustness of our method over arbitrarily placed cameras. Compared with other stereo and multi-camera visual SLAM systems, our method obtains higher pose estimation accuracy with better generalization ability. Our codes and online demos are available at \url{https://github.com/JunhaoWang615/MCVO}  
  </ol>  
</details>  
**comments**: 8 pages, 8 figures  
  
### [An indoor DSO-based ceiling-vision odometry system for indoor industrial environments](http://arxiv.org/abs/2412.02950)  
Abdelhak Bougouffa, Emmanuel Seignez, Samir Bouaziz, Florian Gardes  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Autonomous Mobile Robots operating in indoor industrial environments require a localization system that is reliable and robust. While Visual Odometry (VO) can offer a reasonable estimation of the robot's state, traditional VO methods encounter challenges when confronted with dynamic objects in the scene. Alternatively, an upward-facing camera can be utilized to track the robot's movement relative to the ceiling, which represents a static and consistent space. We introduce in this paper Ceiling-DSO, a ceiling-vision system based on Direct Sparse Odometry (DSO). Unlike other ceiling-vision systems, Ceiling-DSO takes advantage of the versatile formulation of DSO, avoiding assumptions about observable shapes or landmarks on the ceiling. This approach ensures the method's applicability to various ceiling types. Since no publicly available dataset for ceiling-vision exists, we created a custom dataset in a real-world scenario and employed it to evaluate our approach. By adjusting DSO parameters, we identified the optimal fit for online pose estimation, resulting in acceptable error rates compared to ground truth. We provide in this paper a qualitative and quantitative analysis of the obtained results.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Distillation of Diffusion Features for Semantic Correspondence](http://arxiv.org/abs/2412.03512)  
Frank Fundel, Johannes Schusterbauer, Vincent Tao Hu, Björn Ommer  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Semantic correspondence, the task of determining relationships between different parts of images, underpins various applications including 3D reconstruction, image-to-image translation, object tracking, and visual place recognition. Recent studies have begun to explore representations learned in large generative image models for semantic correspondence, demonstrating promising results. Building on this progress, current state-of-the-art methods rely on combining multiple large models, resulting in high computational demands and reduced efficiency. In this work, we address this challenge by proposing a more computationally efficient approach. We propose a novel knowledge distillation technique to overcome the problem of reduced efficiency. We show how to use two large vision foundation models and distill the capabilities of these complementary models into one smaller model that maintains high accuracy at reduced computational cost. Furthermore, we demonstrate that by incorporating 3D data, we are able to further improve performance, without the need for human-annotated correspondences. Overall, our empirical results demonstrate that our distilled model with 3D data augmentation achieves performance superior to current state-of-the-art methods while significantly reducing computational load and enhancing practicality for real-world applications, such as semantic video correspondence. Our code and weights are publicly available on our project page.  
  </ol>  
</details>  
**comments**: WACV 2025, Page: https://compvis.github.io/distilldift  
  
### [Composed Image Retrieval for Training-Free Domain Conversion](http://arxiv.org/abs/2412.03297)  
[[code](https://github.com/nikosefth/freedom)]  
Nikos Efthymiadis, Bill Psomas, Zakaria Laskar, Konstantinos Karantzalos, Yannis Avrithis, Ondřej Chum, Giorgos Tolias  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This work addresses composed image retrieval in the context of domain conversion, where the content of a query image is retrieved in the domain specified by the query text. We show that a strong vision-language model provides sufficient descriptive power without additional training. The query image is mapped to the text input space using textual inversion. Unlike common practice that invert in the continuous space of text tokens, we use the discrete word space via a nearest-neighbor search in a text vocabulary. With this inversion, the image is softly mapped across the vocabulary and is made more robust using retrieval-based augmentation. Database images are retrieved by a weighted ensemble of text queries combining mapped words with the domain text. Our method outperforms prior art by a large margin on standard and newly introduced benchmarks. Code: https://github.com/NikosEfth/freedom  
  </ol>  
</details>  
**comments**: WACV 2025  
  
### [A Minimalistic 3D Self-Organized UAV Flocking Approach for Desert Exploration](http://arxiv.org/abs/2412.02881)  
Thulio Amorim, Tiago Nascimento, Akash Chaudhary, Eliseo Ferrante, Martin Saska  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this work, we propose a minimalistic swarm flocking approach for multirotor unmanned aerial vehicles (UAVs). Our approach allows the swarm to achieve cohesively and aligned flocking (collective motion), in a random direction, without externally provided directional information exchange (alignment control). The method relies on minimalistic sensory requirements as it uses only the relative range and bearing of swarm agents in local proximity obtained through onboard sensors on the UAV. Thus, our method is able to stabilize and control the flock of a general shape above a steep terrain without any explicit communication between swarm members. To implement proximal control in a three-dimensional manner, the Lennard-Jones potential function is used to maintain cohesiveness and avoid collisions between robots. The performance of the proposed approach was tested in real-world conditions by experiments with a team of nine UAVs. Experiments also present the usage of our approach on UAVs that are independent of external positioning systems such as the Global Navigation Satellite System (GNSS). Relying only on a relative visual localization through the ultraviolet direction and ranging (UVDAR) system, previously proposed by our group, the experiments verify that our system can be applied in GNSS-denied environments. The degree achieved of alignment and cohesiveness was evaluated using the metrics of order and steady-state value.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Measure Anything: Real-time, Multi-stage Vision-based Dimensional Measurement using Segment Anything](http://arxiv.org/abs/2412.03472)  
Yongkyu Lee, Shivam Kumar Panda, Wei Wang, Mohammad Khalid Jawed  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present Measure Anything, a comprehensive vision-based framework for dimensional measurement of objects with circular cross-sections, leveraging the Segment Anything Model (SAM). Our approach estimates key geometric features -- including diameter, length, and volume -- for rod-like geometries with varying curvature and general objects with constant skeleton slope. The framework integrates segmentation, mask processing, skeleton construction, and 2D-3D transformation, packaged in a user-friendly interface. We validate our framework by estimating the diameters of Canola stems -- collected from agricultural fields in North Dakota -- which are thin and non-uniform, posing challenges for existing methods. Measuring its diameters is critical, as it is a phenotypic traits that correlates with the health and yield of Canola crops. This application also exemplifies the potential of Measure Anything, where integrating intelligent models -- such as keypoint detection -- extends its scalability to fully automate the measurement process for high-throughput applications. Furthermore, we showcase its versatility in robotic grasping, leveraging extracted geometric features to identify optimal grasp points.  
  </ol>  
</details>  
  
  



## Image Matching  

### [Appearance Matching Adapter for Exemplar-based Semantic Image Synthesis](http://arxiv.org/abs/2412.03150)  
Siyoon Jin, Jisu Nam, Jiyoung Kim, Dahyun Chung, Yeong-Seok Kim, Joonhyung Park, Heonjeong Chu, Seungryong Kim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Exemplar-based semantic image synthesis aims to generate images aligned with given semantic content while preserving the appearance of an exemplar image. Conventional structure-guidance models, such as ControlNet, are limited in that they cannot directly utilize exemplar images as input, relying instead solely on text prompts to control appearance. Recent tuning-free approaches address this limitation by transferring local appearance from the exemplar image to the synthesized image through implicit cross-image matching in the augmented self-attention mechanism of pre-trained diffusion models. However, these methods face challenges when applied to content-rich scenes with significant geometric deformations, such as driving scenes. In this paper, we propose the Appearance Matching Adapter (AM-Adapter), a learnable framework that enhances cross-image matching within augmented self-attention by incorporating semantic information from segmentation maps. To effectively disentangle generation and matching processes, we adopt a stage-wise training approach. Initially, we train the structure-guidance and generation networks, followed by training the AM-Adapter while keeping the other networks frozen. During inference, we introduce an automated exemplar retrieval method to efficiently select exemplar image-segmentation pairs. Despite utilizing a limited number of learnable parameters, our method achieves state-of-the-art performance, excelling in both semantic alignment preservation and local appearance fidelity. Extensive ablation studies further validate our design choices. Code and pre-trained weights will be publicly available.: https://cvlab-kaist.github.io/AM-Adapter/  
  </ol>  
</details>  
  
  



## NeRF  

### [NeRF and Gaussian Splatting SLAM in the Wild](http://arxiv.org/abs/2412.03263)  
Fabian Schmidt, Markus Enzweiler, Abhinav Valada  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Navigating outdoor environments with visual Simultaneous Localization and Mapping (SLAM) systems poses significant challenges due to dynamic scenes, lighting variations, and seasonal changes, requiring robust solutions. While traditional SLAM methods struggle with adaptability, deep learning-based approaches and emerging neural radiance fields as well as Gaussian Splatting-based SLAM methods, offer promising alternatives. However, these methods have primarily been evaluated in controlled indoor environments with stable conditions, leaving a gap in understanding their performance in unstructured and variable outdoor settings. This study addresses this gap by evaluating these methods in natural outdoor environments, focusing on camera tracking accuracy, robustness to environmental factors, and computational efficiency, highlighting distinct trade-offs. Extensive evaluations demonstrate that neural SLAM methods achieve superior robustness, particularly under challenging conditions such as low light, but at a high computational cost. At the same time, traditional methods perform the best across seasons but are highly sensitive to variations in lighting conditions. The code of the benchmark is publicly available at https://github.com/iis-esslingen/nerf-3dgs-benchmark.  
  </ol>  
</details>  
**comments**: 5 pages, 2 figures, 4 tables  
  
  



