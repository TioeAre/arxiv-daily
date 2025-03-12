<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#A-Framework-for-Reducing-the-Complexity-of-Geometric-Vision-Problems-and-its-Application-to-Two-View-Triangulation-with-Approximation-Bounds>A Framework for Reducing the Complexity of Geometric Vision Problems and its Application to Two-View Triangulation with Approximation Bounds</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#CQVPR:-Landmark-aware-Contextual-Queries-for-Visual-Place-Recognition>CQVPR: Landmark-aware Contextual Queries for Visual Place Recognition</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Keypoint-Detection-and-Description-for-Raw-Bayer-Images>Keypoint Detection and Description for Raw Bayer Images</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Keypoint-Detection-and-Description-for-Raw-Bayer-Images>Keypoint Detection and Description for Raw Bayer Images</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#GAS-NeRF:-Geometry-Aware-Stylization-of-Dynamic-Radiance-Fields>GAS-NeRF: Geometry-Aware Stylization of Dynamic Radiance Fields</a></li>
        <li><a href=#Uni-Gaussians:-Unifying-Camera-and-Lidar-Simulation-with-Gaussians-for-Dynamic-Driving-Scenarios>Uni-Gaussians: Unifying Camera and Lidar Simulation with Gaussians for Dynamic Driving Scenarios</a></li>
        <li><a href=#GigaSLAM:-Large-Scale-Monocular-SLAM-with-Hierachical-Gaussian-Splats>GigaSLAM: Large-Scale Monocular SLAM with Hierachical Gaussian Splats</a></li>
        <li><a href=#NeRF-VIO:-Map-Based-Visual-Inertial-Odometry-with-Initialization-Leveraging-Neural-Radiance-Fields>NeRF-VIO: Map-Based Visual-Inertial Odometry with Initialization Leveraging Neural Radiance Fields</a></li>
        <li><a href=#Neural-Radiance-and-Gaze-Fields-for-Visual-Attention-Modeling-in-3D-Environments>Neural Radiance and Gaze Fields for Visual Attention Modeling in 3D Environments</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [A Framework for Reducing the Complexity of Geometric Vision Problems and its Application to Two-View Triangulation with Approximation Bounds](http://arxiv.org/abs/2503.08142)  
Felix Rydell, Georg Bökman, Fredrik Kahl, Kathlén Kohn  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we present a new framework for reducing the computational complexity of geometric vision problems through targeted reweighting of the cost functions used to minimize reprojection errors. Triangulation - the task of estimating a 3D point from noisy 2D projections across multiple images - is a fundamental problem in multiview geometry and Structure-from-Motion (SfM) pipelines. We apply our framework to the two-view case and demonstrate that optimal triangulation, which requires solving a univariate polynomial of degree six, can be simplified through cost function reweighting reducing the polynomial degree to two. This reweighting yields a closed-form solution while preserving strong geometric accuracy. We derive optimal weighting strategies, establish theoretical bounds on the approximation error, and provide experimental results on real data demonstrating the effectiveness of the proposed approach compared to standard methods. Although this work focuses on two-view triangulation, the framework generalizes to other geometric vision problems.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [CQVPR: Landmark-aware Contextual Queries for Visual Place Recognition](http://arxiv.org/abs/2503.08170)  
Dongyue Li, Daisuke Deguchi, Hiroshi Murase  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Place Recognition (VPR) aims to estimate the location of the given query image within a database of geo-tagged images. To identify the exact location in an image, detecting landmarks is crucial. However, in some scenarios, such as urban environments, there are numerous landmarks, such as various modern buildings, and the landmarks in different cities often exhibit high visual similarity. Therefore, it is essential not only to leverage the landmarks but also to consider the contextual information surrounding them, such as whether there are trees, roads, or other features around the landmarks. We propose the Contextual Query VPR (CQVPR), which integrates contextual information with detailed pixel-level visual features. By leveraging a set of learnable contextual queries, our method automatically learns the high-level contexts with respect to landmarks and their surrounding areas. Heatmaps depicting regions that each query attends to serve as context-aware features, offering cues that could enhance the understanding of each scene. We further propose a query matching loss to supervise the extraction process of contextual queries. Extensive experiments on several datasets demonstrate that the proposed method outperforms other state-of-the-art methods, especially in challenging scenarios.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Keypoint Detection and Description for Raw Bayer Images](http://arxiv.org/abs/2503.08673)  
Jiakai Lin, Jinchang Zhang, Guoyu Lu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Keypoint detection and local feature description are fundamental tasks in robotic perception, critical for applications such as SLAM, robot localization, feature matching, pose estimation, and 3D mapping. While existing methods predominantly operate on RGB images, we propose a novel network that directly processes raw images, bypassing the need for the Image Signal Processor (ISP). This approach significantly reduces hardware requirements and memory consumption, which is crucial for robotic vision systems. Our method introduces two custom-designed convolutional kernels capable of performing convolutions directly on raw images, preserving inter-channel information without converting to RGB. Experimental results show that our network outperforms existing algorithms on raw images, achieving higher accuracy and stability under large rotations and scale variations. This work represents the first attempt to develop a keypoint detection and feature description network specifically for raw images, offering a more efficient solution for resource-constrained environments.  
  </ol>  
</details>  
  
  



## Image Matching  

### [Keypoint Detection and Description for Raw Bayer Images](http://arxiv.org/abs/2503.08673)  
Jiakai Lin, Jinchang Zhang, Guoyu Lu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Keypoint detection and local feature description are fundamental tasks in robotic perception, critical for applications such as SLAM, robot localization, feature matching, pose estimation, and 3D mapping. While existing methods predominantly operate on RGB images, we propose a novel network that directly processes raw images, bypassing the need for the Image Signal Processor (ISP). This approach significantly reduces hardware requirements and memory consumption, which is crucial for robotic vision systems. Our method introduces two custom-designed convolutional kernels capable of performing convolutions directly on raw images, preserving inter-channel information without converting to RGB. Experimental results show that our network outperforms existing algorithms on raw images, achieving higher accuracy and stability under large rotations and scale variations. This work represents the first attempt to develop a keypoint detection and feature description network specifically for raw images, offering a more efficient solution for resource-constrained environments.  
  </ol>  
</details>  
  
  



## NeRF  

### [GAS-NeRF: Geometry-Aware Stylization of Dynamic Radiance Fields](http://arxiv.org/abs/2503.08483)  
Nhat Phuong Anh Vu, Abhishek Saroha, Or Litany, Daniel Cremers  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Current 3D stylization techniques primarily focus on static scenes, while our world is inherently dynamic, filled with moving objects and changing environments. Existing style transfer methods primarily target appearance -- such as color and texture transformation -- but often neglect the geometric characteristics of the style image, which are crucial for achieving a complete and coherent stylization effect. To overcome these shortcomings, we propose GAS-NeRF, a novel approach for joint appearance and geometry stylization in dynamic Radiance Fields. Our method leverages depth maps to extract and transfer geometric details into the radiance field, followed by appearance transfer. Experimental results on synthetic and real-world datasets demonstrate that our approach significantly enhances the stylization quality while maintaining temporal coherence in dynamic scenes.  
  </ol>  
</details>  
  
### [Uni-Gaussians: Unifying Camera and Lidar Simulation with Gaussians for Dynamic Driving Scenarios](http://arxiv.org/abs/2503.08317)  
Zikang Yuan, Yuechuan Pu, Hongcheng Luo, Fengtian Lang, Cheng Chi, Teng Li, Yingying Shen, Haiyang Sun, Bing Wang, Xin Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Ensuring the safety of autonomous vehicles necessitates comprehensive simulation of multi-sensor data, encompassing inputs from both cameras and LiDAR sensors, across various dynamic driving scenarios. Neural rendering techniques, which utilize collected raw sensor data to simulate these dynamic environments, have emerged as a leading methodology. While NeRF-based approaches can uniformly represent scenes for rendering data from both camera and LiDAR, they are hindered by slow rendering speeds due to dense sampling. Conversely, Gaussian Splatting-based methods employ Gaussian primitives for scene representation and achieve rapid rendering through rasterization. However, these rasterization-based techniques struggle to accurately model non-linear optical sensors. This limitation restricts their applicability to sensors beyond pinhole cameras. To address these challenges and enable unified representation of dynamic driving scenarios using Gaussian primitives, this study proposes a novel hybrid approach. Our method utilizes rasterization for rendering image data while employing Gaussian ray-tracing for LiDAR data rendering. Experimental results on public datasets demonstrate that our approach outperforms current state-of-the-art methods. This work presents a unified and efficient solution for realistic simulation of camera and LiDAR data in autonomous driving scenarios using Gaussian primitives, offering significant advancements in both rendering quality and computational efficiency.  
  </ol>  
</details>  
**comments**: 10 pages  
  
### [GigaSLAM: Large-Scale Monocular SLAM with Hierachical Gaussian Splats](http://arxiv.org/abs/2503.08071)  
Kai Deng, Jian Yang, Shenlong Wang, Jin Xie  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Tracking and mapping in large-scale, unbounded outdoor environments using only monocular RGB input presents substantial challenges for existing SLAM systems. Traditional Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) SLAM methods are typically limited to small, bounded indoor settings. To overcome these challenges, we introduce GigaSLAM, the first NeRF/3DGS-based SLAM framework for kilometer-scale outdoor environments, as demonstrated on the KITTI and KITTI 360 datasets. Our approach employs a hierarchical sparse voxel map representation, where Gaussians are decoded by neural networks at multiple levels of detail. This design enables efficient, scalable mapping and high-fidelity viewpoint rendering across expansive, unbounded scenes. For front-end tracking, GigaSLAM utilizes a metric depth model combined with epipolar geometry and PnP algorithms to accurately estimate poses, while incorporating a Bag-of-Words-based loop closure mechanism to maintain robust alignment over long trajectories. Consequently, GigaSLAM delivers high-precision tracking and visually faithful rendering on urban outdoor benchmarks, establishing a robust SLAM solution for large-scale, long-term scenarios, and significantly extending the applicability of Gaussian Splatting SLAM systems to unbounded outdoor environments.  
  </ol>  
</details>  
  
### [NeRF-VIO: Map-Based Visual-Inertial Odometry with Initialization Leveraging Neural Radiance Fields](http://arxiv.org/abs/2503.07952)  
Yanyu Zhang, Dongming Wang, Jie Xu, Mengyuan Liu, Pengxiang Zhu, Wei Ren  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    A prior map serves as a foundational reference for localization in context-aware applications such as augmented reality (AR). Providing valuable contextual information about the environment, the prior map is a vital tool for mitigating drift. In this paper, we propose a map-based visual-inertial localization algorithm (NeRF-VIO) with initialization using neural radiance fields (NeRF). Our algorithm utilizes a multilayer perceptron model and redefines the loss function as the geodesic distance on \(SE(3)\), ensuring the invariance of the initialization model under a frame change within \(\mathfrak{se}(3)\). The evaluation demonstrates that our model outperforms existing NeRF-based initialization solution in both accuracy and efficiency. By integrating a two-stage update mechanism within a multi-state constraint Kalman filter (MSCKF) framework, the state of NeRF-VIO is constrained by both captured images from an onboard camera and rendered images from a pre-trained NeRF model. The proposed algorithm is validated using a real-world AR dataset, the results indicate that our two-stage update pipeline outperforms MSCKF across all data sequences.  
  </ol>  
</details>  
  
### [Neural Radiance and Gaze Fields for Visual Attention Modeling in 3D Environments](http://arxiv.org/abs/2503.07828)  
Andrei Chubarau, Yinan Wang, James J. Clark  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce Neural Radiance and Gaze Fields (NeRGs) as a novel approach for representing visual attention patterns in 3D scenes. Our system renders a 2D view of a 3D scene with a pre-trained Neural Radiance Field (NeRF) and visualizes the gaze field for arbitrary observer positions, which may be decoupled from the render camera perspective. We achieve this by augmenting a standard NeRF with an additional neural network that models the gaze probability distribution. The output of a NeRG is a rendered image of the scene viewed from the camera perspective and a pixel-wise salience map representing conditional probability that an observer fixates on a given surface within the 3D scene as visible in the rendered image. Much like how NeRFs perform novel view synthesis, NeRGs enable the reconstruction of gaze patterns from arbitrary perspectives within complex 3D scenes. To ensure consistent gaze reconstructions, we constrain gaze prediction on the 3D structure of the scene and model gaze occlusion due to intervening surfaces when the observer's viewpoint is decoupled from the rendering camera. For training, we leverage ground truth head pose data from skeleton tracking data or predictions from 2D salience models. We demonstrate the effectiveness of NeRGs in a real-world convenience store setting, where head pose tracking data is available.  
  </ol>  
</details>  
**comments**: 11 pages, 8 figures  
  
  



