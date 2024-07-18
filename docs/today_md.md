<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Is-That-Rain?-Understanding-Effects-on-Visual-Odometry-Performance-for-Autonomous-UAVs-and-Efficient-DNN-based-Rain-Classification-at-the-Edge>Is That Rain? Understanding Effects on Visual Odometry Performance for Autonomous UAVs and Efficient DNN-based Rain Classification at the Edge</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#NeuSurfEmb:-A-Complete-Pipeline-for-Dense-Correspondence-based-6D-Object-Pose-Estimation-without-CAD-Models>NeuSurfEmb: A Complete Pipeline for Dense Correspondence-based 6D Object Pose Estimation without CAD Models</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Towards-Revisiting-Visual-Place-Recognition-for-Joining-Submaps-in-Multimap-SLAM>Towards Revisiting Visual Place Recognition for Joining Submaps in Multimap SLAM</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Generalizable-Human-Gaussians-for-Sparse-View-Synthesis>Generalizable Human Gaussians for Sparse View Synthesis</a></li>
        <li><a href=#SG-NeRF:-Neural-Surface-Reconstruction-with-Scene-Graph-Optimization>SG-NeRF: Neural Surface Reconstruction with Scene Graph Optimization</a></li>
        <li><a href=#InfoNorm:-Mutual-Information-Shaping-of-Normals-for-Sparse-View-Reconstruction>InfoNorm: Mutual Information Shaping of Normals for Sparse-View Reconstruction</a></li>
        <li><a href=#Invertible-Neural-Warp-for-NeRF>Invertible Neural Warp for NeRF</a></li>
        <li><a href=#Splatfacto-W:-A-Nerfstudio-Implementation-of-Gaussian-Splatting-for-Unconstrained-Photo-Collections>Splatfacto-W: A Nerfstudio Implementation of Gaussian Splatting for Unconstrained Photo Collections</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Is That Rain? Understanding Effects on Visual Odometry Performance for Autonomous UAVs and Efficient DNN-based Rain Classification at the Edge](http://arxiv.org/abs/2407.12663)  
Andrea Albanese, Yanran Wang, Davide Brunelli, David Boyle  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The development of safe and reliable autonomous unmanned aerial vehicles relies on the ability of the system to recognise and adapt to changes in the local environment based on sensor inputs. State-of-the-art local tracking and trajectory planning are typically performed using camera sensor input to the flight control algorithm, but the extent to which environmental disturbances like rain affect the performance of these systems is largely unknown. In this paper, we first describe the development of an open dataset comprising ~335k images to examine these effects for seven different classes of precipitation conditions and show that a worst-case average tracking error of 1.5 m is possible for a state-of-the-art visual odometry system (VINS-Fusion). We then use the dataset to train a set of deep neural network models suited to mobile and constrained deployment scenarios to determine the extent to which it may be possible to efficiently and accurately classify these `rainy' conditions. The most lightweight of these models (MobileNetV3 small) can achieve an accuracy of 90% with a memory footprint of just 1.28 MB and a frame rate of 93 FPS, which is suitable for deployment in resource-constrained and latency-sensitive systems. We demonstrate a classification latency in the order of milliseconds using typical flight computer hardware. Accordingly, such a model can feed into the disturbance estimation component of an autonomous flight controller. In addition, data from unmanned aerial vehicles with the ability to accurately determine environmental conditions in real time may contribute to developing more granular timely localised weather forecasting.  
  </ol>  
</details>  
  
  



## SFM  

### [NeuSurfEmb: A Complete Pipeline for Dense Correspondence-based 6D Object Pose Estimation without CAD Models](http://arxiv.org/abs/2407.12207)  
[[code](https://github.com/ethz-asl/neusurfemb)]  
Francesco Milano, Jen Jen Chung, Hermann Blum, Roland Siegwart, Lionel Ott  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    State-of-the-art approaches for 6D object pose estimation assume the availability of CAD models and require the user to manually set up physically-based rendering (PBR) pipelines for synthetic training data generation. Both factors limit the application of these methods in real-world scenarios. In this work, we present a pipeline that does not require CAD models and allows training a state-of-the-art pose estimator requiring only a small set of real images as input. Our method is based on a NeuS2 object representation, that we learn through a semi-automated procedure based on Structure-from-Motion (SfM) and object-agnostic segmentation. We exploit the novel-view synthesis ability of NeuS2 and simple cut-and-paste augmentation to automatically generate photorealistic object renderings, which we use to train the correspondence-based SurfEmb pose estimator. We evaluate our method on the LINEMOD-Occlusion dataset, extensively studying the impact of its individual components and showing competitive performance with respect to approaches based on CAD models and PBR data. We additionally demonstrate the ease of use and effectiveness of our pipeline on self-collected real-world objects, showing that our method outperforms state-of-the-art CAD-model-free approaches, with better accuracy and robustness to mild occlusions. To allow the robotics community to benefit from this system, we will publicly release it at https://www.github.com/ethz-asl/neusurfemb.  
  </ol>  
</details>  
**comments**: Accepted by the IEEE/RSJ International Conference on Intelligent
  Robots and Systems (IROS) 2024. 8 pages, 4 figures, 5 tables  
  
  



## Visual Localization  

### [Towards Revisiting Visual Place Recognition for Joining Submaps in Multimap SLAM](http://arxiv.org/abs/2407.12408)  
Markus Weißflog, Stefan Schubert, Peter Protzel, Peer Neubert  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual SLAM is a key technology for many autonomous systems. However, tracking loss can lead to the creation of disjoint submaps in multimap SLAM systems like ORB-SLAM3. Because of that, these systems employ submap merging strategies. As we show, these strategies are not always successful. In this paper, we investigate the impact of using modern VPR approaches for submap merging in visual SLAM. We argue that classical evaluation metrics are not sufficient to estimate the impact of a modern VPR component on the overall system. We show that naively replacing the VPR component does not leverage its full potential without requiring substantial interference in the original system. Because of that, we present a post-processing pipeline along with a set of metrics that allow us to estimate the impact of modern VPR components. We evaluate our approach on the NCLT and Newer College datasets using ORB-SLAM3 with NetVLAD and HDC-DELF as VPR components. Additionally, we present a simple approach for combining VPR with temporal consistency for map merging. We show that the map merging performance of ORB-SLAM3 can be improved. Building on these results, researchers in VPR can assess the potential of their approaches for SLAM systems.  
  </ol>  
</details>  
**comments**: Accepted at TAROS 2024. This is the submitted version  
  
  



## NeRF  

### [Generalizable Human Gaussians for Sparse View Synthesis](http://arxiv.org/abs/2407.12777)  
Youngjoong Kwon, Baole Fang, Yixing Lu, Haoye Dong, Cheng Zhang, Francisco Vicente Carrasco, Albert Mosella-Montoro, Jianjin Xu, Shingo Takagi, Daeil Kim, Aayush Prakash, Fernando De la Torre  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent progress in neural rendering has brought forth pioneering methods, such as NeRF and Gaussian Splatting, which revolutionize view rendering across various domains like AR/VR, gaming, and content creation. While these methods excel at interpolating {\em within the training data}, the challenge of generalizing to new scenes and objects from very sparse views persists. Specifically, modeling 3D humans from sparse views presents formidable hurdles due to the inherent complexity of human geometry, resulting in inaccurate reconstructions of geometry and textures. To tackle this challenge, this paper leverages recent advancements in Gaussian Splatting and introduces a new method to learn generalizable human Gaussians that allows photorealistic and accurate view-rendering of a new human subject from a limited set of sparse views in a feed-forward manner. A pivotal innovation of our approach involves reformulating the learning of 3D Gaussian parameters into a regression process defined on the 2D UV space of a human template, which allows leveraging the strong geometry prior and the advantages of 2D convolutions. In addition, a multi-scaffold is proposed to effectively represent the offset details. Our method outperforms recent methods on both within-dataset generalization as well as cross-dataset generalization settings.  
  </ol>  
</details>  
  
### [SG-NeRF: Neural Surface Reconstruction with Scene Graph Optimization](http://arxiv.org/abs/2407.12667)  
[[code](https://github.com/iris-cyy/sg-nerf)]  
Yiyang Chen, Siyan Dong, Xulong Wang, Lulu Cai, Youyi Zheng, Yanchao Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D surface reconstruction from images is essential for numerous applications. Recently, Neural Radiance Fields (NeRFs) have emerged as a promising framework for 3D modeling. However, NeRFs require accurate camera poses as input, and existing methods struggle to handle significantly noisy pose estimates (i.e., outliers), which are commonly encountered in real-world scenarios. To tackle this challenge, we present a novel approach that optimizes radiance fields with scene graphs to mitigate the influence of outlier poses. Our method incorporates an adaptive inlier-outlier confidence estimation scheme based on scene graphs, emphasizing images of high compatibility with the neighborhood and consistency in the rendering quality. We also introduce an effective intersection-over-union (IoU) loss to optimize the camera pose and surface geometry, together with a coarse-to-fine strategy to facilitate the training. Furthermore, we propose a new dataset containing typical outlier poses for a detailed evaluation. Experimental results on various datasets consistently demonstrate the effectiveness and superiority of our method over existing approaches, showcasing its robustness in handling outliers and producing high-quality 3D reconstructions. Our code and data are available at: \url{https://github.com/Iris-cyy/SG-NeRF}.  
  </ol>  
</details>  
**comments**: ECCV 2024  
  
### [InfoNorm: Mutual Information Shaping of Normals for Sparse-View Reconstruction](http://arxiv.org/abs/2407.12661)  
[[code](https://github.com/muliphein/infonorm)]  
Xulong Wang, Siyan Dong, Youyi Zheng, Yanchao Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D surface reconstruction from multi-view images is essential for scene understanding and interaction. However, complex indoor scenes pose challenges such as ambiguity due to limited observations. Recent implicit surface representations, such as Neural Radiance Fields (NeRFs) and signed distance functions (SDFs), employ various geometric priors to resolve the lack of observed information. Nevertheless, their performance heavily depends on the quality of the pre-trained geometry estimation models. To ease such dependence, we propose regularizing the geometric modeling by explicitly encouraging the mutual information among surface normals of highly correlated scene points. In this way, the geometry learning process is modulated by the second-order correlations from noisy (first-order) geometric priors, thus eliminating the bias due to poor generalization. Additionally, we introduce a simple yet effective scheme that utilizes semantic and geometric features to identify correlated points, enhancing their mutual information accordingly. The proposed technique can serve as a plugin for SDF-based neural surface representations. Our experiments demonstrate the effectiveness of the proposed in improving the surface reconstruction quality of major states of the arts. Our code is available at: \url{https://github.com/Muliphein/InfoNorm}.  
  </ol>  
</details>  
**comments**: ECCV 2024  
  
### [Invertible Neural Warp for NeRF](http://arxiv.org/abs/2407.12354)  
Shin-Fang Chng, Ravi Garg, Hemanth Saratchandran, Simon Lucey  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper tackles the simultaneous optimization of pose and Neural Radiance Fields (NeRF). Departing from the conventional practice of using explicit global representations for camera pose, we propose a novel overparameterized representation that models camera poses as learnable rigid warp functions. We establish that modeling the rigid warps must be tightly coupled with constraints and regularization imposed. Specifically, we highlight the critical importance of enforcing invertibility when learning rigid warp functions via neural network and propose the use of an Invertible Neural Network (INN) coupled with a geometry-informed constraint for this purpose. We present results on synthetic and real-world datasets, and demonstrate that our approach outperforms existing baselines in terms of pose estimation and high-fidelity reconstruction due to enhanced optimization convergence.  
  </ol>  
</details>  
**comments**: Accepted to ECCV 2024. Project page:
  https://sfchng.github.io/ineurowarping-github.io/  
  
### [Splatfacto-W: A Nerfstudio Implementation of Gaussian Splatting for Unconstrained Photo Collections](http://arxiv.org/abs/2407.12306)  
Congrong Xu, Justin Kerr, Angjoo Kanazawa  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Novel view synthesis from unconstrained in-the-wild image collections remains a significant yet challenging task due to photometric variations and transient occluders that complicate accurate scene reconstruction. Previous methods have approached these issues by integrating per-image appearance features embeddings in Neural Radiance Fields (NeRFs). Although 3D Gaussian Splatting (3DGS) offers faster training and real-time rendering, adapting it for unconstrained image collections is non-trivial due to the substantially different architecture. In this paper, we introduce Splatfacto-W, an approach that integrates per-Gaussian neural color features and per-image appearance embeddings into the rasterization process, along with a spherical harmonics-based background model to represent varying photometric appearances and better depict backgrounds. Our key contributions include latent appearance modeling, efficient transient object handling, and precise background modeling. Splatfacto-W delivers high-quality, real-time novel view synthesis with improved scene consistency in in-the-wild scenarios. Our method improves the Peak Signal-to-Noise Ratio (PSNR) by an average of 5.3 dB compared to 3DGS, enhances training speed by 150 times compared to NeRF-based methods, and achieves a similar rendering speed to 3DGS. Additional video results and code integrated into Nerfstudio are available at https://kevinxu02.github.io/splatfactow/.  
  </ol>  
</details>  
**comments**: 9 pages  
  
  



