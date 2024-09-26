<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Frequency-based-View-Selection-in-Gaussian-Splatting-Reconstruction>Frequency-based View Selection in Gaussian Splatting Reconstruction</a></li>
        <li><a href=#Initialization-of-Monocular-Visual-Navigation-for-Autonomous-Agents-Using-Modified-Structure-from-Small-Motion>Initialization of Monocular Visual Navigation for Autonomous Agents Using Modified Structure from Small Motion</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#GSplatLoc:-Grounding-Keypoint-Descriptors-into-3D-Gaussian-Splatting-for-Improved-Visual-Localization>GSplatLoc: Grounding Keypoint Descriptors into 3D Gaussian Splatting for Improved Visual Localization</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#GSplatLoc:-Grounding-Keypoint-Descriptors-into-3D-Gaussian-Splatting-for-Improved-Visual-Localization>GSplatLoc: Grounding Keypoint Descriptors into 3D Gaussian Splatting for Improved Visual Localization</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Game4Loc:-A-UAV-Geo-Localization-Benchmark-from-Game-Data>Game4Loc: A UAV Geo-Localization Benchmark from Game Data</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#TalkinNeRF:-Animatable-Neural-Fields-for-Full-Body-Talking-Humans>TalkinNeRF: Animatable Neural Fields for Full-Body Talking Humans</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Frequency-based View Selection in Gaussian Splatting Reconstruction](http://arxiv.org/abs/2409.16470)  
Monica M. Q. Li, Pierre-Yves Lajoie, Giovanni Beltrame  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Three-dimensional reconstruction is a fundamental problem in robotics perception. We examine the problem of active view selection to perform 3D Gaussian Splatting reconstructions with as few input images as possible. Although 3D Gaussian Splatting has made significant progress in image rendering and 3D reconstruction, the quality of the reconstruction is strongly impacted by the selection of 2D images and the estimation of camera poses through Structure-from-Motion (SfM) algorithms. Current methods to select views that rely on uncertainties from occlusions, depth ambiguities, or neural network predictions directly are insufficient to handle the issue and struggle to generalize to new scenes. By ranking the potential views in the frequency domain, we are able to effectively estimate the potential information gain of new viewpoints without ground truth data. By overcoming current constraints on model architecture and efficacy, our method achieves state-of-the-art results in view selection, demonstrating its potential for efficient image-based 3D reconstruction.  
  </ol>  
</details>  
**comments**: 8 pages, 4 figures  
  
### [Initialization of Monocular Visual Navigation for Autonomous Agents Using Modified Structure from Small Motion](http://arxiv.org/abs/2409.16465)  
Juan-Diego Florez, Mehregan Dor, Panagiotis Tsiotras  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose a standalone monocular visual Simultaneous Localization and Mapping (vSLAM) initialization pipeline for autonomous robots in space. Our method, a state-of-the-art factor graph optimization pipeline, enhances classical Structure from Small Motion (SfSM) to robustly initialize a monocular agent in weak-perspective projection scenes. Furthermore, it overcomes visual estimation challenges introduced by spacecraft inspection trajectories, such as: center-pointing motion, which exacerbates the bas-relief ambiguity, and the presence of a dominant plane in the scene, which causes motion estimation degeneracies in classical Structure from Motion (SfM). We validate our method on realistic, simulated satellite inspection images exhibiting weak-perspective projection, and we demonstrate its effectiveness and improved performance compared to other monocular initialization procedures.  
  </ol>  
</details>  
**comments**: 6 pages, 1 page for references, 6 figures, 1 table, IEEEtran format
  This work has been submitted to the IEEE for possible publication. Copyright
  may be transferred without notice, after which this version may no longer be
  accessible  
  
  



## Visual Localization  

### [GSplatLoc: Grounding Keypoint Descriptors into 3D Gaussian Splatting for Improved Visual Localization](http://arxiv.org/abs/2409.16502)  
Gennady Sidorov, Malik Mohrat, Ksenia Lebedeva, Ruslan Rakhimov, Sergey Kolyubin  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Although various visual localization approaches exist, such as scene coordinate and pose regression, these methods often struggle with high memory consumption or extensive optimization requirements. To address these challenges, we utilize recent advancements in novel view synthesis, particularly 3D Gaussian Splatting (3DGS), to enhance localization. 3DGS allows for the compact encoding of both 3D geometry and scene appearance with its spatial features. Our method leverages the dense description maps produced by XFeat's lightweight keypoint detection and description model. We propose distilling these dense keypoint descriptors into 3DGS to improve the model's spatial understanding, leading to more accurate camera pose predictions through 2D-3D correspondences. After estimating an initial pose, we refine it using a photometric warping loss. Benchmarking on popular indoor and outdoor datasets shows that our approach surpasses state-of-the-art Neural Render Pose (NRP) methods, including NeRFMatch and PNeRFLoc.  
  </ol>  
</details>  
**comments**: Project website at https://gsplatloc.github.io/  
  
  



## Keypoint Detection  

### [GSplatLoc: Grounding Keypoint Descriptors into 3D Gaussian Splatting for Improved Visual Localization](http://arxiv.org/abs/2409.16502)  
Gennady Sidorov, Malik Mohrat, Ksenia Lebedeva, Ruslan Rakhimov, Sergey Kolyubin  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Although various visual localization approaches exist, such as scene coordinate and pose regression, these methods often struggle with high memory consumption or extensive optimization requirements. To address these challenges, we utilize recent advancements in novel view synthesis, particularly 3D Gaussian Splatting (3DGS), to enhance localization. 3DGS allows for the compact encoding of both 3D geometry and scene appearance with its spatial features. Our method leverages the dense description maps produced by XFeat's lightweight keypoint detection and description model. We propose distilling these dense keypoint descriptors into 3DGS to improve the model's spatial understanding, leading to more accurate camera pose predictions through 2D-3D correspondences. After estimating an initial pose, we refine it using a photometric warping loss. Benchmarking on popular indoor and outdoor datasets shows that our approach surpasses state-of-the-art Neural Render Pose (NRP) methods, including NeRFMatch and PNeRFLoc.  
  </ol>  
</details>  
**comments**: Project website at https://gsplatloc.github.io/  
  
  



## Image Matching  

### [Game4Loc: A UAV Geo-Localization Benchmark from Game Data](http://arxiv.org/abs/2409.16925)  
Yuxiang Ji, Boyong He, Zhuoyue Tan, Liaoni Wu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The vision-based geo-localization technology for UAV, serving as a secondary source of GPS information in addition to the global navigation satellite systems (GNSS), can still operate independently in the GPS-denied environment. Recent deep learning based methods attribute this as the task of image matching and retrieval. By retrieving drone-view images in geo-tagged satellite image database, approximate localization information can be obtained. However, due to high costs and privacy concerns, it is usually difficult to obtain large quantities of drone-view images from a continuous area. Existing drone-view datasets are mostly composed of small-scale aerial photography with a strong assumption that there exists a perfect one-to-one aligned reference image for any query, leaving a significant gap from the practical localization scenario. In this work, we construct a large-range contiguous area UAV geo-localization dataset named GTA-UAV, featuring multiple flight altitudes, attitudes, scenes, and targets using modern computer games. Based on this dataset, we introduce a more practical UAV geo-localization task including partial matches of cross-view paired data, and expand the image-level retrieval to the actual localization in terms of distance (meters). For the construction of drone-view and satellite-view pairs, we adopt a weight-based contrastive learning approach, which allows for effective learning while avoiding additional post-processing matching steps. Experiments demonstrate the effectiveness of our data and training method for UAV geo-localization, as well as the generalization capabilities to real-world scenarios.  
  </ol>  
</details>  
**comments**: Project page: https://yux1angji.github.io/game4loc/  
  
  



## NeRF  

### [TalkinNeRF: Animatable Neural Fields for Full-Body Talking Humans](http://arxiv.org/abs/2409.16666)  
Aggelina Chatziagapi, Bindita Chaudhuri, Amit Kumar, Rakesh Ranjan, Dimitris Samaras, Nikolaos Sarafianos  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce a novel framework that learns a dynamic neural radiance field (NeRF) for full-body talking humans from monocular videos. Prior work represents only the body pose or the face. However, humans communicate with their full body, combining body pose, hand gestures, as well as facial expressions. In this work, we propose TalkinNeRF, a unified NeRF-based network that represents the holistic 4D human motion. Given a monocular video of a subject, we learn corresponding modules for the body, face, and hands, that are combined together to generate the final result. To capture complex finger articulation, we learn an additional deformation field for the hands. Our multi-identity representation enables simultaneous training for multiple subjects, as well as robust animation under completely unseen poses. It can also generalize to novel identities, given only a short video as input. We demonstrate state-of-the-art performance for animating full-body talking humans, with fine-grained hand articulation and facial expressions.  
  </ol>  
</details>  
**comments**: Accepted by ECCVW 2024. Project page:
  https://aggelinacha.github.io/TalkinNeRF/  
  
  



