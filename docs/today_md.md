<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#MAC-VO:-Metrics-aware-Covariance-for-Learning-based-Stereo-Visual-Odometry>MAC-VO: Metrics-aware Covariance for Learning-based Stereo Visual Odometry</a></li>
        <li><a href=#GEVO:-Memory-Efficient-Monocular-Visual-Odometry-Using-Gaussians>GEVO: Memory-Efficient Monocular Visual Odometry Using Gaussians</a></li>
        <li><a href=#Panoramic-Direct-LiDAR-assisted-Visual-Odometry>Panoramic Direct LiDAR-assisted Visual Odometry</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#SOLVR:-Submap-Oriented-LiDAR-Visual-Re-Localisation>SOLVR: Submap Oriented LiDAR-Visual Re-Localisation</a></li>
        <li><a href=#Garment-Attribute-Manipulation-with-Multi-level-Attention>Garment Attribute Manipulation with Multi-level Attention</a></li>
        <li><a href=#Evaluating-Pre-trained-Convolutional-Neural-Networks-and-Foundation-Models-as-Feature-Extractors-for-Content-based-Medical-Image-Retrieval>Evaluating Pre-trained Convolutional Neural Networks and Foundation Models as Feature Extractors for Content-based Medical Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Baking-Relightable-NeRF-for-Real-time-Direct/Indirect-Illumination-Rendering>Baking Relightable NeRF for Real-time Direct/Indirect Illumination Rendering</a></li>
        <li><a href=#DENSER:-3D-Gaussians-Splatting-for-Scene-Reconstruction-of-Dynamic-Urban-Environments>DENSER: 3D Gaussians Splatting for Scene Reconstruction of Dynamic Urban Environments</a></li>
        <li><a href=#NARF24:-Estimating-Articulated-Object-Structure-for-Implicit-Rendering>NARF24: Estimating Articulated Object Structure for Implicit Rendering</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [MAC-VO: Metrics-aware Covariance for Learning-based Stereo Visual Odometry](http://arxiv.org/abs/2409.09479)  
Yuheng Qiu, Yutian Chen, Zihao Zhang, Wenshan Wang, Sebastian Scherer  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose the MAC-VO, a novel learning-based stereo VO that leverages the learned metrics-aware matching uncertainty for dual purposes: selecting keypoint and weighing the residual in pose graph optimization. Compared to traditional geometric methods prioritizing texture-affluent features like edges, our keypoint selector employs the learned uncertainty to filter out the low-quality features based on global inconsistency. In contrast to the learning-based algorithms that model the scale-agnostic diagonal weight matrix for covariance, we design a metrics-aware covariance model to capture the spatial error during keypoint registration and the correlations between different axes. Integrating this covariance model into pose graph optimization enhances the robustness and reliability of pose estimation, particularly in challenging environments with varying illumination, feature density, and motion patterns. On public benchmark datasets, MAC-VO outperforms existing VO algorithms and even some SLAM algorithms in challenging environments. The covariance map also provides valuable information about the reliability of the estimated poses, which can benefit decision-making for autonomous systems.  
  </ol>  
</details>  
  
### [GEVO: Memory-Efficient Monocular Visual Odometry Using Gaussians](http://arxiv.org/abs/2409.09295)  
Dasong Gao, Peter Zhi Xuan Li, Vivienne Sze, Sertac Karaman  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Constructing a high-fidelity representation of the 3D scene using a monocular camera can enable a wide range of applications on mobile devices, such as micro-robots, smartphones, and AR/VR headsets. On these devices, memory is often limited in capacity and its access often dominates the consumption of compute energy. Although Gaussian Splatting (GS) allows for high-fidelity reconstruction of 3D scenes, current GS-based SLAM is not memory efficient as a large number of past images is stored to retrain Gaussians for reducing catastrophic forgetting. These images often require two-orders-of-magnitude higher memory than the map itself and thus dominate the total memory usage. In this work, we present GEVO, a GS-based monocular SLAM framework that achieves comparable fidelity as prior methods by rendering (instead of storing) them from the existing map. Novel Gaussian initialization and optimization techniques are proposed to remove artifacts from the map and delay the degradation of the rendered images over time. Across a variety of environments, GEVO achieves comparable map fidelity while reducing the memory overhead to around 58 MBs, which is up to 94x lower than prior works.  
  </ol>  
</details>  
**comments**: 8 pages  
  
### [Panoramic Direct LiDAR-assisted Visual Odometry](http://arxiv.org/abs/2409.09287)  
Zikang Yuan, Tianle Xu, Xiaoxiang Wang, Jinni Geng, Xin Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Enhancing visual odometry by exploiting sparse depth measurements from LiDAR is a promising solution for improving tracking accuracy of an odometry. Most existing works utilize a monocular pinhole camera, yet could suffer from poor robustness due to less available information from limited field-of-view (FOV). This paper proposes a panoramic direct LiDAR-assisted visual odometry, which fully associates the 360-degree FOV LiDAR points with the 360-degree FOV panoramic image datas. 360-degree FOV panoramic images can provide more available information, which can compensate inaccurate pose estimation caused by insufficient texture or motion blur from a single view. In addition to constraints between a specific view at different times, constraints can also be built between different views at the same moment. Experimental results on public datasets demonstrate the benefit of large FOV of our panoramic direct LiDAR-assisted visual odometry to state-of-the-art approaches.  
  </ol>  
</details>  
**comments**: 6 pages, 6 figures  
  
  



## Visual Localization  

### [SOLVR: Submap Oriented LiDAR-Visual Re-Localisation](http://arxiv.org/abs/2409.10247)  
Joshua Knights, Sebastián Barbas Laina, Peyman Moghadam, Stefan Leutenegger  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper proposes SOLVR, a unified pipeline for learning based LiDAR-Visual re-localisation which performs place recognition and 6-DoF registration across sensor modalities. We propose a strategy to align the input sensor modalities by leveraging stereo image streams to produce metric depth predictions with pose information, followed by fusing multiple scene views from a local window using a probabilistic occupancy framework to expand the limited field-of-view of the camera. Additionally, SOLVR adopts a flexible definition of what constitutes positive examples for different training losses, allowing us to simultaneously optimise place recognition and registration performance. Furthermore, we replace RANSAC with a registration function that weights a simple least-squares fitting with the estimated inlier likelihood of sparse keypoint correspondences, improving performance in scenarios with a low inlier ratio between the query and retrieved place. Our experiments on the KITTI and KITTI360 datasets show that SOLVR achieves state-of-the-art performance for LiDAR-Visual place recognition and registration, particularly improving registration accuracy over larger distances between the query and retrieved place.  
  </ol>  
</details>  
**comments**: Submitted to ICRA2025  
  
### [Garment Attribute Manipulation with Multi-level Attention](http://arxiv.org/abs/2409.10206)  
Vittorio Casula, Lorenzo Berlincioni, Luca Cultrera, Federico Becattini, Chiara Pero, Carmen Bisogni, Marco Bertini, Alberto Del Bimbo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In the rapidly evolving field of online fashion shopping, the need for more personalized and interactive image retrieval systems has become paramount. Existing methods often struggle with precisely manipulating specific garment attributes without inadvertently affecting others. To address this challenge, we propose GAMMA (Garment Attribute Manipulation with Multi-level Attention), a novel framework that integrates attribute-disentangled representations with a multi-stage attention-based architecture. GAMMA enables targeted manipulation of fashion image attributes, allowing users to refine their searches with high accuracy. By leveraging a dual-encoder Transformer and memory block, our model achieves state-of-the-art performance on popular datasets like Shopping100k and DeepFashion.  
  </ol>  
</details>  
**comments**: Accepted for publication at the ECCV 2024 workshop FashionAI  
  
### [Evaluating Pre-trained Convolutional Neural Networks and Foundation Models as Feature Extractors for Content-based Medical Image Retrieval](http://arxiv.org/abs/2409.09430)  
Amirreza Mahbod, Nematollah Saeidi, Sepideh Hatamikia, Ramona Woitek  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Medical image retrieval refers to the task of finding similar images for given query images in a database, with applications such as diagnosis support, treatment planning, and educational tools for inexperienced medical practitioners. While traditional medical image retrieval was performed using clinical metadata, content-based medical image retrieval (CBMIR) relies on the characteristic features of the images, such as color, texture, shape, and spatial features. Many approaches have been proposed for CBMIR, and among them, using pre-trained convolutional neural networks (CNNs) is a widely utilized approach. However, considering the recent advances in the development of foundation models for various computer vision tasks, their application for CBMIR can be also investigated for its potentially superior performance.   In this study, we used several pre-trained feature extractors from well-known pre-trained CNNs (VGG19, ResNet-50, DenseNet121, and EfficientNetV2M) and pre-trained foundation models (MedCLIP, BioMedCLIP, OpenCLIP, CONCH and UNI) and investigated the CBMIR performance on a subset of the MedMNIST V2 dataset, including eight types of 2D and 3D medical images. Furthermore, we also investigated the effect of image size on the CBMIR performance.   Our results show that, overall, for the 2D datasets, foundation models deliver superior performance by a large margin compared to CNNs, with UNI providing the best overall performance across all datasets and image sizes. For 3D datasets, CNNs and foundation models deliver more competitive performance, with CONCH achieving the best overall performance. Moreover, our findings confirm that while using larger image sizes (especially for 2D datasets) yields slightly better performance, competitive CBMIR performance can still be achieved even with smaller image sizes. Our codes to generate and reproduce the results are available on GitHub.  
  </ol>  
</details>  
**comments**: 29 pages  
  
  



## NeRF  

### [Baking Relightable NeRF for Real-time Direct/Indirect Illumination Rendering](http://arxiv.org/abs/2409.10327)  
Euntae Choi, Vincent Carpentier, Seunghun Shin, Sungjoo Yoo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Relighting, which synthesizes a novel view under a given lighting condition (unseen in training time), is a must feature for immersive photo-realistic experience. However, real-time relighting is challenging due to high computation cost of the rendering equation which requires shape and material decomposition and visibility test to model shadow. Additionally, for indirect illumination, additional computation of rendering equation on each secondary surface point (where reflection occurs) is required rendering real-time relighting challenging. We propose a novel method that executes a CNN renderer to compute primary surface points and rendering parameters, required for direct illumination. We also present a lightweight hash grid-based renderer, for indirect illumination, which is recursively executed to perform the secondary ray tracing process. Both renderers are trained in a distillation from a pre-trained teacher model and provide real-time physically-based rendering under unseen lighting condition at a negligible loss of rendering quality.  
  </ol>  
</details>  
**comments**: Under review  
  
### [DENSER: 3D Gaussians Splatting for Scene Reconstruction of Dynamic Urban Environments](http://arxiv.org/abs/2409.10041)  
Mahmud A. Mohamad, Gamal Elghazaly, Arthur Hubert, Raphael Frank  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper presents DENSER, an efficient and effective approach leveraging 3D Gaussian splatting (3DGS) for the reconstruction of dynamic urban environments. While several methods for photorealistic scene representations, both implicitly using neural radiance fields (NeRF) and explicitly using 3DGS have shown promising results in scene reconstruction of relatively complex dynamic scenes, modeling the dynamic appearance of foreground objects tend to be challenging, limiting the applicability of these methods to capture subtleties and details of the scenes, especially far dynamic objects. To this end, we propose DENSER, a framework that significantly enhances the representation of dynamic objects and accurately models the appearance of dynamic objects in the driving scene. Instead of directly using Spherical Harmonics (SH) to model the appearance of dynamic objects, we introduce and integrate a new method aiming at dynamically estimating SH bases using wavelets, resulting in better representation of dynamic objects appearance in both space and time. Besides object appearance, DENSER enhances object shape representation through densification of its point cloud across multiple scene frames, resulting in faster convergence of model training. Extensive evaluations on KITTI dataset show that the proposed approach significantly outperforms state-of-the-art methods by a wide margin. Source codes and models will be uploaded to this repository https://github.com/sntubix/denser  
  </ol>  
</details>  
  
### [NARF24: Estimating Articulated Object Structure for Implicit Rendering](http://arxiv.org/abs/2409.09829)  
Stanley Lewis, Tom Gao, Odest Chadwicke Jenkins  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Articulated objects and their representations pose a difficult problem for robots. These objects require not only representations of geometry and texture, but also of the various connections and joint parameters that make up each articulation. We propose a method that learns a common Neural Radiance Field (NeRF) representation across a small number of collected scenes. This representation is combined with a parts-based image segmentation to produce an implicit space part localization, from which the connectivity and joint parameters of the articulated object can be estimated, thus enabling configuration-conditioned rendering.  
  </ol>  
</details>  
**comments**: extended abstract as submitted to ICRA@40 anniversary conference  
  
  



