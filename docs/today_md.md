<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Bayesian-Simultaneous-Localization-and-Multi-Lane-Tracking-Using-Onboard-Sensors-and-a-SD-Map>Bayesian Simultaneous Localization and Multi-Lane Tracking Using Onboard Sensors and a SD Map</a></li>
        <li><a href=#IMU-Aided-Event-based-Stereo-Visual-Odometry>IMU-Aided Event-based Stereo Visual Odometry</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Novel-View-Synthesis-with-Neural-Radiance-Fields-for-Industrial-Robot-Applications>Novel View Synthesis with Neural Radiance Fields for Industrial Robot Applications</a></li>
        <li><a href=#Non-rigid-Structure-from-Motion:-Temporally-smooth-Procrustean-Alignment-and-Spatially-variant-Deformation-Modeling>Non-rigid Structure-from-Motion: Temporally-smooth Procrustean Alignment and Spatially-variant Deformation Modeling</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Breast-Histopathology-Image-Retrieval-by-Attention-based-Adversarially-Regularized-Variational-Graph-Autoencoder-with-Contrastive-Learning-Based-Feature-Extraction>Breast Histopathology Image Retrieval by Attention-based Adversarially Regularized Variational Graph Autoencoder with Contrastive Learning-Based Feature Extraction</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#DistGrid:-Scalable-Scene-Reconstruction-with-Distributed-Multi-resolution-Hash-Grid>DistGrid: Scalable Scene Reconstruction with Distributed Multi-resolution Hash Grid</a></li>
        <li><a href=#Novel-View-Synthesis-with-Neural-Radiance-Fields-for-Industrial-Robot-Applications>Novel View Synthesis with Neural Radiance Fields for Industrial Robot Applications</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Bayesian Simultaneous Localization and Multi-Lane Tracking Using Onboard Sensors and a SD Map](http://arxiv.org/abs/2405.04290)  
Yuxuan Xia, Erik Stenborg, Junsheng Fu, Gustaf Hendeby  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    High-definition map with accurate lane-level information is crucial for autonomous driving, but the creation of these maps is a resource-intensive process. To this end, we present a cost-effective solution to create lane-level roadmaps using only the global navigation satellite system (GNSS) and a camera on customer vehicles. Our proposed solution utilizes a prior standard-definition (SD) map, GNSS measurements, visual odometry, and lane marking edge detection points, to simultaneously estimate the vehicle's 6D pose, its position within a SD map, and also the 3D geometry of traffic lines. This is achieved using a Bayesian simultaneous localization and multi-object tracking filter, where the estimation of traffic lines is formulated as a multiple extended object tracking problem, solved using a trajectory Poisson multi-Bernoulli mixture (TPMBM) filter. In TPMBM filtering, traffic lines are modeled using B-spline trajectories, and each trajectory is parameterized by a sequence of control points. The proposed solution has been evaluated using experimental data collected by a test vehicle driving on highway. Preliminary results show that the traffic line estimates, overlaid on the satellite image, generally align with the lane markings up to some lateral offsets.  
  </ol>  
</details>  
**comments**: 27th International Conference on Information Fusion  
  
### [IMU-Aided Event-based Stereo Visual Odometry](http://arxiv.org/abs/2405.04071)  
[[code](https://github.com/nail-hnu/esvio_aa)]  
Junkai Niu, Sheng Zhong, Yi Zhou  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Direct methods for event-based visual odometry solve the mapping and camera pose tracking sub-problems by establishing implicit data association in a way that the generative model of events is exploited. The main bottlenecks faced by state-of-the-art work in this field include the high computational complexity of mapping and the limited accuracy of tracking. In this paper, we improve our previous direct pipeline \textit{Event-based Stereo Visual Odometry} in terms of accuracy and efficiency. To speed up the mapping operation, we propose an efficient strategy of edge-pixel sampling according to the local dynamics of events. The mapping performance in terms of completeness and local smoothness is also improved by combining the temporal stereo results and the static stereo results. To circumvent the degeneracy issue of camera pose tracking in recovering the yaw component of general 6-DoF motion, we introduce as a prior the gyroscope measurements via pre-integration. Experiments on publicly available datasets justify our improvement. We release our pipeline as an open-source software for future research in this field.  
  </ol>  
</details>  
**comments**: 10 pages, 7 figures, ICRA  
  
  



## SFM  

### [Novel View Synthesis with Neural Radiance Fields for Industrial Robot Applications](http://arxiv.org/abs/2405.04345)  
Markus Hillemann, Robert Langendörfer, Max Heiken, Max Mehltretter, Andreas Schenk, Martin Weinmann, Stefan Hinz, Christian Heipke, Markus Ulrich  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRFs) have become a rapidly growing research field with the potential to revolutionize typical photogrammetric workflows, such as those used for 3D scene reconstruction. As input, NeRFs require multi-view images with corresponding camera poses as well as the interior orientation. In the typical NeRF workflow, the camera poses and the interior orientation are estimated in advance with Structure from Motion (SfM). But the quality of the resulting novel views, which depends on different parameters such as the number and distribution of available images, as well as the accuracy of the related camera poses and interior orientation, is difficult to predict. In addition, SfM is a time-consuming pre-processing step, and its quality strongly depends on the image content. Furthermore, the undefined scaling factor of SfM hinders subsequent steps in which metric information is required. In this paper, we evaluate the potential of NeRFs for industrial robot applications. We propose an alternative to SfM pre-processing: we capture the input images with a calibrated camera that is attached to the end effector of an industrial robot and determine accurate camera poses with metric scale based on the robot kinematics. We then investigate the quality of the novel views by comparing them to ground truth, and by computing an internal quality measure based on ensemble methods. For evaluation purposes, we acquire multiple datasets that pose challenges for reconstruction typical of industrial applications, like reflective objects, poor texture, and fine structures. We show that the robot-based pose determination reaches similar accuracy as SfM in non-demanding cases, while having clear advantages in more challenging scenarios. Finally, we present first results of applying the ensemble method to estimate the quality of the synthetic novel view in the absence of a ground truth.  
  </ol>  
</details>  
**comments**: 8 pages, 8 figures, accepted for publication in The International
  Archives of the Photogrammetry, Remote Sensing and Spatial Information
  Sciences (ISPRS Archives) 2024  
  
### [Non-rigid Structure-from-Motion: Temporally-smooth Procrustean Alignment and Spatially-variant Deformation Modeling](http://arxiv.org/abs/2405.04309)  
Jiawei Shi, Hui Deng, Yuchao Dai  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Even though Non-rigid Structure-from-Motion (NRSfM) has been extensively studied and great progress has been made, there are still key challenges that hinder their broad real-world applications: 1) the inherent motion/rotation ambiguity requires either explicit camera motion recovery with extra constraint or complex Procrustean Alignment; 2) existing low-rank modeling of the global shape can over-penalize drastic deformations in the 3D shape sequence. This paper proposes to resolve the above issues from a spatial-temporal modeling perspective. First, we propose a novel Temporally-smooth Procrustean Alignment module that estimates 3D deforming shapes and adjusts the camera motion by aligning the 3D shape sequence consecutively. Our new alignment module remedies the requirement of complex reference 3D shape during alignment, which is more conductive to non-isotropic deformation modeling. Second, we propose a spatial-weighted approach to enforce the low-rank constraint adaptively at different locations to accommodate drastic spatially-variant deformation reconstruction better. Our modeling outperform existing low-rank based methods, and extensive experiments across different datasets validate the effectiveness of our method.  
  </ol>  
</details>  
**comments**: Accepted by CVPR 2024  
  
  



## Visual Localization  

### [Breast Histopathology Image Retrieval by Attention-based Adversarially Regularized Variational Graph Autoencoder with Contrastive Learning-Based Feature Extraction](http://arxiv.org/abs/2405.04211)  
Nematollah Saeidi, Hossein Karshenas, Bijan Shoushtarian, Sepideh Hatamikia, Ramona Woitek, Amirreza Mahbod  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Breast cancer is a significant global health concern, particularly for women. Early detection and appropriate treatment are crucial in mitigating its impact, with histopathology examinations playing a vital role in swift diagnosis. However, these examinations often require a substantial workforce and experienced medical experts for proper recognition and cancer grading. Automated image retrieval systems have the potential to assist pathologists in identifying cancerous tissues, thereby accelerating the diagnostic process. Nevertheless, due to considerable variability among the tissue and cell patterns in histological images, proposing an accurate image retrieval model is very challenging.   This work introduces a novel attention-based adversarially regularized variational graph autoencoder model for breast histological image retrieval. Additionally, we incorporated cluster-guided contrastive learning as the graph feature extractor to boost the retrieval performance. We evaluated the proposed model's performance on two publicly available datasets of breast cancer histological images and achieved superior or very competitive retrieval performance, with average mAP scores of 96.5% for the BreakHis dataset and 94.7% for the BACH dataset, and mVP scores of 91.9% and 91.3%, respectively.   Our proposed retrieval model has the potential to be used in clinical settings to enhance diagnostic performance and ultimately benefit patients.  
  </ol>  
</details>  
**comments**: 31 pages  
  
  



## NeRF  

### [DistGrid: Scalable Scene Reconstruction with Distributed Multi-resolution Hash Grid](http://arxiv.org/abs/2405.04416)  
Sidun Liu, Peng Qiao, Zongxin Ye, Wenyu Li, Yong Dou  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Field~(NeRF) achieves extremely high quality in object-scaled and indoor scene reconstruction. However, there exist some challenges when reconstructing large-scale scenes. MLP-based NeRFs suffer from limited network capacity, while volume-based NeRFs are heavily memory-consuming when the scene resolution increases. Recent approaches propose to geographically partition the scene and learn each sub-region using an individual NeRF. Such partitioning strategies help volume-based NeRF exceed the single GPU memory limit and scale to larger scenes. However, this approach requires multiple background NeRF to handle out-of-partition rays, which leads to redundancy of learning. Inspired by the fact that the background of current partition is the foreground of adjacent partition, we propose a scalable scene reconstruction method based on joint Multi-resolution Hash Grids, named DistGrid. In this method, the scene is divided into multiple closely-paved yet non-overlapped Axis-Aligned Bounding Boxes, and a novel segmented volume rendering method is proposed to handle cross-boundary rays, thereby eliminating the need for background NeRFs. The experiments demonstrate that our method outperforms existing methods on all evaluated large-scale scenes, and provides visually plausible scene reconstruction. The scalability of our method on reconstruction quality is further evaluated qualitatively and quantitatively.  
  </ol>  
</details>  
**comments**: Originally submitted to Siggraph Asia 2023  
  
### [Novel View Synthesis with Neural Radiance Fields for Industrial Robot Applications](http://arxiv.org/abs/2405.04345)  
Markus Hillemann, Robert Langendörfer, Max Heiken, Max Mehltretter, Andreas Schenk, Martin Weinmann, Stefan Hinz, Christian Heipke, Markus Ulrich  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRFs) have become a rapidly growing research field with the potential to revolutionize typical photogrammetric workflows, such as those used for 3D scene reconstruction. As input, NeRFs require multi-view images with corresponding camera poses as well as the interior orientation. In the typical NeRF workflow, the camera poses and the interior orientation are estimated in advance with Structure from Motion (SfM). But the quality of the resulting novel views, which depends on different parameters such as the number and distribution of available images, as well as the accuracy of the related camera poses and interior orientation, is difficult to predict. In addition, SfM is a time-consuming pre-processing step, and its quality strongly depends on the image content. Furthermore, the undefined scaling factor of SfM hinders subsequent steps in which metric information is required. In this paper, we evaluate the potential of NeRFs for industrial robot applications. We propose an alternative to SfM pre-processing: we capture the input images with a calibrated camera that is attached to the end effector of an industrial robot and determine accurate camera poses with metric scale based on the robot kinematics. We then investigate the quality of the novel views by comparing them to ground truth, and by computing an internal quality measure based on ensemble methods. For evaluation purposes, we acquire multiple datasets that pose challenges for reconstruction typical of industrial applications, like reflective objects, poor texture, and fine structures. We show that the robot-based pose determination reaches similar accuracy as SfM in non-demanding cases, while having clear advantages in more challenging scenarios. Finally, we present first results of applying the ensemble method to estimate the quality of the synthetic novel view in the absence of a ground truth.  
  </ol>  
</details>  
**comments**: 8 pages, 8 figures, accepted for publication in The International
  Archives of the Photogrammetry, Remote Sensing and Spatial Information
  Sciences (ISPRS Archives) 2024  
  
  



