<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Re-localization-acceleration-with-Medoid-Silhouette-Clustering>Re-localization acceleration with Medoid Silhouette Clustering</a></li>
        <li><a href=#A-flexible-framework-for-accurate-LiDAR-odometry,-map-manipulation,-and-localization>A flexible framework for accurate LiDAR odometry, map manipulation, and localization</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Dynamic-Scene-Understanding-through-Object-Centric-Voxelization-and-Neural-Rendering>Dynamic Scene Understanding through Object-Centric Voxelization and Neural Rendering</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Re-localization acceleration with Medoid Silhouette Clustering](http://arxiv.org/abs/2407.20749)  
Hongyi Zhang, Walterio Mayol-Cuevas  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Two crucial performance criteria for the deployment of visual localization are speed and accuracy. Current research on visual localization with neural networks is limited to examining methods for enhancing the accuracy of networks across various datasets. How to expedite the re-localization process within deep neural network architectures still needs further investigation. In this paper, we present a novel approach for accelerating visual re-localization in practice. A tree-like search strategy, built on the keyframes extracted by a visual clustering algorithm, is designed for matching acceleration. Our method has been validated on two tasks across three public datasets, allowing for 50 up to 90 percent time saving over the baseline while not reducing location accuracy.  
  </ol>  
</details>  
**comments**: 11 pages, 6 figures  
  
### [A flexible framework for accurate LiDAR odometry, map manipulation, and localization](http://arxiv.org/abs/2407.20465)  
José Luis Blanco-Claraco  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    LiDAR-based SLAM is a core technology for autonomous vehicles and robots. Despite the intense research activity in this field, each proposed system uses a particular sensor post-processing pipeline and a single map representation format. The present work aims at introducing a revolutionary point of view for 3D LiDAR SLAM and localization: (1) using view-based maps as the fundamental representation of maps ("simple-maps"), which can then be used to generate arbitrary metric maps optimized for particular tasks; and (2) by introducing a new framework in which mapping pipelines can be defined without coding, defining the connections of a network of reusable blocks much like deep-learning networks are designed by connecting layers of standardized elements. Moreover, the idea of including the current linear and angular velocity vectors as variables to be optimized within the ICP loop is also introduced, leading to superior robustness against aggressive motion profiles without an IMU. The presented open-source ecosystem, released to ROS 2, includes tools and prebuilt pipelines covering all the way from data acquisition to map editing and visualization, real-time localization, loop-closure detection, or map georeferencing from consumer-grade GNSS receivers. Extensive experimental validation reveals that the proposal compares well to, or improves, former state-of-the-art (SOTA) LiDAR odometry systems, while also successfully mapping some hard sequences where others diverge. A proposed self-adaptive configuration has been used, without parameter changes, for all 3D LiDAR datasets with sensors between 16 and 128 rings, extensively tested on 83 sequences over more than 250~km of automotive, hand-held, airborne, and quadruped LiDAR datasets, both indoors and outdoors. The open-sourced implementation is available online at https://github.com/MOLAorg/mola  
  </ol>  
</details>  
**comments**: 41 pages, 30 figures  
  
  



## NeRF  

### [Dynamic Scene Understanding through Object-Centric Voxelization and Neural Rendering](http://arxiv.org/abs/2407.20908)  
[[code](https://github.com/zyp123494/dynavol)]  
Yanpeng Zhao, Yiwei Hao, Siyu Gao, Yunbo Wang, Xiaokang Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Learning object-centric representations from unsupervised videos is challenging. Unlike most previous approaches that focus on decomposing 2D images, we present a 3D generative model named DynaVol-S for dynamic scenes that enables object-centric learning within a differentiable volume rendering framework. The key idea is to perform object-centric voxelization to capture the 3D nature of the scene, which infers per-object occupancy probabilities at individual spatial locations. These voxel features evolve through a canonical-space deformation function and are optimized in an inverse rendering pipeline with a compositional NeRF. Additionally, our approach integrates 2D semantic features to create 3D semantic grids, representing the scene through multiple disentangled voxel grids. DynaVol-S significantly outperforms existing models in both novel view synthesis and unsupervised decomposition tasks for dynamic scenes. By jointly considering geometric structures and semantic features, it effectively addresses challenging real-world scenarios involving complex object interactions. Furthermore, once trained, the explicitly meaningful voxel features enable additional capabilities that 2D scene decomposition methods cannot achieve, such as novel scene generation through editing geometric shapes or manipulating the motion trajectories of objects.  
  </ol>  
</details>  
  
  



