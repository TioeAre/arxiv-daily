<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#3D-Reconstruction-with-Fast-Dipole-Sums>3D Reconstruction with Fast Dipole Sums</a></li>
        <li><a href=#MCGMapper:-Light-Weight-Incremental-Structure-from-Motion-and-Visual-Localization-With-Planar-Markers-and-Camera-Groups>MCGMapper: Light-Weight Incremental Structure from Motion and Visual Localization With Planar Markers and Camera Groups</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#MCGMapper:-Light-Weight-Incremental-Structure-from-Motion-and-Visual-Localization-With-Planar-Markers-and-Camera-Groups>MCGMapper: Light-Weight Incremental Structure from Motion and Visual Localization With Planar Markers and Camera Groups</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Deep-PE:-A-Learning-Based-Pose-Evaluator-for-Point-Cloud-Registration>Deep-PE: A Learning-Based Pose Evaluator for Point Cloud Registration</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#F-3DGS:-Factorized-Coordinates-and-Representations-for-3D-Gaussian-Splatting>F-3DGS: Factorized Coordinates and Representations for 3D Gaussian Splatting</a></li>
        <li><a href=#PyGS:-Large-scale-Scene-Representation-with-Pyramidal-3D-Gaussian-Splatting>PyGS: Large-scale Scene Representation with Pyramidal 3D Gaussian Splatting</a></li>
        <li><a href=#Sp2360:-Sparse-view-360-Scene-Reconstruction-using-Cascaded-2D-Diffusion-Priors>Sp2360: Sparse-view 360 Scene Reconstruction using Cascaded 2D Diffusion Priors</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [3D Reconstruction with Fast Dipole Sums](http://arxiv.org/abs/2405.16788)  
Hanyu Chen, Bailey Miller, Ioannis Gkioulekas  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce a technique for the reconstruction of high-fidelity surfaces from multi-view images. Our technique uses a new point-based representation, the dipole sum, which generalizes the winding number to allow for interpolation of arbitrary per-point attributes in point clouds with noisy or outlier points. Using dipole sums allows us to represent implicit geometry and radiance fields as per-point attributes of a point cloud, which we initialize directly from structure from motion. We additionally derive Barnes-Hut fast summation schemes for accelerated forward and reverse-mode dipole sum queries. These queries facilitate the use of ray tracing to efficiently and differentiably render images with our point-based representations, and thus update their point attributes to optimize scene geometry and appearance. We evaluate this inverse rendering framework against state-of-the-art alternatives, based on ray tracing of neural representations or rasterization of Gaussian point-based representations. Our technique significantly improves reconstruction quality at equal runtimes, while also supporting more general rendering techniques such as shadow rays for direct illumination. In the supplement, we provide interactive visualizations of our results.  
  </ol>  
</details>  
**comments**: HTML supplement with interactive visualizations of all experiments
  included, download under ancillary files  
  
### [MCGMapper: Light-Weight Incremental Structure from Motion and Visual Localization With Planar Markers and Camera Groups](http://arxiv.org/abs/2405.16599)  
Yusen Xie, Zhenmin Huang, Kai Chen, Lei Zhu, Jun Ma  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Structure from Motion (SfM) and visual localization in indoor texture-less scenes and industrial scenarios present prevalent yet challenging research topics. Existing SfM methods designed for natural scenes typically yield low accuracy or map-building failures due to insufficient robust feature extraction in such settings. Visual markers, with their artificially designed features, can effectively address these issues. Nonetheless, existing marker-assisted SfM methods encounter problems like slow running speed and difficulties in convergence; and also, they are governed by the strong assumption of unique marker size. In this paper, we propose a novel SfM framework that utilizes planar markers and multiple cameras with known extrinsics to capture the surrounding environment and reconstruct the marker map. In our algorithm, the initial poses of markers and cameras are calculated with Perspective-n-Points (PnP) in the front-end, while bundle adjustment methods customized for markers and camera groups are designed in the back-end to optimize the 6-DOF pose directly. Our algorithm facilitates the reconstruction of large scenes with different marker sizes, and its accuracy and speed of map building are shown to surpass existing methods. Our approach is suitable for a wide range of scenarios, including laboratories, basements, warehouses, and other industrial settings. Furthermore, we incorporate representative scenarios into simulations and also supply our datasets with pose labels to address the scarcity of quantitative ground-truth datasets in this research field. The datasets and source code are available on GitHub.  
  </ol>  
</details>  
**comments**: 8 pages,8 figures  
  
  



## Visual Localization  

### [MCGMapper: Light-Weight Incremental Structure from Motion and Visual Localization With Planar Markers and Camera Groups](http://arxiv.org/abs/2405.16599)  
Yusen Xie, Zhenmin Huang, Kai Chen, Lei Zhu, Jun Ma  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Structure from Motion (SfM) and visual localization in indoor texture-less scenes and industrial scenarios present prevalent yet challenging research topics. Existing SfM methods designed for natural scenes typically yield low accuracy or map-building failures due to insufficient robust feature extraction in such settings. Visual markers, with their artificially designed features, can effectively address these issues. Nonetheless, existing marker-assisted SfM methods encounter problems like slow running speed and difficulties in convergence; and also, they are governed by the strong assumption of unique marker size. In this paper, we propose a novel SfM framework that utilizes planar markers and multiple cameras with known extrinsics to capture the surrounding environment and reconstruct the marker map. In our algorithm, the initial poses of markers and cameras are calculated with Perspective-n-Points (PnP) in the front-end, while bundle adjustment methods customized for markers and camera groups are designed in the back-end to optimize the 6-DOF pose directly. Our algorithm facilitates the reconstruction of large scenes with different marker sizes, and its accuracy and speed of map building are shown to surpass existing methods. Our approach is suitable for a wide range of scenarios, including laboratories, basements, warehouses, and other industrial settings. Furthermore, we incorporate representative scenarios into simulations and also supply our datasets with pose labels to address the scarcity of quantitative ground-truth datasets in this research field. The datasets and source code are available on GitHub.  
  </ol>  
</details>  
**comments**: 8 pages,8 figures  
  
  



## Keypoint Detection  

### [Deep-PE: A Learning-Based Pose Evaluator for Point Cloud Registration](http://arxiv.org/abs/2405.16085)  
Junjie Gao, Chongjian Wang, Zhongjun Ding, Shuangmin Chen, Shiqing Xin, Changhe Tu, Wenping Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In the realm of point cloud registration, the most prevalent pose evaluation approaches are statistics-based, identifying the optimal transformation by maximizing the number of consistent correspondences. However, registration recall decreases significantly when point clouds exhibit a low overlap rate, despite efforts in designing feature descriptors and establishing correspondences. In this paper, we introduce Deep-PE, a lightweight, learning-based pose evaluator designed to enhance the accuracy of pose selection, especially in challenging point cloud scenarios with low overlap. Our network incorporates a Pose-Aware Attention (PAA) module to simulate and learn the alignment status of point clouds under various candidate poses, alongside a Pose Confidence Prediction (PCP) module that predicts the likelihood of successful registration. These two modules facilitate the learning of both local and global alignment priors. Extensive tests across multiple benchmarks confirm the effectiveness of Deep-PE. Notably, on 3DLoMatch with a low overlap rate, Deep-PE significantly outperforms state-of-the-art methods by at least 8% and 11% in registration recall under handcrafted FPFH and learning-based FCGF descriptors, respectively. To the best of our knowledge, this is the first study to utilize deep learning to select the optimal pose without the explicit need for input correspondences.  
  </ol>  
</details>  
**comments**: 22 pages, 16 figures  
  
  



## NeRF  

### [F-3DGS: Factorized Coordinates and Representations for 3D Gaussian Splatting](http://arxiv.org/abs/2405.17083)  
Xiangyu Sun, Joo Chan Lee, Daniel Rho, Jong Hwan Ko, Usman Ali, Eunbyung Park  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The neural radiance field (NeRF) has made significant strides in representing 3D scenes and synthesizing novel views. Despite its advancements, the high computational costs of NeRF have posed challenges for its deployment in resource-constrained environments and real-time applications. As an alternative to NeRF-like neural rendering methods, 3D Gaussian Splatting (3DGS) offers rapid rendering speeds while maintaining excellent image quality. However, as it represents objects and scenes using a myriad of Gaussians, it requires substantial storage to achieve high-quality representation. To mitigate the storage overhead, we propose Factorized 3D Gaussian Splatting (F-3DGS), a novel approach that drastically reduces storage requirements while preserving image quality. Inspired by classical matrix and tensor factorization techniques, our method represents and approximates dense clusters of Gaussians with significantly fewer Gaussians through efficient factorization. We aim to efficiently represent dense 3D Gaussians by approximating them with a limited amount of information for each axis and their combinations. This method allows us to encode a substantially large number of Gaussians along with their essential attributes -- such as color, scale, and rotation -- necessary for rendering using a relatively small number of elements. Extensive experimental results demonstrate that F-3DGS achieves a significant reduction in storage costs while maintaining comparable quality in rendered images.  
  </ol>  
</details>  
  
### [PyGS: Large-scale Scene Representation with Pyramidal 3D Gaussian Splatting](http://arxiv.org/abs/2405.16829)  
Zipeng Wang, Dan Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRFs) have demonstrated remarkable proficiency in synthesizing photorealistic images of large-scale scenes. However, they are often plagued by a loss of fine details and long rendering durations. 3D Gaussian Splatting has recently been introduced as a potent alternative, achieving both high-fidelity visual results and accelerated rendering performance. Nonetheless, scaling 3D Gaussian Splatting is fraught with challenges. Specifically, large-scale scenes grapples with the integration of objects across multiple scales and disparate viewpoints, which often leads to compromised efficacy as the Gaussians need to balance between detail levels. Furthermore, the generation of initialization points via COLMAP from large-scale dataset is both computationally demanding and prone to incomplete reconstructions. To address these challenges, we present Pyramidal 3D Gaussian Splatting (PyGS) with NeRF Initialization. Our approach represent the scene with a hierarchical assembly of Gaussians arranged in a pyramidal fashion. The top level of the pyramid is composed of a few large Gaussians, while each subsequent layer accommodates a denser collection of smaller Gaussians. We effectively initialize these pyramidal Gaussians through sampling a rapidly trained grid-based NeRF at various frequencies. We group these pyramidal Gaussians into clusters and use a compact weighting network to dynamically determine the influence of each pyramid level of each cluster considering camera viewpoint during rendering. Our method achieves a significant performance leap across multiple large-scale datasets and attains a rendering time that is over 400 times faster than current state-of-the-art approaches.  
  </ol>  
</details>  
  
### [Sp2360: Sparse-view 360 Scene Reconstruction using Cascaded 2D Diffusion Priors](http://arxiv.org/abs/2405.16517)  
Soumava Paul, Christopher Wewer, Bernt Schiele, Jan Eric Lenssen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We aim to tackle sparse-view reconstruction of a 360 3D scene using priors from latent diffusion models (LDM). The sparse-view setting is ill-posed and underconstrained, especially for scenes where the camera rotates 360 degrees around a point, as no visual information is available beyond some frontal views focused on the central object(s) of interest. In this work, we show that pretrained 2D diffusion models can strongly improve the reconstruction of a scene with low-cost fine-tuning. Specifically, we present SparseSplat360 (Sp2360), a method that employs a cascade of in-painting and artifact removal models to fill in missing details and clean novel views. Due to superior training and rendering speeds, we use an explicit scene representation in the form of 3D Gaussians over NeRF-based implicit representations. We propose an iterative update strategy to fuse generated pseudo novel views with existing 3D Gaussians fitted to the initial sparse inputs. As a result, we obtain a multi-view consistent scene representation with details coherent with the observed inputs. Our evaluation on the challenging Mip-NeRF360 dataset shows that our proposed 2D to 3D distillation algorithm considerably improves the performance of a regularized version of 3DGS adapted to a sparse-view setting and outperforms existing sparse-view reconstruction methods in 360 scene reconstruction. Qualitatively, our method generates entire 360 scenes from as few as 9 input views, with a high degree of foreground and background detail.  
  </ol>  
</details>  
**comments**: 18 pages, 10 figures, 4 tables  
  
  



