<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#LoopSparseGS:-Loop-Based-Sparse-View-Friendly-Gaussian-Splatting>LoopSparseGS: Loop Based Sparse-View Friendly Gaussian Splatting</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Certifying-Robustness-of-Learning-Based-Keypoint-Detection-and-Pose-Estimation-Methods>Certifying Robustness of Learning-Based Keypoint Detection and Pose Estimation Methods</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#StyleRF-VolVis:-Style-Transfer-of-Neural-Radiance-Fields-for-Expressive-Volume-Visualization>StyleRF-VolVis: Style Transfer of Neural Radiance Fields for Expressive Volume Visualization</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [LoopSparseGS: Loop Based Sparse-View Friendly Gaussian Splatting](http://arxiv.org/abs/2408.00254)  
Zhenyu Bao, Guibiao Liao, Kaichen Zhou, Kanglin Liu, Qing Li, Guoping Qiu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Despite the photorealistic novel view synthesis (NVS) performance achieved by the original 3D Gaussian splatting (3DGS), its rendering quality significantly degrades with sparse input views. This performance drop is mainly caused by the limited number of initial points generated from the sparse input, insufficient supervision during the training process, and inadequate regularization of the oversized Gaussian ellipsoids. To handle these issues, we propose the LoopSparseGS, a loop-based 3DGS framework for the sparse novel view synthesis task. In specific, we propose a loop-based Progressive Gaussian Initialization (PGI) strategy that could iteratively densify the initialized point cloud using the rendered pseudo images during the training process. Then, the sparse and reliable depth from the Structure from Motion, and the window-based dense monocular depth are leveraged to provide precise geometric supervision via the proposed Depth-alignment Regularization (DAR). Additionally, we introduce a novel Sparse-friendly Sampling (SFS) strategy to handle oversized Gaussian ellipsoids leading to large pixel errors. Comprehensive experiments on four datasets demonstrate that LoopSparseGS outperforms existing state-of-the-art methods for sparse-input novel view synthesis, across indoor, outdoor, and object-level scenes with various image resolutions.  
  </ol>  
</details>  
**comments**: 13 pages, 10 figures  
  
  



## Keypoint Detection  

### [Certifying Robustness of Learning-Based Keypoint Detection and Pose Estimation Methods](http://arxiv.org/abs/2408.00117)  
Xusheng Luo, Tianhao Wei, Simin Liu, Ziwei Wang, Luis Mattei-Mendez, Taylor Loper, Joshua Neighbor, Casidhe Hutchison, Changliu Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This work addresses the certification of the local robustness of vision-based two-stage 6D object pose estimation. The two-stage method for object pose estimation achieves superior accuracy by first employing deep neural network-driven keypoint regression and then applying a Perspective-n-Point (PnP) technique. Despite advancements, the certification of these methods' robustness remains scarce. This research aims to fill this gap with a focus on their local robustness on the system level--the capacity to maintain robust estimations amidst semantic input perturbations. The core idea is to transform the certification of local robustness into neural network verification for classification tasks. The challenge is to develop model, input, and output specifications that align with off-the-shelf verification tools. To facilitate verification, we modify the keypoint detection model by substituting nonlinear operations with those more amenable to the verification processes. Instead of injecting random noise into images, as is common, we employ a convex hull representation of images as input specifications to more accurately depict semantic perturbations. Furthermore, by conducting a sensitivity analysis, we propagate the robustness criteria from pose to keypoint accuracy, and then formulating an optimal error threshold allocation problem that allows for the setting of a maximally permissible keypoint deviation thresholds. Viewing each pixel as an individual class, these thresholds result in linear, classification-akin output specifications. Under certain conditions, we demonstrate that the main components of our certification framework are both sound and complete, and validate its effects through extensive evaluations on realistic perturbations. To our knowledge, this is the first study to certify the robustness of large-scale, keypoint-based pose estimation given images in real-world scenarios.  
  </ol>  
</details>  
**comments**: 25 pages, 10 figures, 5 tables  
  
  



## NeRF  

### [StyleRF-VolVis: Style Transfer of Neural Radiance Fields for Expressive Volume Visualization](http://arxiv.org/abs/2408.00150)  
Kaiyuan Tang, Chaoli Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In volume visualization, visualization synthesis has attracted much attention due to its ability to generate novel visualizations without following the conventional rendering pipeline. However, existing solutions based on generative adversarial networks often require many training images and take significant training time. Still, issues such as low quality, consistency, and flexibility persist. This paper introduces StyleRF-VolVis, an innovative style transfer framework for expressive volume visualization (VolVis) via neural radiance field (NeRF). The expressiveness of StyleRF-VolVis is upheld by its ability to accurately separate the underlying scene geometry (i.e., content) and color appearance (i.e., style), conveniently modify color, opacity, and lighting of the original rendering while maintaining visual content consistency across the views, and effectively transfer arbitrary styles from reference images to the reconstructed 3D scene. To achieve these, we design a base NeRF model for scene geometry extraction, a palette color network to classify regions of the radiance field for photorealistic editing, and an unrestricted color network to lift the color palette constraint via knowledge distillation for non-photorealistic editing. We demonstrate the superior quality, consistency, and flexibility of StyleRF-VolVis by experimenting with various volume rendering scenes and reference images and comparing StyleRF-VolVis against other image-based (AdaIN), video-based (ReReVST), and NeRF-based (ARF and SNeRF) style rendering solutions.  
  </ol>  
</details>  
**comments**: Accepted by IEEE VIS 2024  
  
  



