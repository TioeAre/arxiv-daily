<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Key-Grid:-Unsupervised-3D-Keypoints-Detection-using-Grid-Heatmap-Features>Key-Grid: Unsupervised 3D Keypoints Detection using Grid Heatmap Features</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#MVGS:-Multi-view-regulated-Gaussian-Splatting-for-Novel-View-Synthesis>MVGS: Multi-view-regulated Gaussian Splatting for Novel View Synthesis</a></li>
      </ul>
    </li>
  </ol>
</details>

## Keypoint Detection  

### [Key-Grid: Unsupervised 3D Keypoints Detection using Grid Heatmap Features](http://arxiv.org/abs/2410.02237)  
Chengkai Hou, Zhengrong Xue, Bingyang Zhou, Jinghan Ke, Lin Shao, Huazhe Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Detecting 3D keypoints with semantic consistency is widely used in many scenarios such as pose estimation, shape registration and robotics. Currently, most unsupervised 3D keypoint detection methods focus on the rigid-body objects. However, when faced with deformable objects, the keypoints they identify do not preserve semantic consistency well. In this paper, we introduce an innovative unsupervised keypoint detector Key-Grid for both the rigid-body and deformable objects, which is an autoencoder framework. The encoder predicts keypoints and the decoder utilizes the generated keypoints to reconstruct the objects. Unlike previous work, we leverage the identified keypoint in formation to form a 3D grid feature heatmap called grid heatmap, which is used in the decoder section. Grid heatmap is a novel concept that represents the latent variables for grid points sampled uniformly in the 3D cubic space, where these variables are the shortest distance between the grid points and the skeleton connected by keypoint pairs. Meanwhile, we incorporate the information from each layer of the encoder into the decoder section. We conduct an extensive evaluation of Key-Grid on a list of benchmark datasets. Key-Grid achieves the state-of-the-art performance on the semantic consistency and position accuracy of keypoints. Moreover, we demonstrate the robustness of Key-Grid to noise and downsampling. In addition, we achieve SE-(3) invariance of keypoints though generalizing Key-Grid to a SE(3)-invariant backbone.  
  </ol>  
</details>  
  
  



## NeRF  

### [MVGS: Multi-view-regulated Gaussian Splatting for Novel View Synthesis](http://arxiv.org/abs/2410.02103)  
Xiaobiao Du, Yida Wang, Xin Yu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent works in volume rendering, \textit{e.g.} NeRF and 3D Gaussian Splatting (3DGS), significantly advance the rendering quality and efficiency with the help of the learned implicit neural radiance field or 3D Gaussians. Rendering on top of an explicit representation, the vanilla 3DGS and its variants deliver real-time efficiency by optimizing the parametric model with single-view supervision per iteration during training which is adopted from NeRF. Consequently, certain views are overfitted, leading to unsatisfying appearance in novel-view synthesis and imprecise 3D geometries. To solve aforementioned problems, we propose a new 3DGS optimization method embodying four key novel contributions: 1) We transform the conventional single-view training paradigm into a multi-view training strategy. With our proposed multi-view regulation, 3D Gaussian attributes are further optimized without overfitting certain training views. As a general solution, we improve the overall accuracy in a variety of scenarios and different Gaussian variants. 2) Inspired by the benefit introduced by additional views, we further propose a cross-intrinsic guidance scheme, leading to a coarse-to-fine training procedure concerning different resolutions. 3) Built on top of our multi-view regulated training, we further propose a cross-ray densification strategy, densifying more Gaussian kernels in the ray-intersect regions from a selection of views. 4) By further investigating the densification strategy, we found that the effect of densification should be enhanced when certain views are distinct dramatically. As a solution, we propose a novel multi-view augmented densification strategy, where 3D Gaussians are encouraged to get densified to a sufficient number accordingly, resulting in improved reconstruction accuracy.  
  </ol>  
</details>  
**comments**: Project Page:https://xiaobiaodu.github.io/mvgs-project/  
  
  



