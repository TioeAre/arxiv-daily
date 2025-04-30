<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Large-scale-visual-SLAM-for-in-the-wild-videos>Large-scale visual SLAM for in-the-wild videos</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Large-scale-visual-SLAM-for-in-the-wild-videos>Large-scale visual SLAM for in-the-wild videos</a></li>
        <li><a href=#Sparse2DGS:-Geometry-Prioritized-Gaussian-Splatting-for-Surface-Reconstruction-from-Sparse-Views>Sparse2DGS: Geometry-Prioritized Gaussian Splatting for Surface Reconstruction from Sparse Views</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#GSFeatLoc:-Visual-Localization-Using-Feature-Correspondence-on-3D-Gaussian-Splatting>GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Learning-a-General-Model:-Folding-Clothing-with-Topological-Dynamics>Learning a General Model: Folding Clothing with Topological Dynamics</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Large-scale-visual-SLAM-for-in-the-wild-videos>Large-scale visual SLAM for in-the-wild videos</a></li>
        <li><a href=#GSFeatLoc:-Visual-Localization-Using-Feature-Correspondence-on-3D-Gaussian-Splatting>GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting</a></li>
        <li><a href=#Sparse2DGS:-Geometry-Prioritized-Gaussian-Splatting-for-Surface-Reconstruction-from-Sparse-Views>Sparse2DGS: Geometry-Prioritized Gaussian Splatting for Surface Reconstruction from Sparse Views</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Large-scale visual SLAM for in-the-wild videos](http://arxiv.org/abs/2504.20496)  
Shuo Sun, Torsten Sattler, Malcolm Mielle, Achim J. Lilienthal, Martin Magnusson  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Accurate and robust 3D scene reconstruction from casual, in-the-wild videos can significantly simplify robot deployment to new environments. However, reliable camera pose estimation and scene reconstruction from such unconstrained videos remains an open challenge. Existing visual-only SLAM methods perform well on benchmark datasets but struggle with real-world footage which often exhibits uncontrolled motion including rapid rotations and pure forward movements, textureless regions, and dynamic objects. We analyze the limitations of current methods and introduce a robust pipeline designed to improve 3D reconstruction from casual videos. We build upon recent deep visual odometry methods but increase robustness in several ways. Camera intrinsics are automatically recovered from the first few frames using structure-from-motion. Dynamic objects and less-constrained areas are masked with a predictive model. Additionally, we leverage monocular depth estimates to regularize bundle adjustment, mitigating errors in low-parallax situations. Finally, we integrate place recognition and loop closure to reduce long-term drift and refine both intrinsics and pose estimates through global bundle adjustment. We demonstrate large-scale contiguous 3D models from several online videos in various environments. In contrast, baseline methods typically produce locally inconsistent results at several points, producing separate segments or distorted maps. In lieu of ground-truth pose data, we evaluate map consistency, execution time and visual accuracy of re-rendered NeRF models. Our proposed system establishes a new baseline for visual reconstruction from casual uncontrolled videos found online, demonstrating more consistent reconstructions over longer sequences of in-the-wild videos than previously achieved.  
  </ol>  
</details>  
**comments**: fix the overview figure  
  
  



## SFM  

### [Large-scale visual SLAM for in-the-wild videos](http://arxiv.org/abs/2504.20496)  
Shuo Sun, Torsten Sattler, Malcolm Mielle, Achim J. Lilienthal, Martin Magnusson  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Accurate and robust 3D scene reconstruction from casual, in-the-wild videos can significantly simplify robot deployment to new environments. However, reliable camera pose estimation and scene reconstruction from such unconstrained videos remains an open challenge. Existing visual-only SLAM methods perform well on benchmark datasets but struggle with real-world footage which often exhibits uncontrolled motion including rapid rotations and pure forward movements, textureless regions, and dynamic objects. We analyze the limitations of current methods and introduce a robust pipeline designed to improve 3D reconstruction from casual videos. We build upon recent deep visual odometry methods but increase robustness in several ways. Camera intrinsics are automatically recovered from the first few frames using structure-from-motion. Dynamic objects and less-constrained areas are masked with a predictive model. Additionally, we leverage monocular depth estimates to regularize bundle adjustment, mitigating errors in low-parallax situations. Finally, we integrate place recognition and loop closure to reduce long-term drift and refine both intrinsics and pose estimates through global bundle adjustment. We demonstrate large-scale contiguous 3D models from several online videos in various environments. In contrast, baseline methods typically produce locally inconsistent results at several points, producing separate segments or distorted maps. In lieu of ground-truth pose data, we evaluate map consistency, execution time and visual accuracy of re-rendered NeRF models. Our proposed system establishes a new baseline for visual reconstruction from casual uncontrolled videos found online, demonstrating more consistent reconstructions over longer sequences of in-the-wild videos than previously achieved.  
  </ol>  
</details>  
**comments**: fix the overview figure  
  
### [Sparse2DGS: Geometry-Prioritized Gaussian Splatting for Surface Reconstruction from Sparse Views](http://arxiv.org/abs/2504.20378)  
Jiang Wu, Rui Li, Yu Zhu, Rong Guo, Jinqiu Sun, Yanning Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present a Gaussian Splatting method for surface reconstruction using sparse input views. Previous methods relying on dense views struggle with extremely sparse Structure-from-Motion points for initialization. While learning-based Multi-view Stereo (MVS) provides dense 3D points, directly combining it with Gaussian Splatting leads to suboptimal results due to the ill-posed nature of sparse-view geometric optimization. We propose Sparse2DGS, an MVS-initialized Gaussian Splatting pipeline for complete and accurate reconstruction. Our key insight is to incorporate the geometric-prioritized enhancement schemes, allowing for direct and robust geometric learning under ill-posed conditions. Sparse2DGS outperforms existing methods by notable margins while being ${2}\times$ faster than the NeRF-based fine-tuning approach.  
  </ol>  
</details>  
**comments**: CVPR 2025  
  
  



## Visual Localization  

### [GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting](http://arxiv.org/abs/2504.20379)  
Jongwon Lee, Timothy Bretl  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we present a method for localizing a query image with respect to a precomputed 3D Gaussian Splatting (3DGS) scene representation. First, the method uses 3DGS to render a synthetic RGBD image at some initial pose estimate. Second, it establishes 2D-2D correspondences between the query image and this synthetic image. Third, it uses the depth map to lift the 2D-2D correspondences to 2D-3D correspondences and solves a perspective-n-point (PnP) problem to produce a final pose estimate. Results from evaluation across three existing datasets with 38 scenes and over 2,700 test images show that our method significantly reduces both inference time (by over two orders of magnitude, from more than 10 seconds to as fast as 0.1 seconds) and estimation error compared to baseline methods that use photometric loss minimization. Results also show that our method tolerates large errors in the initial pose estimate of up to 55{\deg} in rotation and 1.1 units in translation (normalized by scene scale), achieving final pose errors of less than 5{\deg} in rotation and 0.05 units in translation on 90% of images from the Synthetic NeRF and Mip-NeRF360 datasets and on 42% of images from the more challenging Tanks and Temples dataset.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Learning a General Model: Folding Clothing with Topological Dynamics](http://arxiv.org/abs/2504.20720)  
Yiming Liu, Lijun Han, Enlin Gu, Hesheng Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The high degrees of freedom and complex structure of garments present significant challenges for clothing manipulation. In this paper, we propose a general topological dynamics model to fold complex clothing. By utilizing the visible folding structure as the topological skeleton, we design a novel topological graph to represent the clothing state. This topological graph is low-dimensional and applied for complex clothing in various folding states. It indicates the constraints of clothing and enables predictions regarding clothing movement. To extract graphs from self-occlusion, we apply semantic segmentation to analyze the occlusion relationships and decompose the clothing structure. The decomposed structure is then combined with keypoint detection to generate the topological graph. To analyze the behavior of the topological graph, we employ an improved Graph Neural Network (GNN) to learn the general dynamics. The GNN model can predict the deformation of clothing and is employed to calculate the deformation Jacobi matrix for control. Experiments using jackets validate the algorithm's effectiveness to recognize and fold complex clothing with self-occlusion.  
  </ol>  
</details>  
  
  



## NeRF  

### [Large-scale visual SLAM for in-the-wild videos](http://arxiv.org/abs/2504.20496)  
Shuo Sun, Torsten Sattler, Malcolm Mielle, Achim J. Lilienthal, Martin Magnusson  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Accurate and robust 3D scene reconstruction from casual, in-the-wild videos can significantly simplify robot deployment to new environments. However, reliable camera pose estimation and scene reconstruction from such unconstrained videos remains an open challenge. Existing visual-only SLAM methods perform well on benchmark datasets but struggle with real-world footage which often exhibits uncontrolled motion including rapid rotations and pure forward movements, textureless regions, and dynamic objects. We analyze the limitations of current methods and introduce a robust pipeline designed to improve 3D reconstruction from casual videos. We build upon recent deep visual odometry methods but increase robustness in several ways. Camera intrinsics are automatically recovered from the first few frames using structure-from-motion. Dynamic objects and less-constrained areas are masked with a predictive model. Additionally, we leverage monocular depth estimates to regularize bundle adjustment, mitigating errors in low-parallax situations. Finally, we integrate place recognition and loop closure to reduce long-term drift and refine both intrinsics and pose estimates through global bundle adjustment. We demonstrate large-scale contiguous 3D models from several online videos in various environments. In contrast, baseline methods typically produce locally inconsistent results at several points, producing separate segments or distorted maps. In lieu of ground-truth pose data, we evaluate map consistency, execution time and visual accuracy of re-rendered NeRF models. Our proposed system establishes a new baseline for visual reconstruction from casual uncontrolled videos found online, demonstrating more consistent reconstructions over longer sequences of in-the-wild videos than previously achieved.  
  </ol>  
</details>  
**comments**: fix the overview figure  
  
### [GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting](http://arxiv.org/abs/2504.20379)  
Jongwon Lee, Timothy Bretl  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we present a method for localizing a query image with respect to a precomputed 3D Gaussian Splatting (3DGS) scene representation. First, the method uses 3DGS to render a synthetic RGBD image at some initial pose estimate. Second, it establishes 2D-2D correspondences between the query image and this synthetic image. Third, it uses the depth map to lift the 2D-2D correspondences to 2D-3D correspondences and solves a perspective-n-point (PnP) problem to produce a final pose estimate. Results from evaluation across three existing datasets with 38 scenes and over 2,700 test images show that our method significantly reduces both inference time (by over two orders of magnitude, from more than 10 seconds to as fast as 0.1 seconds) and estimation error compared to baseline methods that use photometric loss minimization. Results also show that our method tolerates large errors in the initial pose estimate of up to 55{\deg} in rotation and 1.1 units in translation (normalized by scene scale), achieving final pose errors of less than 5{\deg} in rotation and 0.05 units in translation on 90% of images from the Synthetic NeRF and Mip-NeRF360 datasets and on 42% of images from the more challenging Tanks and Temples dataset.  
  </ol>  
</details>  
  
### [Sparse2DGS: Geometry-Prioritized Gaussian Splatting for Surface Reconstruction from Sparse Views](http://arxiv.org/abs/2504.20378)  
Jiang Wu, Rui Li, Yu Zhu, Rong Guo, Jinqiu Sun, Yanning Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present a Gaussian Splatting method for surface reconstruction using sparse input views. Previous methods relying on dense views struggle with extremely sparse Structure-from-Motion points for initialization. While learning-based Multi-view Stereo (MVS) provides dense 3D points, directly combining it with Gaussian Splatting leads to suboptimal results due to the ill-posed nature of sparse-view geometric optimization. We propose Sparse2DGS, an MVS-initialized Gaussian Splatting pipeline for complete and accurate reconstruction. Our key insight is to incorporate the geometric-prioritized enhancement schemes, allowing for direct and robust geometric learning under ill-posed conditions. Sparse2DGS outperforms existing methods by notable margins while being ${2}\times$ faster than the NeRF-based fine-tuning approach.  
  </ol>  
</details>  
**comments**: CVPR 2025  
  
  



