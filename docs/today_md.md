<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Building-Rome-with-Convex-Optimization>Building Rome with Convex Optimization</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Learning-Street-View-Representations-with-Spatiotemporal-Contrast>Learning Street View Representations with Spatiotemporal Contrast</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#PoI:-Pixel-of-Interest-for-Novel-View-Synthesis-Assisted-Scene-Coordinate-Regression>PoI: Pixel of Interest for Novel View Synthesis Assisted Scene Coordinate Regression</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Building Rome with Convex Optimization](http://arxiv.org/abs/2502.04640)  
Haoyu Han, Heng Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Global bundle adjustment is made easy by depth prediction and convex optimization. We (i) propose a scaled bundle adjustment (SBA) formulation that lifts 2D keypoint measurements to 3D with learned depth, (ii) design an empirically tight convex semidfinite program (SDP) relaxation that solves SBA to certfiable global optimality, (iii) solve the SDP relaxations at extreme scale with Burer-Monteiro factorization and a CUDA-based trust-region Riemannian optimizer (dubbed XM), (iv) build a structure from motion (SfM) pipeline with XM as the optimization engine and show that XM-SfM dominates or compares favorably with existing SfM pipelines in terms of reconstruction quality while being faster, more scalable, and initialization-free.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Learning Street View Representations with Spatiotemporal Contrast](http://arxiv.org/abs/2502.04638)  
Yong Li, Yingjing Huang, Gengchen Mai, Fan Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Street view imagery is extensively utilized in representation learning for urban visual environments, supporting various sustainable development tasks such as environmental perception and socio-economic assessment. However, it is challenging for existing image representations to specifically encode the dynamic urban environment (such as pedestrians, vehicles, and vegetation), the built environment (including buildings, roads, and urban infrastructure), and the environmental ambiance (such as the cultural and socioeconomic atmosphere) depicted in street view imagery to address downstream tasks related to the city. In this work, we propose an innovative self-supervised learning framework that leverages temporal and spatial attributes of street view imagery to learn image representations of the dynamic urban environment for diverse downstream tasks. By employing street view images captured at the same location over time and spatially nearby views at the same time, we construct contrastive learning tasks designed to learn the temporal-invariant characteristics of the built environment and the spatial-invariant neighborhood ambiance. Our approach significantly outperforms traditional supervised and unsupervised methods in tasks such as visual place recognition, socioeconomic estimation, and human-environment perception. Moreover, we demonstrate the varying behaviors of image representations learned through different contrastive learning objectives across various downstream tasks. This study systematically discusses representation learning strategies for urban studies based on street view images, providing a benchmark that enhances the applicability of visual data in urban science. The code is available at https://github.com/yonglleee/UrbanSTCL.  
  </ol>  
</details>  
  
  



## NeRF  

### [PoI: Pixel of Interest for Novel View Synthesis Assisted Scene Coordinate Regression](http://arxiv.org/abs/2502.04843)  
Feifei Li, Qi Song, Chi Zhang, Hui Shuai, Rui Huang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The task of estimating camera poses can be enhanced through novel view synthesis techniques such as NeRF and Gaussian Splatting to increase the diversity and extension of training data. However, these techniques often produce rendered images with issues like blurring and ghosting, which compromise their reliability. These issues become particularly pronounced for Scene Coordinate Regression (SCR) methods, which estimate 3D coordinates at the pixel level. To mitigate the problems associated with unreliable rendered images, we introduce a novel filtering approach, which selectively extracts well-rendered pixels while discarding the inferior ones. This filter simultaneously measures the SCR model's real-time reprojection loss and gradient during training. Building on this filtering technique, we also develop a new strategy to improve scene coordinate regression using sparse inputs, drawing on successful applications of sparse input techniques in novel view synthesis. Our experimental results validate the effectiveness of our method, demonstrating state-of-the-art performance on indoor and outdoor datasets.  
  </ol>  
</details>  
  
  



