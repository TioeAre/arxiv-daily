<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#GT-Loc:-Unifying-When-and-Where-in-Images-Through-a-Joint-Embedding-Space>GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space</a></li>
        <li><a href=#Text-to-Remote-Sensing-Image-Retrieval-beyond-RGB-Sources>Text-to-Remote-Sensing-Image Retrieval beyond RGB Sources</a></li>
        <li><a href=#Kaleidoscopic-Background-Attack:-Disrupting-Pose-Estimation-with-Multi-Fold-Radial-Symmetry-Textures>Kaleidoscopic Background Attack: Disrupting Pose Estimation with Multi-Fold Radial Symmetry Textures</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#VoxelRF:-Voxelized-Radiance-Field-for-Fast-Wireless-Channel-Modeling>VoxelRF: Voxelized Radiance Field for Fast Wireless Channel Modeling</a></li>
        <li><a href=#Stable-Score-Distillation>Stable Score Distillation</a></li>
        <li><a href=#From-images-to-properties:-a-NeRF-driven-framework-for-granular-material-parameter-inversion>From images to properties: a NeRF-driven framework for granular material parameter inversion</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space](http://arxiv.org/abs/2507.10473)  
David G. Shatwell, Ishan Rajendrakumar Dave, Sirnam Swetha, Mubarak Shah  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Timestamp prediction aims to determine when an image was captured using only visual information, supporting applications such as metadata correction, retrieval, and digital forensics. In outdoor scenarios, hourly estimates rely on cues like brightness, hue, and shadow positioning, while seasonal changes and weather inform date estimation. However, these visual cues significantly depend on geographic context, closely linking timestamp prediction to geo-localization. To address this interdependence, we introduce GT-Loc, a novel retrieval-based method that jointly predicts the capture time (hour and month) and geo-location (GPS coordinates) of an image. Our approach employs separate encoders for images, time, and location, aligning their embeddings within a shared high-dimensional feature space. Recognizing the cyclical nature of time, instead of conventional contrastive learning with hard positives and negatives, we propose a temporal metric-learning objective providing soft targets by modeling pairwise time differences over a cyclical toroidal surface. We present new benchmarks demonstrating that our joint optimization surpasses previous time prediction methods, even those using the ground-truth geo-location as an input during inference. Additionally, our approach achieves competitive results on standard geo-localization tasks, and the unified embedding space facilitates compositional and text-based image retrieval.  
  </ol>  
</details>  
**comments**: Accepted in ICCV2025  
  
### [Text-to-Remote-Sensing-Image Retrieval beyond RGB Sources](http://arxiv.org/abs/2507.10403)  
Daniele Rege Cambrin, Lorenzo Vaiani, Giuseppe Gallipoli, Luca Cagliero, Paolo Garza  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Retrieving relevant imagery from vast satellite archives is crucial for applications like disaster response and long-term climate monitoring. However, most text-to-image retrieval systems are limited to RGB data, failing to exploit the unique physical information captured by other sensors, such as the all-weather structural sensitivity of Synthetic Aperture Radar (SAR) or the spectral signatures in optical multispectral data. To bridge this gap, we introduce CrisisLandMark, a new large-scale corpus of over 647,000 Sentinel-1 SAR and Sentinel-2 multispectral images paired with structured textual annotations for land cover, land use, and crisis events harmonized from authoritative land cover systems (CORINE and Dynamic World) and crisis-specific sources. We then present CLOSP (Contrastive Language Optical SAR Pretraining), a novel framework that uses text as a bridge to align unpaired optical and SAR images into a unified embedding space. Our experiments show that CLOSP achieves a new state-of-the-art, improving retrieval nDGC by 54% over existing models. Additionally, we find that the unified training strategy overcomes the inherent difficulty of interpreting SAR imagery by transferring rich semantic knowledge from the optical domain with indirect interaction. Furthermore, GeoCLOSP, which integrates geographic coordinates into our framework, creates a powerful trade-off between generality and specificity: while the CLOSP excels at general semantic tasks, the GeoCLOSP becomes a specialized expert for retrieving location-dependent crisis events and rare geographic features. This work highlights that the integration of diverse sensor data and geographic context is essential for unlocking the full potential of remote sensing archives.  
  </ol>  
</details>  
  
### [Kaleidoscopic Background Attack: Disrupting Pose Estimation with Multi-Fold Radial Symmetry Textures](http://arxiv.org/abs/2507.10265)  
Xinlong Ding, Hongwei Yu, Jiawei Li, Feifan Li, Yu Shang, Bochao Zou, Huimin Ma, Jiansheng Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Camera pose estimation is a fundamental computer vision task that is essential for applications like visual localization and multi-view stereo reconstruction. In the object-centric scenarios with sparse inputs, the accuracy of pose estimation can be significantly influenced by background textures that occupy major portions of the images across different viewpoints. In light of this, we introduce the Kaleidoscopic Background Attack (KBA), which uses identical segments to form discs with multi-fold radial symmetry. These discs maintain high similarity across different viewpoints, enabling effective attacks on pose estimation models even with natural texture segments. Additionally, a projected orientation consistency loss is proposed to optimize the kaleidoscopic segments, leading to significant enhancement in the attack effectiveness. Experimental results show that optimized adversarial kaleidoscopic backgrounds can effectively attack various camera pose estimation models.  
  </ol>  
</details>  
**comments**: Accepted at ICCV 2025. Project page is available at
  https://wakuwu.github.io/KBA  
  
  



## NeRF  

### [VoxelRF: Voxelized Radiance Field for Fast Wireless Channel Modeling](http://arxiv.org/abs/2507.09987)  
Zihang Zeng, Shu Sun, Meixia Tao, Yin Xu, Xianghao Yu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Wireless channel modeling in complex environments is crucial for wireless communication system design and deployment. Traditional channel modeling approaches face challenges in balancing accuracy, efficiency, and scalability, while recent neural approaches such as neural radiance field (NeRF) suffer from long training and slow inference. To tackle these challenges, we propose voxelized radiance field (VoxelRF), a novel neural representation for wireless channel modeling that enables fast and accurate synthesis of spatial spectra. VoxelRF replaces the costly multilayer perception (MLP) used in NeRF-based methods with trilinear interpolation of voxel grid-based representation, and two shallow MLPs to model both propagation and transmitter-dependent effects. To further accelerate training and improve generalization, we introduce progressive learning, empty space skipping, and an additional background entropy loss function. Experimental results demonstrate that VoxelRF achieves competitive accuracy with significantly reduced computation and limited training data, making it more practical for real-time and resource-constrained wireless applications.  
  </ol>  
</details>  
  
### [Stable Score Distillation](http://arxiv.org/abs/2507.09168)  
Haiming Zhu, Yangyang Xu, Chenshu Xu, Tingrui Shen, Wenxi Liu, Yong Du, Jun Yu, Shengfeng He  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Text-guided image and 3D editing have advanced with diffusion-based models, yet methods like Delta Denoising Score often struggle with stability, spatial control, and editing strength. These limitations stem from reliance on complex auxiliary structures, which introduce conflicting optimization signals and restrict precise, localized edits. We introduce Stable Score Distillation (SSD), a streamlined framework that enhances stability and alignment in the editing process by anchoring a single classifier to the source prompt. Specifically, SSD utilizes Classifier-Free Guidance (CFG) equation to achieves cross-prompt alignment, and introduces a constant term null-text branch to stabilize the optimization process. This approach preserves the original content's structure and ensures that editing trajectories are closely aligned with the source prompt, enabling smooth, prompt-specific modifications while maintaining coherence in surrounding regions. Additionally, SSD incorporates a prompt enhancement branch to boost editing strength, particularly for style transformations. Our method achieves state-of-the-art results in 2D and 3D editing tasks, including NeRF and text-driven style edits, with faster convergence and reduced complexity, providing a robust and efficient solution for text-guided editing.  
  </ol>  
</details>  
  
### [From images to properties: a NeRF-driven framework for granular material parameter inversion](http://arxiv.org/abs/2507.09005)  
Cheng-Hsi Hsiao, Krishna Kumar  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce a novel framework that integrates Neural Radiance Fields (NeRF) with Material Point Method (MPM) simulation to infer granular material properties from visual observations. Our approach begins by generating synthetic experimental data, simulating an plow interacting with sand. The experiment is rendered into realistic images as the photographic observations. These observations include multi-view images of the experiment's initial state and time-sequenced images from two fixed cameras. Using NeRF, we reconstruct the 3D geometry from the initial multi-view images, leveraging its capability to synthesize novel viewpoints and capture intricate surface details. The reconstructed geometry is then used to initialize material point positions for the MPM simulation, where the friction angle remains unknown. We render images of the simulation under the same camera setup and compare them to the observed images. By employing Bayesian optimization, we minimize the image loss to estimate the best-fitting friction angle. Our results demonstrate that friction angle can be estimated with an error within 2 degrees, highlighting the effectiveness of inverse analysis through purely visual observations. This approach offers a promising solution for characterizing granular materials in real-world scenarios where direct measurement is impractical or impossible.  
  </ol>  
</details>  
  
  



