<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Volume-Encoding-Gaussians:-Transfer-Function-Agnostic-3D-Gaussians-for-Volume-Rendering>Volume Encoding Gaussians: Transfer Function-Agnostic 3D Gaussians for Volume Rendering</a></li>
        <li><a href=#EDGS:-Eliminating-Densification-for-Efficient-Convergence-of-3DGS>EDGS: Eliminating Densification for Efficient Convergence of 3DGS</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Outlier-Robust-Multi-Model-Fitting-on-Quantum-Annealers>Outlier-Robust Multi-Model Fitting on Quantum Annealers</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#SLAM&Render:-A-Benchmark-for-the-Intersection-Between-Neural-Rendering,-Gaussian-Splatting-and-SLAM>SLAM&Render: A Benchmark for the Intersection Between Neural Rendering, Gaussian Splatting and SLAM</a></li>
        <li><a href=#BEV-GS:-Feed-forward-Gaussian-Splatting-in-Bird's-Eye-View-for-Road-Reconstruction>BEV-GS: Feed-forward Gaussian Splatting in Bird's-Eye-View for Road Reconstruction</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Volume Encoding Gaussians: Transfer Function-Agnostic 3D Gaussians for Volume Rendering](http://arxiv.org/abs/2504.13339)  
Landon Dyken, Andres Sewell, Will Usher, Steve Petruzza, Sidharth Kumar  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    While HPC resources are increasingly being used to produce adaptively refined or unstructured volume datasets, current research in applying machine learning-based representation to visualization has largely ignored this type of data. To address this, we introduce Volume Encoding Gaussians (VEG), a novel 3D Gaussian-based representation for scientific volume visualization focused on unstructured volumes. Unlike prior 3D Gaussian Splatting (3DGS) methods that store view-dependent color and opacity for each Gaussian, VEG decouple the visual appearance from the data representation by encoding only scalar values, enabling transfer-function-agnostic rendering of 3DGS models for interactive scientific visualization. VEG are directly initialized from volume datasets, eliminating the need for structure-from-motion pipelines like COLMAP. To ensure complete scalar field coverage, we introduce an opacity-guided training strategy, using differentiable rendering with multiple transfer functions to optimize our data representation. This allows VEG to preserve fine features across the full scalar range of a dataset while remaining independent of any specific transfer function. Each Gaussian is scaled and rotated to adapt to local geometry, allowing for efficient representation of unstructured meshes without storing mesh connectivity and while using far fewer primitives. Across a diverse set of data, VEG achieve high reconstruction quality, compress large volume datasets by up to 3600x, and support lightning-fast rendering on commodity GPUs, enabling interactive visualization of large-scale structured and unstructured volumes.  
  </ol>  
</details>  
  
### [EDGS: Eliminating Densification for Efficient Convergence of 3DGS](http://arxiv.org/abs/2504.13204)  
Dmytro Kotovenko, Olga Grebenkova, Björn Ommer  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D Gaussian Splatting reconstructs scenes by starting from a sparse Structure-from-Motion initialization and iteratively refining under-reconstructed regions. This process is inherently slow, as it requires multiple densification steps where Gaussians are repeatedly split and adjusted, following a lengthy optimization path. Moreover, this incremental approach often leads to suboptimal renderings, particularly in high-frequency regions where detail is critical.   We propose a fundamentally different approach: we eliminate densification process with a one-step approximation of scene geometry using triangulated pixels from dense image correspondences. This dense initialization allows us to estimate rough geometry of the scene while preserving rich details from input RGB images, providing each Gaussian with well-informed colors, scales, and positions. As a result, we dramatically shorten the optimization path and remove the need for densification. Unlike traditional methods that rely on sparse keypoints, our dense initialization ensures uniform detail across the scene, even in high-frequency regions where 3DGS and other methods struggle. Moreover, since all splats are initialized in parallel at the start of optimization, we eliminate the need to wait for densification to adjust new Gaussians.   Our method not only outperforms speed-optimized models in training efficiency but also achieves higher rendering quality than state-of-the-art approaches, all while using only half the splats of standard 3DGS. It is fully compatible with other 3DGS acceleration techniques, making it a versatile and efficient solution that can be integrated with existing approaches.  
  </ol>  
</details>  
  
  



## Image Matching  

### [Outlier-Robust Multi-Model Fitting on Quantum Annealers](http://arxiv.org/abs/2504.13836)  
Saurabh Pandey, Luca Magri, Federica Arrigoni, Vladislav Golyanik  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Multi-model fitting (MMF) presents a significant challenge in Computer Vision, particularly due to its combinatorial nature. While recent advancements in quantum computing offer promise for addressing NP-hard problems, existing quantum-based approaches for model fitting are either limited to a single model or consider multi-model scenarios within outlier-free datasets. This paper introduces a novel approach, the robust quantum multi-model fitting (R-QuMF) algorithm, designed to handle outliers effectively. Our method leverages the intrinsic capabilities of quantum hardware to tackle combinatorial challenges inherent in MMF tasks, and it does not require prior knowledge of the exact number of models, thereby enhancing its practical applicability. By formulating the problem as a maximum set coverage task for adiabatic quantum computers (AQC), R-QuMF outperforms existing quantum techniques, demonstrating superior performance across various synthetic and real-world 3D datasets. Our findings underscore the potential of quantum computing in addressing the complexities of MMF, especially in real-world scenarios with noisy and outlier-prone data.  
  </ol>  
</details>  
**comments**: Accepted at CVPR 2025 Workshop "Image Matching: Local Features &
  Beyond"  
  
  



## NeRF  

### [SLAM&Render: A Benchmark for the Intersection Between Neural Rendering, Gaussian Splatting and SLAM](http://arxiv.org/abs/2504.13713)  
[[code](https://github.com/samuel-cerezo/SLAM-Render)]  
Samuel Cerezo, Gaetano Meli, Tomás Berriel Martins, Kirill Safronov, Javier Civera  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Models and methods originally developed for novel view synthesis and scene rendering, such as Neural Radiance Fields (NeRF) and Gaussian Splatting, are increasingly being adopted as representations in Simultaneous Localization and Mapping (SLAM). However, existing datasets fail to include the specific challenges of both fields, such as multimodality and sequentiality in SLAM or generalization across viewpoints and illumination conditions in neural rendering. To bridge this gap, we introduce SLAM&Render, a novel dataset designed to benchmark methods in the intersection between SLAM and novel view rendering. It consists of 40 sequences with synchronized RGB, depth, IMU, robot kinematic data, and ground-truth pose streams. By releasing robot kinematic data, the dataset also enables the assessment of novel SLAM strategies when applied to robot manipulators. The dataset sequences span five different setups featuring consumer and industrial objects under four different lighting conditions, with separate training and test trajectories per scene, as well as object rearrangements. Our experimental results, obtained with several baselines from the literature, validate SLAM&Render as a relevant benchmark for this emerging research area.  
  </ol>  
</details>  
  
### [BEV-GS: Feed-forward Gaussian Splatting in Bird's-Eye-View for Road Reconstruction](http://arxiv.org/abs/2504.13207)  
Wenhua Wu, Tong Zhao, Chensheng Peng, Lei Yang, Yintao Wei, Zhe Liu, Hesheng Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Road surface is the sole contact medium for wheels or robot feet. Reconstructing road surface is crucial for unmanned vehicles and mobile robots. Recent studies on Neural Radiance Fields (NeRF) and Gaussian Splatting (GS) have achieved remarkable results in scene reconstruction. However, they typically rely on multi-view image inputs and require prolonged optimization times. In this paper, we propose BEV-GS, a real-time single-frame road surface reconstruction method based on feed-forward Gaussian splatting. BEV-GS consists of a prediction module and a rendering module. The prediction module introduces separate geometry and texture networks following Bird's-Eye-View paradigm. Geometric and texture parameters are directly estimated from a single frame, avoiding per-scene optimization. In the rendering module, we utilize grid Gaussian for road surface representation and novel view synthesis, which better aligns with road surface characteristics. Our method achieves state-of-the-art performance on the real-world dataset RSRD. The road elevation error reduces to 1.73 cm, and the PSNR of novel view synthesis reaches 28.36 dB. The prediction and rendering FPS is 26, and 2061, respectively, enabling high-accuracy and real-time applications. The code will be available at: \href{https://github.com/cat-wwh/BEV-GS}{\texttt{https://github.com/cat-wwh/BEV-GS}}  
  </ol>  
</details>  
  
  



