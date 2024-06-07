<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Localized-Gaussian-Point-Management>Localized Gaussian Point Management</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#GLACE:-Global-Local-Accelerated-Coordinate-Encoding>GLACE: Global Local Accelerated Coordinate Encoding</a></li>
        <li><a href=#Monocular-Localization-with-Semantics-Map-for-Autonomous-Vehicles>Monocular Localization with Semantics Map for Autonomous Vehicles</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Improving-Physics-Augmented-Continuum-Neural-Radiance-Field-Based-Geometry-Agnostic-System-Identification-with-Lagrangian-Particle-Optimization>Improving Physics-Augmented Continuum Neural Radiance Field-Based Geometry-Agnostic System Identification with Lagrangian Particle Optimization</a></li>
        <li><a href=#How-Far-Can-We-Compress-Instant-NGP-Based-NeRF?>How Far Can We Compress Instant-NGP-Based NeRF?</a></li>
        <li><a href=#Gear-NeRF:-Free-Viewpoint-Rendering-and-Tracking-with-Motion-aware-Spatio-Temporal-Sampling>Gear-NeRF: Free-Viewpoint Rendering and Tracking with Motion-aware Spatio-Temporal Sampling</a></li>
        <li><a href=#Superpoint-Gaussian-Splatting-for-Real-Time-High-Fidelity-Dynamic-Scene-Reconstruction>Superpoint Gaussian Splatting for Real-Time High-Fidelity Dynamic Scene Reconstruction</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Localized Gaussian Point Management](http://arxiv.org/abs/2406.04251)  
Haosen Yang, Chenhao Zhang, Wenqing Wang, Marco Volino, Adrian Hilton, Li Zhang, Xiatian Zhu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Point management is a critical component in optimizing 3D Gaussian Splatting (3DGS) models, as the point initiation (e.g., via structure from motion) is distributionally inappropriate. Typically, the Adaptive Density Control (ADC) algorithm is applied, leveraging view-averaged gradient magnitude thresholding for point densification, opacity thresholding for pruning, and regular all-points opacity reset. However, we reveal that this strategy is limited in tackling intricate/special image regions (e.g., transparent) as it is unable to identify all the 3D zones that require point densification, and lacking an appropriate mechanism to handle the ill-conditioned points with negative impacts (occlusion due to false high opacity). To address these limitations, we propose a Localized Point Management (LPM) strategy, capable of identifying those error-contributing zones in the highest demand for both point addition and geometry calibration. Zone identification is achieved by leveraging the underlying multiview geometry constraints, with the guidance of image rendering errors. We apply point densification in the identified zone, whilst resetting the opacity of those points residing in front of these regions so that a new opportunity is created to correct ill-conditioned points. Serving as a versatile plugin, LPM can be seamlessly integrated into existing 3D Gaussian Splatting models. Experimental evaluation across both static 3D and dynamic 4D scenes validate the efficacy of our LPM strategy in boosting a variety of existing 3DGS models both quantitatively and qualitatively. Notably, LPM improves both vanilla 3DGS and SpaceTimeGS to achieve state-of-the-art rendering quality while retaining real-time speeds, outperforming on challenging datasets such as Tanks & Temples and the Neural 3D Video Dataset.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [GLACE: Global Local Accelerated Coordinate Encoding](http://arxiv.org/abs/2406.04340)  
[[code](https://github.com/cvg/glace)]  
Fangjinhua Wang, Xudong Jiang, Silvano Galliani, Christoph Vogel, Marc Pollefeys  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Scene coordinate regression (SCR) methods are a family of visual localization methods that directly regress 2D-3D matches for camera pose estimation. They are effective in small-scale scenes but face significant challenges in large-scale scenes that are further amplified in the absence of ground truth 3D point clouds for supervision. Here, the model can only rely on reprojection constraints and needs to implicitly triangulate the points. The challenges stem from a fundamental dilemma: The network has to be invariant to observations of the same landmark at different viewpoints and lighting conditions, etc., but at the same time discriminate unrelated but similar observations. The latter becomes more relevant and severe in larger scenes. In this work, we tackle this problem by introducing the concept of co-visibility to the network. We propose GLACE, which integrates pre-trained global and local encodings and enables SCR to scale to large scenes with only a single small-sized network. Specifically, we propose a novel feature diffusion technique that implicitly groups the reprojection constraints with co-visibility and avoids overfitting to trivial solutions. Additionally, our position decoder parameterizes the output positions for large-scale scenes more effectively. Without using 3D models or depth maps for supervision, our method achieves state-of-the-art results on large-scale scenes with a low-map-size model. On Cambridge landmarks, with a single model, we achieve 17% lower median position error than Poker, the ensemble variant of the state-of-the-art SCR method ACE. Code is available at: https://github.com/cvg/glace.  
  </ol>  
</details>  
**comments**: Large-scale visual localization with a single optimizable MLP. CVPR
  2024. Code: https://github.com/cvg/glace. Project page:
  https://xjiangan.github.io/glace  
  
### [Monocular Localization with Semantics Map for Autonomous Vehicles](http://arxiv.org/abs/2406.03835)  
Jixiang Wan, Xudong Zhang, Shuzhou Dong, Yuwei Zhang, Yuchen Yang, Ruoxi Wu, Ye Jiang, Jijunnan Li, Jinquan Lin, Ming Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Accurate and robust localization remains a significant challenge for autonomous vehicles. The cost of sensors and limitations in local computational efficiency make it difficult to scale to large commercial applications. Traditional vision-based approaches focus on texture features that are susceptible to changes in lighting, season, perspective, and appearance. Additionally, the large storage size of maps with descriptors and complex optimization processes hinder system performance. To balance efficiency and accuracy, we propose a novel lightweight visual semantic localization algorithm that employs stable semantic features instead of low-level texture features. First, semantic maps are constructed offline by detecting semantic objects, such as ground markers, lane lines, and poles, using cameras or LiDAR sensors. Then, online visual localization is performed through data association of semantic features and map objects. We evaluated our proposed localization framework in the publicly available KAIST Urban dataset and in scenarios recorded by ourselves. The experimental results demonstrate that our method is a reliable and practical localization solution in various autonomous driving localization tasks.  
  </ol>  
</details>  
  
  



## NeRF  

### [Improving Physics-Augmented Continuum Neural Radiance Field-Based Geometry-Agnostic System Identification with Lagrangian Particle Optimization](http://arxiv.org/abs/2406.04155)  
Takuhiro Kaneko  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Geometry-agnostic system identification is a technique for identifying the geometry and physical properties of an object from video sequences without any geometric assumptions. Recently, physics-augmented continuum neural radiance fields (PAC-NeRF) has demonstrated promising results for this technique by utilizing a hybrid Eulerian-Lagrangian representation, in which the geometry is represented by the Eulerian grid representations of NeRF, the physics is described by a material point method (MPM), and they are connected via Lagrangian particles. However, a notable limitation of PAC-NeRF is that its performance is sensitive to the learning of the geometry from the first frames owing to its two-step optimization. First, the grid representations are optimized with the first frames of video sequences, and then the physical properties are optimized through video sequences utilizing the fixed first-frame grid representations. This limitation can be critical when learning of the geometric structure is difficult, for example, in a few-shot (sparse view) setting. To overcome this limitation, we propose Lagrangian particle optimization (LPO), in which the positions and features of particles are optimized through video sequences in Lagrangian space. This method allows for the optimization of the geometric structure across the entire video sequence within the physical constraints imposed by the MPM. The experimental results demonstrate that the LPO is useful for geometric correction and physical identification in sparse-view settings.  
  </ol>  
</details>  
**comments**: Accepted to CVPR 2024. Project page:
  https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/lpo/  
  
### [How Far Can We Compress Instant-NGP-Based NeRF?](http://arxiv.org/abs/2406.04101)  
[[code](https://github.com/yihangchen-ee/cnc)]  
Yihang Chen, Qianyi Wu, Mehrtash Harandi, Jianfei Cai  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In recent years, Neural Radiance Field (NeRF) has demonstrated remarkable capabilities in representing 3D scenes. To expedite the rendering process, learnable explicit representations have been introduced for combination with implicit NeRF representation, which however results in a large storage space requirement. In this paper, we introduce the Context-based NeRF Compression (CNC) framework, which leverages highly efficient context models to provide a storage-friendly NeRF representation. Specifically, we excavate both level-wise and dimension-wise context dependencies to enable probability prediction for information entropy reduction. Additionally, we exploit hash collision and occupancy grids as strong prior knowledge for better context modeling. To the best of our knowledge, we are the first to construct and exploit context models for NeRF compression. We achieve a size reduction of 100 $\times$ and 70$\times$ with improved fidelity against the baseline Instant-NGP on Synthesic-NeRF and Tanks and Temples datasets, respectively. Additionally, we attain 86.7\% and 82.3\% storage size reduction against the SOTA NeRF compression method BiRF. Our code is available here: https://github.com/YihangChen-ee/CNC.  
  </ol>  
</details>  
**comments**: Project Page: https://yihangchen-ee.github.io/project_cnc/ Code:
  https://github.com/yihangchen-ee/cnc/. We further propose a 3DGS compression
  method HAC, which is based on CNC:
  https://yihangchen-ee.github.io/project_hac/  
  
### [Gear-NeRF: Free-Viewpoint Rendering and Tracking with Motion-aware Spatio-Temporal Sampling](http://arxiv.org/abs/2406.03723)  
Xinhang Liu, Yu-Wing Tai, Chi-Keung Tang, Pedro Miraldo, Suhas Lohit, Moitreya Chatterjee  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Extensions of Neural Radiance Fields (NeRFs) to model dynamic scenes have enabled their near photo-realistic, free-viewpoint rendering. Although these methods have shown some potential in creating immersive experiences, two drawbacks limit their ubiquity: (i) a significant reduction in reconstruction quality when the computing budget is limited, and (ii) a lack of semantic understanding of the underlying scenes. To address these issues, we introduce Gear-NeRF, which leverages semantic information from powerful image segmentation models. Our approach presents a principled way for learning a spatio-temporal (4D) semantic embedding, based on which we introduce the concept of gears to allow for stratified modeling of dynamic regions of the scene based on the extent of their motion. Such differentiation allows us to adjust the spatio-temporal sampling resolution for each region in proportion to its motion scale, achieving more photo-realistic dynamic novel view synthesis. At the same time, almost for free, our approach enables free-viewpoint tracking of objects of interest - a functionality not yet achieved by existing NeRF-based methods. Empirical studies validate the effectiveness of our method, where we achieve state-of-the-art rendering and tracking performance on multiple challenging datasets.  
  </ol>  
</details>  
**comments**: Paper accepted to IEEE/CVF CVPR 2024 (Spotlight). Work done when XL
  was an intern at MERL. Project Page Link:
  https://merl.com/research/highlights/gear-nerf  
  
### [Superpoint Gaussian Splatting for Real-Time High-Fidelity Dynamic Scene Reconstruction](http://arxiv.org/abs/2406.03697)  
Diwen Wan, Ruijie Lu, Gang Zeng  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Rendering novel view images in dynamic scenes is a crucial yet challenging task. Current methods mainly utilize NeRF-based methods to represent the static scene and an additional time-variant MLP to model scene deformations, resulting in relatively low rendering quality as well as slow inference speed. To tackle these challenges, we propose a novel framework named Superpoint Gaussian Splatting (SP-GS). Specifically, our framework first employs explicit 3D Gaussians to reconstruct the scene and then clusters Gaussians with similar properties (e.g., rotation, translation, and location) into superpoints. Empowered by these superpoints, our method manages to extend 3D Gaussian splatting to dynamic scenes with only a slight increase in computational expense. Apart from achieving state-of-the-art visual quality and real-time rendering under high resolutions, the superpoint representation provides a stronger manipulation capability. Extensive experiments demonstrate the practicality and effectiveness of our approach on both synthetic and real-world datasets. Please see our project page at https://dnvtmf.github.io/SP_GS.github.io.  
  </ol>  
</details>  
**comments**: Accepted by ICML 2024  
  
  



