<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#VOCAL:-Visual-Odometry-via-ContrAstive-Learning>VOCAL: Visual Odometry via ContrAstive Learning</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#LoD-Loc-v2:-Aerial-Visual-Localization-over-Low-Level-of-Detail-City-Models-using-Explicit-Silhouette-Alignment>LoD-Loc v2: Aerial Visual Localization over Low Level-of-Detail City Models using Explicit Silhouette Alignment</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#What-does-really-matter-in-image-goal-navigation?>What does really matter in image goal navigation?</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Tile-and-Slide-:-A-New-Framework-for-Scaling-NeRF-from-Local-to-Global-3D-Earth-Observation>Tile and Slide : A New Framework for Scaling NeRF from Local to Global 3D Earth Observation</a></li>
        <li><a href=#Surgical-Neural-Radiance-Fields-from-One-Image>Surgical Neural Radiance Fields from One Image</a></li>
        <li><a href=#PlantSegNeRF:-A-few-shot,-cross-dataset-method-for-plant-3D-instance-point-cloud-reconstruction-via-joint-channel-NeRF-with-multi-view-image-instance-matching>PlantSegNeRF: A few-shot, cross-dataset method for plant 3D instance point cloud reconstruction via joint-channel NeRF with multi-view image instance matching</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [VOCAL: Visual Odometry via ContrAstive Learning](http://arxiv.org/abs/2507.00243)  
Chi-Yao Huang, Zeel Bhatt, Yezhou Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Breakthroughs in visual odometry (VO) have fundamentally reshaped the landscape of robotics, enabling ultra-precise camera state estimation that is crucial for modern autonomous systems. Despite these advances, many learning-based VO techniques rely on rigid geometric assumptions, which often fall short in interpretability and lack a solid theoretical basis within fully data-driven frameworks. To overcome these limitations, we introduce VOCAL (Visual Odometry via ContrAstive Learning), a novel framework that reimagines VO as a label ranking challenge. By integrating Bayesian inference with a representation learning framework, VOCAL organizes visual features to mirror camera states. The ranking mechanism compels similar camera states to converge into consistent and spatially coherent representations within the latent space. This strategic alignment not only bolsters the interpretability of the learned features but also ensures compatibility with multimodal data sources. Extensive evaluations on the KITTI dataset highlight VOCAL's enhanced interpretability and flexibility, pushing VO toward more general and explainable spatial intelligence.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [LoD-Loc v2: Aerial Visual Localization over Low Level-of-Detail City Models using Explicit Silhouette Alignment](http://arxiv.org/abs/2507.00659)  
Juelin Zhu, Shuaibang Peng, Long Wang, Hanlin Tan, Yu Liu, Maojun Zhang, Shen Yan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose a novel method for aerial visual localization over low Level-of-Detail (LoD) city models. Previous wireframe-alignment-based method LoD-Loc has shown promising localization results leveraging LoD models. However, LoD-Loc mainly relies on high-LoD (LoD3 or LoD2) city models, but the majority of available models and those many countries plan to construct nationwide are low-LoD (LoD1). Consequently, enabling localization on low-LoD city models could unlock drones' potential for global urban localization. To address these issues, we introduce LoD-Loc v2, which employs a coarse-to-fine strategy using explicit silhouette alignment to achieve accurate localization over low-LoD city models in the air. Specifically, given a query image, LoD-Loc v2 first applies a building segmentation network to shape building silhouettes. Then, in the coarse pose selection stage, we construct a pose cost volume by uniformly sampling pose hypotheses around a prior pose to represent the pose probability distribution. Each cost of the volume measures the degree of alignment between the projected and predicted silhouettes. We select the pose with maximum value as the coarse pose. In the fine pose estimation stage, a particle filtering method incorporating a multi-beam tracking approach is used to efficiently explore the hypothesis space and obtain the final pose estimation. To further facilitate research in this field, we release two datasets with LoD1 city models covering 10.7 km , along with real RGB queries and ground-truth pose annotations. Experimental results show that LoD-Loc v2 improves estimation accuracy with high-LoD models and enables localization with low-LoD models for the first time. Moreover, it outperforms state-of-the-art baselines by large margins, even surpassing texture-model-based methods, and broadens the convergence basin to accommodate larger prior errors.  
  </ol>  
</details>  
**comments**: Accepted by ICCV 2025  
  
  



## Image Matching  

### [What does really matter in image goal navigation?](http://arxiv.org/abs/2507.01667)  
Gianluca Monaci, Philippe Weinzaepfel, Christian Wolf  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image goal navigation requires two different skills: firstly, core navigation skills, including the detection of free space and obstacles, and taking decisions based on an internal representation; and secondly, computing directional information by comparing visual observations to the goal image. Current state-of-the-art methods either rely on dedicated image-matching, or pre-training of computer vision modules on relative pose estimation. In this paper, we study whether this task can be efficiently solved with end-to-end training of full agents with RL, as has been claimed by recent work. A positive answer would have impact beyond Embodied AI and allow training of relative pose estimation from reward for navigation alone. In a large study we investigate the effect of architectural choices like late fusion, channel stacking, space-to-depth projections and cross-attention, and their role in the emergence of relative pose estimators from navigation training. We show that the success of recent methods is influenced up to a certain extent by simulator settings, leading to shortcuts in simulation. However, we also show that these capabilities can be transferred to more realistic setting, up to some extend. We also find evidence for correlations between navigation performance and probed (emerging) relative pose estimation performance, an important sub skill.  
  </ol>  
</details>  
  
  



## NeRF  

### [Tile and Slide : A New Framework for Scaling NeRF from Local to Global 3D Earth Observation](http://arxiv.org/abs/2507.01631)  
Camille Billouard, Dawa Derksen, Alexandre Constantin, Bruno Vallet  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) have recently emerged as a paradigm for 3D reconstruction from multiview satellite imagery. However, state-of-the-art NeRF methods are typically constrained to small scenes due to the memory footprint during training, which we study in this paper. Previous work on large-scale NeRFs palliate this by dividing the scene into NeRFs. This paper introduces Snake-NeRF, a framework that scales to large scenes. Our out-of-core method eliminates the need to load all images and networks simultaneously, and operates on a single device. We achieve this by dividing the region of interest into NeRFs that 3D tile without overlap. Importantly, we crop the images with overlap to ensure each NeRFs is trained with all the necessary pixels. We introduce a novel $2\times 2$ 3D tile progression strategy and segmented sampler, which together prevent 3D reconstruction errors along the tile edges. Our experiments conclude that large satellite images can effectively be processed with linear time complexity, on a single GPU, and without compromise in quality.  
  </ol>  
</details>  
**comments**: Accepted at ICCV 2025 Workshop 3D-VAST (From street to space: 3D
  Vision Across Altitudes). Version before camera ready. Our code will be made
  public after the conference  
  
### [Surgical Neural Radiance Fields from One Image](http://arxiv.org/abs/2507.00969)  
Alberto Neri, Maximilan Fehrentz, Veronica Penza, Leonardo S. Mattos, Nazim Haouchine  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Purpose: Neural Radiance Fields (NeRF) offer exceptional capabilities for 3D reconstruction and view synthesis, yet their reliance on extensive multi-view data limits their application in surgical intraoperative settings where only limited data is available. In particular, collecting such extensive data intraoperatively is impractical due to time constraints. This work addresses this challenge by leveraging a single intraoperative image and preoperative data to train NeRF efficiently for surgical scenarios.   Methods: We leverage preoperative MRI data to define the set of camera viewpoints and images needed for robust and unobstructed training. Intraoperatively, the appearance of the surgical image is transferred to the pre-constructed training set through neural style transfer, specifically combining WTC2 and STROTSS to prevent over-stylization. This process enables the creation of a dataset for instant and fast single-image NeRF training.   Results: The method is evaluated with four clinical neurosurgical cases. Quantitative comparisons to NeRF models trained on real surgical microscope images demonstrate strong synthesis agreement, with similarity metrics indicating high reconstruction fidelity and stylistic alignment. When compared with ground truth, our method demonstrates high structural similarity, confirming good reconstruction quality and texture preservation.   Conclusion: Our approach demonstrates the feasibility of single-image NeRF training in surgical settings, overcoming the limitations of traditional multi-view methods.  
  </ol>  
</details>  
  
### [PlantSegNeRF: A few-shot, cross-dataset method for plant 3D instance point cloud reconstruction via joint-channel NeRF with multi-view image instance matching](http://arxiv.org/abs/2507.00371)  
Xin Yang, Ruiming Du, Hanyang Huang, Jiayang Xie, Pengyao Xie, Leisen Fang, Ziyue Guo, Nanjun Jiang, Yu Jiang, Haiyan Cen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Organ segmentation of plant point clouds is a prerequisite for the high-resolution and accurate extraction of organ-level phenotypic traits. Although the fast development of deep learning has boosted much research on segmentation of plant point clouds, the existing techniques for organ segmentation still face limitations in resolution, segmentation accuracy, and generalizability across various plant species. In this study, we proposed a novel approach called plant segmentation neural radiance fields (PlantSegNeRF), aiming to directly generate high-precision instance point clouds from multi-view RGB image sequences for a wide range of plant species. PlantSegNeRF performed 2D instance segmentation on the multi-view images to generate instance masks for each organ with a corresponding ID. The multi-view instance IDs corresponding to the same plant organ were then matched and refined using a specially designed instance matching module. The instance NeRF was developed to render an implicit scene, containing color, density, semantic and instance information. The implicit scene was ultimately converted into high-precision plant instance point clouds based on the volume density. The results proved that in semantic segmentation of point clouds, PlantSegNeRF outperformed the commonly used methods, demonstrating an average improvement of 16.1%, 18.3%, 17.8%, and 24.2% in precision, recall, F1-score, and IoU compared to the second-best results on structurally complex datasets. More importantly, PlantSegNeRF exhibited significant advantages in plant point cloud instance segmentation tasks. Across all plant datasets, it achieved average improvements of 11.7%, 38.2%, 32.2% and 25.3% in mPrec, mRec, mCov, mWCov, respectively. This study extends the organ-level plant phenotyping and provides a high-throughput way to supply high-quality 3D data for the development of large-scale models in plant science.  
  </ol>  
</details>  
  
  



