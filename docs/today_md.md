<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#RoMo:-Robust-Motion-Segmentation-Improves-Structure-from-Motion>RoMo: Robust Motion Segmentation Improves Structure from Motion</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#A-Visual-inertial-Localization-Algorithm-using-Opportunistic-Visual-Beacons-and-Dead-Reckoning-for-GNSS-Denied-Large-scale-Applications>A Visual-inertial Localization Algorithm using Opportunistic Visual Beacons and Dead-Reckoning for GNSS-Denied Large-scale Applications</a></li>
        <li><a href=#Optimizing-Image-Retrieval-with-an-Extended-b-Metric-Space>Optimizing Image Retrieval with an Extended b-Metric Space</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#$C^{3}$-NeRF:-Modeling-Multiple-Scenes-via-Conditional-cum-Continual-Neural-Radiance-Fields>$C^{3}$-NeRF: Modeling Multiple Scenes via Conditional-cum-Continual Neural Radiance Fields</a></li>
        <li><a href=#Gaussian-Splashing:-Direct-Volumetric-Rendering-Underwater>Gaussian Splashing: Direct Volumetric Rendering Underwater</a></li>
        <li><a href=#ReconDreamer:-Crafting-World-Models-for-Driving-Scene-Reconstruction-via-Online-Restoration>ReconDreamer: Crafting World Models for Driving Scene Reconstruction via Online Restoration</a></li>
        <li><a href=#LokiTalk:-Learning-Fine-Grained-and-Generalizable-Correspondences-to-Enhance-NeRF-based-Talking-Head-Synthesis>LokiTalk: Learning Fine-Grained and Generalizable Correspondences to Enhance NeRF-based Talking Head Synthesis</a></li>
        <li><a href=#SAMa:-Material-aware-3D-Selection-and-Segmentation>SAMa: Material-aware 3D Selection and Segmentation</a></li>
        <li><a href=#Surf-NeRF:-Surface-Regularised-Neural-Radiance-Fields>Surf-NeRF: Surface Regularised Neural Radiance Fields</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [RoMo: Robust Motion Segmentation Improves Structure from Motion](http://arxiv.org/abs/2411.18650)  
Lily Goli, Sara Sabour, Mark Matthews, Marcus Brubaker, Dmitry Lagun, Alec Jacobson, David J. Fleet, Saurabh Saxena, Andrea Tagliasacchi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    There has been extensive progress in the reconstruction and generation of 4D scenes from monocular casually-captured video. While these tasks rely heavily on known camera poses, the problem of finding such poses using structure-from-motion (SfM) often depends on robustly separating static from dynamic parts of a video. The lack of a robust solution to this problem limits the performance of SfM camera-calibration pipelines. We propose a novel approach to video-based motion segmentation to identify the components of a scene that are moving w.r.t. a fixed world frame. Our simple but effective iterative method, RoMo, combines optical flow and epipolar cues with a pre-trained video segmentation model. It outperforms unsupervised baselines for motion segmentation as well as supervised baselines trained from synthetic data. More importantly, the combination of an off-the-shelf SfM pipeline with our segmentation masks establishes a new state-of-the-art on camera calibration for scenes with dynamic content, outperforming existing methods by a substantial margin.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [A Visual-inertial Localization Algorithm using Opportunistic Visual Beacons and Dead-Reckoning for GNSS-Denied Large-scale Applications](http://arxiv.org/abs/2411.19845)  
Liqiang Zhang Ye Tian Dongyan Wei  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    With the development of smart cities, the demand for continuous pedestrian navigation in large-scale urban environments has significantly increased. While global navigation satellite systems (GNSS) provide low-cost and reliable positioning services, they are often hindered in complex urban canyon environments. Thus, exploring opportunistic signals for positioning in urban areas has become a key solution. Augmented reality (AR) allows pedestrians to acquire real-time visual information. Accordingly, we propose a low-cost visual-inertial positioning solution. This method comprises a lightweight multi-scale group convolution (MSGC)-based visual place recognition (VPR) neural network, a pedestrian dead reckoning (PDR) algorithm, and a visual/inertial fusion approach based on a Kalman filter with gross error suppression. The VPR serves as a conditional observation to the Kalman filter, effectively correcting the errors accumulated through the PDR method. This enables the entire algorithm to ensure the reliability of long-term positioning in GNSS-denied areas. Extensive experimental results demonstrate that our method maintains stable positioning during large-scale movements. Compared to the lightweight MobileNetV3-based VPR method, our proposed VPR solution improves Recall@1 by at least 3\% on two public datasets while reducing the number of parameters by 63.37\%. It also achieves performance that is comparable to the VGG16-based method. The VPR-PDR algorithm improves localization accuracy by more than 40\% compared to the original PDR.  
  </ol>  
</details>  
  
### [Optimizing Image Retrieval with an Extended b-Metric Space](http://arxiv.org/abs/2411.18800)  
Abdelkader Belhenniche, Roman Chertovskih  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This article provides a new approach on how to enhance data storage and retrieval in the Query By Image Content Systems (QBIC) by introducing the ${\rm NEM}_{\sigma}$ distance measure, satisfying the relaxed triangle inequality. By leveraging the concept of extended $b$-metric spaces, we address complex distance relationships, thereby improving the accuracy and efficiency of image database management. The use of ${\rm NEM}_{\sigma}$ facilitates better scalability and accuracy in large-scale image retrieval systems, optimizing both the storage and retrieval processes. The proposed method represents a significant advancement over traditional distance measures, offering enhanced flexibility and precision in the context of image content-based querying. Additionally, we take inspiration from ice flow models using ${\rm NEM}_{\sigma}$ and ${\rm NEM}_r$ , adding dynamic and location-based factors to better capture details in images.  
  </ol>  
</details>  
  
  



## NeRF  

### [ $C^{3}$ -NeRF: Modeling Multiple Scenes via Conditional-cum-Continual Neural Radiance Fields](http://arxiv.org/abs/2411.19903)  
Prajwal Singh, Ashish Tiwari, Gautam Vashishtha, Shanmuganathan Raman  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural radiance fields (NeRF) have exhibited highly photorealistic rendering of novel views through per-scene optimization over a single 3D scene. With the growing popularity of NeRF and its variants, they have become ubiquitous and have been identified as efficient 3D resources. However, they are still far from being scalable since a separate model needs to be stored for each scene, and the training time increases linearly with every newly added scene. Surprisingly, the idea of encoding multiple 3D scenes into a single NeRF model is heavily under-explored. In this work, we propose a novel conditional-cum-continual framework, called $C^{3}$-NeRF, to accommodate multiple scenes into the parameters of a single neural radiance field. Unlike conventional approaches that leverage feature extractors and pre-trained priors for scene conditioning, we use simple pseudo-scene labels to model multiple scenes in NeRF. Interestingly, we observe the framework is also inherently continual (via generative replay) with minimal, if not no, forgetting of the previously learned scenes. Consequently, the proposed framework adapts to multiple new scenes without necessarily accessing the old data. Through extensive qualitative and quantitative evaluation using synthetic and real datasets, we demonstrate the inherent capacity of the NeRF model to accommodate multiple scenes with high-quality novel-view renderings without adding additional parameters. We provide implementation details and dynamic visualizations of our results in the supplementary file.  
  </ol>  
</details>  
  
### [Gaussian Splashing: Direct Volumetric Rendering Underwater](http://arxiv.org/abs/2411.19588)  
Nir Mualem, Roy Amoyal, Oren Freifeld, Derya Akkaynak  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In underwater images, most useful features are occluded by water. The extent of the occlusion depends on imaging geometry and can vary even across a sequence of burst images. As a result, 3D reconstruction methods robust on in-air scenes, like Neural Radiance Field methods (NeRFs) or 3D Gaussian Splatting (3DGS), fail on underwater scenes. While a recent underwater adaptation of NeRFs achieved state-of-the-art results, it is impractically slow: reconstruction takes hours and its rendering rate, in frames per second (FPS), is less than 1. Here, we present a new method that takes only a few minutes for reconstruction and renders novel underwater scenes at 140 FPS. Named Gaussian Splashing, our method unifies the strengths and speed of 3DGS with an image formation model for capturing scattering, introducing innovations in the rendering and depth estimation procedures and in the 3DGS loss function. Despite the complexities of underwater adaptation, our method produces images at unparalleled speeds with superior details. Moreover, it reveals distant scene details with far greater clarity than other methods, dramatically improving reconstructed and rendered images. We demonstrate results on existing datasets and a new dataset we have collected.   Additional visual results are available at: https://bgu-cs-vil.github.io/gaussiansplashingUW.github.io/ .  
  </ol>  
</details>  
  
### [ReconDreamer: Crafting World Models for Driving Scene Reconstruction via Online Restoration](http://arxiv.org/abs/2411.19548)  
Chaojun Ni, Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Wenkang Qin, Guan Huang, Chen Liu, Yuyin Chen, Yida Wang, Xueyang Zhang, Yifei Zhan, Kun Zhan, Peng Jia, Xianpeng Lang, Xingang Wang, Wenjun Mei  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Closed-loop simulation is crucial for end-to-end autonomous driving. Existing sensor simulation methods (e.g., NeRF and 3DGS) reconstruct driving scenes based on conditions that closely mirror training data distributions. However, these methods struggle with rendering novel trajectories, such as lane changes. Recent works have demonstrated that integrating world model knowledge alleviates these issues. Despite their efficiency, these approaches still encounter difficulties in the accurate representation of more complex maneuvers, with multi-lane shifts being a notable example. Therefore, we introduce ReconDreamer, which enhances driving scene reconstruction through incremental integration of world model knowledge. Specifically, DriveRestorer is proposed to mitigate artifacts via online restoration. This is complemented by a progressive data update strategy designed to ensure high-quality rendering for more complex maneuvers. To the best of our knowledge, ReconDreamer is the first method to effectively render in large maneuvers. Experimental results demonstrate that ReconDreamer outperforms Street Gaussians in the NTA-IoU, NTL-IoU, and FID, with relative improvements by 24.87%, 6.72%, and 29.97%. Furthermore, ReconDreamer surpasses DriveDreamer4D with PVG during large maneuver rendering, as verified by a relative improvement of 195.87% in the NTA-IoU metric and a comprehensive user study.  
  </ol>  
</details>  
**comments**: Project Page: https://recondreamer.github.io  
  
### [LokiTalk: Learning Fine-Grained and Generalizable Correspondences to Enhance NeRF-based Talking Head Synthesis](http://arxiv.org/abs/2411.19525)  
Tianqi Li, Ruobing Zheng, Bonan Li, Zicheng Zhang, Meng Wang, Jingdong Chen, Ming Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Despite significant progress in talking head synthesis since the introduction of Neural Radiance Fields (NeRF), visual artifacts and high training costs persist as major obstacles to large-scale commercial adoption. We propose that identifying and establishing fine-grained and generalizable correspondences between driving signals and generated results can simultaneously resolve both problems. Here we present LokiTalk, a novel framework designed to enhance NeRF-based talking heads with lifelike facial dynamics and improved training efficiency. To achieve fine-grained correspondences, we introduce Region-Specific Deformation Fields, which decompose the overall portrait motion into lip movements, eye blinking, head pose, and torso movements. By hierarchically modeling the driving signals and their associated regions through two cascaded deformation fields, we significantly improve dynamic accuracy and minimize synthetic artifacts. Furthermore, we propose ID-Aware Knowledge Transfer, a plug-and-play module that learns generalizable dynamic and static correspondences from multi-identity videos, while simultaneously extracting ID-specific dynamic and static features to refine the depiction of individual characters. Comprehensive evaluations demonstrate that LokiTalk delivers superior high-fidelity results and training efficiency compared to previous methods. The code will be released upon acceptance.  
  </ol>  
</details>  
  
### [SAMa: Material-aware 3D Selection and Segmentation](http://arxiv.org/abs/2411.19322)  
Michael Fischer, Iliyan Georgiev, Thibault Groueix, Vladimir G. Kim, Tobias Ritschel, Valentin Deschaintre  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Decomposing 3D assets into material parts is a common task for artists and creators, yet remains a highly manual process. In this work, we introduce Select Any Material (SAMa), a material selection approach for various 3D representations. Building on the recently introduced SAM2 video selection model, we extend its capabilities to the material domain. We leverage the model's cross-view consistency to create a 3D-consistent intermediate material-similarity representation in the form of a point cloud from a sparse set of views. Nearest-neighbour lookups in this similarity cloud allow us to efficiently reconstruct accurate continuous selection masks over objects' surfaces that can be inspected from any view. Our method is multiview-consistent by design, alleviating the need for contrastive learning or feature-field pre-processing, and performs optimization-free selection in seconds. Our approach works on arbitrary 3D representations and outperforms several strong baselines in terms of selection accuracy and multiview consistency. It enables several compelling applications, such as replacing the diffuse-textured materials on a text-to-3D output, or selecting and editing materials on NeRFs and 3D-Gaussians.  
  </ol>  
</details>  
**comments**: Project Page: https://mfischer-ucl.github.io/sama  
  
### [Surf-NeRF: Surface Regularised Neural Radiance Fields](http://arxiv.org/abs/2411.18652)  
Jack Naylor, Viorela Ila, Donald G. Dansereau  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRFs) provide a high fidelity, continuous scene representation that can realistically represent complex behaviour of light. Despite recent works like Ref-NeRF improving geometry through physics-inspired models, the ability for a NeRF to overcome shape-radiance ambiguity and converge to a representation consistent with real geometry remains limited. We demonstrate how curriculum learning of a surface light field model helps a NeRF converge towards a more geometrically accurate scene representation. We introduce four additional regularisation terms to impose geometric smoothness, consistency of normals and a separation of Lambertian and specular appearance at geometry in the scene, conforming to physical models. Our approach yields improvements of 14.4% to normals on positionally encoded NeRFs and 9.2% on grid-based models compared to current reflection-based NeRF variants. This includes a separated view-dependent appearance, conditioning a NeRF to have a geometric representation consistent with the captured scene. We demonstrate compatibility of our method with existing NeRF variants, as a key step in enabling radiance-based representations for geometry critical applications.  
  </ol>  
</details>  
**comments**: 20 pages, 17 figures, 9 tables, project page can be found at
  http://roboticimaging.org/Projects/SurfNeRF  
  
  



