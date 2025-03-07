<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Geometry-Constrained-Monocular-Scale-Estimation-Using-Semantic-Segmentation-for-Dynamic-Scenes>Geometry-Constrained Monocular Scale Estimation Using Semantic Segmentation for Dynamic Scenes</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#PLMP----Point-Line-Minimal-Problems-for-Projective-SfM>PLMP -- Point-Line Minimal Problems for Projective SfM</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#RadIR:-A-Scalable-Framework-for-Multi-Grained-Medical-Image-Retrieval-via-Radiology-Report-Mining>RadIR: A Scalable Framework for Multi-Grained Medical Image Retrieval via Radiology Report Mining</a></li>
        <li><a href=#ForestLPR:-LiDAR-Place-Recognition-in-Forests-Attentioning-Multiple-BEV-Density-Images>ForestLPR: LiDAR Place Recognition in Forests Attentioning Multiple BEV Density Images</a></li>
        <li><a href=#Geometry-Constrained-Monocular-Scale-Estimation-Using-Semantic-Segmentation-for-Dynamic-Scenes>Geometry-Constrained Monocular Scale Estimation Using Semantic Segmentation for Dynamic Scenes</a></li>
        <li><a href=#Bridging-the-Vision-Brain-Gap-with-an-Uncertainty-Aware-Blur-Prior>Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior</a></li>
        <li><a href=#Image-Based-Relocalization-and-Alignment-for-Long-Term-Monitoring-of-Dynamic-Underwater-Environments>Image-Based Relocalization and Alignment for Long-Term Monitoring of Dynamic Underwater Environments</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Spatial-regularisation-for-improved-accuracy-and-interpretability-in-keypoint-based-registration>Spatial regularisation for improved accuracy and interpretability in keypoint-based registration</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Learning-3D-Medical-Image-Models-From-Brain-Functional-Connectivity-Network-Supervision-For-Mental-Disorder-Diagnosis>Learning 3D Medical Image Models From Brain Functional Connectivity Network Supervision For Mental Disorder Diagnosis</a></li>
        <li><a href=#Diff-Reg-v2:-Diffusion-Based-Matching-Matrix-Estimation-for-Image-Matching-and-3D-Registration>Diff-Reg v2: Diffusion-Based Matching Matrix Estimation for Image Matching and 3D Registration</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Surgical-Gaussian-Surfels:-Highly-Accurate-Real-time-Surgical-Scene-Rendering>Surgical Gaussian Surfels: Highly Accurate Real-time Surgical Scene Rendering</a></li>
        <li><a href=#LensDFF:-Language-enhanced-Sparse-Feature-Distillation-for-Efficient-Few-Shot-Dexterous-Manipulation>LensDFF: Language-enhanced Sparse Feature Distillation for Efficient Few-Shot Dexterous Manipulation</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Geometry-Constrained Monocular Scale Estimation Using Semantic Segmentation for Dynamic Scenes](http://arxiv.org/abs/2503.04235)  
Hui Zhang, Zhiyang Wu, Qianqian Shangguan, Kang An  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Monocular visual localization plays a pivotal role in advanced driver assistance systems and autonomous driving by estimating a vehicle's ego-motion from a single pinhole camera. Nevertheless, conventional monocular visual odometry encoun-ters challenges in scale estimation due to the absence of depth information during projection. Previous methodologies, whether rooted in physical constraints or deep learning paradigms, con-tend with issues related to computational complexity and the management of dynamic objects. This study extends our prior research, presenting innovative strategies for ego-motion estima-tion and the selection of ground points. Striving for a nuanced equilibrium between computational efficiency and precision, we propose a hybrid method that leverages the SegNeXt model for real-time applications, encompassing both ego-motion estimation and ground point selection. Our methodology incorporates dy-namic object masks to eliminate unstable features and employs ground plane masks for meticulous triangulation. Furthermore, we exploit Geometry-constraint to delineate road regions for scale recovery. The integration of this approach with the mo-nocular version of ORB-SLAM3 culminates in the accurate esti-mation of a road model, a pivotal component in our scale recov-ery process. Rigorous experiments, conducted on the KITTI da-taset, systematically compare our method with existing monocu-lar visual odometry algorithms and contemporary scale recovery methodologies. The results undeniably confirm the superior ef-fectiveness of our approach, surpassing state-of-the-art visual odometry algorithms. Our source code is available at https://git hub.com/bFr0zNq/MVOSegScale.  
  </ol>  
</details>  
  
  



## SFM  

### [PLMP -- Point-Line Minimal Problems for Projective SfM](http://arxiv.org/abs/2503.04351)  
Kim Kiehn, Albin Ahlbäck, Kathlén Kohn  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We completely classify all minimal problems for Structure-from-Motion (SfM) where arrangements of points and lines are fully observed by multiple uncalibrated pinhole cameras. We find 291 minimal problems, 73 of which have unique solutions and can thus be solved linearly. Two of the linear problems allow an arbitrary number of views, while all other minimal problems have at most 9 cameras. All minimal problems have at most 7 points and at most 12 lines. We compute the number of solutions of each minimal problem, as this gives a measurement of the problem's intrinsic difficulty, and find that these number are relatively low (e.g., when comparing with minimal problems for calibrated cameras). Finally, by exploring stabilizer subgroups of subarrangements, we develop a geometric and systematic way to 1) factorize minimal problems into smaller problems, 2) identify minimal problems in underconstrained problems, and 3) formally prove non-minimality.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [RadIR: A Scalable Framework for Multi-Grained Medical Image Retrieval via Radiology Report Mining](http://arxiv.org/abs/2503.04653)  
Tengfei Zhang, Ziheng Zhao, Chaoyi Wu, Xiao Zhou, Ya Zhang, Yangfeng Wang, Weidi Xie  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Developing advanced medical imaging retrieval systems is challenging due to the varying definitions of `similar images' across different medical contexts. This challenge is compounded by the lack of large-scale, high-quality medical imaging retrieval datasets and benchmarks. In this paper, we propose a novel methodology that leverages dense radiology reports to define image-wise similarity ordering at multiple granularities in a scalable and fully automatic manner. Using this approach, we construct two comprehensive medical imaging retrieval datasets: MIMIC-IR for Chest X-rays and CTRATE-IR for CT scans, providing detailed image-image ranking annotations conditioned on diverse anatomical structures. Furthermore, we develop two retrieval systems, RadIR-CXR and model-ChestCT, which demonstrate superior performance in traditional image-image and image-report retrieval tasks. These systems also enable flexible, effective image retrieval conditioned on specific anatomical structures described in text, achieving state-of-the-art results on 77 out of 78 metrics.  
  </ol>  
</details>  
  
### [ForestLPR: LiDAR Place Recognition in Forests Attentioning Multiple BEV Density Images](http://arxiv.org/abs/2503.04475)  
Yanqing Shen, Turcan Tuna, Marco Hutter, Cesar Cadena, Nanning Zheng  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Place recognition is essential to maintain global consistency in large-scale localization systems. While research in urban environments has progressed significantly using LiDARs or cameras, applications in natural forest-like environments remain largely under-explored. Furthermore, forests present particular challenges due to high self-similarity and substantial variations in vegetation growth over time. In this work, we propose a robust LiDAR-based place recognition method for natural forests, ForestLPR. We hypothesize that a set of cross-sectional images of the forest's geometry at different heights contains the information needed to recognize revisiting a place. The cross-sectional images are represented by \ac{bev} density images of horizontal slices of the point cloud at different heights. Our approach utilizes a visual transformer as the shared backbone to produce sets of local descriptors and introduces a multi-BEV interaction module to attend to information at different heights adaptively. It is followed by an aggregation layer that produces a rotation-invariant place descriptor. We evaluated the efficacy of our method extensively on real-world data from public benchmarks as well as robotic datasets and compared it against the state-of-the-art (SOTA) methods. The results indicate that ForestLPR has consistently good performance on all evaluations and achieves an average increase of 7.38\% and 9.11\% on Recall@1 over the closest competitor on intra-sequence loop closure detection and inter-sequence re-localization, respectively, validating our hypothesis  
  </ol>  
</details>  
**comments**: accepted by CVPR2025  
  
### [Geometry-Constrained Monocular Scale Estimation Using Semantic Segmentation for Dynamic Scenes](http://arxiv.org/abs/2503.04235)  
Hui Zhang, Zhiyang Wu, Qianqian Shangguan, Kang An  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Monocular visual localization plays a pivotal role in advanced driver assistance systems and autonomous driving by estimating a vehicle's ego-motion from a single pinhole camera. Nevertheless, conventional monocular visual odometry encoun-ters challenges in scale estimation due to the absence of depth information during projection. Previous methodologies, whether rooted in physical constraints or deep learning paradigms, con-tend with issues related to computational complexity and the management of dynamic objects. This study extends our prior research, presenting innovative strategies for ego-motion estima-tion and the selection of ground points. Striving for a nuanced equilibrium between computational efficiency and precision, we propose a hybrid method that leverages the SegNeXt model for real-time applications, encompassing both ego-motion estimation and ground point selection. Our methodology incorporates dy-namic object masks to eliminate unstable features and employs ground plane masks for meticulous triangulation. Furthermore, we exploit Geometry-constraint to delineate road regions for scale recovery. The integration of this approach with the mo-nocular version of ORB-SLAM3 culminates in the accurate esti-mation of a road model, a pivotal component in our scale recov-ery process. Rigorous experiments, conducted on the KITTI da-taset, systematically compare our method with existing monocu-lar visual odometry algorithms and contemporary scale recovery methodologies. The results undeniably confirm the superior ef-fectiveness of our approach, surpassing state-of-the-art visual odometry algorithms. Our source code is available at https://git hub.com/bFr0zNq/MVOSegScale.  
  </ol>  
</details>  
  
### [Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior](http://arxiv.org/abs/2503.04207)  
Haitao Wu, Qing Li, Changqing Zhang, Zhen He, Xiaomin Ying  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Can our brain signals faithfully reflect the original visual stimuli, even including high-frequency details? Although human perceptual and cognitive capacities enable us to process and remember visual information, these abilities are constrained by several factors, such as limited attentional resources and the finite capacity of visual memory. When visual stimuli are processed by human visual system into brain signals, some information is inevitably lost, leading to a discrepancy known as the \textbf{System GAP}. Additionally, perceptual and cognitive dynamics, along with technical noise in signal acquisition, degrade the fidelity of brain signals relative to the visual stimuli, known as the \textbf{Random GAP}. When encoded brain representations are directly aligned with the corresponding pretrained image features, the System GAP and Random GAP between paired data challenge the model, requiring it to bridge these gaps. However, in the context of limited paired data, these gaps are difficult for the model to learn, leading to overfitting and poor generalization to new data. To address these GAPs, we propose a simple yet effective approach called the \textbf{Uncertainty-aware Blur Prior (UBP)}. It estimates the uncertainty within the paired data, reflecting the mismatch between brain signals and visual stimuli. Based on this uncertainty, UBP dynamically blurs the high-frequency details of the original images, reducing the impact of the mismatch and improving alignment. Our method achieves a top-1 accuracy of \textbf{50.9\%} and a top-5 accuracy of \textbf{79.7\%} on the zero-shot brain-to-image retrieval task, surpassing previous state-of-the-art methods by margins of \textbf{13.7\%} and \textbf{9.8\%}, respectively. Code is available at \href{https://github.com/HaitaoWuTJU/Uncertainty-aware-Blur-Prior}{GitHub}.  
  </ol>  
</details>  
  
### [Image-Based Relocalization and Alignment for Long-Term Monitoring of Dynamic Underwater Environments](http://arxiv.org/abs/2503.04096)  
Beverley Gorry, Tobias Fischer, Michael Milford, Alejandro Fontan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Effective monitoring of underwater ecosystems is crucial for tracking environmental changes, guiding conservation efforts, and ensuring long-term ecosystem health. However, automating underwater ecosystem management with robotic platforms remains challenging due to the complexities of underwater imagery, which pose significant difficulties for traditional visual localization methods. We propose an integrated pipeline that combines Visual Place Recognition (VPR), feature matching, and image segmentation on video-derived images. This method enables robust identification of revisited areas, estimation of rigid transformations, and downstream analysis of ecosystem changes. Furthermore, we introduce the SQUIDLE+ VPR Benchmark-the first large-scale underwater VPR benchmark designed to leverage an extensive collection of unstructured data from multiple robotic platforms, spanning time intervals from days to years. The dataset encompasses diverse trajectories, arbitrary overlap and diverse seafloor types captured under varying environmental conditions, including differences in depth, lighting, and turbidity. Our code is available at: https://github.com/bev-gorry/underloc  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Spatial regularisation for improved accuracy and interpretability in keypoint-based registration](http://arxiv.org/abs/2503.04499)  
Benjamin Billot, Ramya Muthukrishnan, Esra Abaci-Turk, Ellen P. Grant, Nicholas Ayache, Hervé Delingette, Polina Golland  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Unsupervised registration strategies bypass requirements in ground truth transforms or segmentations by optimising similarity metrics between fixed and moved volumes. Among these methods, a recent subclass of approaches based on unsupervised keypoint detection stand out as very promising for interpretability. Specifically, these methods train a network to predict feature maps for fixed and moving images, from which explainable centres of mass are computed to obtain point clouds, that are then aligned in closed-form. However, the features returned by the network often yield spatially diffuse patterns that are hard to interpret, thus undermining the purpose of keypoint-based registration. Here, we propose a three-fold loss to regularise the spatial distribution of the features. First, we use the KL divergence to model features as point spread functions that we interpret as probabilistic keypoints. Then, we sharpen the spatial distributions of these features to increase the precision of the detected landmarks. Finally, we introduce a new repulsive loss across keypoints to encourage spatial diversity. Overall, our loss considerably improves the interpretability of the features, which now correspond to precise and anatomically meaningful landmarks. We demonstrate our three-fold loss in foetal rigid motion tracking and brain MRI affine registration tasks, where it not only outperforms state-of-the-art unsupervised strategies, but also bridges the gap with state-of-the-art supervised methods. Our code is available at https://github.com/BenBillot/spatial_regularisation.  
  </ol>  
</details>  
**comments**: under review  
  
  



## Image Matching  

### [Learning 3D Medical Image Models From Brain Functional Connectivity Network Supervision For Mental Disorder Diagnosis](http://arxiv.org/abs/2503.04205)  
Xingcan Hu, Wei Wang, Li Xiao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In MRI-based mental disorder diagnosis, most previous studies focus on functional connectivity network (FCN) derived from functional MRI (fMRI). However, the small size of annotated fMRI datasets restricts its wide application. Meanwhile, structural MRIs (sMRIs), such as 3D T1-weighted (T1w) MRI, which are commonly used and readily accessible in clinical settings, are often overlooked. To integrate the complementary information from both function and structure for improved diagnostic accuracy, we propose CINP (Contrastive Image-Network Pre-training), a framework that employs contrastive learning between sMRI and FCN. During pre-training, we incorporate masked image modeling and network-image matching to enhance visual representation learning and modality alignment. Since the CINP facilitates knowledge transfer from FCN to sMRI, we introduce network prompting. It utilizes only sMRI from suspected patients and a small amount of FCNs from different patient classes for diagnosing mental disorders, which is practical in real-world clinical scenario. The competitive performance on three mental disorder diagnosis tasks demonstrate the effectiveness of the CINP in integrating multimodal MRI information, as well as the potential of incorporating sMRI into clinical diagnosis using network prompting.  
  </ol>  
</details>  
  
### [Diff-Reg v2: Diffusion-Based Matching Matrix Estimation for Image Matching and 3D Registration](http://arxiv.org/abs/2503.04127)  
Qianliang Wu, Haobo Jiang, Yaqing Ding, Lei Luo, Jin Xie, Jian Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Establishing reliable correspondences is crucial for all registration tasks, including 2D image registration, 3D point cloud registration, and 2D-3D image-to-point cloud registration. However, these tasks are often complicated by challenges such as scale inconsistencies, symmetry, and large deformations, which can lead to ambiguous matches. Previous feature-based and correspondence-based methods typically rely on geometric or semantic features to generate or polish initial potential correspondences. Some methods typically leverage specific geometric priors, such as topological preservation, to devise diverse and innovative strategies tailored to a given enhancement goal, which cannot be exhaustively enumerated. Additionally, many previous approaches rely on a single-step prediction head, which can struggle with local minima in complex matching scenarios. To address these challenges, we introduce an innovative paradigm that leverages a diffusion model in matrix space for robust matching matrix estimation. Our model treats correspondence estimation as a denoising diffusion process in the matching matrix space, gradually refining the intermediate matching matrix to the optimal one. Specifically, we apply the diffusion model in the doubly stochastic matrix space for 3D-3D and 2D-3D registration tasks. In the 2D image registration task, we deploy the diffusion model in a matrix subspace where dual-softmax projection regularization is applied. For all three registration tasks, we provide adaptive matching matrix embedding implementations tailored to the specific characteristics of each task while maintaining a consistent "match-to-warp" encoding pattern. Furthermore, we adopt a lightweight design for the denoising module. In inference, once points or image features are extracted and fixed, this module performs multi-step denoising predictions through reverse sampling.  
  </ol>  
</details>  
**comments**: arXiv admin note: text overlap with arXiv:2403.19919  
  
  



## NeRF  

### [Surgical Gaussian Surfels: Highly Accurate Real-time Surgical Scene Rendering](http://arxiv.org/abs/2503.04079)  
Idris O. Sunmola, Zhenjun Zhao, Samuel Schmidgall, Yumeng Wang, Paul Maria Scheikl, Axel Krieger  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Accurate geometric reconstruction of deformable tissues in monocular endoscopic video remains a fundamental challenge in robot-assisted minimally invasive surgery. Although recent volumetric and point primitive methods based on neural radiance fields (NeRF) and 3D Gaussian primitives have efficiently rendered surgical scenes, they still struggle with handling artifact-free tool occlusions and preserving fine anatomical details. These limitations stem from unrestricted Gaussian scaling and insufficient surface alignment constraints during reconstruction. To address these issues, we introduce Surgical Gaussian Surfels (SGS), which transforms anisotropic point primitives into surface-aligned elliptical splats by constraining the scale component of the Gaussian covariance matrix along the view-aligned axis. We predict accurate surfel motion fields using a lightweight Multi-Layer Perceptron (MLP) coupled with locality constraints to handle complex tissue deformations. We use homodirectional view-space positional gradients to capture fine image details by splitting Gaussian Surfels in over-reconstructed regions. In addition, we define surface normals as the direction of the steepest density change within each Gaussian surfel primitive, enabling accurate normal estimation without requiring monocular normal priors. We evaluate our method on two in-vivo surgical datasets, where it outperforms current state-of-the-art methods in surface geometry, normal map quality, and rendering efficiency, while remaining competitive in real-time rendering performance. We make our code available at https://github.com/aloma85/SurgicalGaussianSurfels  
  </ol>  
</details>  
  
### [LensDFF: Language-enhanced Sparse Feature Distillation for Efficient Few-Shot Dexterous Manipulation](http://arxiv.org/abs/2503.03890)  
Qian Feng, David S. Martinez Lema, Jianxiang Feng, Zhaopeng Chen, Alois Knoll  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Learning dexterous manipulation from few-shot demonstrations is a significant yet challenging problem for advanced, human-like robotic systems. Dense distilled feature fields have addressed this challenge by distilling rich semantic features from 2D visual foundation models into the 3D domain. However, their reliance on neural rendering models such as Neural Radiance Fields (NeRF) or Gaussian Splatting results in high computational costs. In contrast, previous approaches based on sparse feature fields either suffer from inefficiencies due to multi-view dependencies and extensive training or lack sufficient grasp dexterity. To overcome these limitations, we propose Language-ENhanced Sparse Distilled Feature Field (LensDFF), which efficiently distills view-consistent 2D features onto 3D points using our novel language-enhanced feature fusion strategy, thereby enabling single-view few-shot generalization. Based on LensDFF, we further introduce a few-shot dexterous manipulation framework that integrates grasp primitives into the demonstrations to generate stable and highly dexterous grasps. Moreover, we present a real2sim grasp evaluation pipeline for efficient grasp assessment and hyperparameter tuning. Through extensive simulation experiments based on the real2sim pipeline and real-world experiments, our approach achieves competitive grasping performance, outperforming state-of-the-art approaches.  
  </ol>  
</details>  
**comments**: 8 pages  
  
  



