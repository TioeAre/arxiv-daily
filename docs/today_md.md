<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#The-Empirical-Impact-of-Forgetting-and-Transfer-in-Continual-Visual-Odometry>The Empirical Impact of Forgetting and Transfer in Continual Visual Odometry</a></li>
        <li><a href=#Self-Supervised-Geometry-Guided-Initialization-for-Robust-Monocular-Visual-Odometry>Self-Supervised Geometry-Guided Initialization for Robust Monocular Visual Odometry</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#CamCo:-Camera-Controllable-3D-Consistent-Image-to-Video-Generation>CamCo: Camera-Controllable 3D-Consistent Image-to-Video Generation</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Can-CLIP-help-CLIP-in-learning-3D?>Can CLIP help CLIP in learning 3D?</a></li>
        <li><a href=#Decomposing-and-Interpreting-Image-Representations-via-Text-in-ViTs-Beyond-CLIP>Decomposing and Interpreting Image Representations via Text in ViTs Beyond CLIP</a></li>
        <li><a href=#Scale-Free-Image-Keypoints-Using-Differentiable-Persistent-Homology>Scale-Free Image Keypoints Using Differentiable Persistent Homology</a></li>
        <li><a href=#Visual-place-recognition-for-aerial-imagery:-A-survey>Visual place recognition for aerial imagery: A survey</a></li>
        <li><a href=#NuRF:-Nudging-the-Particle-Filter-in-Radiance-Fields-for-Robot-Visual-Localization>NuRF: Nudging the Particle Filter in Radiance Fields for Robot Visual Localization</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Scale-Free-Image-Keypoints-Using-Differentiable-Persistent-Homology>Scale-Free Image Keypoints Using Differentiable Persistent Homology</a></li>
        <li><a href=#W-Net:-A-Facial-Feature-Guided-Face-Super-Resolution-Network>W-Net: A Facial Feature-Guided Face Super-Resolution Network</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Enhancing-Temporal-Consistency-in-Video-Editing-by-Reconstructing-Videos-with-3D-Gaussian-Splatting>Enhancing Temporal Consistency in Video Editing by Reconstructing Videos with 3D Gaussian Splatting</a></li>
        <li><a href=#Query-based-Semantic-Gaussian-Field-for-Scene-Representation-in-Reinforcement-Learning>Query-based Semantic Gaussian Field for Scene Representation in Reinforcement Learning</a></li>
        <li><a href=#Reconstructing-and-Simulating-Dynamic-3D-Objects-with-Mesh-adsorbed-Gaussian-Splatting>Reconstructing and Simulating Dynamic 3D Objects with Mesh-adsorbed Gaussian Splatting</a></li>
        <li><a href=#Tetrahedron-Splatting-for-3D-Generation>Tetrahedron Splatting for 3D Generation</a></li>
        <li><a href=#Self-Calibrating-4D-Novel-View-Synthesis-from-Monocular-Videos-Using-Gaussian-Splatting>Self-Calibrating 4D Novel View Synthesis from Monocular Videos Using Gaussian Splatting</a></li>
        <li><a href=#PruNeRF:-Segment-Centric-Dataset-Pruning-via-3D-Spatial-Consistency>PruNeRF: Segment-Centric Dataset Pruning via 3D Spatial Consistency</a></li>
        <li><a href=#Representing-Animatable-Avatar-via-Factorized-Neural-Fields>Representing Animatable Avatar via Factorized Neural Fields</a></li>
        <li><a href=#SuperGaussian:-Repurposing-Video-Models-for-3D-Super-Resolution>SuperGaussian: Repurposing Video Models for 3D Super Resolution</a></li>
        <li><a href=#Efficient-Neural-Light-Fields-(ENeLF)-for-Mobile-Devices>Efficient Neural Light Fields (ENeLF) for Mobile Devices</a></li>
        <li><a href=#Bilateral-Guided-Radiance-Field-Processing>Bilateral Guided Radiance Field Processing</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [The Empirical Impact of Forgetting and Transfer in Continual Visual Odometry](http://arxiv.org/abs/2406.01797)  
Paolo Cudrano, Xiaoyu Luo, Matteo Matteucci  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    As robotics continues to advance, the need for adaptive and continuously-learning embodied agents increases, particularly in the realm of assistance robotics. Quick adaptability and long-term information retention are essential to operate in dynamic environments typical of humans' everyday lives. A lifelong learning paradigm is thus required, but it is scarcely addressed by current robotics literature. This study empirically investigates the impact of catastrophic forgetting and the effectiveness of knowledge transfer in neural networks trained continuously in an embodied setting. We focus on the task of visual odometry, which holds primary importance for embodied agents in enabling their self-localization. We experiment on the simple continual scenario of discrete transitions between indoor locations, akin to a robot navigating different apartments. In this regime, we observe initial satisfactory performance with high transferability between environments, followed by a specialization phase where the model prioritizes current environment-specific knowledge at the expense of generalization. Conventional regularization strategies and increased model capacity prove ineffective in mitigating this phenomenon. Rehearsal is instead mildly beneficial but with the addition of a substantial memory cost. Incorporating action information, as commonly done in embodied settings, facilitates quicker convergence but exacerbates specialization, making the model overly reliant on its motion expectations and less adept at correctly interpreting visual cues. These findings emphasize the open challenges of balancing adaptation and memory retention in lifelong robotics and contribute valuable insights into the application of a lifelong paradigm on embodied agents.  
  </ol>  
</details>  
**comments**: Accepted to CoLLAs 2024  
  
### [Self-Supervised Geometry-Guided Initialization for Robust Monocular Visual Odometry](http://arxiv.org/abs/2406.00929)  
Takayuki Kanai, Igor Vasiljevic, Vitor Guizilini, Kazuhiro Shintani  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Monocular visual odometry is a key technology in a wide variety of autonomous systems. Relative to traditional feature-based methods, that suffer from failures due to poor lighting, insufficient texture, large motions, etc., recent learning-based SLAM methods exploit iterative dense bundle adjustment to address such failure cases and achieve robust accurate localization in a wide variety of real environments, without depending on domain-specific training data. However, despite its potential, learning-based SLAM still struggles with scenarios involving large motion and object dynamics. In this paper, we diagnose key weaknesses in a popular learning-based SLAM model (DROID-SLAM) by analyzing major failure cases on outdoor benchmarks and exposing various shortcomings of its optimization process. We then propose the use of self-supervised priors leveraging a frozen large-scale pre-trained monocular depth estimation to initialize the dense bundle adjustment process, leading to robust visual odometry without the need to fine-tune the SLAM backbone. Despite its simplicity, our proposed method demonstrates significant improvements on KITTI odometry, as well as the challenging DDAD benchmark. Code and pre-trained models will be released upon publication.  
  </ol>  
</details>  
**comments**: 8 pages. 5 figures. This work has been submitted to the IEEE for
  possible publication. Copyright may be transferred without notice, after
  which this version may no longer be accessible  
  
  



## SFM  

### [CamCo: Camera-Controllable 3D-Consistent Image-to-Video Generation](http://arxiv.org/abs/2406.02509)  
Dejia Xu, Weili Nie, Chao Liu, Sifei Liu, Jan Kautz, Zhangyang Wang, Arash Vahdat  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recently video diffusion models have emerged as expressive generative tools for high-quality video content creation readily available to general users. However, these models often do not offer precise control over camera poses for video generation, limiting the expression of cinematic language and user control. To address this issue, we introduce CamCo, which allows fine-grained Camera pose Control for image-to-video generation. We equip a pre-trained image-to-video generator with accurately parameterized camera pose input using Pl\"ucker coordinates. To enhance 3D consistency in the videos produced, we integrate an epipolar attention module in each attention block that enforces epipolar constraints to the feature maps. Additionally, we fine-tune CamCo on real-world videos with camera poses estimated through structure-from-motion algorithms to better synthesize object motion. Our experiments show that CamCo significantly improves 3D consistency and camera control capabilities compared to previous models while effectively generating plausible object motion. Project page: https://ir1d.github.io/CamCo/  
  </ol>  
</details>  
**comments**: Project page: https://ir1d.github.io/CamCo/  
  
  



## Visual Localization  

### [Can CLIP help CLIP in learning 3D?](http://arxiv.org/abs/2406.02202)  
Cristian Sbrolli, Matteo Matteucci  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this study, we explore an alternative approach to enhance contrastive text-image-3D alignment in the absence of textual descriptions for 3D objects. We introduce two unsupervised methods, $I2I$ and $(I2L)^2$ , which leverage CLIP knowledge about textual and 2D data to compute the neural perceived similarity between two 3D samples. We employ the proposed methods to mine 3D hard negatives, establishing a multimodal contrastive pipeline with hard negative weighting via a custom loss function. We train on different configurations of the proposed hard negative mining approach, and we evaluate the accuracy of our models in 3D classification and on the cross-modal retrieval benchmark, testing image-to-shape and shape-to-image retrieval. Results demonstrate that our approach, even without explicit text alignment, achieves comparable or superior performance on zero-shot and standard 3D classification, while significantly improving both image-to-shape and shape-to-image retrieval compared to previous methods.  
  </ol>  
</details>  
  
### [Decomposing and Interpreting Image Representations via Text in ViTs Beyond CLIP](http://arxiv.org/abs/2406.01583)  
Sriram Balasubramanian, Samyadeep Basu, Soheil Feizi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent works have explored how individual components of the CLIP-ViT model contribute to the final representation by leveraging the shared image-text representation space of CLIP. These components, such as attention heads and MLPs, have been shown to capture distinct image features like shape, color or texture. However, understanding the role of these components in arbitrary vision transformers (ViTs) is challenging. To this end, we introduce a general framework which can identify the roles of various components in ViTs beyond CLIP. Specifically, we (a) automate the decomposition of the final representation into contributions from different model components, and (b) linearly map these contributions to CLIP space to interpret them via text. Additionally, we introduce a novel scoring function to rank components by their importance with respect to specific features. Applying our framework to various ViT variants (e.g. DeiT, DINO, DINOv2, Swin, MaxViT), we gain insights into the roles of different components concerning particular image features.These insights facilitate applications such as image retrieval using text descriptions or reference images, visualizing token importance heatmaps, and mitigating spurious correlations.  
  </ol>  
</details>  
**comments**: 22 pages, 15 figures  
  
### [Scale-Free Image Keypoints Using Differentiable Persistent Homology](http://arxiv.org/abs/2406.01315)  
Giovanni Barbarani, Francesco Vaccarino, Gabriele Trivigno, Marco Guerra, Gabriele Berton, Carlo Masone  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In computer vision, keypoint detection is a fundamental task, with applications spanning from robotics to image retrieval; however, existing learning-based methods suffer from scale dependency and lack flexibility. This paper introduces a novel approach that leverages Morse theory and persistent homology, powerful tools rooted in algebraic topology. We propose a novel loss function based on the recent introduction of a notion of subgradient in persistent homology, paving the way toward topological learning. Our detector, MorseDet, is the first topology-based learning model for feature detection, which achieves competitive performance in keypoint repeatability and introduces a principled and theoretically robust approach to the problem.  
  </ol>  
</details>  
**comments**: Accepted to ICML 2024  
  
### [Visual place recognition for aerial imagery: A survey](http://arxiv.org/abs/2406.00885)  
[[code](https://github.com/prime-slam/aero-vloc)]  
Ivan Moskalenko, Anastasiia Kornilova, Gonzalo Ferrer  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Aerial imagery and its direct application to visual localization is an essential problem for many Robotics and Computer Vision tasks. While Global Navigation Satellite Systems (GNSS) are the standard default solution for solving the aerial localization problem, it is subject to a number of limitations, such as, signal instability or solution unreliability that make this option not so desirable. Consequently, visual geolocalization is emerging as a viable alternative. However, adapting Visual Place Recognition (VPR) task to aerial imagery presents significant challenges, including weather variations and repetitive patterns. Current VPR reviews largely neglect the specific context of aerial data. This paper introduces a methodology tailored for evaluating VPR techniques specifically in the domain of aerial imagery, providing a comprehensive assessment of various methods and their performance. However, we not only compare various VPR methods, but also demonstrate the importance of selecting appropriate zoom and overlap levels when constructing map tiles to achieve maximum efficiency of VPR algorithms in the case of aerial imagery. The code is available on our GitHub repository -- https://github.com/prime-slam/aero-vloc.  
  </ol>  
</details>  
  
### [NuRF: Nudging the Particle Filter in Radiance Fields for Robot Visual Localization](http://arxiv.org/abs/2406.00312)  
Wugang Meng, Tianfu Wu, Huan Yin, Fumin Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Can we localize a robot in radiance fields only using monocular vision? This study presents NuRF, a nudged particle filter framework for 6-DoF robot visual localization in radiance fields. NuRF sets anchors in SE(3) to leverage visual place recognition, which provides image comparisons to guide the sampling process. This guidance could improve the convergence and robustness of particle filters for robot localization. Additionally, an adaptive scheme is designed to enhance the performance of NuRF, thus enabling both global visual localization and local pose tracking. Real-world experiments are conducted with comprehensive tests to demonstrate the effectiveness of NuRF. The results showcase the advantages of NuRF in terms of accuracy and efficiency, including comparisons with alternative approaches. Furthermore, we report our findings for future studies and advancements in robot navigation in radiance fields.  
  </ol>  
</details>  
**comments**: 11 pages, 14 figures  
  
  



## Keypoint Detection  

### [Scale-Free Image Keypoints Using Differentiable Persistent Homology](http://arxiv.org/abs/2406.01315)  
Giovanni Barbarani, Francesco Vaccarino, Gabriele Trivigno, Marco Guerra, Gabriele Berton, Carlo Masone  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In computer vision, keypoint detection is a fundamental task, with applications spanning from robotics to image retrieval; however, existing learning-based methods suffer from scale dependency and lack flexibility. This paper introduces a novel approach that leverages Morse theory and persistent homology, powerful tools rooted in algebraic topology. We propose a novel loss function based on the recent introduction of a notion of subgradient in persistent homology, paving the way toward topological learning. Our detector, MorseDet, is the first topology-based learning model for feature detection, which achieves competitive performance in keypoint repeatability and introduces a principled and theoretically robust approach to the problem.  
  </ol>  
</details>  
**comments**: Accepted to ICML 2024  
  
### [W-Net: A Facial Feature-Guided Face Super-Resolution Network](http://arxiv.org/abs/2406.00676)  
Hao Liu, Yang Yang, Yunxia Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Face Super-Resolution (FSR) aims to recover high-resolution (HR) face images from low-resolution (LR) ones. Despite the progress made by convolutional neural networks in FSR, the results of existing approaches are not ideal due to their low reconstruction efficiency and insufficient utilization of prior information. Considering that faces are highly structured objects, effectively leveraging facial priors to improve FSR results is a worthwhile endeavor. This paper proposes a novel network architecture called W-Net to address this challenge. W-Net leverages meticulously designed Parsing Block to fully exploit the resolution potential of LR image. We use this parsing map as an attention prior, effectively integrating information from both the parsing map and LR images. Simultaneously, we perform multiple fusions in various dimensions through the W-shaped network structure combined with the LPF(LR-Parsing Map Fusion Module). Additionally, we utilize a facial parsing graph as a mask, assigning different weights and loss functions to key facial areas to balance the performance of our reconstructed facial images between perceptual quality and pixel accuracy. We conducted extensive comparative experiments, not only limited to conventional facial super-resolution metrics but also extending to downstream tasks such as facial recognition and facial keypoint detection. The experiments demonstrate that W-Net exhibits outstanding performance in quantitative metrics, visual quality, and downstream tasks.  
  </ol>  
</details>  
**comments**: 15 pages,9 figures  
  
  



## NeRF  

### [Enhancing Temporal Consistency in Video Editing by Reconstructing Videos with 3D Gaussian Splatting](http://arxiv.org/abs/2406.02541)  
Inkyu Shin, Qihang Yu, Xiaohui Shen, In So Kweon, Kuk-Jin Yoon, Liang-Chieh Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in zero-shot video diffusion models have shown promise for text-driven video editing, but challenges remain in achieving high temporal consistency. To address this, we introduce Video-3DGS, a 3D Gaussian Splatting (3DGS)-based video refiner designed to enhance temporal consistency in zero-shot video editors. Our approach utilizes a two-stage 3D Gaussian optimizing process tailored for editing dynamic monocular videos. In the first stage, Video-3DGS employs an improved version of COLMAP, referred to as MC-COLMAP, which processes original videos using a Masked and Clipped approach. For each video clip, MC-COLMAP generates the point clouds for dynamic foreground objects and complex backgrounds. These point clouds are utilized to initialize two sets of 3D Gaussians (Frg-3DGS and Bkg-3DGS) aiming to represent foreground and background views. Both foreground and background views are then merged with a 2D learnable parameter map to reconstruct full views. In the second stage, we leverage the reconstruction ability developed in the first stage to impose the temporal constraints on the video diffusion model. To demonstrate the efficacy of Video-3DGS on both stages, we conduct extensive experiments across two related tasks: Video Reconstruction and Video Editing. Video-3DGS trained with 3k iterations significantly improves video reconstruction quality (+3 PSNR, +7 PSNR increase) and training efficiency (x1.9, x4.5 times faster) over NeRF-based and 3DGS-based state-of-art methods on DAVIS dataset, respectively. Moreover, it enhances video editing by ensuring temporal consistency across 58 dynamic monocular videos.  
  </ol>  
</details>  
  
### [Query-based Semantic Gaussian Field for Scene Representation in Reinforcement Learning](http://arxiv.org/abs/2406.02370)  
Jiaxu Wang, Ziyi Zhang, Qiang Zhang, Jia Li, Jingkai Sun, Mingyuan Sun, Junhao He, Renjing Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Latent scene representation plays a significant role in training reinforcement learning (RL) agents. To obtain good latent vectors describing the scenes, recent works incorporate the 3D-aware latent-conditioned NeRF pipeline into scene representation learning. However, these NeRF-related methods struggle to perceive 3D structural information due to the inefficient dense sampling in volumetric rendering. Moreover, they lack fine-grained semantic information included in their scene representation vectors because they evenly consider free and occupied spaces. Both of them can destroy the performance of downstream RL tasks. To address the above challenges, we propose a novel framework that adopts the efficient 3D Gaussian Splatting (3DGS) to learn 3D scene representation for the first time. In brief, we present the Query-based Generalizable 3DGS to bridge the 3DGS technique and scene representations with more geometrical awareness than those in NeRFs. Moreover, we present the Hierarchical Semantics Encoding to ground the fine-grained semantic features to 3D Gaussians and further distilled to the scene representation vectors. We conduct extensive experiments on two RL platforms including Maniskill2 and Robomimic across 10 different tasks. The results show that our method outperforms the other 5 baselines by a large margin. We achieve the best success rates on 8 tasks and the second-best on the other two tasks.  
  </ol>  
</details>  
  
### [Reconstructing and Simulating Dynamic 3D Objects with Mesh-adsorbed Gaussian Splatting](http://arxiv.org/abs/2406.01593)  
Shaojie Ma, Yawei Luo, Yi Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D reconstruction and simulation, while interrelated, have distinct objectives: reconstruction demands a flexible 3D representation adaptable to diverse scenes, whereas simulation requires a structured representation to model motion principles effectively. This paper introduces the Mesh-adsorbed Gaussian Splatting (MaGS) method to resolve such a dilemma. MaGS constrains 3D Gaussians to hover on the mesh surface, creating a mutual-adsorbed mesh-Gaussian 3D representation that combines the rendering flexibility of 3D Gaussians with the spatial coherence of meshes. Leveraging this representation, we introduce a learnable Relative Deformation Field (RDF) to model the relative displacement between the mesh and 3D Gaussians, extending traditional mesh-driven deformation paradigms that only rely on ARAP prior, thus capturing the motion of each 3D Gaussian more precisely. By joint optimizing meshes, 3D Gaussians, and RDF, MaGS achieves both high rendering accuracy and realistic deformation. Extensive experiments on the D-NeRF and NeRF-DS datasets demonstrate that MaGS can generate competitive results in both reconstruction and simulation.  
  </ol>  
</details>  
**comments**: Project Page: see https://wcwac.github.io/MaGS-page/  
  
### [Tetrahedron Splatting for 3D Generation](http://arxiv.org/abs/2406.01579)  
[[code](https://github.com/fudan-zvg/tet-splatting)]  
Chun Gu, Zeyu Yang, Zijie Pan, Xiatian Zhu, Li Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D representation is essential to the significant advance of 3D generation with 2D diffusion priors. As a flexible representation, NeRF has been first adopted for 3D representation. With density-based volumetric rendering, it however suffers both intensive computational overhead and inaccurate mesh extraction. Using a signed distance field and Marching Tetrahedra, DMTet allows for precise mesh extraction and real-time rendering but is limited in handling large topological changes in meshes, leading to optimization challenges. Alternatively, 3D Gaussian Splatting (3DGS) is favored in both training and rendering efficiency while falling short in mesh extraction. In this work, we introduce a novel 3D representation, Tetrahedron Splatting (TeT-Splatting), that supports easy convergence during optimization, precise mesh extraction, and real-time rendering simultaneously. This is achieved by integrating surface-based volumetric rendering within a structured tetrahedral grid while preserving the desired ability of precise mesh extraction, and a tile-based differentiable tetrahedron rasterizer. Furthermore, we incorporate eikonal and normal consistency regularization terms for the signed distance field to improve generation quality and stability. Critically, our representation can be trained without mesh extraction, making the optimization process easier to converge. Our TeT-Splatting can be readily integrated in existing 3D generation pipelines, along with polygonal mesh for texture optimization. Extensive experiments show that our TeT-Splatting strikes a superior tradeoff among convergence speed, render efficiency, and mesh quality as compared to previous alternatives under varying 3D generation settings.  
  </ol>  
</details>  
**comments**: Code: https://github.com/fudan-zvg/tet-splatting  
  
### [Self-Calibrating 4D Novel View Synthesis from Monocular Videos Using Gaussian Splatting](http://arxiv.org/abs/2406.01042)  
[[code](https://github.com/fangli333/sc-4dgs)]  
Fang Li, Hao Zhang, Narendra Ahuja  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Gaussian Splatting (GS) has significantly elevated scene reconstruction efficiency and novel view synthesis (NVS) accuracy compared to Neural Radiance Fields (NeRF), particularly for dynamic scenes. However, current 4D NVS methods, whether based on GS or NeRF, primarily rely on camera parameters provided by COLMAP and even utilize sparse point clouds generated by COLMAP for initialization, which lack accuracy as well are time-consuming. This sometimes results in poor dynamic scene representation, especially in scenes with large object movements, or extreme camera conditions e.g. small translations combined with large rotations. Some studies simultaneously optimize the estimation of camera parameters and scenes, supervised by additional information like depth, optical flow, etc. obtained from off-the-shelf models. Using this unverified information as ground truth can reduce robustness and accuracy, which does frequently occur for long monocular videos (with e.g. > hundreds of frames). We propose a novel approach that learns a high-fidelity 4D GS scene representation with self-calibration of camera parameters. It includes the extraction of 2D point features that robustly represent 3D structure, and their use for subsequent joint optimization of camera parameters and 3D structure towards overall 4D scene optimization. We demonstrate the accuracy and time efficiency of our method through extensive quantitative and qualitative experimental results on several standard benchmarks. The results show significant improvements over state-of-the-art methods for 4D novel view synthesis. The source code will be released soon at https://github.com/fangli333/SC-4DGS.  
  </ol>  
</details>  
**comments**: GitHub Page: https://github.com/fangli333/SC-4DGS  
  
### [PruNeRF: Segment-Centric Dataset Pruning via 3D Spatial Consistency](http://arxiv.org/abs/2406.00798)  
Yeonsung Jung, Heecheol Yun, Joonhyung Park, Jin-Hwa Kim, Eunho Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) have shown remarkable performance in learning 3D scenes. However, NeRF exhibits vulnerability when confronted with distractors in the training images -- unexpected objects are present only within specific views, such as moving entities like pedestrians or birds. Excluding distractors during dataset construction is a straightforward solution, but without prior knowledge of their types and quantities, it becomes prohibitively expensive. In this paper, we propose PruNeRF, a segment-centric dataset pruning framework via 3D spatial consistency, that effectively identifies and prunes the distractors. We first examine existing metrics for measuring pixel-wise distraction and introduce Influence Functions for more accurate measurements. Then, we assess 3D spatial consistency using a depth-based reprojection technique to obtain 3D-aware distraction. Furthermore, we incorporate segmentation for pixel-to-segment refinement, enabling more precise identification. Our experiments on benchmark datasets demonstrate that PruNeRF consistently outperforms state-of-the-art methods in robustness against distractors.  
  </ol>  
</details>  
  
### [Representing Animatable Avatar via Factorized Neural Fields](http://arxiv.org/abs/2406.00637)  
Chunjin Song, Zhijie Wu, Bastian Wandt, Leonid Sigal, Helge Rhodin  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    For reconstructing high-fidelity human 3D models from monocular videos, it is crucial to maintain consistent large-scale body shapes along with finely matched subtle wrinkles. This paper explores the observation that the per-frame rendering results can be factorized into a pose-independent component and a corresponding pose-dependent equivalent to facilitate frame consistency. Pose adaptive textures can be further improved by restricting frequency bands of these two components. In detail, pose-independent outputs are expected to be low-frequency, while highfrequency information is linked to pose-dependent factors. We achieve a coherent preservation of both coarse body contours across the entire input video and finegrained texture features that are time variant with a dual-branch network with distinct frequency components. The first branch takes coordinates in canonical space as input, while the second branch additionally considers features outputted by the first branch and pose information of each frame. Our network integrates the information predicted by both branches and utilizes volume rendering to generate photo-realistic 3D human images. Through experiments, we demonstrate that our network surpasses the neural radiance fields (NeRF) based state-of-the-art methods in preserving high-frequency details and ensuring consistent body contours.  
  </ol>  
</details>  
  
### [SuperGaussian: Repurposing Video Models for 3D Super Resolution](http://arxiv.org/abs/2406.00609)  
Yuan Shen, Duygu Ceylan, Paul Guerrero, Zexiang Xu, Niloy J. Mitra, Shenlong Wang, Anna Frühstück  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present a simple, modular, and generic method that upsamples coarse 3D models by adding geometric and appearance details. While generative 3D models now exist, they do not yet match the quality of their counterparts in image and video domains. We demonstrate that it is possible to directly repurpose existing (pretrained) video models for 3D super-resolution and thus sidestep the problem of the shortage of large repositories of high-quality 3D training models. We describe how to repurpose video upsampling models, which are not 3D consistent, and combine them with 3D consolidation to produce 3D-consistent results. As output, we produce high quality Gaussian Splat models, which are object centric and effective. Our method is category agnostic and can be easily incorporated into existing 3D workflows. We evaluate our proposed SuperGaussian on a variety of 3D inputs, which are diverse both in terms of complexity and representation (e.g., Gaussian Splats or NeRFs), and demonstrate that our simple method significantly improves the fidelity of the final 3D models. Check our project website for details: supergaussian.github.io  
  </ol>  
</details>  
**comments**: Check our project website for details:
  https://supergaussian.github.io  
  
### [Efficient Neural Light Fields (ENeLF) for Mobile Devices](http://arxiv.org/abs/2406.00598)  
Austin Peng  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Novel view synthesis (NVS) is a challenge in computer vision and graphics, focusing on generating realistic images of a scene from unobserved camera poses, given a limited set of authentic input images. Neural radiance fields (NeRF) achieved impressive results in rendering quality by utilizing volumetric rendering. However, NeRF and its variants are unsuitable for mobile devices due to the high computational cost of volumetric rendering. Emerging research in neural light fields (NeLF) eliminates the need for volumetric rendering by directly learning a mapping from ray representation to pixel color. NeLF has demonstrated its capability to achieve results similar to NeRF but requires a more extensive, computationally intensive network that is not mobile-friendly. Unlike existing works, this research builds upon the novel network architecture introduced by MobileR2L and aggressively applies a compression technique (channel-wise structure pruning) to produce a model that runs efficiently on mobile devices with lower latency and smaller sizes, with a slight decrease in performance.  
  </ol>  
</details>  
  
### [Bilateral Guided Radiance Field Processing](http://arxiv.org/abs/2406.00448)  
Yuehao Wang, Chaoyi Wang, Bingchen Gong, Tianfan Xue  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) achieves unprecedented performance in synthesizing novel view synthesis, utilizing multi-view consistency. When capturing multiple inputs, image signal processing (ISP) in modern cameras will independently enhance them, including exposure adjustment, color correction, local tone mapping, etc. While these processings greatly improve image quality, they often break the multi-view consistency assumption, leading to "floaters" in the reconstructed radiance fields. To address this concern without compromising visual aesthetics, we aim to first disentangle the enhancement by ISP at the NeRF training stage and re-apply user-desired enhancements to the reconstructed radiance fields at the finishing stage. Furthermore, to make the re-applied enhancements consistent between novel views, we need to perform imaging signal processing in 3D space (i.e. "3D ISP"). For this goal, we adopt the bilateral grid, a locally-affine model, as a generalized representation of ISP processing. Specifically, we optimize per-view 3D bilateral grids with radiance fields to approximate the effects of camera pipelines for each input view. To achieve user-adjustable 3D finishing, we propose to learn a low-rank 4D bilateral grid from a given single view edit, lifting photo enhancements to the whole 3D scene. We demonstrate our approach can boost the visual quality of novel view synthesis by effectively removing floaters and performing enhancements from user retouching. The source code and our data are available at: https://bilarfpro.github.io.  
  </ol>  
</details>  
**comments**: SIGGRAPH (ACM TOG), 2024. Project page: https://bilarfpro.github.io  
  
  



