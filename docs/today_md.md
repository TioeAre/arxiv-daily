<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Hybrid-Structure-from-Motion-and-Camera-Relocalization-for-Enhanced-Egocentric-Localization>Hybrid Structure-from-Motion and Camera Relocalization for Enhanced Egocentric Localization</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Improving-Visual-Place-Recognition-Based-Robot-Navigation-Through-Verification-of-Localization-Estimates>Improving Visual Place Recognition Based Robot Navigation Through Verification of Localization Estimates</a></li>
        <li><a href=#Lifelong-Histopathology-Whole-Slide-Image-Retrieval-via-Distance-Consistency-Rehearsal>Lifelong Histopathology Whole Slide Image Retrieval via Distance Consistency Rehearsal</a></li>
        <li><a href=#SGLC:-Semantic-Graph-Guided-Coarse-Fine-Refine-Full-Loop-Closing-for-LiDAR-SLAM>SGLC: Semantic Graph-Guided Coarse-Fine-Refine Full Loop Closing for LiDAR SLAM</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#WildGaussians:-3D-Gaussian-Splatting-in-the-Wild>WildGaussians: 3D Gaussian Splatting in the Wild</a></li>
        <li><a href=#MeshAvatar:-Learning-High-quality-Triangular-Human-Avatars-from-Multi-view-Videos>MeshAvatar: Learning High-quality Triangular Human Avatars from Multi-view Videos</a></li>
        <li><a href=#Explicit_NeRF_QA:-A-Quality-Assessment-Database-for-Explicit-NeRF-Model-Compression>Explicit_NeRF_QA: A Quality Assessment Database for Explicit NeRF Model Compression</a></li>
        <li><a href=#Bayesian-uncertainty-analysis-for-underwater-3D-reconstruction-with-neural-radiance-fields>Bayesian uncertainty analysis for underwater 3D reconstruction with neural radiance fields</a></li>
        <li><a href=#Survey-on-Fundamental-Deep-Learning-3D-Reconstruction-Techniques>Survey on Fundamental Deep Learning 3D Reconstruction Techniques</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Hybrid Structure-from-Motion and Camera Relocalization for Enhanced Egocentric Localization](http://arxiv.org/abs/2407.08023)  
[[code](https://github.com/wayne-mai/egoloc_v1)]  
Jinjie Mai, Abdullah Hamdi, Silvio Giancola, Chen Zhao, Bernard Ghanem  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We built our pipeline EgoLoc-v1, mainly inspired by EgoLoc. We propose a model ensemble strategy to improve the camera pose estimation part of the VQ3D task, which has been proven to be essential in previous work. The core idea is not only to do SfM for egocentric videos but also to do 2D-3D matching between existing 3D scans and 2D video frames. In this way, we have a hybrid SfM and camera relocalization pipeline, which can provide us with more camera poses, leading to higher QwP and overall success rate. Our method achieves the best performance regarding the most important metric, the overall success rate. We surpass previous state-of-the-art, the competitive EgoLoc, by $1.5\%$ . The code is available at \url{https://github.com/Wayne-Mai/egoloc_v1}.  
  </ol>  
</details>  
**comments**: 1st place winner of the 2024 Ego4D-Ego-Exo4D Challenge in VQ3D  
  
  



## Visual Localization  

### [Improving Visual Place Recognition Based Robot Navigation Through Verification of Localization Estimates](http://arxiv.org/abs/2407.08162)  
Owen Claxton, Connor Malone, Helen Carson, Jason Ford, Gabe Bolton, Iman Shames, Michael Milford  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Place Recognition (VPR) systems often have imperfect performance, which affects robot navigation decisions. This research introduces a novel Multi-Layer Perceptron (MLP) integrity monitor for VPR which demonstrates improved performance and generalizability over the previous state-of-the-art SVM approach, removing per-environment training and reducing manual tuning requirements. We test our proposed system in extensive real-world experiments, where we also present two real-time integrity-based VPR verification methods: an instantaneous rejection method for a robot navigating to a goal zone (Experiment 1); and a historical method that takes a best, verified, match from its recent trajectory and uses an odometer to extrapolate forwards to a current position estimate (Experiment 2). Noteworthy results for Experiment 1 include a decrease in aggregate mean along-track goal error from ~9.8m to ~3.1m in missions the robot pursued to completion, and an increase in the aggregate rate of successful mission completion from ~41% to ~55%. Experiment 2 showed a decrease in aggregate mean along-track localization error from ~2.0m to ~0.5m, and an increase in the aggregate precision of localization attempts from ~97% to ~99%. Overall, our results demonstrate the practical usefulness of a VPR integrity monitor in real-world robotics to improve VPR localization and consequent navigation performance.  
  </ol>  
</details>  
**comments**: Currently Under Review  
  
### [Lifelong Histopathology Whole Slide Image Retrieval via Distance Consistency Rehearsal](http://arxiv.org/abs/2407.08153)  
Xinyu Zhu, Zhiguo Jiang, Kun Wu, Jun Shi, Yushan Zheng  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Content-based histopathological image retrieval (CBHIR) has gained attention in recent years, offering the capability to return histopathology images that are content-wise similar to the query one from an established database. However, in clinical practice, the continuously expanding size of WSI databases limits the practical application of the current CBHIR methods. In this paper, we propose a Lifelong Whole Slide Retrieval (LWSR) framework to address the challenges of catastrophic forgetting by progressive model updating on continuously growing retrieval database. Our framework aims to achieve the balance between stability and plasticity during continuous learning. To preserve system plasticity, we utilize local memory bank with reservoir sampling method to save instances, which can comprehensively encompass the feature spaces of both old and new tasks. Furthermore, A distance consistency rehearsal (DCR) module is designed to ensure the retrieval queue's consistency for previous tasks, which is regarded as stability within a lifelong CBHIR system. We evaluated the proposed method on four public WSI datasets from TCGA projects. The experimental results have demonstrated the proposed method is effective and is superior to the state-of-the-art methods.  
  </ol>  
</details>  
  
### [SGLC: Semantic Graph-Guided Coarse-Fine-Refine Full Loop Closing for LiDAR SLAM](http://arxiv.org/abs/2407.08106)  
Neng Wang, Xieyuanli Chen, Chenghao Shi, Zhiqiang Zheng, Hongshan Yu, Huimin Lu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Loop closing is a crucial component in SLAM that helps eliminate accumulated errors through two main steps: loop detection and loop pose correction. The first step determines whether loop closing should be performed, while the second estimates the 6-DoF pose to correct odometry drift. Current methods mostly focus on developing robust descriptors for loop closure detection, often neglecting loop pose estimation. A few methods that do include pose estimation either suffer from low accuracy or incur high computational costs. To tackle this problem, we introduce SGLC, a real-time semantic graph-guided full loop closing method, with robust loop closure detection and 6-DoF pose estimation capabilities. SGLC takes into account the distinct characteristics of foreground and background points. For foreground instances, it builds a semantic graph that not only abstracts point cloud representation for fast descriptor generation and matching but also guides the subsequent loop verification and initial pose estimation. Background points, meanwhile, are exploited to provide more geometric features for scan-wise descriptor construction and stable planar information for further pose refinement. Loop pose estimation employs a coarse-fine-refine registration scheme that considers the alignment of both instance points and background points, offering high efficiency and accuracy. We evaluate the loop closing performance of SGLC through extensive experiments on the KITTI and KITTI-360 datasets, demonstrating its superiority over existing state-of-the-art methods. Additionally, we integrate SGLC into a SLAM system, eliminating accumulated errors and improving overall SLAM performance. The implementation of SGLC will be released at https://github.com/nubot-nudt/SGLC.  
  </ol>  
</details>  
**comments**: 8 pages, 4 figures  
  
  



## NeRF  

### [WildGaussians: 3D Gaussian Splatting in the Wild](http://arxiv.org/abs/2407.08447)  
Jonas Kulhanek, Songyou Peng, Zuzana Kukelova, Marc Pollefeys, Torsten Sattler  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    While the field of 3D scene reconstruction is dominated by NeRFs due to their photorealistic quality, 3D Gaussian Splatting (3DGS) has recently emerged, offering similar quality with real-time rendering speeds. However, both methods primarily excel with well-controlled 3D scenes, while in-the-wild data - characterized by occlusions, dynamic objects, and varying illumination - remains challenging. NeRFs can adapt to such conditions easily through per-image embedding vectors, but 3DGS struggles due to its explicit representation and lack of shared parameters. To address this, we introduce WildGaussians, a novel approach to handle occlusions and appearance changes with 3DGS. By leveraging robust DINO features and integrating an appearance modeling module within 3DGS, our method achieves state-of-the-art results. We demonstrate that WildGaussians matches the real-time rendering speed of 3DGS while surpassing both 3DGS and NeRF baselines in handling in-the-wild data, all within a simple architectural framework.  
  </ol>  
</details>  
**comments**: https://wild-gaussians.github.io/  
  
### [MeshAvatar: Learning High-quality Triangular Human Avatars from Multi-view Videos](http://arxiv.org/abs/2407.08414)  
[[code](https://github.com/shad0wta9/meshavatar)]  
Yushuo Chen, Zerong Zheng, Zhe Li, Chao Xu, Yebin Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present a novel pipeline for learning high-quality triangular human avatars from multi-view videos. Recent methods for avatar learning are typically based on neural radiance fields (NeRF), which is not compatible with traditional graphics pipeline and poses great challenges for operations like editing or synthesizing under different environments. To overcome these limitations, our method represents the avatar with an explicit triangular mesh extracted from an implicit SDF field, complemented by an implicit material field conditioned on given poses. Leveraging this triangular avatar representation, we incorporate physics-based rendering to accurately decompose geometry and texture. To enhance both the geometric and appearance details, we further employ a 2D UNet as the network backbone and introduce pseudo normal ground-truth as additional supervision. Experiments show that our method can learn triangular avatars with high-quality geometry reconstruction and plausible material decomposition, inherently supporting editing, manipulation or relighting operations.  
  </ol>  
</details>  
**comments**: Project Page: https://shad0wta9.github.io/meshavatar-page/  
  
### [Explicit_NeRF_QA: A Quality Assessment Database for Explicit NeRF Model Compression](http://arxiv.org/abs/2407.08165)  
Yuke Xing, Qi Yang, Kaifa Yang, Yilin Xu, Zhu Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In recent years, Neural Radiance Fields (NeRF) have demonstrated significant advantages in representing and synthesizing 3D scenes. Explicit NeRF models facilitate the practical NeRF applications with faster rendering speed, and also attract considerable attention in NeRF compression due to its huge storage cost. To address the challenge of the NeRF compression study, in this paper, we construct a new dataset, called Explicit_NeRF_QA. We use 22 3D objects with diverse geometries, textures, and material complexities to train four typical explicit NeRF models across five parameter levels. Lossy compression is introduced during the model generation, pivoting the selection of key parameters such as hash table size for InstantNGP and voxel grid resolution for Plenoxels. By rendering NeRF samples to processed video sequences (PVS), a large scale subjective experiment with lab environment is conducted to collect subjective scores from 21 viewers. The diversity of content, accuracy of mean opinion scores (MOS), and characteristics of NeRF distortion are comprehensively presented, establishing the heterogeneity of the proposed dataset. The state-of-the-art objective metrics are tested in the new dataset. Best Person correlation, which is around 0.85, is collected from the full-reference objective metric. All tested no-reference metrics report very poor results with 0.4 to 0.6 correlations, demonstrating the need for further development of more robust no-reference metrics. The dataset, including NeRF samples, source 3D objects, multiview images for NeRF generation, PVSs, MOS, is made publicly available at the following location: https://github.com/LittlericeChloe/Explicit_NeRF_QA.  
  </ol>  
</details>  
**comments**: 5 pages, 4 figures, 2 tables, conference  
  
### [Bayesian uncertainty analysis for underwater 3D reconstruction with neural radiance fields](http://arxiv.org/abs/2407.08154)  
Haojie Lian, Xinhao Li, Yilin Qu, Jing Du, Zhuxuan Meng, Jie Liu, Leilei Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural radiance fields (NeRFs) are a deep learning technique that can generate novel views of 3D scenes using sparse 2D images from different viewing directions and camera poses. As an extension of conventional NeRFs in underwater environment, where light can get absorbed and scattered by water, SeaThru-NeRF was proposed to separate the clean appearance and geometric structure of underwater scene from the effects of the scattering medium. Since the quality of the appearance and structure of underwater scenes is crucial for downstream tasks such as underwater infrastructure inspection, the reliability of the 3D reconstruction model should be considered and evaluated. Nonetheless, owing to the lack of ability to quantify uncertainty in 3D reconstruction of underwater scenes under natural ambient illumination, the practical deployment of NeRFs in unmanned autonomous underwater navigation is limited. To address this issue, we introduce a spatial perturbation field D_omega based on Bayes' rays in SeaThru-NeRF and perform Laplace approximation to obtain a Gaussian distribution N(0,Sigma) of the parameters omega, where the diagonal elements of Sigma correspond to the uncertainty at each spatial location. We also employ a simple thresholding method to remove artifacts from the rendered results of underwater scenes. Numerical experiments are provided to demonstrate the effectiveness of this approach.  
  </ol>  
</details>  
  
### [Survey on Fundamental Deep Learning 3D Reconstruction Techniques](http://arxiv.org/abs/2407.08137)  
Yonge Bai, LikHang Wong, TszYin Twan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This survey aims to investigate fundamental deep learning (DL) based 3D reconstruction techniques that produce photo-realistic 3D models and scenes, highlighting Neural Radiance Fields (NeRFs), Latent Diffusion Models (LDM), and 3D Gaussian Splatting. We dissect the underlying algorithms, evaluate their strengths and tradeoffs, and project future research trajectories in this rapidly evolving field. We provide a comprehensive overview of the fundamental in DL-driven 3D scene reconstruction, offering insights into their potential applications and limitations.  
  </ol>  
</details>  
  
  



