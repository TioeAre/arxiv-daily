<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Robust-Incremental-Structure-from-Motion-with-Hybrid-Features>Robust Incremental Structure-from-Motion with Hybrid Features</a></li>
        <li><a href=#MASt3R-SfM:-a-Fully-Integrated-Solution-for-Unconstrained-Structure-from-Motion>MASt3R-SfM: a Fully-Integrated Solution for Unconstrained Structure-from-Motion</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#CELLmap:-Enhancing-LiDAR-SLAM-through-Elastic-and-Lightweight-Spherical-Map-Representation>CELLmap: Enhancing LiDAR SLAM through Elastic and Lightweight Spherical Map Representation</a></li>
        <li><a href=#VLAD-BuFF:-Burst-aware-Fast-Feature-Aggregation-for-Visual-Place-Recognition>VLAD-BuFF: Burst-aware Fast Feature Aggregation for Visual Place Recognition</a></li>
        <li><a href=#MASt3R-SfM:-a-Fully-Integrated-Solution-for-Unconstrained-Structure-from-Motion>MASt3R-SfM: a Fully-Integrated Solution for Unconstrained Structure-from-Motion</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#OpenKD:-Opening-Prompt-Diversity-for-Zero--and-Few-shot-Keypoint-Detection>OpenKD: Opening Prompt Diversity for Zero- and Few-shot Keypoint Detection</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Distributed-NeRF-Learning-for-Collaborative-Multi-Robot-Perception>Distributed NeRF Learning for Collaborative Multi-Robot Perception</a></li>
        <li><a href=#Active-Neural-Mapping-at-Scale>Active Neural Mapping at Scale</a></li>
        <li><a href=#OPONeRF:-One-Point-One-NeRF-for-Robust-Neural-Rendering>OPONeRF: One-Point-One NeRF for Robust Neural Rendering</a></li>
        <li><a href=#G3R:-Gradient-Guided-Generalizable-Reconstruction>G3R: Gradient Guided Generalizable Reconstruction</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Robust Incremental Structure-from-Motion with Hybrid Features](http://arxiv.org/abs/2409.19811)  
Shaohui Liu, Yidan Gao, Tianyi Zhang, Rémi Pautrat, Johannes L. Schönberger, Viktor Larsson, Marc Pollefeys  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Structure-from-Motion (SfM) has become a ubiquitous tool for camera calibration and scene reconstruction with many downstream applications in computer vision and beyond. While the state-of-the-art SfM pipelines have reached a high level of maturity in well-textured and well-configured scenes over the last decades, they still fall short of robustly solving the SfM problem in challenging scenarios. In particular, weakly textured scenes and poorly constrained configurations oftentimes cause catastrophic failures or large errors for the primarily keypoint-based pipelines. In these scenarios, line segments are often abundant and can offer complementary geometric constraints. Their large spatial extent and typically structured configurations lead to stronger geometric constraints as compared to traditional keypoint-based methods. In this work, we introduce an incremental SfM system that, in addition to points, leverages lines and their structured geometric relations. Our technical contributions span the entire pipeline (mapping, triangulation, registration) and we integrate these into a comprehensive end-to-end SfM system that we share as an open-source software with the community. We also present the first analytical method to propagate uncertainties for 3D optimized lines via sensitivity analysis. Experiments show that our system is consistently more robust and accurate compared to the widely used point-based state of the art in SfM -- achieving richer maps and more precise camera registrations, especially under challenging conditions. In addition, our uncertainty-aware localization module alone is able to consistently improve over the state of the art under both point-alone and hybrid setups.  
  </ol>  
</details>  
**comments**: 40 pages, 16 figures, 9 tables. To appear in ECCV 2024  
  
### [MASt3R-SfM: a Fully-Integrated Solution for Unconstrained Structure-from-Motion](http://arxiv.org/abs/2409.19152)  
Bardienus Duisterhof, Lojze Zust, Philippe Weinzaepfel, Vincent Leroy, Yohann Cabon, Jerome Revaud  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Structure-from-Motion (SfM), a task aiming at jointly recovering camera poses and 3D geometry of a scene given a set of images, remains a hard problem with still many open challenges despite decades of significant progress. The traditional solution for SfM consists of a complex pipeline of minimal solvers which tends to propagate errors and fails when images do not sufficiently overlap, have too little motion, etc. Recent methods have attempted to revisit this paradigm, but we empirically show that they fall short of fixing these core issues. In this paper, we propose instead to build upon a recently released foundation model for 3D vision that can robustly produce local 3D reconstructions and accurate matches. We introduce a low-memory approach to accurately align these local reconstructions in a global coordinate system. We further show that such foundation models can serve as efficient image retrievers without any overhead, reducing the overall complexity from quadratic to linear. Overall, our novel SfM pipeline is simple, scalable, fast and truly unconstrained, i.e. it can handle any collection of images, ordered or not. Extensive experiments on multiple benchmarks show that our method provides steady performance across diverse settings, especially outperforming existing methods in small- and medium-scale settings.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [CELLmap: Enhancing LiDAR SLAM through Elastic and Lightweight Spherical Map Representation](http://arxiv.org/abs/2409.19597)  
Yifan Duan, Xinran Zhang, Yao Li, Guoliang You, Xiaomeng Chu, Jianmin Ji, Yanyong Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    SLAM is a fundamental capability of unmanned systems, with LiDAR-based SLAM gaining widespread adoption due to its high precision. Current SLAM systems can achieve centimeter-level accuracy within a short period. However, there are still several challenges when dealing with largescale mapping tasks including significant storage requirements and difficulty of reusing the constructed maps. To address this, we first design an elastic and lightweight map representation called CELLmap, composed of several CELLs, each representing the local map at the corresponding location. Then, we design a general backend including CELL-based bidirectional registration module and loop closure detection module to improve global map consistency. Our experiments have demonstrated that CELLmap can represent the precise geometric structure of large-scale maps of KITTI dataset using only about 60 MB. Additionally, our general backend achieves up to a 26.88% improvement over various LiDAR odometry methods.  
  </ol>  
</details>  
**comments**: 7 pages, 5 figures  
  
### [VLAD-BuFF: Burst-aware Fast Feature Aggregation for Visual Place Recognition](http://arxiv.org/abs/2409.19293)  
[[code](https://github.com/ahmedest61/vlad-buff)]  
Ahmad Khaliq, Ming Xu, Stephen Hausler, Michael Milford, Sourav Garg  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Place Recognition (VPR) is a crucial component of many visual localization pipelines for embodied agents. VPR is often formulated as an image retrieval task aimed at jointly learning local features and an aggregation method. The current state-of-the-art VPR methods rely on VLAD aggregation, which can be trained to learn a weighted contribution of features through their soft assignment to cluster centers. However, this process has two key limitations. Firstly, the feature-to-cluster weighting does not account for over-represented repetitive structures within a cluster, e.g., shadows or window panes; this phenomenon is also referred to as the `burstiness' problem, classically solved by discounting repetitive features before aggregation. Secondly, feature to cluster comparisons are compute-intensive for state-of-the-art image encoders with high-dimensional local features. This paper addresses these limitations by introducing VLAD-BuFF with two novel contributions: i) a self-similarity based feature discounting mechanism to learn Burst-aware features within end-to-end VPR training, and ii) Fast Feature aggregation by reducing local feature dimensions specifically through PCA-initialized learnable pre-projection. We benchmark our method on 9 public datasets, where VLAD-BuFF sets a new state of the art. Our method is able to maintain its high recall even for 12x reduced local feature dimensions, thus enabling fast feature aggregation without compromising on recall. Through additional qualitative studies, we show how our proposed weighting method effectively downweights the non-distinctive features. Source code: https://github.com/Ahmedest61/VLAD-BuFF/.  
  </ol>  
</details>  
**comments**: Presented at ECCV 2024; Includes supplementary; 29 pages; 7 figures  
  
### [MASt3R-SfM: a Fully-Integrated Solution for Unconstrained Structure-from-Motion](http://arxiv.org/abs/2409.19152)  
Bardienus Duisterhof, Lojze Zust, Philippe Weinzaepfel, Vincent Leroy, Yohann Cabon, Jerome Revaud  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Structure-from-Motion (SfM), a task aiming at jointly recovering camera poses and 3D geometry of a scene given a set of images, remains a hard problem with still many open challenges despite decades of significant progress. The traditional solution for SfM consists of a complex pipeline of minimal solvers which tends to propagate errors and fails when images do not sufficiently overlap, have too little motion, etc. Recent methods have attempted to revisit this paradigm, but we empirically show that they fall short of fixing these core issues. In this paper, we propose instead to build upon a recently released foundation model for 3D vision that can robustly produce local 3D reconstructions and accurate matches. We introduce a low-memory approach to accurately align these local reconstructions in a global coordinate system. We further show that such foundation models can serve as efficient image retrievers without any overhead, reducing the overall complexity from quadratic to linear. Overall, our novel SfM pipeline is simple, scalable, fast and truly unconstrained, i.e. it can handle any collection of images, ordered or not. Extensive experiments on multiple benchmarks show that our method provides steady performance across diverse settings, especially outperforming existing methods in small- and medium-scale settings.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [OpenKD: Opening Prompt Diversity for Zero- and Few-shot Keypoint Detection](http://arxiv.org/abs/2409.19899)  
Changsheng Lu, Zheyuan Liu, Piotr Koniusz  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Exploiting the foundation models (e.g., CLIP) to build a versatile keypoint detector has gained increasing attention. Most existing models accept either the text prompt (e.g., ``the nose of a cat''), or the visual prompt (e.g., support image with keypoint annotations), to detect the corresponding keypoints in query image, thereby, exhibiting either zero-shot or few-shot detection ability. However, the research on taking multimodal prompt is still underexplored, and the prompt diversity in semantics and language is far from opened. For example, how to handle unseen text prompts for novel keypoint detection and the diverse text prompts like ``Can you detect the nose and ears of a cat?'' In this work, we open the prompt diversity from three aspects: modality, semantics (seen v.s. unseen), and language, to enable a more generalized zero- and few-shot keypoint detection (Z-FSKD). We propose a novel OpenKD model which leverages multimodal prototype set to support both visual and textual prompting. Further, to infer the keypoint location of unseen texts, we add the auxiliary keypoints and texts interpolated from visual and textual domains into training, which improves the spatial reasoning of our model and significantly enhances zero-shot novel keypoint detection. We also found large language model (LLM) is a good parser, which achieves over 96% accuracy to parse keypoints from texts. With LLM, OpenKD can handle diverse text prompts. Experimental results show that our method achieves state-of-the-art performance on Z-FSKD and initiates new ways to deal with unseen text and diverse texts. The source code and data are available at https://github.com/AlanLuSun/OpenKD.  
  </ol>  
</details>  
**comments**: Accepted by ECCV 2024  
  
  



## NeRF  

### [Distributed NeRF Learning for Collaborative Multi-Robot Perception](http://arxiv.org/abs/2409.20289)  
Hongrui Zhao, Boris Ivanovic, Negar Mehr  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Effective environment perception is crucial for enabling downstream robotic applications. Individual robotic agents often face occlusion and limited visibility issues, whereas multi-agent systems can offer a more comprehensive mapping of the environment, quicker coverage, and increased fault tolerance. In this paper, we propose a collaborative multi-agent perception system where agents collectively learn a neural radiance field (NeRF) from posed RGB images to represent a scene. Each agent processes its local sensory data and shares only its learned NeRF model with other agents, reducing communication overhead. Given NeRF's low memory footprint, this approach is well-suited for robotic systems with limited bandwidth, where transmitting all raw data is impractical. Our distributed learning framework ensures consistency across agents' local NeRF models, enabling convergence to a unified scene representation. We show the effectiveness of our method through an extensive set of experiments on datasets containing challenging real-world scenes, achieving performance comparable to centralized mapping of the environment where data is sent to a central server for processing. Additionally, we find that multi-agent learning provides regularization benefits, improving geometric consistency in scenarios with sparse input views. We show that in such scenarios, multi-agent mapping can even outperform centralized training.  
  </ol>  
</details>  
  
### [Active Neural Mapping at Scale](http://arxiv.org/abs/2409.20276)  
Zijia Kuang, Zike Yan, Hao Zhao, Guyue Zhou, Hongbin Zha  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce a NeRF-based active mapping system that enables efficient and robust exploration of large-scale indoor environments. The key to our approach is the extraction of a generalized Voronoi graph (GVG) from the continually updated neural map, leading to the synergistic integration of scene geometry, appearance, topology, and uncertainty. Anchoring uncertain areas induced by the neural map to the vertices of GVG allows the exploration to undergo adaptive granularity along a safe path that traverses unknown areas efficiently. Harnessing a modern hybrid NeRF representation, the proposed system achieves competitive results in terms of reconstruction accuracy, coverage completeness, and exploration efficiency even when scaling up to large indoor environments. Extensive results at different scales validate the efficacy of the proposed system.  
  </ol>  
</details>  
  
### [OPONeRF: One-Point-One NeRF for Robust Neural Rendering](http://arxiv.org/abs/2409.20043)  
Yu Zheng, Yueqi Duan, Kangfu Zheng, Hongru Yan, Jiwen Lu, Jie Zhou  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we propose a One-Point-One NeRF (OPONeRF) framework for robust scene rendering. Existing NeRFs are designed based on a key assumption that the target scene remains unchanged between the training and test time. However, small but unpredictable perturbations such as object movements, light changes and data contaminations broadly exist in real-life 3D scenes, which lead to significantly defective or failed rendering results even for the recent state-of-the-art generalizable methods. To address this, we propose a divide-and-conquer framework in OPONeRF that adaptively responds to local scene variations via personalizing appropriate point-wise parameters, instead of fitting a single set of NeRF parameters that are inactive to test-time unseen changes. Moreover, to explicitly capture the local uncertainty, we decompose the point representation into deterministic mapping and probabilistic inference. In this way, OPONeRF learns the sharable invariance and unsupervisedly models the unexpected scene variations between the training and testing scenes. To validate the effectiveness of the proposed method, we construct benchmarks from both realistic and synthetic data with diverse test-time perturbations including foreground motions, illumination variations and multi-modality noises, which are more challenging than conventional generalization and temporal reconstruction benchmarks. Experimental results show that our OPONeRF outperforms state-of-the-art NeRFs on various evaluation metrics through benchmark experiments and cross-scene evaluations. We further show the efficacy of the proposed method via experimenting on other existing generalization-based benchmarks and incorporating the idea of One-Point-One NeRF into other advanced baseline methods.  
  </ol>  
</details>  
  
### [G3R: Gradient Guided Generalizable Reconstruction](http://arxiv.org/abs/2409.19405)  
Yun Chen, Jingkang Wang, Ze Yang, Sivabalan Manivasagam, Raquel Urtasun  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Large scale 3D scene reconstruction is important for applications such as virtual reality and simulation. Existing neural rendering approaches (e.g., NeRF, 3DGS) have achieved realistic reconstructions on large scenes, but optimize per scene, which is expensive and slow, and exhibit noticeable artifacts under large view changes due to overfitting. Generalizable approaches or large reconstruction models are fast, but primarily work for small scenes/objects and often produce lower quality rendering results. In this work, we introduce G3R, a generalizable reconstruction approach that can efficiently predict high-quality 3D scene representations for large scenes. We propose to learn a reconstruction network that takes the gradient feedback signals from differentiable rendering to iteratively update a 3D scene representation, combining the benefits of high photorealism from per-scene optimization with data-driven priors from fast feed-forward prediction methods. Experiments on urban-driving and drone datasets show that G3R generalizes across diverse large scenes and accelerates the reconstruction process by at least 10x while achieving comparable or better realism compared to 3DGS, and also being more robust to large view changes.  
  </ol>  
</details>  
**comments**: ECCV 2024. Project page: https://waabi.ai/g3r  
  
  



