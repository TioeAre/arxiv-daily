<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#cuVSLAM:-CUDA-accelerated-visual-odometry>cuVSLAM: CUDA accelerated visual odometry</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#SupeRANSAC:-One-RANSAC-to-Rule-Them-All>SupeRANSAC: One RANSAC to Rule Them All</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#HypeVPR:-Exploring-Hyperbolic-Space-for-Perspective-to-Equirectangular-Visual-Place-Recognition>HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition</a></li>
        <li><a href=#Deep-Learning-Reforms-Image-Matching:-A-Survey-and-Outlook>Deep Learning Reforms Image Matching: A Survey and Outlook</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Vanishing-arcs-for-isolated-plane-curve-singularities>Vanishing arcs for isolated plane curve singularities</a></li>
        <li><a href=#Deep-Learning-Reforms-Image-Matching:-A-Survey-and-Outlook>Deep Learning Reforms Image Matching: A Survey and Outlook</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#ProJo4D:-Progressive-Joint-Optimization-for-Sparse-View-Inverse-Physics-Estimation>ProJo4D: Progressive Joint Optimization for Sparse-View Inverse Physics Estimation</a></li>
        <li><a href=#Unifying-Appearance-Codes-and-Bilateral-Grids-for-Driving-Scene-Gaussian-Splatting>Unifying Appearance Codes and Bilateral Grids for Driving Scene Gaussian Splatting</a></li>
        <li><a href=#Generating-Synthetic-Stereo-Datasets-using-3D-Gaussian-Splatting-and-Expert-Knowledge-Transfer>Generating Synthetic Stereo Datasets using 3D Gaussian Splatting and Expert Knowledge Transfer</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [cuVSLAM: CUDA accelerated visual odometry](http://arxiv.org/abs/2506.04359)  
Alexander Korovko, Dmitry Slepichev, Alexander Efitorov, Aigul Dzhumamuratova, Viktor Kuznetsov, Hesam Rabeti, Joydeep Biswas  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Accurate and robust pose estimation is a key requirement for any autonomous robot. We present cuVSLAM, a state-of-the-art solution for visual simultaneous localization and mapping, which can operate with a variety of visual-inertial sensor suites, including multiple RGB and depth cameras, and inertial measurement units. cuVSLAM supports operation with as few as one RGB camera to as many as 32 cameras, in arbitrary geometric configurations, thus supporting a wide range of robotic setups. cuVSLAM is specifically optimized using CUDA to deploy in real-time applications with minimal computational overhead on edge-computing devices such as the NVIDIA Jetson. We present the design and implementation of cuVSLAM, example use cases, and empirical results on several state-of-the-art benchmarks demonstrating the best-in-class performance of cuVSLAM.  
  </ol>  
</details>  
  
  



## SFM  

### [SupeRANSAC: One RANSAC to Rule Them All](http://arxiv.org/abs/2506.04803)  
Daniel Barath  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Robust estimation is a cornerstone in computer vision, particularly for tasks like Structure-from-Motion and Simultaneous Localization and Mapping. RANSAC and its variants are the gold standard for estimating geometric models (e.g., homographies, relative/absolute poses) from outlier-contaminated data. Despite RANSAC's apparent simplicity, achieving consistently high performance across different problems is challenging. While recent research often focuses on improving specific RANSAC components (e.g., sampling, scoring), overall performance is frequently more influenced by the "bells and whistles" (i.e., the implementation details and problem-specific optimizations) within a given library. Popular frameworks like OpenCV and PoseLib demonstrate varying performance, excelling in some tasks but lagging in others. We introduce SupeRANSAC, a novel unified RANSAC pipeline, and provide a detailed analysis of the techniques that make RANSAC effective for specific vision tasks, including homography, fundamental/essential matrix, and absolute/rigid pose estimation. SupeRANSAC is designed for consistent accuracy across these tasks, improving upon the best existing methods by, for example, 6 AUC points on average for fundamental matrix estimation. We demonstrate significant performance improvements over the state-of-the-art on multiple problems and datasets. Code: https://github.com/danini/superansac  
  </ol>  
</details>  
  
  



## Visual Localization  

### [HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition](http://arxiv.org/abs/2506.04764)  
Suhan Woo, Seongwon Lee, Jinwoo Jang, Euntai Kim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    When applying Visual Place Recognition (VPR) to real-world mobile robots and similar applications, perspective-to-equirectangular (P2E) formulation naturally emerges as a suitable approach to accommodate diverse query images captured from various viewpoints. In this paper, we introduce HypeVPR, a novel hierarchical embedding framework in hyperbolic space, designed to address the unique challenges of P2E VPR. The key idea behind HypeVPR is that visual environments captured by panoramic views exhibit inherent hierarchical structures. To leverage this property, we employ hyperbolic space to represent hierarchical feature relationships and preserve distance properties within the feature space. To achieve this, we propose a hierarchical feature aggregation mechanism that organizes local-to-global feature representations within hyperbolic space. Additionally, HypeVPR adopts an efficient coarse-to-fine search strategy, optimally balancing speed and accuracy to ensure robust matching, even between descriptors from different image types. This approach enables HypeVPR to outperform state-of-the-art methods while significantly reducing retrieval time, achieving up to 5x faster retrieval across diverse benchmark datasets. The code and models will be released at https://github.com/suhan-woo/HypeVPR.git.  
  </ol>  
</details>  
  
### [Deep Learning Reforms Image Matching: A Survey and Outlook](http://arxiv.org/abs/2506.04619)  
Shihua Zhang, Zizhuo Li, Kaining Zhang, Yifan Lu, Yuxin Deng, Linfeng Tang, Xingyu Jiang, Jiayi Ma  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image matching, which establishes correspondences between two-view images to recover 3D structure and camera geometry, serves as a cornerstone in computer vision and underpins a wide range of applications, including visual localization, 3D reconstruction, and simultaneous localization and mapping (SLAM). Traditional pipelines composed of ``detector-descriptor, feature matcher, outlier filter, and geometric estimator'' falter in challenging scenarios. Recent deep-learning advances have significantly boosted both robustness and accuracy. This survey adopts a unique perspective by comprehensively reviewing how deep learning has incrementally transformed the classical image matching pipeline. Our taxonomy highly aligns with the traditional pipeline in two key aspects: i) the replacement of individual steps in the traditional pipeline with learnable alternatives, including learnable detector-descriptor, outlier filter, and geometric estimator; and ii) the merging of multiple steps into end-to-end learnable modules, encompassing middle-end sparse matcher, end-to-end semi-dense/dense matcher, and pose regressor. We first examine the design principles, advantages, and limitations of both aspects, and then benchmark representative methods on relative pose recovery, homography estimation, and visual localization tasks. Finally, we discuss open challenges and outline promising directions for future research. By systematically categorizing and evaluating deep learning-driven strategies, this survey offers a clear overview of the evolving image matching landscape and highlights key avenues for further innovation.  
  </ol>  
</details>  
  
  



## Image Matching  

### [Vanishing arcs for isolated plane curve singularities](http://arxiv.org/abs/2506.04917)  
Hanwool Bae, Cheol-Hyun Cho, Dongwook Choa, Wonbo Jeong, Pablo Portilla Cuadrado  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The variation operator associated with an isolated hypersurface singularity is a classical topological invariant that relates relative and absolute homologies of the Milnor fiber via a non trivial isomorphism. Here we work with a topological version of this operator that deals with proper arcs and closed curves instead of homology cycles. Building on the classical framework of geometric vanishing cycles, we introduce the concept of vanishing arcsets as their counterpart using this geometric variation operator. We characterize which properly embedded arcs are sent to geometric vanishing cycles by the geometric variation operator in terms of intersections numbers of the arcs and their images by the geometric monodromy. Furthermore, we prove that for any distinguished collection of vanishing cycles arising from an A'Campo's divide, there exists a topological exceptional collection of arcsets whose variation images match this collection.  
  </ol>  
</details>  
**comments**: 42 pages  
  
### [Deep Learning Reforms Image Matching: A Survey and Outlook](http://arxiv.org/abs/2506.04619)  
Shihua Zhang, Zizhuo Li, Kaining Zhang, Yifan Lu, Yuxin Deng, Linfeng Tang, Xingyu Jiang, Jiayi Ma  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image matching, which establishes correspondences between two-view images to recover 3D structure and camera geometry, serves as a cornerstone in computer vision and underpins a wide range of applications, including visual localization, 3D reconstruction, and simultaneous localization and mapping (SLAM). Traditional pipelines composed of ``detector-descriptor, feature matcher, outlier filter, and geometric estimator'' falter in challenging scenarios. Recent deep-learning advances have significantly boosted both robustness and accuracy. This survey adopts a unique perspective by comprehensively reviewing how deep learning has incrementally transformed the classical image matching pipeline. Our taxonomy highly aligns with the traditional pipeline in two key aspects: i) the replacement of individual steps in the traditional pipeline with learnable alternatives, including learnable detector-descriptor, outlier filter, and geometric estimator; and ii) the merging of multiple steps into end-to-end learnable modules, encompassing middle-end sparse matcher, end-to-end semi-dense/dense matcher, and pose regressor. We first examine the design principles, advantages, and limitations of both aspects, and then benchmark representative methods on relative pose recovery, homography estimation, and visual localization tasks. Finally, we discuss open challenges and outline promising directions for future research. By systematically categorizing and evaluating deep learning-driven strategies, this survey offers a clear overview of the evolving image matching landscape and highlights key avenues for further innovation.  
  </ol>  
</details>  
  
  



## NeRF  

### [ProJo4D: Progressive Joint Optimization for Sparse-View Inverse Physics Estimation](http://arxiv.org/abs/2506.05317)  
Daniel Rho, Jun Myeong Choi, Biswadip Dey, Roni Sengupta  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural rendering has made significant strides in 3D reconstruction and novel view synthesis. With the integration with physics, it opens up new applications. The inverse problem of estimating physics from visual data, however, still remains challenging, limiting its effectiveness for applications like physically accurate digital twin creation in robotics and XR. Existing methods that incorporate physics into neural rendering frameworks typically require dense multi-view videos as input, making them impractical for scalable, real-world use. When presented with sparse multi-view videos, the sequential optimization strategy used by existing approaches introduces significant error accumulation, e.g., poor initial 3D reconstruction leads to bad material parameter estimation in subsequent stages. Instead of sequential optimization, directly optimizing all parameters at the same time also fails due to the highly non-convex and often non-differentiable nature of the problem. We propose ProJo4D, a progressive joint optimization framework that gradually increases the set of jointly optimized parameters guided by their sensitivity, leading to fully joint optimization over geometry, appearance, physical state, and material property. Evaluations on PAC-NeRF and Spring-Gaus datasets show that ProJo4D outperforms prior work in 4D future state prediction, novel view rendering of future state, and material parameter estimation, demonstrating its effectiveness in physically grounded 4D scene understanding. For demos, please visit the project webpage: https://daniel03c1.github.io/ProJo4D/  
  </ol>  
</details>  
  
### [Unifying Appearance Codes and Bilateral Grids for Driving Scene Gaussian Splatting](http://arxiv.org/abs/2506.05280)  
Nan Wang, Yuantao Chen, Lixing Xiao, Weiqing Xiao, Bohan Li, Zhaoxi Chen, Chongjie Ye, Shaocong Xu, Saining Zhang, Ziyang Yan, Pierre Merriaux, Lei Lei, Tianfan Xue, Hao Zhao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural rendering techniques, including NeRF and Gaussian Splatting (GS), rely on photometric consistency to produce high-quality reconstructions. However, in real-world scenarios, it is challenging to guarantee perfect photometric consistency in acquired images. Appearance codes have been widely used to address this issue, but their modeling capability is limited, as a single code is applied to the entire image. Recently, the bilateral grid was introduced to perform pixel-wise color mapping, but it is difficult to optimize and constrain effectively. In this paper, we propose a novel multi-scale bilateral grid that unifies appearance codes and bilateral grids. We demonstrate that this approach significantly improves geometric accuracy in dynamic, decoupled autonomous driving scene reconstruction, outperforming both appearance codes and bilateral grids. This is crucial for autonomous driving, where accurate geometry is important for obstacle avoidance and control. Our method shows strong results across four datasets: Waymo, NuScenes, Argoverse, and PandaSet. We further demonstrate that the improvement in geometry is driven by the multi-scale bilateral grid, which effectively reduces floaters caused by photometric inconsistency.  
  </ol>  
</details>  
**comments**: Project page: https://bigcileng.github.io/bilateral-driving; Code:
  https://github.com/BigCiLeng/bilateral-driving  
  
### [Generating Synthetic Stereo Datasets using 3D Gaussian Splatting and Expert Knowledge Transfer](http://arxiv.org/abs/2506.04908)  
Filip Slezak, Magnus K. Gjerde, Joakim B. Haurum, Ivan Nikolov, Morten S. Laursen, Thomas B. Moeslund  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we introduce a 3D Gaussian Splatting (3DGS)-based pipeline for stereo dataset generation, offering an efficient alternative to Neural Radiance Fields (NeRF)-based methods. To obtain useful geometry estimates, we explore utilizing the reconstructed geometry from the explicit 3D representations as well as depth estimates from the FoundationStereo model in an expert knowledge transfer setup. We find that when fine-tuning stereo models on 3DGS-generated datasets, we demonstrate competitive performance in zero-shot generalization benchmarks. When using the reconstructed geometry directly, we observe that it is often noisy and contains artifacts, which propagate noise to the trained model. In contrast, we find that the disparity estimates from FoundationStereo are cleaner and consequently result in a better performance on the zero-shot generalization benchmarks. Our method highlights the potential for low-cost, high-fidelity dataset creation and fast fine-tuning for deep stereo models. Moreover, we also reveal that while the latest Gaussian Splatting based methods have achieved superior performance on established benchmarks, their robustness falls short in challenging in-the-wild settings warranting further exploration.  
  </ol>  
</details>  
  
  



