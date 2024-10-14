<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Voxel-SLAM:-A-Complete,-Accurate,-and-Versatile-LiDAR-Inertial-SLAM-System>Voxel-SLAM: A Complete, Accurate, and Versatile LiDAR-Inertial SLAM System</a></li>
        <li><a href=#Semantic-Token-Reweighting-for-Interpretable-and-Controllable-Text-Embeddings-in-CLIP>Semantic Token Reweighting for Interpretable and Controllable Text Embeddings in CLIP</a></li>
        <li><a href=#A-Unified-Deep-Semantic-Expansion-Framework-for-Domain-Generalized-Person-Re-identification>A Unified Deep Semantic Expansion Framework for Domain-Generalized Person Re-identification</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#SceneCraft:-Layout-Guided-3D-Scene-Generation>SceneCraft: Layout-Guided 3D Scene Generation</a></li>
        <li><a href=#MeshGS:-Adaptive-Mesh-Aligned-Gaussian-Splatting-for-High-Quality-Rendering>MeshGS: Adaptive Mesh-Aligned Gaussian Splatting for High-Quality Rendering</a></li>
        <li><a href=#Optimizing-NeRF-based-SLAM-with-Trajectory-Smoothness-Constraints>Optimizing NeRF-based SLAM with Trajectory Smoothness Constraints</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Voxel-SLAM: A Complete, Accurate, and Versatile LiDAR-Inertial SLAM System](http://arxiv.org/abs/2410.08935)  
Zheng Liu, Haotian Li, Chongjian Yuan, Xiyuan Liu, Jiarong Lin, Rundong Li, Chunran Zheng, Bingyang Zhou, Wenyi Liu, Fu Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this work, we present Voxel-SLAM: a complete, accurate, and versatile LiDAR-inertial SLAM system that fully utilizes short-term, mid-term, long-term, and multi-map data associations to achieve real-time estimation and high precision mapping. The system consists of five modules: initialization, odometry, local mapping, loop closure, and global mapping, all employing the same map representation, an adaptive voxel map. The initialization provides an accurate initial state estimation and a consistent local map for subsequent modules, enabling the system to start with a highly dynamic initial state. The odometry, exploiting the short-term data association, rapidly estimates current states and detects potential system divergence. The local mapping, exploiting the mid-term data association, employs a local LiDAR-inertial bundle adjustment (BA) to refine the states (and the local map) within a sliding window of recent LiDAR scans. The loop closure detects previously visited places in the current and all previous sessions. The global mapping refines the global map with an efficient hierarchical global BA. The loop closure and global mapping both exploit long-term and multi-map data associations. We conducted a comprehensive benchmark comparison with other state-of-the-art methods across 30 sequences from three representative scenes, including narrow indoor environments using hand-held equipment, large-scale wilderness environments with aerial robots, and urban environments on vehicle platforms. Other experiments demonstrate the robustness and efficiency of the initialization, the capacity to work in multiple sessions, and relocalization in degenerated environments.  
  </ol>  
</details>  
  
### [Semantic Token Reweighting for Interpretable and Controllable Text Embeddings in CLIP](http://arxiv.org/abs/2410.08469)  
Eunji Kim, Kyuhong Shim, Simyung Chang, Sungroh Yoon  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    A text encoder within Vision-Language Models (VLMs) like CLIP plays a crucial role in translating textual input into an embedding space shared with images, thereby facilitating the interpretative analysis of vision tasks through natural language. Despite the varying significance of different textual elements within a sentence depending on the context, efforts to account for variation of importance in constructing text embeddings have been lacking. We propose a framework of Semantic Token Reweighting to build Interpretable text embeddings (SToRI), which incorporates controllability as well. SToRI refines the text encoding process in CLIP by differentially weighting semantic elements based on contextual importance, enabling finer control over emphasis responsive to data-driven insights and user preferences. The efficacy of SToRI is demonstrated through comprehensive experiments on few-shot image classification and image retrieval tailored to user preferences.  
  </ol>  
</details>  
**comments**: Accepted at EMNLP 2024 Findings  
  
### [A Unified Deep Semantic Expansion Framework for Domain-Generalized Person Re-identification](http://arxiv.org/abs/2410.08456)  
Eugene P. W. Ang, Shan Lin, Alex C. Kot  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Supervised Person Re-identification (Person ReID) methods have achieved excellent performance when training and testing within one camera network. However, they usually suffer from considerable performance degradation when applied to different camera systems. In recent years, many Domain Adaptation Person ReID methods have been proposed, achieving impressive performance without requiring labeled data from the target domain. However, these approaches still need the unlabeled data of the target domain during the training process, making them impractical in many real-world scenarios. Our work focuses on the more practical Domain Generalized Person Re-identification (DG-ReID) problem. Given one or more source domains, it aims to learn a generalized model that can be applied to unseen target domains. One promising research direction in DG-ReID is the use of implicit deep semantic feature expansion, and our previous method, Domain Embedding Expansion (DEX), is one such example that achieves powerful results in DG-ReID. However, in this work we show that DEX and other similar implicit deep semantic feature expansion methods, due to limitations in their proposed loss function, fail to reach their full potential on large evaluation benchmarks as they have a tendency to saturate too early. Leveraging on this analysis, we propose Unified Deep Semantic Expansion, our novel framework that unifies implicit and explicit semantic feature expansion techniques in a single framework to mitigate this early over-fitting and achieve a new state-of-the-art (SOTA) in all DG-ReID benchmarks. Further, we apply our method on more general image retrieval tasks, also surpassing the current SOTA in all of these benchmarks by wide margins.  
  </ol>  
</details>  
**comments**: Neurocomputing Volume 600, 1 October 2024, 128120. 15 pages  
  
  



## NeRF  

### [SceneCraft: Layout-Guided 3D Scene Generation](http://arxiv.org/abs/2410.09049)  
[[code](https://github.com/orangesodahub/scenecraft)]  
Xiuyu Yang, Yunze Man, Jun-Kun Chen, Yu-Xiong Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The creation of complex 3D scenes tailored to user specifications has been a tedious and challenging task with traditional 3D modeling tools. Although some pioneering methods have achieved automatic text-to-3D generation, they are generally limited to small-scale scenes with restricted control over the shape and texture. We introduce SceneCraft, a novel method for generating detailed indoor scenes that adhere to textual descriptions and spatial layout preferences provided by users. Central to our method is a rendering-based technique, which converts 3D semantic layouts into multi-view 2D proxy maps. Furthermore, we design a semantic and depth conditioned diffusion model to generate multi-view images, which are used to learn a neural radiance field (NeRF) as the final scene representation. Without the constraints of panorama image generation, we surpass previous methods in supporting complicated indoor space generation beyond a single room, even as complicated as a whole multi-bedroom apartment with irregular shapes and layouts. Through experimental analysis, we demonstrate that our method significantly outperforms existing approaches in complex indoor scene generation with diverse textures, consistent geometry, and realistic visual quality. Code and more results are available at: https://orangesodahub.github.io/SceneCraft  
  </ol>  
</details>  
**comments**: NeurIPS 2024. Code: https://github.com/OrangeSodahub/SceneCraft
  Project Page: https://orangesodahub.github.io/SceneCraft  
  
### [MeshGS: Adaptive Mesh-Aligned Gaussian Splatting for High-Quality Rendering](http://arxiv.org/abs/2410.08941)  
Jaehoon Choi, Yonghan Lee, Hyungtae Lee, Heesung Kwon, Dinesh Manocha  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recently, 3D Gaussian splatting has gained attention for its capability to generate high-fidelity rendering results. At the same time, most applications such as games, animation, and AR/VR use mesh-based representations to represent and render 3D scenes. We propose a novel approach that integrates mesh representation with 3D Gaussian splats to perform high-quality rendering of reconstructed real-world scenes. In particular, we introduce a distance-based Gaussian splatting technique to align the Gaussian splats with the mesh surface and remove redundant Gaussian splats that do not contribute to the rendering. We consider the distance between each Gaussian splat and the mesh surface to distinguish between tightly-bound and loosely-bound Gaussian splats. The tightly-bound splats are flattened and aligned well with the mesh geometry. The loosely-bound Gaussian splats are used to account for the artifacts in reconstructed 3D meshes in terms of rendering. We present a training strategy of binding Gaussian splats to the mesh geometry, and take into account both types of splats. In this context, we introduce several regularization techniques aimed at precisely aligning tightly-bound Gaussian splats with the mesh surface during the training process. We validate the effectiveness of our method on large and unbounded scene from mip-NeRF 360 and Deep Blending datasets. Our method surpasses recent mesh-based neural rendering techniques by achieving a 2dB higher PSNR, and outperforms mesh-based Gaussian splatting methods by 1.3 dB PSNR, particularly on the outdoor mip-NeRF 360 dataset, demonstrating better rendering quality. We provide analyses for each type of Gaussian splat and achieve a reduction in the number of Gaussian splats by 30% compared to the original 3D Gaussian splatting.  
  </ol>  
</details>  
**comments**: ACCV (Asian Conference on Computer Vision) 2024  
  
### [Optimizing NeRF-based SLAM with Trajectory Smoothness Constraints](http://arxiv.org/abs/2410.08780)  
Yicheng He, Guangcheng Chen, Hong Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The joint optimization of Neural Radiance Fields (NeRF) and camera trajectories has been widely applied in SLAM tasks due to its superior dense mapping quality and consistency. NeRF-based SLAM learns camera poses using constraints by implicit map representation. A widely observed phenomenon that results from the constraints of this form is jerky and physically unrealistic estimated camera motion, which in turn affects the map quality. To address this deficiency of current NeRF-based SLAM, we propose in this paper TS-SLAM (TS for Trajectory Smoothness). It introduces smoothness constraints on camera trajectories by representing them with uniform cubic B-splines with continuous acceleration that guarantees smooth camera motion. Benefiting from the differentiability and local control properties of B-splines, TS-SLAM can incrementally learn the control points end-to-end using a sliding window paradigm. Additionally, we regularize camera trajectories by exploiting the dynamics prior to further smooth trajectories. Experimental results demonstrate that TS-SLAM achieves superior trajectory accuracy and improves mapping quality versus NeRF-based SLAM that does not employ the above smoothness constraints.  
  </ol>  
</details>  
  
  



