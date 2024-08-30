<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Creating-a-Segmented-Pointcloud-of-Grapevines-by-Combining-Multiple-Viewpoints-Through-Visual-Odometry>Creating a Segmented Pointcloud of Grapevines by Combining Multiple Viewpoints Through Visual Odometry</a></li>
        <li><a href=#Single-Photon-3D-Imaging-with-Equi-Depth-Photon-Histograms>Single-Photon 3D Imaging with Equi-Depth Photon Histograms</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Mismatched:-Evaluating-the-Limits-of-Image-Matching-Approaches-and-Benchmarks>Mismatched: Evaluating the Limits of Image Matching Approaches and Benchmarks</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#A-compact-neuromorphic-system-for-ultra-energy-efficient,-on-device-robot-localization>A compact neuromorphic system for ultra energy-efficient, on-device robot localization</a></li>
        <li><a href=#Rethinking-Sparse-Lexical-Representations-for-Image-Retrieval-in-the-Age-of-Rising-Multi-Modal-Large-Language-Models>Rethinking Sparse Lexical Representations for Image Retrieval in the Age of Rising Multi-Modal Large Language Models</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Super-Resolution-works-for-coastal-simulations>Super-Resolution works for coastal simulations</a></li>
        <li><a href=#Mismatched:-Evaluating-the-Limits-of-Image-Matching-Approaches-and-Benchmarks>Mismatched: Evaluating the Limits of Image Matching Approaches and Benchmarks</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Generic-Objects-as-Pose-Probes-for-Few-Shot-View-Synthesis>Generic Objects as Pose Probes for Few-Shot View Synthesis</a></li>
        <li><a href=#Spurfies:-Sparse-Surface-Reconstruction-using-Local-Geometry-Priors>Spurfies: Sparse Surface Reconstruction using Local Geometry Priors</a></li>
        <li><a href=#NeRF-CA:-Dynamic-Reconstruction-of-X-ray-Coronary-Angiography-with-Extremely-Sparse-views>NeRF-CA: Dynamic Reconstruction of X-ray Coronary Angiography with Extremely Sparse-views</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Creating a Segmented Pointcloud of Grapevines by Combining Multiple Viewpoints Through Visual Odometry](http://arxiv.org/abs/2408.16472)  
Michael Adlerstein, Angelo Bratta, João Carlos Virgolino Soares, Giovanni Dessy, Miguel Fernandes, Matteo Gatti, Claudio Semini  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Grapevine winter pruning is a labor-intensive and repetitive process that significantly influences the quality and quantity of the grape harvest and produced wine of the following season. It requires a careful and expert detection of the point to be cut. Because of its complexity, repetitive nature and time constraint, the task requires skilled labor that needs to be trained. This extended abstract presents the computer vision pipeline employed in project Vinum, using detectron2 as a segmentation network and keypoint visual odometry to merge different observation into a single pointcloud used to make informed pruning decisions.  
  </ol>  
</details>  
  
### [Single-Photon 3D Imaging with Equi-Depth Photon Histograms](http://arxiv.org/abs/2408.16150)  
Kaustubh Sadekar, David Maier, Atul Ingle  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Single-photon cameras present a promising avenue for high-resolution 3D imaging. They have ultra-high sensitivity -- down to individual photons -- and can record photon arrival times with extremely high (sub-nanosecond) resolution. Single-photon 3D cameras estimate the round-trip time of a laser pulse by forming equi-width (EW) histograms of detected photon timestamps. Acquiring and transferring such EW histograms requires high bandwidth and in-pixel memory, making SPCs less attractive in resource-constrained settings such as mobile devices and AR/VR headsets. In this work we propose a 3D sensing technique based on equi-depth (ED) histograms. ED histograms compress timestamp data more efficiently than EW histograms, reducing the bandwidth requirement. Moreover, to reduce the in-pixel memory requirement, we propose a lightweight algorithm to estimate ED histograms in an online fashion without explicitly storing the photon timestamps. This algorithm is amenable to future in-pixel implementations. We propose algorithms that process ED histograms to perform 3D computer-vision tasks of estimating scene distance maps and performing visual odometry under challenging conditions such as high ambient light. Our work paves the way towards lower bandwidth and reduced in-pixel memory requirements for SPCs, making them attractive for resource-constrained 3D vision applications. Project page: $\href{https://www.computational.camera/pedh}{https://www.computational.camera/pedh}$  
  </ol>  
</details>  
  
  



## SFM  

### [Mismatched: Evaluating the Limits of Image Matching Approaches and Benchmarks](http://arxiv.org/abs/2408.16445)  
[[code](https://github.com/surgical-vision/colmap-match-converter)]  
Sierra Bonilla, Chiara Di Vece, Rema Daher, Xinwei Ju, Danail Stoyanov, Francisco Vasconcelos, Sophia Bano  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Three-dimensional (3D) reconstruction from two-dimensional images is an active research field in computer vision, with applications ranging from navigation and object tracking to segmentation and three-dimensional modeling. Traditionally, parametric techniques have been employed for this task. However, recent advancements have seen a shift towards learning-based methods. Given the rapid pace of research and the frequent introduction of new image matching methods, it is essential to evaluate them. In this paper, we present a comprehensive evaluation of various image matching methods using a structure-from-motion pipeline. We assess the performance of these methods on both in-domain and out-of-domain datasets, identifying key limitations in both the methods and benchmarks. We also investigate the impact of edge detection as a pre-processing step. Our analysis reveals that image matching for 3D reconstruction remains an open challenge, necessitating careful selection and tuning of models for specific scenarios, while also highlighting mismatches in how metrics currently represent method performance.  
  </ol>  
</details>  
**comments**: 19 pages, 5 figures  
  
  



## Visual Localization  

### [A compact neuromorphic system for ultra energy-efficient, on-device robot localization](http://arxiv.org/abs/2408.16754)  
[[code](https://github.com/AdamDHines/LENS)]  
Adam D. Hines, Michael Milford, Tobias Fischer  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neuromorphic computing offers a transformative pathway to overcome the computational and energy challenges faced in deploying robotic localization and navigation systems at the edge. Visual place recognition, a critical component for navigation, is often hampered by the high resource demands of conventional systems, making them unsuitable for small-scale robotic platforms which still require to perform complex, long-range tasks. Although neuromorphic approaches offer potential for greater efficiency, real-time edge deployment remains constrained by the complexity and limited scalability of bio-realistic networks. Here, we demonstrate a neuromorphic localization system that performs accurate place recognition in up to 8km of traversal using models as small as 180 KB with 44k parameters, while consuming less than 1% of the energy required by conventional methods. Our Locational Encoding with Neuromorphic Systems (LENS) integrates spiking neural networks, an event-based dynamic vision sensor, and a neuromorphic processor within a single SPECK(TM) chip, enabling real-time, energy-efficient localization on a hexapod robot. LENS represents the first fully neuromorphic localization system capable of large-scale, on-device deployment, setting a new benchmark for energy efficient robotic place recognition.  
  </ol>  
</details>  
**comments**: 28 pages, 4 main figures, 4 supplementary figures, 1 supplementary
  table, and 1 movie. Under review  
  
### [Rethinking Sparse Lexical Representations for Image Retrieval in the Age of Rising Multi-Modal Large Language Models](http://arxiv.org/abs/2408.16296)  
Kengo Nakata, Daisuke Miyashita, Youyang Ng, Yasuto Hoshi, Jun Deguchi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we rethink sparse lexical representations for image retrieval. By utilizing multi-modal large language models (M-LLMs) that support visual prompting, we can extract image features and convert them into textual data, enabling us to utilize efficient sparse retrieval algorithms employed in natural language processing for image retrieval tasks. To assist the LLM in extracting image features, we apply data augmentation techniques for key expansion and analyze the impact with a metric for relevance between images and textual data. We empirically show the superior precision and recall performance of our image retrieval method compared to conventional vision-language model-based methods on the MS-COCO, PASCAL VOC, and NUS-WIDE datasets in a keyword-based image retrieval scenario, where keywords serve as search queries. We also demonstrate that the retrieval performance can be improved by iteratively incorporating keywords into search queries.  
  </ol>  
</details>  
**comments**: Accepted to ECCV 2024 Workshops: 2nd Workshop on Traditional Computer
  Vision in the Age of Deep Learning (TradiCV)  
  
  



## Image Matching  

### [Super-Resolution works for coastal simulations](http://arxiv.org/abs/2408.16553)  
Zhi-Song Liu, Markus Buttner, Vadym Aizinger, Andreas Rupp  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Learning fine-scale details of a coastal ocean simulation from a coarse representation is a challenging task. For real-world applications, high-resolution simulations are necessary to advance understanding of many coastal processes, specifically, to predict flooding resulting from tsunamis and storm surges. We propose a Deep Network for Coastal Super-Resolution (DNCSR) for spatiotemporal enhancement to efficiently learn the high-resolution numerical solution. Given images of coastal simulations produced on low-resolution computational meshes using low polynomial order discontinuous Galerkin discretizations and a coarse temporal resolution, the proposed DNCSR learns to produce high-resolution free surface elevation and velocity visualizations in both time and space. To efficiently model the dynamic changes over time and space, we propose grid-aware spatiotemporal attention to project the temporal features to the spatial domain for non-local feature matching. The coordinate information is also utilized via positional encoding. For the final reconstruction, we use the spatiotemporal bilinear operation to interpolate the missing frames and then expand the feature maps to the frequency domain for residual mapping. Besides data-driven losses, the proposed physics-informed loss guarantees gradient consistency and momentum changes. Their combination contributes to the overall 24% improvements in RMSE. To train the proposed model, we propose a large-scale coastal simulation dataset and use it for model optimization and evaluation. Our method shows superior super-resolution quality and fast computation compared to the state-of-the-art methods.  
  </ol>  
</details>  
**comments**: 13 pages, 12 figures  
  
### [Mismatched: Evaluating the Limits of Image Matching Approaches and Benchmarks](http://arxiv.org/abs/2408.16445)  
[[code](https://github.com/surgical-vision/colmap-match-converter)]  
Sierra Bonilla, Chiara Di Vece, Rema Daher, Xinwei Ju, Danail Stoyanov, Francisco Vasconcelos, Sophia Bano  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Three-dimensional (3D) reconstruction from two-dimensional images is an active research field in computer vision, with applications ranging from navigation and object tracking to segmentation and three-dimensional modeling. Traditionally, parametric techniques have been employed for this task. However, recent advancements have seen a shift towards learning-based methods. Given the rapid pace of research and the frequent introduction of new image matching methods, it is essential to evaluate them. In this paper, we present a comprehensive evaluation of various image matching methods using a structure-from-motion pipeline. We assess the performance of these methods on both in-domain and out-of-domain datasets, identifying key limitations in both the methods and benchmarks. We also investigate the impact of edge detection as a pre-processing step. Our analysis reveals that image matching for 3D reconstruction remains an open challenge, necessitating careful selection and tuning of models for specific scenarios, while also highlighting mismatches in how metrics currently represent method performance.  
  </ol>  
</details>  
**comments**: 19 pages, 5 figures  
  
  



## NeRF  

### [Generic Objects as Pose Probes for Few-Shot View Synthesis](http://arxiv.org/abs/2408.16690)  
Zhirui Gao, Renjiao Yi, Chenyang Zhu, Ke Zhuang, Wei Chen, Kai Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Radiance fields including NeRFs and 3D Gaussians demonstrate great potential in high-fidelity rendering and scene reconstruction, while they require a substantial number of posed images as inputs. COLMAP is frequently employed for preprocessing to estimate poses, while it necessitates a large number of feature matches to operate effectively, and it struggles with scenes characterized by sparse features, large baselines between images, or a limited number of input images. We aim to tackle few-view NeRF reconstruction using only 3 to 6 unposed scene images. Traditional methods often use calibration boards but they are not common in images. We propose a novel idea of utilizing everyday objects, commonly found in both images and real life, as "pose probes". The probe object is automatically segmented by SAM, whose shape is initialized from a cube. We apply a dual-branch volume rendering optimization (object NeRF and scene NeRF) to constrain the pose optimization and jointly refine the geometry. Specifically, object poses of two views are first estimated by PnP matching in an SDF representation, which serves as initial poses. PnP matching, requiring only a few features, is suitable for feature-sparse scenes. Additional views are incrementally incorporated to refine poses from preceding views. In experiments, PoseProbe achieves state-of-the-art performance in both pose estimation and novel view synthesis across multiple datasets. We demonstrate its effectiveness, particularly in few-view and large-baseline scenes where COLMAP struggles. In ablations, using different objects in a scene yields comparable performance.  
  </ol>  
</details>  
  
### [Spurfies: Sparse Surface Reconstruction using Local Geometry Priors](http://arxiv.org/abs/2408.16544)  
Kevin Raj, Christopher Wewer, Raza Yunus, Eddy Ilg, Jan Eric Lenssen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce Spurfies, a novel method for sparse-view surface reconstruction that disentangles appearance and geometry information to utilize local geometry priors trained on synthetic data. Recent research heavily focuses on 3D reconstruction using dense multi-view setups, typically requiring hundreds of images. However, these methods often struggle with few-view scenarios. Existing sparse-view reconstruction techniques often rely on multi-view stereo networks that need to learn joint priors for geometry and appearance from a large amount of data. In contrast, we introduce a neural point representation that disentangles geometry and appearance to train a local geometry prior using a subset of the synthetic ShapeNet dataset only. During inference, we utilize this surface prior as additional constraint for surface and appearance reconstruction from sparse input views via differentiable volume rendering, restricting the space of possible solutions. We validate the effectiveness of our method on the DTU dataset and demonstrate that it outperforms previous state of the art by 35% in surface quality while achieving competitive novel view synthesis quality. Moreover, in contrast to previous works, our method can be applied to larger, unbounded scenes, such as Mip-NeRF 360.  
  </ol>  
</details>  
**comments**: https://geometric-rl.mpi-inf.mpg.de/spurfies/  
  
### [NeRF-CA: Dynamic Reconstruction of X-ray Coronary Angiography with Extremely Sparse-views](http://arxiv.org/abs/2408.16355)  
[[code](https://github.com/kirstenmaas/nerf-ca)]  
Kirsten W. H. Maas, Danny Ruijters, Anna Vilanova, Nicola Pezzotti  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Dynamic three-dimensional (4D) reconstruction from two-dimensional X-ray coronary angiography (CA) remains a significant clinical problem. Challenges include sparse-view settings, intra-scan motion, and complex vessel morphology such as structure sparsity and background occlusion. Existing CA reconstruction methods often require extensive user interaction or large training datasets. On the other hand, Neural Radiance Field (NeRF), a promising deep learning technique, has successfully reconstructed high-fidelity static scenes for natural and medical scenes. Recent work, however, identified that sparse-views, background occlusion, and dynamics still pose a challenge when applying NeRF in the X-ray angiography context. Meanwhile, many successful works for natural scenes propose regularization for sparse-view reconstruction or scene decomposition to handle dynamics. However, these techniques do not directly translate to the CA context, where both challenges and background occlusion are significant. This paper introduces NeRF-CA, the first step toward a 4D CA reconstruction method that achieves reconstructions from sparse coronary angiograms with cardiac motion. We leverage the motion of the coronary artery to decouple the scene into a dynamic coronary artery component and static background. We combine this scene decomposition with tailored regularization techniques. These techniques enforce the separation of the coronary artery from the background by enforcing dynamic structure sparsity and scene smoothness. By uniquely combining these approaches, we achieve 4D reconstructions from as few as four angiogram sequences. This setting aligns with clinical workflows while outperforming state-of-the-art X-ray sparse-view NeRF reconstruction techniques. We validate our approach quantitatively and qualitatively using 4D phantom datasets and ablation studies.  
  </ol>  
</details>  
  
  



