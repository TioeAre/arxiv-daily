<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Preserving-Relative-Localization-of-FoV-Limited-Drone-Swarm-via-Active-Mutual-Observation>Preserving Relative Localization of FoV-Limited Drone Swarm via Active Mutual Observation</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Indoor-3D-Reconstruction-with-an-Unknown-Camera-Projector-Pair>Indoor 3D Reconstruction with an Unknown Camera-Projector Pair</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Close,-But-Not-There:-Boosting-Geographic-Distance-Sensitivity-in-Visual-Place-Recognition>Close, But Not There: Boosting Geographic Distance Sensitivity in Visual Place Recognition</a></li>
        <li><a href=#Freeview-Sketching:-View-Aware-Fine-Grained-Sketch-Based-Image-Retrieval>Freeview Sketching: View-Aware Fine-Grained Sketch-Based Image Retrieval</a></li>
        <li><a href=#Cross-Modal-Attention-Alignment-Network-with-Auxiliary-Text-Description-for-zero-shot-sketch-based-image-retrieval>Cross-Modal Attention Alignment Network with Auxiliary Text Description for zero-shot sketch-based image retrieval</a></li>
        <li><a href=#Dynamically-Modulating-Visual-Place-Recognition-Sequence-Length-For-Minimum-Acceptable-Performance-Scenarios>Dynamically Modulating Visual Place Recognition Sequence Length For Minimum Acceptable Performance Scenarios</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Multi-Grained-Contrast-for-Data-Efficient-Unsupervised-Representation-Learning>Multi-Grained Contrast for Data-Efficient Unsupervised Representation Learning</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#BeNeRF:-Neural-Radiance-Fields-from-a-Single-Blurry-Image-and-Event-Stream>BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream</a></li>
        <li><a href=#Active-Human-Pose-Estimation-via-an-Autonomous-UAV-Agent>Active Human Pose Estimation via an Autonomous UAV Agent</a></li>
        <li><a href=#DRAGON:-Drone-and-Ground-Gaussian-Splatting-for-3D-Building-Reconstruction>DRAGON: Drone and Ground Gaussian Splatting for 3D Building Reconstruction</a></li>
        <li><a href=#Fast-and-Efficient:-Mask-Neural-Fields-for-3D-Scene-Segmentation>Fast and Efficient: Mask Neural Fields for 3D Scene Segmentation</a></li>
        <li><a href=#Intrinsic-PAPR-for-Point-level-3D-Scene-Albedo-and-Shading-Editing>Intrinsic PAPR for Point-level 3D Scene Albedo and Shading Editing</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Preserving Relative Localization of FoV-Limited Drone Swarm via Active Mutual Observation](http://arxiv.org/abs/2407.01292)  
Lianjie Guo, Zaitian Gongye, Ziyi Xu, Yingjian Wang, Xin Zhou, Jinni Zhou, Fei Gao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Relative state estimation is crucial for vision-based swarms to estimate and compensate for the unavoidable drift of visual odometry. For autonomous drones equipped with the most compact sensor setting -- a stereo camera that provides a limited field of view (FoV), the demand for mutual observation for relative state estimation conflicts with the demand for environment observation. To balance the two demands for FoV limited swarms by acquiring mutual observations with a safety guarantee, this paper proposes an active localization correction system, which plans camera orientations via a yaw planner during the flight. The yaw planner manages the contradiction by calculating suitable timing and yaw angle commands based on the evaluation of localization uncertainty estimated by the Kalman Filter. Simulation validates the scalability of our algorithm. In real-world experiments, we reduce positioning drift by up to 65% and managed to maintain a given formation in both indoor and outdoor GPS-denied flight, from which the accuracy, efficiency, and robustness of the proposed system are verified.  
  </ol>  
</details>  
**comments**: Accepted by IROS 2024, 8 pages, 10 figures  
  
  



## SFM  

### [Indoor 3D Reconstruction with an Unknown Camera-Projector Pair](http://arxiv.org/abs/2407.01945)  
Zhaoshuai Qi, Yifeng Hao, Rui Hu, Wenyou Chang, Jiaqi Yang, Yanning Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Structured light-based method with a camera-projector pair (CPP) plays a vital role in indoor 3D reconstruction, especially for scenes with weak textures. Previous methods usually assume known intrinsics, which are pre-calibrated from known objects, or self-calibrated from multi-view observations. It is still challenging to reliably recover CPP intrinsics from only two views without any known objects. In this paper, we provide a simple yet reliable solution. We demonstrate that, for the first time, sufficient constraints on CPP intrinsics can be derived from an unknown cuboid corner (C2), e.g. a room's corner, which is a common structure in indoor scenes. In addition, with only known camera principal point, the complex multi-variable estimation of all CPP intrinsics can be simplified to a simple univariable optimization problem, leading to reliable calibration and thus direct 3D reconstruction with unknown CPP. Extensive results have demonstrated the superiority of the proposed method over both traditional and learning-based counterparts. Furthermore, the proposed method also demonstrates impressive potential to solve similar tasks without active lighting, such as sparse-view structure from motion.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Close, But Not There: Boosting Geographic Distance Sensitivity in Visual Place Recognition](http://arxiv.org/abs/2407.02422)  
Sergio Izquierdo, Javier Civera  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Place Recognition (VPR) plays a critical role in many localization and mapping pipelines. It consists of retrieving the closest sample to a query image, in a certain embedding space, from a database of geotagged references. The image embedding is learned to effectively describe a place despite variations in visual appearance, viewpoint, and geometric changes. In this work, we formulate how limitations in the Geographic Distance Sensitivity of current VPR embeddings result in a high probability of incorrectly sorting the top-k retrievals, negatively impacting the recall. In order to address this issue in single-stage VPR, we propose a novel mining strategy, CliqueMining, that selects positive and negative examples by sampling cliques from a graph of visually similar images. Our approach boosts the sensitivity of VPR embeddings at small distance ranges, significantly improving the state of the art on relevant benchmarks. In particular, we raise recall@1 from 75% to 82% in MSLS Challenge, and from 76% to 90% in Nordland. Models and code are available at https://github.com/serizba/cliquemining.  
  </ol>  
</details>  
  
### [Freeview Sketching: View-Aware Fine-Grained Sketch-Based Image Retrieval](http://arxiv.org/abs/2407.01810)  
Aneeshan Sain, Pinaki Nath Chowdhury, Subhadeep Koley, Ayan Kumar Bhunia, Yi-Zhe Song  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we delve into the intricate dynamics of Fine-Grained Sketch-Based Image Retrieval (FG-SBIR) by addressing a critical yet overlooked aspect -- the choice of viewpoint during sketch creation. Unlike photo systems that seamlessly handle diverse views through extensive datasets, sketch systems, with limited data collected from fixed perspectives, face challenges. Our pilot study, employing a pre-trained FG-SBIR model, highlights the system's struggle when query-sketches differ in viewpoint from target instances. Interestingly, a questionnaire however shows users desire autonomy, with a significant percentage favouring view-specific retrieval. To reconcile this, we advocate for a view-aware system, seamlessly accommodating both view-agnostic and view-specific tasks. Overcoming dataset limitations, our first contribution leverages multi-view 2D projections of 3D objects, instilling cross-modal view awareness. The second contribution introduces a customisable cross-modal feature through disentanglement, allowing effortless mode switching. Extensive experiments on standard datasets validate the effectiveness of our method.  
  </ol>  
</details>  
**comments**: Accepted in European Conference on Computer Vision (ECCV) 2024  
  
### [Cross-Modal Attention Alignment Network with Auxiliary Text Description for zero-shot sketch-based image retrieval](http://arxiv.org/abs/2407.00979)  
Hanwen Su, Ge Song, Kai Huang, Jiyan Wang, Ming Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we study the problem of zero-shot sketch-based image retrieval (ZS-SBIR). The prior methods tackle the problem in a two-modality setting with only category labels or even no textual information involved. However, the growing prevalence of Large-scale pre-trained Language Models (LLMs), which have demonstrated great knowledge learned from web-scale data, can provide us with an opportunity to conclude collective textual information. Our key innovation lies in the usage of text data as auxiliary information for images, thus leveraging the inherent zero-shot generalization ability that language offers. To this end, we propose an approach called Cross-Modal Attention Alignment Network with Auxiliary Text Description for zero-shot sketch-based image retrieval. The network consists of three components: (i) a Description Generation Module that generates textual descriptions for each training category by prompting an LLM with several interrogative sentences, (ii) a Feature Extraction Module that includes two ViTs for sketch and image data, a transformer for extracting tokens of sentences of each training category, finally (iii) a Cross-modal Alignment Module that exchanges the token features of both text-sketch and text-image using cross-attention mechanism, and align the tokens locally and globally. Extensive experiments on three benchmark datasets show our superior performances over the state-of-the-art ZS-SBIR methods.  
  </ol>  
</details>  
  
### [Dynamically Modulating Visual Place Recognition Sequence Length For Minimum Acceptable Performance Scenarios](http://arxiv.org/abs/2407.00863)  
Connor Malone, Ankit Vora, Thierry Peynot, Michael Milford  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Mobile robots and autonomous vehicles are often required to function in environments where critical position estimates from sensors such as GPS become uncertain or unreliable. Single image visual place recognition (VPR) provides an alternative for localization but often requires techniques such as sequence matching to improve robustness, which incurs additional computation and latency costs. Even then, the sequence length required to localize at an acceptable performance level varies widely; and simply setting overly long fixed sequence lengths creates unnecessary latency, computational overhead, and can even degrade performance. In these scenarios it is often more desirable to meet or exceed a set target performance at minimal expense. In this paper we present an approach which uses a calibration set of data to fit a model that modulates sequence length for VPR as needed to exceed a target localization performance. We make use of a coarse position prior, which could be provided by any other localization system, and capture the variation in appearance across this region. We use the correlation between appearance variation and sequence length to curate VPR features and fit a multilayer perceptron (MLP) for selecting the optimal length. We demonstrate that this method is effective at modulating sequence length to maximize the number of sections in a dataset which meet or exceed a target performance whilst minimizing the median length used. We show applicability across several datasets and reveal key phenomena like generalization capabilities, the benefits of curating features and the utility of non-state-of-the-art feature extractors with nuanced properties.  
  </ol>  
</details>  
**comments**: DOI TBC  
  
  



## Keypoint Detection  

### [Multi-Grained Contrast for Data-Efficient Unsupervised Representation Learning](http://arxiv.org/abs/2407.02014)  
[[code](https://github.com/visresearch/mgc)]  
Chengchao Shen, Jianzhong Chen, Jianxin Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The existing contrastive learning methods mainly focus on single-grained representation learning, e.g., part-level, object-level or scene-level ones, thus inevitably neglecting the transferability of representations on other granularity levels. In this paper, we aim to learn multi-grained representations, which can effectively describe the image on various granularity levels, thus improving generalization on extensive downstream tasks. To this end, we propose a novel Multi-Grained Contrast method (MGC) for unsupervised representation learning. Specifically, we construct delicate multi-grained correspondences between positive views and then conduct multi-grained contrast by the correspondences to learn more general unsupervised representations.   Without pretrained on large-scale dataset, our method significantly outperforms the existing state-of-the-art methods on extensive downstream tasks, including object detection, instance segmentation, scene parsing, semantic segmentation and keypoint detection. Moreover, experimental results support the data-efficient property and excellent representation transferability of our method. The source code and trained weights are available at \url{https://github.com/visresearch/mgc}.  
  </ol>  
</details>  
  
  



## NeRF  

### [BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream](http://arxiv.org/abs/2407.02174)  
[[code](https://github.com/WU-CVGL/BeNeRF)]  
Wenpu Li, Pian Wan, Peng Wang, Jinhang Li, Yi Zhou, Peidong Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural implicit representation of visual scenes has attracted a lot of attention in recent research of computer vision and graphics. Most prior methods focus on how to reconstruct 3D scene representation from a set of images. In this work, we demonstrate the possibility to recover the neural radiance fields (NeRF) from a single blurry image and its corresponding event stream. We model the camera motion with a cubic B-Spline in SE(3) space. Both the blurry image and the brightness change within a time interval, can then be synthesized from the 3D scene representation given the 6-DoF poses interpolated from the cubic B-Spline. Our method can jointly learn both the implicit neural scene representation and recover the camera motion by minimizing the differences between the synthesized data and the real measurements without pre-computed camera poses from COLMAP. We evaluate the proposed method with both synthetic and real datasets. The experimental results demonstrate that we are able to render view-consistent latent sharp images from the learned NeRF and bring a blurry image alive in high quality. Code and data are available at https://github.com/WU-CVGL/BeNeRF.  
  </ol>  
</details>  
**comments**: Accepted to ECCV 2024  
  
### [Active Human Pose Estimation via an Autonomous UAV Agent](http://arxiv.org/abs/2407.01811)  
Jingxi Chen, Botao He, Chahat Deep Singh, Cornelia Fermuller, Yiannis Aloimonos  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    One of the core activities of an active observer involves moving to secure a "better" view of the scene, where the definition of "better" is task-dependent. This paper focuses on the task of human pose estimation from videos capturing a person's activity. Self-occlusions within the scene can complicate or even prevent accurate human pose estimation. To address this, relocating the camera to a new vantage point is necessary to clarify the view, thereby improving 2D human pose estimation. This paper formalizes the process of achieving an improved viewpoint. Our proposed solution to this challenge comprises three main components: a NeRF-based Drone-View Data Generation Framework, an On-Drone Network for Camera View Error Estimation, and a Combined Planner for devising a feasible motion plan to reposition the camera based on the predicted errors for camera views. The Data Generation Framework utilizes NeRF-based methods to generate a comprehensive dataset of human poses and activities, enhancing the drone's adaptability in various scenarios. The Camera View Error Estimation Network is designed to evaluate the current human pose and identify the most promising next viewing angles for the drone, ensuring a reliable and precise pose estimation from those angles. Finally, the combined planner incorporates these angles while considering the drone's physical and environmental limitations, employing efficient algorithms to navigate safe and effective flight paths. This system represents a significant advancement in active 2D human pose estimation for an autonomous UAV agent, offering substantial potential for applications in aerial cinematography by improving the performance of autonomous human pose estimation and maintaining the operational safety and efficiency of UAVs.  
  </ol>  
</details>  
  
### [DRAGON: Drone and Ground Gaussian Splatting for 3D Building Reconstruction](http://arxiv.org/abs/2407.01761)  
Yujin Ham, Mateusz Michalkiewicz, Guha Balakrishnan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D building reconstruction from imaging data is an important task for many applications ranging from urban planning to reconnaissance. Modern Novel View synthesis (NVS) methods like NeRF and Gaussian Splatting offer powerful techniques for developing 3D models from natural 2D imagery in an unsupervised fashion. These algorithms generally require input training views surrounding the scene of interest, which, in the case of large buildings, is typically not available across all camera elevations. In particular, the most readily available camera viewpoints at scale across most buildings are at near-ground (e.g., with mobile phones) and aerial (drones) elevations. However, due to the significant difference in viewpoint between drone and ground image sets, camera registration - a necessary step for NVS algorithms - fails. In this work we propose a method, DRAGON, that can take drone and ground building imagery as input and produce a 3D NVS model. The key insight of DRAGON is that intermediate elevation imagery may be extrapolated by an NVS algorithm itself in an iterative procedure with perceptual regularization, thereby bridging the visual feature gap between the two elevations and enabling registration. We compiled a semi-synthetic dataset of 9 large building scenes using Google Earth Studio, and quantitatively and qualitatively demonstrate that DRAGON can generate compelling renderings on this dataset compared to baseline strategies.  
  </ol>  
</details>  
**comments**: 12 pages, 9 figures, accepted to ICCP 2024  
  
### [Fast and Efficient: Mask Neural Fields for 3D Scene Segmentation](http://arxiv.org/abs/2407.01220)  
Zihan Gao, Lingling Li, Licheng Jiao, Fang Liu, Xu Liu, Wenping Ma, Yuwei Guo, Shuyuan Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Understanding 3D scenes is a crucial challenge in computer vision research with applications spanning multiple domains. Recent advancements in distilling 2D vision-language foundation models into neural fields, like NeRF and 3DGS, enables open-vocabulary segmentation of 3D scenes from 2D multi-view images without the need for precise 3D annotations. While effective, however, the per-pixel distillation of high-dimensional CLIP features introduces ambiguity and necessitates complex regularization strategies, adding inefficiencies during training. This paper presents MaskField, which enables fast and efficient 3D open-vocabulary segmentation with neural fields under weak supervision. Unlike previous methods, MaskField distills masks rather than dense high-dimensional CLIP features. MaskFields employ neural fields as binary mask generators and supervise them with masks generated by SAM and classified by coarse CLIP features. MaskField overcomes the ambiguous object boundaries by naturally introducing SAM segmented object shapes without extra regularization during training. By circumventing the direct handling of high-dimensional CLIP features during training, MaskField is particularly compatible with explicit scene representations like 3DGS. Our extensive experiments show that MaskField not only surpasses prior state-of-the-art methods but also achieves remarkably fast convergence, outperforming previous methods with just 5 minutes of training. We hope that MaskField will inspire further exploration into how neural fields can be trained to comprehend 3D scenes from 2D models.  
  </ol>  
</details>  
**comments**: 16 pages, 7 figures  
  
### [Intrinsic PAPR for Point-level 3D Scene Albedo and Shading Editing](http://arxiv.org/abs/2407.00500)  
Alireza Moazeni, Shichong Peng, Ke Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in neural rendering have excelled at novel view synthesis from multi-view RGB images. However, they often lack the capability to edit the shading or colour of the scene at a detailed point-level, while ensuring consistency across different viewpoints. In this work, we address the challenge of point-level 3D scene albedo and shading editing from multi-view RGB images, focusing on detailed editing at the point-level rather than at a part or global level. While prior works based on volumetric representation such as NeRF struggle with achieving 3D consistent editing at the point level, recent advancements in point-based neural rendering show promise in overcoming this challenge. We introduce ``Intrinsic PAPR'', a novel method based on the recent point-based neural rendering technique Proximity Attention Point Rendering (PAPR). Unlike other point-based methods that model the intrinsic decomposition of the scene, our approach does not rely on complicated shading models or simplistic priors that may not universally apply. Instead, we directly model scene decomposition into albedo and shading components, leading to better estimation accuracy. Comparative evaluations against the latest point-based inverse rendering methods demonstrate that Intrinsic PAPR achieves higher-quality novel view rendering and superior point-level albedo and shading editing.  
  </ol>  
</details>  
  
  



