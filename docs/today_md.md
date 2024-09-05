<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Robust-Vehicle-Localization-and-Tracking-in-Rain-using-Street-Maps>Robust Vehicle Localization and Tracking in Rain using Street Maps</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Object-Gaussian-for-Monocular-6D-Pose-Estimation-from-Sparse-Views>Object Gaussian for Monocular 6D Pose Estimation from Sparse Views</a></li>
        <li><a href=#Geometry-aware-Feature-Matching-for-Large-Scale-Structure-from-Motion>Geometry-aware Feature Matching for Large-Scale Structure from Motion</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#NUDGE:-Lightweight-Non-Parametric-Fine-Tuning-of-Embeddings-for-Retrieval>NUDGE: Lightweight Non-Parametric Fine-Tuning of Embeddings for Retrieval</a></li>
        <li><a href=#Optimizing-CLIP-Models-for-Image-Retrieval-with-Maintained-Joint-Embedding-Alignment>Optimizing CLIP Models for Image Retrieval with Maintained Joint-Embedding Alignment</a></li>
        <li><a href=#A-Review-of-Image-Retrieval-Techniques:-Data-Augmentation-and-Adversarial-Learning-Approaches>A Review of Image Retrieval Techniques: Data Augmentation and Adversarial Learning Approaches</a></li>
        <li><a href=#Online-One-Dimensional-Magnetic-Field-SLAM-with-Loop-Closure-Detection>Online One-Dimensional Magnetic Field SLAM with Loop-Closure Detection</a></li>
        <li><a href=#Evidential-Transformers-for-Improved-Image-Retrieval>Evidential Transformers for Improved Image Retrieval</a></li>
        <li><a href=#EgoHDM:-An-Online-Egocentric-Inertial-Human-Motion-Capture,-Localization,-and-Dense-Mapping-System>EgoHDM: An Online Egocentric-Inertial Human Motion Capture, Localization, and Dense Mapping System</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#UC-NeRF:-Uncertainty-aware-Conditional-Neural-Radiance-Fields-from-Endoscopic-Sparse-Views>UC-NeRF: Uncertainty-aware Conditional Neural Radiance Fields from Endoscopic Sparse Views</a></li>
        <li><a href=#GraspSplats:-Efficient-Manipulation-with-3D-Feature-Splatting>GraspSplats: Efficient Manipulation with 3D Feature Splatting</a></li>
        <li><a href=#$S^2$NeRF:-Privacy-preserving-Training-Framework-for-NeRF>$S^2$NeRF: Privacy-preserving Training Framework for NeRF</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Robust Vehicle Localization and Tracking in Rain using Street Maps](http://arxiv.org/abs/2409.01038)  
Yu Xiang Tan, Malika Meghjani  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    GPS-based vehicle localization and tracking suffers from unstable positional information commonly experienced in tunnel segments and in dense urban areas. Also, both Visual Odometry (VO) and Visual Inertial Odometry (VIO) are susceptible to adverse weather conditions that causes occlusions or blur on the visual input. In this paper, we propose a novel approach for vehicle localization that uses street network based map information to correct drifting odometry estimates and intermittent GPS measurements especially, in adversarial scenarios such as driving in rain and tunnels. Specifically, our approach is a flexible fusion algorithm that integrates intermittent GPS, drifting IMU and VO estimates together with 2D map information for robust vehicle localization and tracking. We refer to our approach as Map-Fusion. We robustly evaluate our proposed approach on four geographically diverse datasets from different countries ranging across clear and rain weather conditions. These datasets also include challenging visual segments in tunnels and underpasses. We show that with the integration of the map information, our Map-Fusion algorithm reduces the error of the state-of-the-art VO and VIO approaches across all datasets. We also validate our proposed algorithm in a real-world environment and in real-time on a hardware constrained mobile robot. Map-Fusion achieved 2.46m error in clear weather and 6.05m error in rain weather for a 150m route.  
  </ol>  
</details>  
  
  



## SFM  

### [Object Gaussian for Monocular 6D Pose Estimation from Sparse Views](http://arxiv.org/abs/2409.02581)  
Luqing Luo, Shichu Sun, Jiangang Yang, Linfang Zheng, Jinwei Du, Jian Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Monocular object pose estimation, as a pivotal task in computer vision and robotics, heavily depends on accurate 2D-3D correspondences, which often demand costly CAD models that may not be readily available. Object 3D reconstruction methods offer an alternative, among which recent advancements in 3D Gaussian Splatting (3DGS) afford a compelling potential. Yet its performance still suffers and tends to overfit with fewer input views. Embracing this challenge, we introduce SGPose, a novel framework for sparse view object pose estimation using Gaussian-based methods. Given as few as ten views, SGPose generates a geometric-aware representation by starting with a random cuboid initialization, eschewing reliance on Structure-from-Motion (SfM) pipeline-derived geometry as required by traditional 3DGS methods. SGPose removes the dependence on CAD models by regressing dense 2D-3D correspondences between images and the reconstructed model from sparse input and random initialization, while the geometric-consistent depth supervision and online synthetic view warping are key to the success. Experiments on typical benchmarks, especially on the Occlusion LM-O dataset, demonstrate that SGPose outperforms existing methods even under sparse view constraints, under-scoring its potential in real-world applications.  
  </ol>  
</details>  
  
### [Geometry-aware Feature Matching for Large-Scale Structure from Motion](http://arxiv.org/abs/2409.02310)  
Gonglin Chen, Jinsen Wu, Haiwei Chen, Wenbin Teng, Zhiyuan Gao, Andrew Feng, Rongjun Qin, Yajie Zhao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Establishing consistent and dense correspondences across multiple images is crucial for Structure from Motion (SfM) systems. Significant view changes, such as air-to-ground with very sparse view overlap, pose an even greater challenge to the correspondence solvers. We present a novel optimization-based approach that significantly enhances existing feature matching methods by introducing geometry cues in addition to color cues. This helps fill gaps when there is less overlap in large-scale scenarios. Our method formulates geometric verification as an optimization problem, guiding feature matching within detector-free methods and using sparse correspondences from detector-based methods as anchor points. By enforcing geometric constraints via the Sampson Distance, our approach ensures that the denser correspondences from detector-free methods are geometrically consistent and more accurate. This hybrid strategy significantly improves correspondence density and accuracy, mitigates multi-view inconsistencies, and leads to notable advancements in camera pose accuracy and point cloud density. It outperforms state-of-the-art feature matching methods on benchmark datasets and enables feature matching in challenging extreme large-scale settings.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [NUDGE: Lightweight Non-Parametric Fine-Tuning of Embeddings for Retrieval](http://arxiv.org/abs/2409.02343)  
Sepanta Zeighami, Zac Wellmer, Aditya Parameswaran  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    $k$-Nearest Neighbor search on dense vector embeddings ($k$-NN retrieval) from pre-trained embedding models is the predominant retrieval method for text and images, as well as Retrieval-Augmented Generation (RAG) pipelines. In practice, application developers often fine-tune the embeddings to improve their accuracy on the dataset and query workload in hand. Existing approaches either fine-tune the pre-trained model itself or, more efficiently, but at the cost of accuracy, train adaptor models to transform the output of the pre-trained model. We present NUDGE, a family of novel non-parametric embedding fine-tuning approaches that are significantly more accurate and efficient than both sets of existing approaches. NUDGE directly modifies the embeddings of data records to maximize the accuracy of $k$ -NN retrieval. We present a thorough theoretical and experimental study of NUDGE's non-parametric approach. We show that even though the underlying problem is NP-Hard, constrained variations can be solved efficiently. These constraints additionally ensure that the changes to the embeddings are modest, avoiding large distortions to the semantics learned during pre-training. In experiments across five pre-trained models and nine standard text and image retrieval datasets, NUDGE runs in minutes and often improves NDCG@10 by more than 10% over existing fine-tuning methods. On average, NUDGE provides 3.3x and 4.3x higher increase in accuracy and runs 200x and 3x faster, respectively, over fine-tuning the pre-trained model and training adaptors.  
  </ol>  
</details>  
  
### [Optimizing CLIP Models for Image Retrieval with Maintained Joint-Embedding Alignment](http://arxiv.org/abs/2409.01936)  
Konstantin Schall, Kai Uwe Barthel, Nico Hezel, Klaus Jung  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Contrastive Language and Image Pairing (CLIP), a transformative method in multimedia retrieval, typically trains two neural networks concurrently to generate joint embeddings for text and image pairs. However, when applied directly, these models often struggle to differentiate between visually distinct images that have similar captions, resulting in suboptimal performance for image-based similarity searches. This paper addresses the challenge of optimizing CLIP models for various image-based similarity search scenarios, while maintaining their effectiveness in text-based search tasks such as text-to-image retrieval and zero-shot classification. We propose and evaluate two novel methods aimed at refining the retrieval capabilities of CLIP without compromising the alignment between text and image embeddings. The first method involves a sequential fine-tuning process: initially optimizing the image encoder for more precise image retrieval and subsequently realigning the text encoder to these optimized image embeddings. The second approach integrates pseudo-captions during the retrieval-optimization phase to foster direct alignment within the embedding space. Through comprehensive experiments, we demonstrate that these methods enhance CLIP's performance on various benchmarks, including image retrieval, k-NN classification, and zero-shot text-based classification, while maintaining robustness in text-to-image retrieval. Our optimized models permit maintaining a single embedding per image, significantly simplifying the infrastructure needed for large-scale multi-modal similarity search systems.  
  </ol>  
</details>  
  
### [A Review of Image Retrieval Techniques: Data Augmentation and Adversarial Learning Approaches](http://arxiv.org/abs/2409.01219)  
Kim Jinwoo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image retrieval is a crucial research topic in computer vision, with broad application prospects ranging from online product searches to security surveillance systems. In recent years, the accuracy and efficiency of image retrieval have significantly improved due to advancements in deep learning. However, existing methods still face numerous challenges, particularly in handling large-scale datasets, cross-domain retrieval, and image perturbations that can arise from real-world conditions such as variations in lighting, occlusion, and viewpoint. Data augmentation techniques and adversarial learning methods have been widely applied in the field of image retrieval to address these challenges. Data augmentation enhances the model's generalization ability and robustness by generating more diverse training samples, simulating real-world variations, and reducing overfitting. Meanwhile, adversarial attacks and defenses introduce perturbations during training to improve the model's robustness against potential attacks, ensuring reliability in practical applications. This review comprehensively summarizes the latest research advancements in image retrieval, with a particular focus on the roles of data augmentation and adversarial learning techniques in enhancing retrieval performance. Future directions and potential challenges are also discussed.  
  </ol>  
</details>  
  
### [Online One-Dimensional Magnetic Field SLAM with Loop-Closure Detection](http://arxiv.org/abs/2409.01091)  
Manon Kok, Arno Solin  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present a lightweight magnetic field simultaneous localisation and mapping (SLAM) approach for drift correction in odometry paths, where the interest is purely in the odometry and not in map building. We represent the past magnetic field readings as a one-dimensional trajectory against which the current magnetic field observations are matched. This approach boils down to sequential loop-closure detection and decision-making, based on the current pose state estimate and the magnetic field. We combine this setup with a path estimation framework using an extended Kalman smoother which fuses the odometry increments with the detected loop-closure timings. We demonstrate the practical applicability of the model with several different real-world examples from a handheld iPad moving in indoor scenes.  
  </ol>  
</details>  
**comments**: To appear in International Conference on Multisensor Fusion and
  Integration for Intelligent Systems (MFI) 2024  
  
### [Evidential Transformers for Improved Image Retrieval](http://arxiv.org/abs/2409.01082)  
Danilo Dordevic, Suryansh Kumar  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce the Evidential Transformer, an uncertainty-driven transformer model for improved and robust image retrieval. In this paper, we make several contributions to content-based image retrieval (CBIR). We incorporate probabilistic methods into image retrieval, achieving robust and reliable results, with evidential classification surpassing traditional training based on multiclass classification as a baseline for deep metric learning. Furthermore, we improve the state-of-the-art retrieval results on several datasets by leveraging the Global Context Vision Transformer (GC ViT) architecture. Our experimental results consistently demonstrate the reliability of our approach, setting a new benchmark in CBIR in all test settings on the Stanford Online Products (SOP) and CUB-200-2011 datasets.  
  </ol>  
</details>  
**comments**: 6 pages, 6 figures, To be presented at the 3rd Workshop on
  Uncertainty Quantification for Computer Vision, at the ECCV 2024 conference
  in Milan, Italy  
  
### [EgoHDM: An Online Egocentric-Inertial Human Motion Capture, Localization, and Dense Mapping System](http://arxiv.org/abs/2409.00343)  
Bonan Liu, Handi Yin, Manuel Kaufmann, Jinhao He, Sammy Christen, Jie Song, Pan Hui  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present EgoHDM, an online egocentric-inertial human motion capture (mocap), localization, and dense mapping system. Our system uses 6 inertial measurement units (IMUs) and a commodity head-mounted RGB camera. EgoHDM is the first human mocap system that offers dense scene mapping in near real-time. Further, it is fast and robust to initialize and fully closes the loop between physically plausible map-aware global human motion estimation and mocap-aware 3D scene reconstruction. Our key idea is integrating camera localization and mapping information with inertial human motion capture bidirectionally in our system. To achieve this, we design a tightly coupled mocap-aware dense bundle adjustment and physics-based body pose correction module leveraging a local body-centric elevation map. The latter introduces a novel terrain-aware contact PD controller, which enables characters to physically contact the given local elevation map thereby reducing human floating or penetration. We demonstrate the performance of our system on established synthetic and real-world benchmarks. The results show that our method reduces human localization, camera pose, and mapping accuracy error by 41%, 71%, 46%, respectively, compared to the state of the art. Our qualitative evaluations on newly captured data further demonstrate that EgoHDM can cover challenging scenarios in non-flat terrain including stepping over stairs and outdoor scenes in the wild.  
  </ol>  
</details>  
  
  



## NeRF  

### [UC-NeRF: Uncertainty-aware Conditional Neural Radiance Fields from Endoscopic Sparse Views](http://arxiv.org/abs/2409.02917)  
[[code](https://github.com/wrld/uc-nerf)]  
Jiaxin Guo, Jiangliu Wang, Ruofeng Wei, Di Kang, Qi Dou, Yun-hui Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visualizing surgical scenes is crucial for revealing internal anatomical structures during minimally invasive procedures. Novel View Synthesis is a vital technique that offers geometry and appearance reconstruction, enhancing understanding, planning, and decision-making in surgical scenes. Despite the impressive achievements of Neural Radiance Field (NeRF), its direct application to surgical scenes produces unsatisfying results due to two challenges: endoscopic sparse views and significant photometric inconsistencies. In this paper, we propose uncertainty-aware conditional NeRF for novel view synthesis to tackle the severe shape-radiance ambiguity from sparse surgical views. The core of UC-NeRF is to incorporate the multi-view uncertainty estimation to condition the neural radiance field for modeling the severe photometric inconsistencies adaptively. Specifically, our UC-NeRF first builds a consistency learner in the form of multi-view stereo network, to establish the geometric correspondence from sparse views and generate uncertainty estimation and feature priors. In neural rendering, we design a base-adaptive NeRF network to exploit the uncertainty estimation for explicitly handling the photometric inconsistencies. Furthermore, an uncertainty-guided geometry distillation is employed to enhance geometry learning. Experiments on the SCARED and Hamlyn datasets demonstrate our superior performance in rendering appearance and geometry, consistently outperforming the current state-of-the-art approaches. Our code will be released at \url{https://github.com/wrld/UC-NeRF}.  
  </ol>  
</details>  
  
### [GraspSplats: Efficient Manipulation with 3D Feature Splatting](http://arxiv.org/abs/2409.02084)  
Mazeyu Ji, Ri-Zhao Qiu, Xueyan Zou, Xiaolong Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The ability for robots to perform efficient and zero-shot grasping of object parts is crucial for practical applications and is becoming prevalent with recent advances in Vision-Language Models (VLMs). To bridge the 2D-to-3D gap for representations to support such a capability, existing methods rely on neural fields (NeRFs) via differentiable rendering or point-based projection methods. However, we demonstrate that NeRFs are inappropriate for scene changes due to their implicitness and point-based methods are inaccurate for part localization without rendering-based optimization. To amend these issues, we propose GraspSplats. Using depth supervision and a novel reference feature computation method, GraspSplats generates high-quality scene representations in under 60 seconds. We further validate the advantages of Gaussian-based representation by showing that the explicit and optimized geometry in GraspSplats is sufficient to natively support (1) real-time grasp sampling and (2) dynamic and articulated object manipulation with point trackers. With extensive experiments on a Franka robot, we demonstrate that GraspSplats significantly outperforms existing methods under diverse task settings. In particular, GraspSplats outperforms NeRF-based methods like F3RM and LERF-TOGO, and 2D detection methods.  
  </ol>  
</details>  
**comments**: Project webpage: https://graspsplats.github.io/  
  
### [ $S^2$ NeRF: Privacy-preserving Training Framework for NeRF](http://arxiv.org/abs/2409.01661)  
Bokang Zhang, Yanglin Zhang, Zhikun Zhang, Jinglan Yang, Lingying Huang, Junfeng Wu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) have revolutionized 3D computer vision and graphics, facilitating novel view synthesis and influencing sectors like extended reality and e-commerce. However, NeRF's dependence on extensive data collection, including sensitive scene image data, introduces significant privacy risks when users upload this data for model training. To address this concern, we first propose SplitNeRF, a training framework that incorporates split learning (SL) techniques to enable privacy-preserving collaborative model training between clients and servers without sharing local data. Despite its benefits, we identify vulnerabilities in SplitNeRF by developing two attack methods, Surrogate Model Attack and Scene-aided Surrogate Model Attack, which exploit the shared gradient data and a few leaked scene images to reconstruct private scene information. To counter these threats, we introduce $S^2$NeRF, secure SplitNeRF that integrates effective defense mechanisms. By introducing decaying noise related to the gradient norm into the shared gradient information, $S^2$NeRF preserves privacy while maintaining a high utility of the NeRF model. Our extensive evaluations across multiple datasets demonstrate the effectiveness of $S^2$NeRF against privacy breaches, confirming its viability for secure NeRF training in sensitive applications.  
  </ol>  
</details>  
**comments**: To appear in the ACM Conference on Computer and Communications
  Security (CCS'24), October 14-18, 2024, Salt Lake City, UT, USA  
  
  



