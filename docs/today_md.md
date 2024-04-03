<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#TSCM:-A-Teacher-Student-Model-for-Vision-Place-Recognition-Using-Cross-Metric-Knowledge-Distillation>TSCM: A Teacher-Student Model for Vision Place Recognition Using Cross-Metric Knowledge Distillation</a></li>
        <li><a href=#On-Train-Test-Class-Overlap-and-Detection-for-Image-Retrieval>On Train-Test Class Overlap and Detection for Image Retrieval</a></li>
        <li><a href=#NVINS:-Robust-Visual-Inertial-Navigation-Fused-with-NeRF-augmented-Camera-Pose-Regressor-and-Uncertainty-Quantification>NVINS: Robust Visual Inertial Navigation Fused with NeRF-augmented Camera Pose Regressor and Uncertainty Quantification</a></li>
        <li><a href=#On-the-Estimation-of-Image-matching-Uncertainty-in-Visual-Place-Recognition>On the Estimation of Image-matching Uncertainty in Visual Place Recognition</a></li>
        <li><a href=#NYC-Indoor-VPR:-A-Long-Term-Indoor-Visual-Place-Recognition-Dataset-with-Semi-Automatic-Annotation>NYC-Indoor-VPR: A Long-Term Indoor Visual Place Recognition Dataset with Semi-Automatic Annotation</a></li>
        <li><a href=#SceneGraphLoc:-Cross-Modal-Coarse-Visual-Localization-on-3D-Scene-Graphs>SceneGraphLoc: Cross-Modal Coarse Visual Localization on 3D Scene Graphs</a></li>
        <li><a href=#Do-Vision-Language-Models-Understand-Compound-Nouns?>Do Vision-Language Models Understand Compound Nouns?</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Marrying-NeRF-with-Feature-Matching-for-One-step-Pose-Estimation>Marrying NeRF with Feature Matching for One-step Pose Estimation</a></li>
        <li><a href=#3MOS:-Multi-sources,-Multi-resolutions,-and-Multi-scenes-dataset-for-Optical-SAR-image-matching>3MOS: Multi-sources, Multi-resolutions, and Multi-scenes dataset for Optical-SAR image matching</a></li>
        <li><a href=#On-the-Estimation-of-Image-matching-Uncertainty-in-Visual-Place-Recognition>On the Estimation of Image-matching Uncertainty in Visual Place Recognition</a></li>
        <li><a href=#Image-to-Image-Matching-via-Foundation-Models:-A-New-Perspective-for-Open-Vocabulary-Semantic-Segmentation>Image-to-Image Matching via Foundation Models: A New Perspective for Open-Vocabulary Semantic Segmentation</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Alpha-Invariance:-On-Inverse-Scaling-Between-Distance-and-Volume-Density-in-Neural-Radiance-Fields>Alpha Invariance: On Inverse Scaling Between Distance and Volume Density in Neural Radiance Fields</a></li>
        <li><a href=#Uncertainty-aware-Active-Learning-of-NeRF-based-Object-Models-for-Robot-Manipulators-using-Visual-and-Re-orientation-Actions>Uncertainty-aware Active Learning of NeRF-based Object Models for Robot Manipulators using Visual and Re-orientation Actions</a></li>
        <li><a href=#NVINS:-Robust-Visual-Inertial-Navigation-Fused-with-NeRF-augmented-Camera-Pose-Regressor-and-Uncertainty-Quantification>NVINS: Robust Visual Inertial Navigation Fused with NeRF-augmented Camera Pose Regressor and Uncertainty Quantification</a></li>
        <li><a href=#NeRF-MAE-:-Masked-AutoEncoders-for-Self-Supervised-3D-representation-Learning-for-Neural-Radiance-Fields>NeRF-MAE : Masked AutoEncoders for Self Supervised 3D representation Learning for Neural Radiance Fields</a></li>
        <li><a href=#MagicMirror:-Fast-and-High-Quality-Avatar-Generation-with-a-Constrained-Search-Space>MagicMirror: Fast and High-Quality Avatar Generation with a Constrained Search Space</a></li>
        <li><a href=#StructLDM:-Structured-Latent-Diffusion-for-3D-Human-Generation>StructLDM: Structured Latent Diffusion for 3D Human Generation</a></li>
        <li><a href=#Mirror-3DGS:-Incorporating-Mirror-Reflections-into-3D-Gaussian-Splatting>Mirror-3DGS: Incorporating Mirror Reflections into 3D Gaussian Splatting</a></li>
        <li><a href=#SGCNeRF:-Few-Shot-Neural-Rendering-via-Sparse-Geometric-Consistency-Guidance>SGCNeRF: Few-Shot Neural Rendering via Sparse Geometric Consistency Guidance</a></li>
        <li><a href=#FlexiDreamer:-Single-Image-to-3D-Generation-with-FlexiCubes>FlexiDreamer: Single Image-to-3D Generation with FlexiCubes</a></li>
        <li><a href=#Marrying-NeRF-with-Feature-Matching-for-One-step-Pose-Estimation>Marrying NeRF with Feature Matching for One-step Pose Estimation</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [TSCM: A Teacher-Student Model for Vision Place Recognition Using Cross-Metric Knowledge Distillation](http://arxiv.org/abs/2404.01587)  
Yehui Shen, Mingmin Liu, Huimin Lu, Xieyuanli Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual place recognition (VPR) plays a pivotal role in autonomous exploration and navigation of mobile robots within complex outdoor environments. While cost-effective and easily deployed, camera sensors are sensitive to lighting and weather changes, and even slight image alterations can greatly affect VPR efficiency and precision. Existing methods overcome this by exploiting powerful yet large networks, leading to significant consumption of computational resources. In this paper, we propose a high-performance teacher and lightweight student distillation framework called TSCM. It exploits our devised cross-metric knowledge distillation to narrow the performance gap between the teacher and student models, maintaining superior performance while enabling minimal computational load during deployment. We conduct comprehensive evaluations on large-scale datasets, namely Pittsburgh30k and Pittsburgh250k. Experimental results demonstrate the superiority of our method over baseline models in terms of recognition accuracy and model parameter efficiency. Moreover, our ablation studies show that the proposed knowledge distillation technique surpasses other counterparts. The code of our method has been released at https://github.com/nubot-nudt/TSCM.  
  </ol>  
</details>  
  
### [On Train-Test Class Overlap and Detection for Image Retrieval](http://arxiv.org/abs/2404.01524)  
[[code](https://github.com/dealicious-inc/rgldv2-clean)]  
Chull Hwan Song, Jooyoung Yoon, Taebaek Hwang, Shunghyun Choi, Yeong Hyeon Gu, Yannis Avrithis  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    How important is it for training and evaluation sets to not have class overlap in image retrieval? We revisit Google Landmarks v2 clean, the most popular training set, by identifying and removing class overlap with Revisited Oxford and Paris [34], the most popular evaluation set. By comparing the original and the new RGLDv2-clean on a benchmark of reproduced state-of-the-art methods, our findings are striking. Not only is there a dramatic drop in performance, but it is inconsistent across methods, changing the ranking.What does it take to focus on objects or interest and ignore background clutter when indexing? Do we need to train an object detector and the representation separately? Do we need location supervision? We introduce Single-stage Detect-to-Retrieve (CiDeR), an end-to-end, single-stage pipeline to detect objects of interest and extract a global image representation. We outperform previous state-of-the-art on both existing training sets and the new RGLDv2-clean. Our dataset is available at https://github.com/dealicious-inc/RGLDv2-clean.  
  </ol>  
</details>  
**comments**: CVPR2024 Accepted  
  
### [NVINS: Robust Visual Inertial Navigation Fused with NeRF-augmented Camera Pose Regressor and Uncertainty Quantification](http://arxiv.org/abs/2404.01400)  
Juyeop Han, Lukas Lao Beyer, Guilherme V. Cavalheiro, Sertac Karaman  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In recent years, Neural Radiance Fields (NeRF) have emerged as a powerful tool for 3D reconstruction and novel view synthesis. However, the computational cost of NeRF rendering and degradation in quality due to the presence of artifacts pose significant challenges for its application in real-time and robust robotic tasks, especially on embedded systems. This paper introduces a novel framework that integrates NeRF-derived localization information with Visual-Inertial Odometry(VIO) to provide a robust solution for robotic navigation in a real-time. By training an absolute pose regression network with augmented image data rendered from a NeRF and quantifying its uncertainty, our approach effectively counters positional drift and enhances system reliability. We also establish a mathematically sound foundation for combining visual inertial navigation with camera localization neural networks, considering uncertainty under a Bayesian framework. Experimental validation in the photorealistic simulation environment demonstrates significant improvements in accuracy compared to a conventional VIO approach.  
  </ol>  
</details>  
**comments**: 8 pages, 5 figures, 2 tables  
  
### [On the Estimation of Image-matching Uncertainty in Visual Place Recognition](http://arxiv.org/abs/2404.00546)  
Mubariz Zaffar, Liangliang Nan, Julian F. P. Kooij  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In Visual Place Recognition (VPR) the pose of a query image is estimated by comparing the image to a map of reference images with known reference poses. As is typical for image retrieval problems, a feature extractor maps the query and reference images to a feature space, where a nearest neighbor search is then performed. However, till recently little attention has been given to quantifying the confidence that a retrieved reference image is a correct match. Highly certain but incorrect retrieval can lead to catastrophic failure of VPR-based localization pipelines. This work compares for the first time the main approaches for estimating the image-matching uncertainty, including the traditional retrieval-based uncertainty estimation, more recent data-driven aleatoric uncertainty estimation, and the compute-intensive geometric verification. We further formulate a simple baseline method, ``SUE'', which unlike the other methods considers the freely-available poses of the reference images in the map. Our experiments reveal that a simple L2-distance between the query and reference descriptors is already a better estimate of image-matching uncertainty than current data-driven approaches. SUE outperforms the other efficient uncertainty estimation methods, and its uncertainty estimates complement the computationally expensive geometric verification approach. Future works for uncertainty estimation in VPR should consider the baselines discussed in this work.  
  </ol>  
</details>  
**comments**: To appear in the proceedings of the IEEE/CVF Conference on Computer
  Vision and Pattern Recognition (CVPR) 2024  
  
### [NYC-Indoor-VPR: A Long-Term Indoor Visual Place Recognition Dataset with Semi-Automatic Annotation](http://arxiv.org/abs/2404.00504)  
Diwei Sheng, Anbang Yang, John-Ross Rizzo, Chen Feng  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Place Recognition (VPR) in indoor environments is beneficial to humans and robots for better localization and navigation. It is challenging due to appearance changes at various frequencies, and difficulties of obtaining ground truth metric trajectories for training and evaluation. This paper introduces the NYC-Indoor-VPR dataset, a unique and rich collection of over 36,000 images compiled from 13 distinct crowded scenes in New York City taken under varying lighting conditions with appearance changes. Each scene has multiple revisits across a year. To establish the ground truth for VPR, we propose a semiautomatic annotation approach that computes the positional information of each image. Our method specifically takes pairs of videos as input and yields matched pairs of images along with their estimated relative locations. The accuracy of this matching is refined by human annotators, who utilize our annotation software to correlate the selected keyframes. Finally, we present a benchmark evaluation of several state-of-the-art VPR algorithms using our annotated dataset, revealing its challenge and thus value for VPR research.  
  </ol>  
</details>  
**comments**: 7 pages, 7 figures, published in 2024 IEEE International Conference
  on Robotics and Automation (ICRA 2024)  
  
### [SceneGraphLoc: Cross-Modal Coarse Visual Localization on 3D Scene Graphs](http://arxiv.org/abs/2404.00469)  
Yang Miao, Francis Engelmann, Olga Vysotska, Federico Tombari, Marc Pollefeys, Dániel Béla Baráth  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce a novel problem, i.e., the localization of an input image within a multi-modal reference map represented by a database of 3D scene graphs. These graphs comprise multiple modalities, including object-level point clouds, images, attributes, and relationships between objects, offering a lightweight and efficient alternative to conventional methods that rely on extensive image databases. Given the available modalities, the proposed method SceneGraphLoc learns a fixed-sized embedding for each node (i.e., representing an object instance) in the scene graph, enabling effective matching with the objects visible in the input query image. This strategy significantly outperforms other cross-modal methods, even without incorporating images into the map embeddings. When images are leveraged, SceneGraphLoc achieves performance close to that of state-of-the-art techniques depending on large image databases, while requiring three orders-of-magnitude less storage and operating orders-of-magnitude faster. The code will be made public.  
  </ol>  
</details>  
  
### [Do Vision-Language Models Understand Compound Nouns?](http://arxiv.org/abs/2404.00419)  
Sonal Kumar, Sreyan Ghosh, S Sakshi, Utkarsh Tyagi, Dinesh Manocha  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Open-vocabulary vision-language models (VLMs) like CLIP, trained using contrastive loss, have emerged as a promising new paradigm for text-to-image retrieval. However, do VLMs understand compound nouns (CNs) (e.g., lab coat) as well as they understand nouns (e.g., lab)? We curate Compun, a novel benchmark with 400 unique and commonly used CNs, to evaluate the effectiveness of VLMs in interpreting CNs. The Compun benchmark challenges a VLM for text-to-image retrieval where, given a text prompt with a CN, the task is to select the correct image that shows the CN among a pair of distractor images that show the constituent nouns that make up the CN. Next, we perform an in-depth analysis to highlight CLIPs' limited understanding of certain types of CNs. Finally, we present an alternative framework that moves beyond hand-written templates for text prompts widely used by CLIP-like models. We employ a Large Language Model to generate multiple diverse captions that include the CN as an object in the scene described by the caption. Our proposed method improves CN understanding of CLIP by 8.25% on Compun. Code and benchmark are available at: https://github.com/sonalkum/Compun  
  </ol>  
</details>  
**comments**: Accepted to NAACL 2024 Main Conference  
  
  



## Image Matching  

### [Marrying NeRF with Feature Matching for One-step Pose Estimation](http://arxiv.org/abs/2404.00891)  
Ronghan Chen, Yang Cong, Yu Ren  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Given the image collection of an object, we aim at building a real-time image-based pose estimation method, which requires neither its CAD model nor hours of object-specific training. Recent NeRF-based methods provide a promising solution by directly optimizing the pose from pixel loss between rendered and target images. However, during inference, they require long converging time, and suffer from local minima, making them impractical for real-time robot applications. We aim at solving this problem by marrying image matching with NeRF. With 2D matches and depth rendered by NeRF, we directly solve the pose in one step by building 2D-3D correspondences between target and initial view, thus allowing for real-time prediction. Moreover, to improve the accuracy of 2D-3D correspondences, we propose a 3D consistent point mining strategy, which effectively discards unfaithful points reconstruted by NeRF. Moreover, current NeRF-based methods naively optimizing pixel loss fail at occluded images. Thus, we further propose a 2D matches based sampling strategy to preclude the occluded area. Experimental results on representative datasets prove that our method outperforms state-of-the-art methods, and improves inference efficiency by 90x, achieving real-time prediction at 6 FPS.  
  </ol>  
</details>  
**comments**: ICRA, 2024. Video https://www.youtube.com/watch?v=70fgUobOFWo  
  
### [3MOS: Multi-sources, Multi-resolutions, and Multi-scenes dataset for Optical-SAR image matching](http://arxiv.org/abs/2404.00838)  
Yibin Ye, Xichao Teng, Shuo Chen, Yijie Bian, Tao Tan, Zhang Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Optical-SAR image matching is a fundamental task for image fusion and visual navigation. However, all large-scale open SAR dataset for methods development are collected from single platform, resulting in limited satellite types and spatial resolutions. Since images captured by different sensors vary significantly in both geometric and radiometric appearance, existing methods may fail to match corresponding regions containing the same content. Besides, most of existing datasets have not been categorized based on the characteristics of different scenes. To encourage the design of more general multi-modal image matching methods, we introduce a large-scale Multi-sources,Multi-resolutions, and Multi-scenes dataset for Optical-SAR image matching(3MOS). It consists of 155K optical-SAR image pairs, including SAR data from six commercial satellites, with resolutions ranging from 1.25m to 12.5m. The data has been classified into eight scenes including urban, rural, plains, hills, mountains, water, desert, and frozen earth. Extensively experiments show that none of state-of-the-art methods achieve consistently superior performance across different sources, resolutions and scenes. In addition, the distribution of data has a substantial impact on the matching capability of deep learning models, this proposes the domain adaptation challenge in optical-SAR image matching. Our data and code will be available at:https://github.com/3M-OS/3MOS.  
  </ol>  
</details>  
**comments**: 20pages 17 figures  
  
### [On the Estimation of Image-matching Uncertainty in Visual Place Recognition](http://arxiv.org/abs/2404.00546)  
Mubariz Zaffar, Liangliang Nan, Julian F. P. Kooij  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In Visual Place Recognition (VPR) the pose of a query image is estimated by comparing the image to a map of reference images with known reference poses. As is typical for image retrieval problems, a feature extractor maps the query and reference images to a feature space, where a nearest neighbor search is then performed. However, till recently little attention has been given to quantifying the confidence that a retrieved reference image is a correct match. Highly certain but incorrect retrieval can lead to catastrophic failure of VPR-based localization pipelines. This work compares for the first time the main approaches for estimating the image-matching uncertainty, including the traditional retrieval-based uncertainty estimation, more recent data-driven aleatoric uncertainty estimation, and the compute-intensive geometric verification. We further formulate a simple baseline method, ``SUE'', which unlike the other methods considers the freely-available poses of the reference images in the map. Our experiments reveal that a simple L2-distance between the query and reference descriptors is already a better estimate of image-matching uncertainty than current data-driven approaches. SUE outperforms the other efficient uncertainty estimation methods, and its uncertainty estimates complement the computationally expensive geometric verification approach. Future works for uncertainty estimation in VPR should consider the baselines discussed in this work.  
  </ol>  
</details>  
**comments**: To appear in the proceedings of the IEEE/CVF Conference on Computer
  Vision and Pattern Recognition (CVPR) 2024  
  
### [Image-to-Image Matching via Foundation Models: A New Perspective for Open-Vocabulary Semantic Segmentation](http://arxiv.org/abs/2404.00262)  
Yuan Wang, Rui Sun, Naisong Luo, Yuwen Pan, Tianzhu Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Open-vocabulary semantic segmentation (OVS) aims to segment images of arbitrary categories specified by class labels or captions. However, most previous best-performing methods, whether pixel grouping methods or region recognition methods, suffer from false matches between image features and category labels. We attribute this to the natural gap between the textual features and visual features. In this work, we rethink how to mitigate false matches from the perspective of image-to-image matching and propose a novel relation-aware intra-modal matching (RIM) framework for OVS based on visual foundation models. RIM achieves robust region classification by firstly constructing diverse image-modal reference features and then matching them with region features based on relation-aware ranking distribution. The proposed RIM enjoys several merits. First, the intra-modal reference features are better aligned, circumventing potential ambiguities that may arise in cross-modal matching. Second, the ranking-based matching process harnesses the structure information implicit in the inter-class relationships, making it more robust than comparing individually. Extensive experiments on three benchmarks demonstrate that RIM outperforms previous state-of-the-art methods by large margins, obtaining a lead of more than 10% in mIoU on PASCAL VOC benchmark.  
  </ol>  
</details>  
**comments**: Accepted to CVPR2024  
  
  



## NeRF  

### [Alpha Invariance: On Inverse Scaling Between Distance and Volume Density in Neural Radiance Fields](http://arxiv.org/abs/2404.02155)  
Joshua Ahn, Haochen Wang, Raymond A. Yeh, Greg Shakhnarovich  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Scale-ambiguity in 3D scene dimensions leads to magnitude-ambiguity of volumetric densities in neural radiance fields, i.e., the densities double when scene size is halved, and vice versa. We call this property alpha invariance. For NeRFs to better maintain alpha invariance, we recommend 1) parameterizing both distance and volume densities in log space, and 2) a discretization-agnostic initialization strategy to guarantee high ray transmittance. We revisit a few popular radiance field models and find that these systems use various heuristics to deal with issues arising from scene scaling. We test their behaviors and show our recipe to be more robust.  
  </ol>  
</details>  
**comments**: CVPR 2024. project page https://pals.ttic.edu/p/alpha-invariance  
  
### [Uncertainty-aware Active Learning of NeRF-based Object Models for Robot Manipulators using Visual and Re-orientation Actions](http://arxiv.org/abs/2404.01812)  
Saptarshi Dasgupta, Akshat Gupta, Shreshth Tuli, Rohan Paul  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Manipulating unseen objects is challenging without a 3D representation, as objects generally have occluded surfaces. This requires physical interaction with objects to build their internal representations. This paper presents an approach that enables a robot to rapidly learn the complete 3D model of a given object for manipulation in unfamiliar orientations. We use an ensemble of partially constructed NeRF models to quantify model uncertainty to determine the next action (a visual or re-orientation action) by optimizing informativeness and feasibility. Further, our approach determines when and how to grasp and re-orient an object given its partial NeRF model and re-estimates the object pose to rectify misalignments introduced during the interaction. Experiments with a simulated Franka Emika Robot Manipulator operating in a tabletop environment with benchmark objects demonstrate an improvement of (i) 14% in visual reconstruction quality (PSNR), (ii) 20% in the geometric/depth reconstruction of the object surface (F-score) and (iii) 71% in the task success rate of manipulating objects a-priori unseen orientations/stable configurations in the scene; over current methods. The project page can be found here: https://actnerf.github.io.  
  </ol>  
</details>  
**comments**: This work has been submitted to the IEEE for possible publication.
  Copyright may be transferred without notice, after which this version may no
  longer be accessible  
  
### [NVINS: Robust Visual Inertial Navigation Fused with NeRF-augmented Camera Pose Regressor and Uncertainty Quantification](http://arxiv.org/abs/2404.01400)  
Juyeop Han, Lukas Lao Beyer, Guilherme V. Cavalheiro, Sertac Karaman  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In recent years, Neural Radiance Fields (NeRF) have emerged as a powerful tool for 3D reconstruction and novel view synthesis. However, the computational cost of NeRF rendering and degradation in quality due to the presence of artifacts pose significant challenges for its application in real-time and robust robotic tasks, especially on embedded systems. This paper introduces a novel framework that integrates NeRF-derived localization information with Visual-Inertial Odometry(VIO) to provide a robust solution for robotic navigation in a real-time. By training an absolute pose regression network with augmented image data rendered from a NeRF and quantifying its uncertainty, our approach effectively counters positional drift and enhances system reliability. We also establish a mathematically sound foundation for combining visual inertial navigation with camera localization neural networks, considering uncertainty under a Bayesian framework. Experimental validation in the photorealistic simulation environment demonstrates significant improvements in accuracy compared to a conventional VIO approach.  
  </ol>  
</details>  
**comments**: 8 pages, 5 figures, 2 tables  
  
### [NeRF-MAE : Masked AutoEncoders for Self Supervised 3D representation Learning for Neural Radiance Fields](http://arxiv.org/abs/2404.01300)  
Muhammad Zubair Irshad, Sergey Zakahrov, Vitor Guizilini, Adrien Gaidon, Zsolt Kira, Rares Ambrus  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural fields excel in computer vision and robotics due to their ability to understand the 3D visual world such as inferring semantics, geometry, and dynamics. Given the capabilities of neural fields in densely representing a 3D scene from 2D images, we ask the question: Can we scale their self-supervised pretraining, specifically using masked autoencoders, to generate effective 3D representations from posed RGB images. Owing to the astounding success of extending transformers to novel data modalities, we employ standard 3D Vision Transformers to suit the unique formulation of NeRFs. We leverage NeRF's volumetric grid as a dense input to the transformer, contrasting it with other 3D representations such as pointclouds where the information density can be uneven, and the representation is irregular. Due to the difficulty of applying masked autoencoders to an implicit representation, such as NeRF, we opt for extracting an explicit representation that canonicalizes scenes across domains by employing the camera trajectory for sampling. Our goal is made possible by masking random patches from NeRF's radiance and density grid and employing a standard 3D Swin Transformer to reconstruct the masked patches. In doing so, the model can learn the semantic and spatial structure of complete scenes. We pretrain this representation at scale on our proposed curated posed-RGB data, totaling over 1.6 million images. Once pretrained, the encoder is used for effective 3D transfer learning. Our novel self-supervised pretraining for NeRFs, NeRF-MAE, scales remarkably well and improves performance on various challenging 3D tasks. Utilizing unlabeled posed 2D data for pretraining, NeRF-MAE significantly outperforms self-supervised 3D pretraining and NeRF scene understanding baselines on Front3D and ScanNet datasets with an absolute performance improvement of over 20% AP50 and 8% AP25 for 3D object detection.  
  </ol>  
</details>  
**comments**: 29 pages, 13 figures. Project Page: https://nerf-mae.github.io/  
  
### [MagicMirror: Fast and High-Quality Avatar Generation with a Constrained Search Space](http://arxiv.org/abs/2404.01296)  
Armand Comas-Massagué, Di Qiu, Menglei Chai, Marcel Bühler, Amit Raj, Ruiqi Gao, Qiangeng Xu, Mark Matthews, Paulo Gotardo, Octavia Camps, Sergio Orts-Escolano, Thabo Beeler  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce a novel framework for 3D human avatar generation and personalization, leveraging text prompts to enhance user engagement and customization. Central to our approach are key innovations aimed at overcoming the challenges in photo-realistic avatar synthesis. Firstly, we utilize a conditional Neural Radiance Fields (NeRF) model, trained on a large-scale unannotated multi-view dataset, to create a versatile initial solution space that accelerates and diversifies avatar generation. Secondly, we develop a geometric prior, leveraging the capabilities of Text-to-Image Diffusion Models, to ensure superior view invariance and enable direct optimization of avatar geometry. These foundational ideas are complemented by our optimization pipeline built on Variational Score Distillation (VSD), which mitigates texture loss and over-saturation issues. As supported by our extensive experiments, these strategies collectively enable the creation of custom avatars with unparalleled visual quality and better adherence to input text prompts. You can find more results and videos in our website: https://syntec-research.github.io/MagicMirror  
  </ol>  
</details>  
  
### [StructLDM: Structured Latent Diffusion for 3D Human Generation](http://arxiv.org/abs/2404.01241)  
Tao Hu, Fangzhou Hong, Ziwei Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent 3D human generative models have achieved remarkable progress by learning 3D-aware GANs from 2D images. However, existing 3D human generative methods model humans in a compact 1D latent space, ignoring the articulated structure and semantics of human body topology. In this paper, we explore more expressive and higher-dimensional latent space for 3D human modeling and propose StructLDM, a diffusion-based unconditional 3D human generative model, which is learned from 2D images. StructLDM solves the challenges imposed due to the high-dimensional growth of latent space with three key designs: 1) A semantic structured latent space defined on the dense surface manifold of a statistical human body template. 2) A structured 3D-aware auto-decoder that factorizes the global latent space into several semantic body parts parameterized by a set of conditional structured local NeRFs anchored to the body template, which embeds the properties learned from the 2D training data and can be decoded to render view-consistent humans under different poses and clothing styles. 3) A structured latent diffusion model for generative human appearance sampling. Extensive experiments validate StructLDM's state-of-the-art generation performance and illustrate the expressiveness of the structured latent space over the well-adopted 1D latent space. Notably, StructLDM enables different levels of controllable 3D human generation and editing, including pose/view/shape control, and high-level tasks including compositional generations, part-aware clothing editing, 3D virtual try-on, etc. Our project page is at: https://taohuumd.github.io/projects/StructLDM/.  
  </ol>  
</details>  
**comments**: Project page: https://taohuumd.github.io/projects/StructLDM/  
  
### [Mirror-3DGS: Incorporating Mirror Reflections into 3D Gaussian Splatting](http://arxiv.org/abs/2404.01168)  
Jiarui Meng, Haijie Li, Yanmin Wu, Qiankun Gao, Shuzhou Yang, Jian Zhang, Siwei Ma  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D Gaussian Splatting (3DGS) has marked a significant breakthrough in the realm of 3D scene reconstruction and novel view synthesis. However, 3DGS, much like its predecessor Neural Radiance Fields (NeRF), struggles to accurately model physical reflections, particularly in mirrors that are ubiquitous in real-world scenes. This oversight mistakenly perceives reflections as separate entities that physically exist, resulting in inaccurate reconstructions and inconsistent reflective properties across varied viewpoints. To address this pivotal challenge, we introduce Mirror-3DGS, an innovative rendering framework devised to master the intricacies of mirror geometries and reflections, paving the way for the generation of realistically depicted mirror reflections. By ingeniously incorporating mirror attributes into the 3DGS and leveraging the principle of plane mirror imaging, Mirror-3DGS crafts a mirrored viewpoint to observe from behind the mirror, enriching the realism of scene renderings. Extensive assessments, spanning both synthetic and real-world scenes, showcase our method's ability to render novel views with enhanced fidelity in real-time, surpassing the state-of-the-art Mirror-NeRF specifically within the challenging mirror regions. Our code will be made publicly available for reproducible research.  
  </ol>  
</details>  
**comments**: 22 pages, 7 figures  
  
### [SGCNeRF: Few-Shot Neural Rendering via Sparse Geometric Consistency Guidance](http://arxiv.org/abs/2404.00992)  
Yuru Xiao, Xianming Liu, Deming Zhai, Kui Jiang, Junjun Jiang, Xiangyang Ji  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Field (NeRF) technology has made significant strides in creating novel viewpoints. However, its effectiveness is hampered when working with sparsely available views, often leading to performance dips due to overfitting. FreeNeRF attempts to overcome this limitation by integrating implicit geometry regularization, which incrementally improves both geometry and textures. Nonetheless, an initial low positional encoding bandwidth results in the exclusion of high-frequency elements. The quest for a holistic approach that simultaneously addresses overfitting and the preservation of high-frequency details remains ongoing. This study introduces a novel feature matching based sparse geometry regularization module. This module excels in pinpointing high-frequency keypoints, thereby safeguarding the integrity of fine details. Through progressive refinement of geometry and textures across NeRF iterations, we unveil an effective few-shot neural rendering architecture, designated as SGCNeRF, for enhanced novel view synthesis. Our experiments demonstrate that SGCNeRF not only achieves superior geometry-consistent outcomes but also surpasses FreeNeRF, with improvements of 0.7 dB and 0.6 dB in PSNR on the LLFF and DTU datasets, respectively.  
  </ol>  
</details>  
  
### [FlexiDreamer: Single Image-to-3D Generation with FlexiCubes](http://arxiv.org/abs/2404.00987)  
Ruowen Zhao, Zhengyi Wang, Yikai Wang, Zihan Zhou, Jun Zhu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D content generation from text prompts or single images has made remarkable progress in quality and speed recently. One of its dominant paradigms involves generating consistent multi-view images followed by a sparse-view reconstruction. However, due to the challenge of directly deforming the mesh representation to approach the target topology, most methodologies learn an implicit representation (such as NeRF) during the sparse-view reconstruction and acquire the target mesh by a post-processing extraction. Although the implicit representation can effectively model rich 3D information, its training typically entails a long convergence time. In addition, the post-extraction operation from the implicit field also leads to undesirable visual artifacts. In this paper, we propose FlexiDreamer, a novel single image-to-3d generation framework that reconstructs the target mesh in an end-to-end manner. By leveraging a flexible gradient-based extraction known as FlexiCubes, our method circumvents the defects brought by the post-processing and facilitates a direct acquisition of the target mesh. Furthermore, we incorporate a multi-resolution hash grid encoding scheme that progressively activates the encoding levels into the implicit field in FlexiCubes to help capture geometric details for per-step optimization. Notably, FlexiDreamer recovers a dense 3D structure from a single-view image in approximately 1 minute on a single NVIDIA A100 GPU, outperforming previous methodologies by a large margin.  
  </ol>  
</details>  
**comments**: project page:https://flexidreamer.github.io  
  
### [Marrying NeRF with Feature Matching for One-step Pose Estimation](http://arxiv.org/abs/2404.00891)  
Ronghan Chen, Yang Cong, Yu Ren  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Given the image collection of an object, we aim at building a real-time image-based pose estimation method, which requires neither its CAD model nor hours of object-specific training. Recent NeRF-based methods provide a promising solution by directly optimizing the pose from pixel loss between rendered and target images. However, during inference, they require long converging time, and suffer from local minima, making them impractical for real-time robot applications. We aim at solving this problem by marrying image matching with NeRF. With 2D matches and depth rendered by NeRF, we directly solve the pose in one step by building 2D-3D correspondences between target and initial view, thus allowing for real-time prediction. Moreover, to improve the accuracy of 2D-3D correspondences, we propose a 3D consistent point mining strategy, which effectively discards unfaithful points reconstruted by NeRF. Moreover, current NeRF-based methods naively optimizing pixel loss fail at occluded images. Thus, we further propose a 2D matches based sampling strategy to preclude the occluded area. Experimental results on representative datasets prove that our method outperforms state-of-the-art methods, and improves inference efficiency by 90x, achieving real-time prediction at 6 FPS.  
  </ol>  
</details>  
**comments**: ICRA, 2024. Video https://www.youtube.com/watch?v=70fgUobOFWo  
  
  



