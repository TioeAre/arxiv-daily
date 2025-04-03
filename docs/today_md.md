<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#ForestVO:-Enhancing-Visual-Odometry-in-Forest-Environments-through-ForestGlue>ForestVO: Enhancing Visual Odometry in Forest Environments through ForestGlue</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#FIORD:-A-Fisheye-Indoor-Outdoor-Dataset-with-LIDAR-Ground-Truth-for-3D-Scene-Reconstruction-and-Benchmarking>FIORD: A Fisheye Indoor-Outdoor Dataset with LIDAR Ground Truth for 3D Scene Reconstruction and Benchmarking</a></li>
        <li><a href=#LITA-GS:-Illumination-Agnostic-Novel-View-Synthesis-via-Reference-Free-3D-Gaussian-Splatting-and-Physical-Priors>LITA-GS: Illumination-Agnostic Novel View Synthesis via Reference-Free 3D Gaussian Splatting and Physical Priors</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Prompt-Guided-Attention-Head-Selection-for-Focus-Oriented-Image-Retrieval>Prompt-Guided Attention Head Selection for Focus-Oriented Image Retrieval</a></li>
        <li><a href=#IDMR:-Towards-Instance-Driven-Precise-Visual-Correspondence-in-Multimodal-Retrieval>IDMR: Towards Instance-Driven Precise Visual Correspondence in Multimodal Retrieval</a></li>
        <li><a href=#Scaling-Prompt-Instructed-Zero-Shot-Composed-Image-Retrieval-with-Image-Only-Data>Scaling Prompt Instructed Zero Shot Composed Image Retrieval with Image-Only Data</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#SuperEvent:-Cross-Modal-Learning-of-Event-based-Keypoint-Detection>SuperEvent: Cross-Modal Learning of Event-based Keypoint Detection</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Scaling-Prompt-Instructed-Zero-Shot-Composed-Image-Retrieval-with-Image-Only-Data>Scaling Prompt Instructed Zero Shot Composed Image Retrieval with Image-Only Data</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Diffusion-Guided-Gaussian-Splatting-for-Large-Scale-Unconstrained-3D-Reconstruction-and-Novel-View-Synthesis>Diffusion-Guided Gaussian Splatting for Large-Scale Unconstrained 3D Reconstruction and Novel View Synthesis</a></li>
        <li><a href=#BOGausS:-Better-Optimized-Gaussian-Splatting>BOGausS: Better Optimized Gaussian Splatting</a></li>
        <li><a href=#FIORD:-A-Fisheye-Indoor-Outdoor-Dataset-with-LIDAR-Ground-Truth-for-3D-Scene-Reconstruction-and-Benchmarking>FIORD: A Fisheye Indoor-Outdoor Dataset with LIDAR Ground Truth for 3D Scene Reconstruction and Benchmarking</a></li>
        <li><a href=#RealityAvatar:-Towards-Realistic-Loose-Clothing-Modeling-in-Animatable-3D-Gaussian-Avatars>RealityAvatar: Towards Realistic Loose Clothing Modeling in Animatable 3D Gaussian Avatars</a></li>
        <li><a href=#Luminance-GS:-Adapting-3D-Gaussian-Splatting-to-Challenging-Lighting-Conditions-with-View-Adaptive-Curve-Adjustment>Luminance-GS: Adapting 3D Gaussian Splatting to Challenging Lighting Conditions with View-Adaptive Curve Adjustment</a></li>
        <li><a href=#Neural-Pruning-for-3D-Scene-Reconstruction:-Efficient-NeRF-Acceleration>Neural Pruning for 3D Scene Reconstruction: Efficient NeRF Acceleration</a></li>
        <li><a href=#NeuRadar:-Neural-Radiance-Fields-for-Automotive-Radar-Point-Clouds>NeuRadar: Neural Radiance Fields for Automotive Radar Point Clouds</a></li>
        <li><a href=#NeRF-Based-defect-detection>NeRF-Based defect detection</a></li>
        <li><a href=#LITA-GS:-Illumination-Agnostic-Novel-View-Synthesis-via-Reference-Free-3D-Gaussian-Splatting-and-Physical-Priors>LITA-GS: Illumination-Agnostic Novel View Synthesis via Reference-Free 3D Gaussian Splatting and Physical Priors</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [ForestVO: Enhancing Visual Odometry in Forest Environments through ForestGlue](http://arxiv.org/abs/2504.01261)  
Thomas Pritchard, Saifullah Ijaz, Ronald Clark, Basaran Bahadir Kocer  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in visual odometry systems have improved autonomous navigation; however, challenges persist in complex environments like forests, where dense foliage, variable lighting, and repetitive textures compromise feature correspondence accuracy. To address these challenges, we introduce ForestGlue, enhancing the SuperPoint feature detector through four configurations - grayscale, RGB, RGB-D, and stereo-vision - optimised for various sensing modalities. For feature matching, we employ LightGlue or SuperGlue, retrained with synthetic forest data. ForestGlue achieves comparable pose estimation accuracy to baseline models but requires only 512 keypoints - just 25% of the baseline's 2048 - to reach an LO-RANSAC AUC score of 0.745 at a 10{\deg} threshold. With only a quarter of keypoints needed, ForestGlue significantly reduces computational overhead, demonstrating effectiveness in dynamic forest environments, and making it suitable for real-time deployment on resource-constrained platforms. By combining ForestGlue with a transformer-based pose estimation model, we propose ForestVO, which estimates relative camera poses using matched 2D pixel coordinates between frames. On challenging TartanAir forest sequences, ForestVO achieves an average relative pose error (RPE) of 1.09 m and a kitti_score of 2.33%, outperforming direct-based methods like DSO by 40% in dynamic scenes. Despite using only 10% of the dataset for training, ForestVO maintains competitive performance with TartanVO while being a significantly lighter model. This work establishes an end-to-end deep learning pipeline specifically tailored for visual odometry in forested environments, leveraging forest-specific training data to optimise feature correspondence and pose estimation, thereby enhancing the accuracy and robustness of autonomous navigation systems.  
  </ol>  
</details>  
**comments**: Accepted to the IEEE Robotics and Automation Letters  
  
  



## SFM  

### [FIORD: A Fisheye Indoor-Outdoor Dataset with LIDAR Ground Truth for 3D Scene Reconstruction and Benchmarking](http://arxiv.org/abs/2504.01732)  
Ulas Gunes, Matias Turkulainen, Xuqian Ren, Arno Solin, Juho Kannala, Esa Rahtu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The development of large-scale 3D scene reconstruction and novel view synthesis methods mostly rely on datasets comprising perspective images with narrow fields of view (FoV). While effective for small-scale scenes, these datasets require large image sets and extensive structure-from-motion (SfM) processing, limiting scalability. To address this, we introduce a fisheye image dataset tailored for scene reconstruction tasks. Using dual 200-degree fisheye lenses, our dataset provides full 360-degree coverage of 5 indoor and 5 outdoor scenes. Each scene has sparse SfM point clouds and precise LIDAR-derived dense point clouds that can be used as geometric ground-truth, enabling robust benchmarking under challenging conditions such as occlusions and reflections. While the baseline experiments focus on vanilla Gaussian Splatting and NeRF based Nerfacto methods, the dataset supports diverse approaches for scene reconstruction, novel view synthesis, and image-based rendering.  
  </ol>  
</details>  
**comments**: SCIA 2025  
  
### [LITA-GS: Illumination-Agnostic Novel View Synthesis via Reference-Free 3D Gaussian Splatting and Physical Priors](http://arxiv.org/abs/2504.00219)  
Han Zhou, Wei Dong, Jun Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Directly employing 3D Gaussian Splatting (3DGS) on images with adverse illumination conditions exhibits considerable difficulty in achieving high-quality, normally-exposed representations due to: (1) The limited Structure from Motion (SfM) points estimated in adverse illumination scenarios fail to capture sufficient scene details; (2) Without ground-truth references, the intensive information loss, significant noise, and color distortion pose substantial challenges for 3DGS to produce high-quality results; (3) Combining existing exposure correction methods with 3DGS does not achieve satisfactory performance due to their individual enhancement processes, which lead to the illumination inconsistency between enhanced images from different viewpoints. To address these issues, we propose LITA-GS, a novel illumination-agnostic novel view synthesis method via reference-free 3DGS and physical priors. Firstly, we introduce an illumination-invariant physical prior extraction pipeline. Secondly, based on the extracted robust spatial structure prior, we develop the lighting-agnostic structure rendering strategy, which facilitates the optimization of the scene structure and object appearance. Moreover, a progressive denoising module is introduced to effectively mitigate the noise within the light-invariant representation. We adopt the unsupervised strategy for the training of LITA-GS and extensive experiments demonstrate that LITA-GS surpasses the state-of-the-art (SOTA) NeRF-based method while enjoying faster inference speed and costing reduced training time. The code is released at https://github.com/LowLevelAI/LITA-GS.  
  </ol>  
</details>  
**comments**: Accepted by CVPR 2025. 3DGS, Adverse illumination conditions,
  Reference-free, Physical priors  
  
  



## Visual Localization  

### [Prompt-Guided Attention Head Selection for Focus-Oriented Image Retrieval](http://arxiv.org/abs/2504.01348)  
Yuji Nozawa, Yu-Chieh Lin, Kazumoto Nakamura, Youyang Ng  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The goal of this paper is to enhance pretrained Vision Transformer (ViT) models for focus-oriented image retrieval with visual prompting. In real-world image retrieval scenarios, both query and database images often exhibit complexity, with multiple objects and intricate backgrounds. Users often want to retrieve images with specific object, which we define as the Focus-Oriented Image Retrieval (FOIR) task. While a standard image encoder can be employed to extract image features for similarity matching, it may not perform optimally in the multi-object-based FOIR task. This is because each image is represented by a single global feature vector. To overcome this, a prompt-based image retrieval solution is required. We propose an approach called Prompt-guided attention Head Selection (PHS) to leverage the head-wise potential of the multi-head attention mechanism in ViT in a promptable manner. PHS selects specific attention heads by matching their attention maps with user's visual prompts, such as a point, box, or segmentation. This empowers the model to focus on specific object of interest while preserving the surrounding visual context. Notably, PHS does not necessitate model re-training and avoids any image alteration. Experimental results show that PHS substantially improves performance on multiple datasets, offering a practical and training-free solution to enhance model performance in the FOIR task.  
  </ol>  
</details>  
**comments**: Accepted to CVPR 2025 PixFoundation Workshop  
  
### [IDMR: Towards Instance-Driven Precise Visual Correspondence in Multimodal Retrieval](http://arxiv.org/abs/2504.00954)  
Bangwei Liu, Yicheng Bao, Shaohui Lin, Xuhong Wang, Xin Tan, Yingchun Wang, Yuan Xie, Chaochao Lu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Multimodal retrieval systems are becoming increasingly vital for cutting-edge AI technologies, such as embodied AI and AI-driven digital content industries. However, current multimodal retrieval tasks lack sufficient complexity and demonstrate limited practical application value. It spires us to design Instance-Driven Multimodal Image Retrieval (IDMR), a novel task that requires models to retrieve images containing the same instance as a query image while matching a text-described scenario. Unlike existing retrieval tasks focused on global image similarity or category-level matching, IDMR demands fine-grained instance-level consistency across diverse contexts. To benchmark this capability, we develop IDMR-bench using real-world object tracking and first-person video data. Addressing the scarcity of training data, we propose a cross-domain synthesis method that creates 557K training samples by cropping objects from standard detection datasets. Our Multimodal Large Language Model (MLLM) based retrieval model, trained on 1.2M samples, outperforms state-of-the-art approaches on both traditional benchmarks and our zero-shot IDMR-bench. Experimental results demonstrate previous models' limitations in instance-aware retrieval and highlight the potential of MLLM for advanced retrieval applications. The whole training dataset, codes and models, with wide ranges of sizes, are available at https://github.com/BwLiu01/IDMR.  
  </ol>  
</details>  
  
### [Scaling Prompt Instructed Zero Shot Composed Image Retrieval with Image-Only Data](http://arxiv.org/abs/2504.00812)  
Yiqun Duan, Sameera Ramasinghe, Stephen Gould, Ajanthan Thalaiyasingam  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed Image Retrieval (CIR) is the task of retrieving images matching a reference image augmented with a text, where the text describes changes to the reference image in natural language. Traditionally, models designed for CIR have relied on triplet data containing a reference image, reformulation text, and a target image. However, curating such triplet data often necessitates human intervention, leading to prohibitive costs. This challenge has hindered the scalability of CIR model training even with the availability of abundant unlabeled data. With the recent advances in foundational models, we advocate a shift in the CIR training paradigm where human annotations can be efficiently replaced by large language models (LLMs). Specifically, we demonstrate the capability of large captioning and language models in efficiently generating data for CIR only relying on unannotated image collections. Additionally, we introduce an embedding reformulation architecture that effectively combines image and text modalities. Our model, named InstructCIR, outperforms state-of-the-art methods in zero-shot composed image retrieval on CIRR and FashionIQ datasets. Furthermore, we demonstrate that by increasing the amount of generated data, our zero-shot model gets closer to the performance of supervised baselines.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [SuperEvent: Cross-Modal Learning of Event-based Keypoint Detection](http://arxiv.org/abs/2504.00139)  
Yannick Burkhardt, Simon Schaefer, Stefan Leutenegger  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Event-based keypoint detection and matching holds significant potential, enabling the integration of event sensors into highly optimized Visual SLAM systems developed for frame cameras over decades of research. Unfortunately, existing approaches struggle with the motion-dependent appearance of keypoints and the complex noise prevalent in event streams, resulting in severely limited feature matching capabilities and poor performance on downstream tasks. To mitigate this problem, we propose SuperEvent, a data-driven approach to predict stable keypoints with expressive descriptors. Due to the absence of event datasets with ground truth keypoint labels, we leverage existing frame-based keypoint detectors on readily available event-aligned and synchronized gray-scale frames for self-supervision: we generate temporally sparse keypoint pseudo-labels considering that events are a product of both scene appearance and camera motion. Combined with our novel, information-rich event representation, we enable SuperEvent to effectively learn robust keypoint detection and description in event streams. Finally, we demonstrate the usefulness of SuperEvent by its integration into a modern sparse keypoint and descriptor-based SLAM framework originally developed for traditional cameras, surpassing the state-of-the-art in event-based SLAM by a wide margin. Source code and multimedia material are available at smartroboticslab.github.io/SuperEvent.  
  </ol>  
</details>  
**comments**: In Review for ICCV25  
  
  



## Image Matching  

### [Scaling Prompt Instructed Zero Shot Composed Image Retrieval with Image-Only Data](http://arxiv.org/abs/2504.00812)  
Yiqun Duan, Sameera Ramasinghe, Stephen Gould, Ajanthan Thalaiyasingam  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed Image Retrieval (CIR) is the task of retrieving images matching a reference image augmented with a text, where the text describes changes to the reference image in natural language. Traditionally, models designed for CIR have relied on triplet data containing a reference image, reformulation text, and a target image. However, curating such triplet data often necessitates human intervention, leading to prohibitive costs. This challenge has hindered the scalability of CIR model training even with the availability of abundant unlabeled data. With the recent advances in foundational models, we advocate a shift in the CIR training paradigm where human annotations can be efficiently replaced by large language models (LLMs). Specifically, we demonstrate the capability of large captioning and language models in efficiently generating data for CIR only relying on unannotated image collections. Additionally, we introduce an embedding reformulation architecture that effectively combines image and text modalities. Our model, named InstructCIR, outperforms state-of-the-art methods in zero-shot composed image retrieval on CIRR and FashionIQ datasets. Furthermore, we demonstrate that by increasing the amount of generated data, our zero-shot model gets closer to the performance of supervised baselines.  
  </ol>  
</details>  
  
  



## NeRF  

### [Diffusion-Guided Gaussian Splatting for Large-Scale Unconstrained 3D Reconstruction and Novel View Synthesis](http://arxiv.org/abs/2504.01960)  
Niluthpol Chowdhury Mithun, Tuan Pham, Qiao Wang, Ben Southall, Kshitij Minhas, Bogdan Matei, Stephan Mandt, Supun Samarasekera, Rakesh Kumar  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in 3D Gaussian Splatting (3DGS) and Neural Radiance Fields (NeRF) have achieved impressive results in real-time 3D reconstruction and novel view synthesis. However, these methods struggle in large-scale, unconstrained environments where sparse and uneven input coverage, transient occlusions, appearance variability, and inconsistent camera settings lead to degraded quality. We propose GS-Diff, a novel 3DGS framework guided by a multi-view diffusion model to address these limitations. By generating pseudo-observations conditioned on multi-view inputs, our method transforms under-constrained 3D reconstruction problems into well-posed ones, enabling robust optimization even with sparse data. GS-Diff further integrates several enhancements, including appearance embedding, monocular depth priors, dynamic object modeling, anisotropy regularization, and advanced rasterization techniques, to tackle geometric and photometric challenges in real-world settings. Experiments on four benchmarks demonstrate that GS-Diff consistently outperforms state-of-the-art baselines by significant margins.  
  </ol>  
</details>  
**comments**: WACV ULTRRA Workshop 2025  
  
### [BOGausS: Better Optimized Gaussian Splatting](http://arxiv.org/abs/2504.01844)  
Stéphane Pateux, Matthieu Gendrin, Luce Morin, Théo Ladune, Xiaoran Jiang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D Gaussian Splatting (3DGS) proposes an efficient solution for novel view synthesis. Its framework provides fast and high-fidelity rendering. Although less complex than other solutions such as Neural Radiance Fields (NeRF), there are still some challenges building smaller models without sacrificing quality. In this study, we perform a careful analysis of 3DGS training process and propose a new optimization methodology. Our Better Optimized Gaussian Splatting (BOGausS) solution is able to generate models up to ten times lighter than the original 3DGS with no quality degradation, thus significantly boosting the performance of Gaussian Splatting compared to the state of the art.  
  </ol>  
</details>  
  
### [FIORD: A Fisheye Indoor-Outdoor Dataset with LIDAR Ground Truth for 3D Scene Reconstruction and Benchmarking](http://arxiv.org/abs/2504.01732)  
Ulas Gunes, Matias Turkulainen, Xuqian Ren, Arno Solin, Juho Kannala, Esa Rahtu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The development of large-scale 3D scene reconstruction and novel view synthesis methods mostly rely on datasets comprising perspective images with narrow fields of view (FoV). While effective for small-scale scenes, these datasets require large image sets and extensive structure-from-motion (SfM) processing, limiting scalability. To address this, we introduce a fisheye image dataset tailored for scene reconstruction tasks. Using dual 200-degree fisheye lenses, our dataset provides full 360-degree coverage of 5 indoor and 5 outdoor scenes. Each scene has sparse SfM point clouds and precise LIDAR-derived dense point clouds that can be used as geometric ground-truth, enabling robust benchmarking under challenging conditions such as occlusions and reflections. While the baseline experiments focus on vanilla Gaussian Splatting and NeRF based Nerfacto methods, the dataset supports diverse approaches for scene reconstruction, novel view synthesis, and image-based rendering.  
  </ol>  
</details>  
**comments**: SCIA 2025  
  
### [RealityAvatar: Towards Realistic Loose Clothing Modeling in Animatable 3D Gaussian Avatars](http://arxiv.org/abs/2504.01559)  
Yahui Li, Zhi Zeng, Liming Pang, Guixuan Zhang, Shuwu Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Modeling animatable human avatars from monocular or multi-view videos has been widely studied, with recent approaches leveraging neural radiance fields (NeRFs) or 3D Gaussian Splatting (3DGS) achieving impressive results in novel-view and novel-pose synthesis. However, existing methods often struggle to accurately capture the dynamics of loose clothing, as they primarily rely on global pose conditioning or static per-frame representations, leading to oversmoothing and temporal inconsistencies in non-rigid regions. To address this, We propose RealityAvatar, an efficient framework for high-fidelity digital human modeling, specifically targeting loosely dressed avatars. Our method leverages 3D Gaussian Splatting to capture complex clothing deformations and motion dynamics while ensuring geometric consistency. By incorporating a motion trend module and a latentbone encoder, we explicitly model pose-dependent deformations and temporal variations in clothing behavior. Extensive experiments on benchmark datasets demonstrate the effectiveness of our approach in capturing fine-grained clothing deformations and motion-driven shape variations. Our method significantly enhances structural fidelity and perceptual quality in dynamic human reconstruction, particularly in non-rigid regions, while achieving better consistency across temporal frames.  
  </ol>  
</details>  
  
### [Luminance-GS: Adapting 3D Gaussian Splatting to Challenging Lighting Conditions with View-Adaptive Curve Adjustment](http://arxiv.org/abs/2504.01503)  
Ziteng Cui, Xuangeng Chu, Tatsuya Harada  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Capturing high-quality photographs under diverse real-world lighting conditions is challenging, as both natural lighting (e.g., low-light) and camera exposure settings (e.g., exposure time) significantly impact image quality. This challenge becomes more pronounced in multi-view scenarios, where variations in lighting and image signal processor (ISP) settings across viewpoints introduce photometric inconsistencies. Such lighting degradations and view-dependent variations pose substantial challenges to novel view synthesis (NVS) frameworks based on Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). To address this, we introduce Luminance-GS, a novel approach to achieving high-quality novel view synthesis results under diverse challenging lighting conditions using 3DGS. By adopting per-view color matrix mapping and view-adaptive curve adjustments, Luminance-GS achieves state-of-the-art (SOTA) results across various lighting conditions -- including low-light, overexposure, and varying exposure -- while not altering the original 3DGS explicit representation. Compared to previous NeRF- and 3DGS-based baselines, Luminance-GS provides real-time rendering speed with improved reconstruction quality.  
  </ol>  
</details>  
**comments**: CVPR 2025, project page:
  https://cuiziteng.github.io/Luminance_GS_web/  
  
### [Neural Pruning for 3D Scene Reconstruction: Efficient NeRF Acceleration](http://arxiv.org/abs/2504.00950)  
Tianqi Ding, Dawei Xiang, Pablo Rivas, Liang Dong  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) have become a popular 3D reconstruction approach in recent years. While they produce high-quality results, they also demand lengthy training times, often spanning days. This paper studies neural pruning as a strategy to address these concerns. We compare pruning approaches, including uniform sampling, importance-based methods, and coreset-based techniques, to reduce the model size and speed up training. Our findings show that coreset-driven pruning can achieve a 50% reduction in model size and a 35% speedup in training, with only a slight decrease in accuracy. These results suggest that pruning can be an effective method for improving the efficiency of NeRF models in resource-limited settings.  
  </ol>  
</details>  
**comments**: 12 pages, 4 figures, accepted by International Conference on the AI
  Revolution: Research, Ethics, and Society (AIR-RES 2025)  
  
### [NeuRadar: Neural Radiance Fields for Automotive Radar Point Clouds](http://arxiv.org/abs/2504.00859)  
Mahan Rafidashti, Ji Lan, Maryam Fatemi, Junsheng Fu, Lars Hammarstrand, Lennart Svensson  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Radar is an important sensor for autonomous driving (AD) systems due to its robustness to adverse weather and different lighting conditions. Novel view synthesis using neural radiance fields (NeRFs) has recently received considerable attention in AD due to its potential to enable efficient testing and validation but remains unexplored for radar point clouds. In this paper, we present NeuRadar, a NeRF-based model that jointly generates radar point clouds, camera images, and lidar point clouds. We explore set-based object detection methods such as DETR, and propose an encoder-based solution grounded in the NeRF geometry for improved generalizability. We propose both a deterministic and a probabilistic point cloud representation to accurately model the radar behavior, with the latter being able to capture radar's stochastic behavior. We achieve realistic reconstruction results for two automotive datasets, establishing a baseline for NeRF-based radar point cloud simulation models. In addition, we release radar data for ZOD's Sequences and Drives to enable further research in this field. To encourage further development of radar NeRFs, we release the source code for NeuRadar.  
  </ol>  
</details>  
  
### [NeRF-Based defect detection](http://arxiv.org/abs/2504.00270)  
Tianqi, Ding, Dawei Xiang, Yijiashun Qi, Ze Yang, Zunduo Zhao, Tianyao Sun, Pengbin Feng, Haoyu Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The rapid growth of industrial automation has highlighted the need for precise and efficient defect detection in large-scale machinery. Traditional inspection techniques, involving manual procedures such as scaling tall structures for visual evaluation, are labor-intensive, subjective, and often hazardous. To overcome these challenges, this paper introduces an automated defect detection framework built on Neural Radiance Fields (NeRF) and the concept of digital twins. The system utilizes UAVs to capture images and reconstruct 3D models of machinery, producing both a standard reference model and a current-state model for comparison. Alignment of the models is achieved through the Iterative Closest Point (ICP) algorithm, enabling precise point cloud analysis to detect deviations that signify potential defects. By eliminating manual inspection, this method improves accuracy, enhances operational safety, and offers a scalable solution for defect detection. The proposed approach demonstrates great promise for reliable and efficient industrial applications.  
  </ol>  
</details>  
**comments**: 6 pages, 11 figures, 2025 2nd International Conference on Remote
  Sensing, Mapping and Image Processing (RSMIP 2025)  
  
### [LITA-GS: Illumination-Agnostic Novel View Synthesis via Reference-Free 3D Gaussian Splatting and Physical Priors](http://arxiv.org/abs/2504.00219)  
Han Zhou, Wei Dong, Jun Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Directly employing 3D Gaussian Splatting (3DGS) on images with adverse illumination conditions exhibits considerable difficulty in achieving high-quality, normally-exposed representations due to: (1) The limited Structure from Motion (SfM) points estimated in adverse illumination scenarios fail to capture sufficient scene details; (2) Without ground-truth references, the intensive information loss, significant noise, and color distortion pose substantial challenges for 3DGS to produce high-quality results; (3) Combining existing exposure correction methods with 3DGS does not achieve satisfactory performance due to their individual enhancement processes, which lead to the illumination inconsistency between enhanced images from different viewpoints. To address these issues, we propose LITA-GS, a novel illumination-agnostic novel view synthesis method via reference-free 3DGS and physical priors. Firstly, we introduce an illumination-invariant physical prior extraction pipeline. Secondly, based on the extracted robust spatial structure prior, we develop the lighting-agnostic structure rendering strategy, which facilitates the optimization of the scene structure and object appearance. Moreover, a progressive denoising module is introduced to effectively mitigate the noise within the light-invariant representation. We adopt the unsupervised strategy for the training of LITA-GS and extensive experiments demonstrate that LITA-GS surpasses the state-of-the-art (SOTA) NeRF-based method while enjoying faster inference speed and costing reduced training time. The code is released at https://github.com/LowLevelAI/LITA-GS.  
  </ol>  
</details>  
**comments**: Accepted by CVPR 2025. 3DGS, Adverse illumination conditions,
  Reference-free, Physical priors  
  
  



