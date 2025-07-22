<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#DiffPF:-Differentiable-Particle-Filtering-with-Generative-Sampling-via-Conditional-Diffusion-Models>DiffPF: Differentiable Particle Filtering with Generative Sampling via Conditional Diffusion Models</a></li>
        <li><a href=#Dense-depth-map-guided-deep-Lidar-Visual-Odometry-with-Sparse-Point-Clouds-and-Images>Dense-depth map guided deep Lidar-Visual Odometry with Sparse Point Clouds and Images</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Hi^2-GSLoc:-Dual-Hierarchical-Gaussian-Specific-Visual-Relocalization-for-Remote-Sensing>Hi^2-GSLoc: Dual-Hierarchical Gaussian-Specific Visual Relocalization for Remote Sensing</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#LoopNet:-A-Multitasking-Few-Shot-Learning-Approach-for-Loop-Closure-in-Large-Scale-SLAM>LoopNet: A Multitasking Few-Shot Learning Approach for Loop Closure in Large Scale SLAM</a></li>
        <li><a href=#Visual-Place-Recognition-for-Large-Scale-UAV-Applications>Visual Place Recognition for Large-Scale UAV Applications</a></li>
        <li><a href=#U-MARVEL:-Unveiling-Key-Factors-for-Universal-Multimodal-Retrieval-via-Embedding-Learning-with-MLLMs>U-MARVEL: Unveiling Key Factors for Universal Multimodal Retrieval via Embedding Learning with MLLMs</a></li>
        <li><a href=#OptiCorNet:-Optimizing-Sequence-Based-Context-Correlation-for-Visual-Place-Recognition>OptiCorNet: Optimizing Sequence-Based Context Correlation for Visual Place Recognition</a></li>
        <li><a href=#Developing-an-AI-Guided-Assistant-Device-for-the-Deaf-and-Hearing-Impaired>Developing an AI-Guided Assistant Device for the Deaf and Hearing Impaired</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#DiSCO-3D-:-Discovering-and-segmenting-Sub-Concepts-from-Open-vocabulary-queries-in-NeRF>DiSCO-3D : Discovering and segmenting Sub-Concepts from Open-vocabulary queries in NeRF</a></li>
        <li><a href=#Advances-in-Feed-Forward-3D-Reconstruction-and-View-Synthesis:-A-Survey>Advances in Feed-Forward 3D Reconstruction and View Synthesis: A Survey</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [DiffPF: Differentiable Particle Filtering with Generative Sampling via Conditional Diffusion Models](http://arxiv.org/abs/2507.15716)  
Ziyu Wan, Lin Zhao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper proposes DiffPF, a differentiable particle filter that leverages diffusion models for state estimation in dynamic systems. Unlike conventional differentiable particle filters, which require importance weighting and typically rely on predefined or low-capacity proposal distributions. DiffPF learns a flexible posterior sampler by conditioning a diffusion model on predicted particles and the current observation. This enables accurate, equally-weighted sampling from complex, high-dimensional, and multimodal filtering distributions. We evaluate DiffPF across a range of scenarios, including both unimodal and highly multimodal distributions, and test it on simulated as well as real-world tasks, where it consistently outperforms existing filtering baselines. In particular, DiffPF achieves an 82.8% improvement in estimation accuracy on a highly multimodal global localization benchmark, and a 26% improvement on the real-world KITTI visual odometry benchmark, compared to state-of-the-art differentiable filters. To the best of our knowledge, DiffPF is the first method to integrate conditional diffusion models into particle filtering, enabling high-quality posterior sampling that produces more informative particles and significantly improves state estimation.  
  </ol>  
</details>  
  
### [Dense-depth map guided deep Lidar-Visual Odometry with Sparse Point Clouds and Images](http://arxiv.org/abs/2507.15496)  
JunYing Huang, Ao Xu, DongSun Yong, KeRen Li, YuanFeng Wang, Qi Qin  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Odometry is a critical task for autonomous systems for self-localization and navigation. We propose a novel LiDAR-Visual odometry framework that integrates LiDAR point clouds and images for accurate and robust pose estimation. Our method utilizes a dense-depth map estimated from point clouds and images through depth completion, and incorporates a multi-scale feature extraction network with attention mechanisms, enabling adaptive depth-aware representations. Furthermore, we leverage dense depth information to refine flow estimation and mitigate errors in occlusion-prone regions. Our hierarchical pose refinement module optimizes motion estimation progressively, ensuring robust predictions against dynamic environments and scale ambiguities. Comprehensive experiments on the KITTI odometry benchmark demonstrate that our approach achieves similar or superior accuracy and robustness compared to state-of-the-art visual and LiDAR odometry methods.  
  </ol>  
</details>  
  
  



## SFM  

### [Hi^2-GSLoc: Dual-Hierarchical Gaussian-Specific Visual Relocalization for Remote Sensing](http://arxiv.org/abs/2507.15683)  
Boni Hu, Zhenyu Xia, Lin Chen, Pengcheng Han, Shuhui Bu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual relocalization, which estimates the 6-degree-of-freedom (6-DoF) camera pose from query images, is fundamental to remote sensing and UAV applications. Existing methods face inherent trade-offs: image-based retrieval and pose regression approaches lack precision, while structure-based methods that register queries to Structure-from-Motion (SfM) models suffer from computational complexity and limited scalability. These challenges are particularly pronounced in remote sensing scenarios due to large-scale scenes, high altitude variations, and domain gaps of existing visual priors. To overcome these limitations, we leverage 3D Gaussian Splatting (3DGS) as a novel scene representation that compactly encodes both 3D geometry and appearance. We introduce $\mathrm{Hi}^2$ -GSLoc, a dual-hierarchical relocalization framework that follows a sparse-to-dense and coarse-to-fine paradigm, fully exploiting the rich semantic information and geometric constraints inherent in Gaussian primitives. To handle large-scale remote sensing scenarios, we incorporate partitioned Gaussian training, GPU-accelerated parallel matching, and dynamic memory management strategies. Our approach consists of two stages: (1) a sparse stage featuring a Gaussian-specific consistent render-aware sampling strategy and landmark-guided detector for robust and accurate initial pose estimation, and (2) a dense stage that iteratively refines poses through coarse-to-fine dense rasterization matching while incorporating reliability verification. Through comprehensive evaluation on simulation data, public datasets, and real flight experiments, we demonstrate that our method delivers competitive localization accuracy, recall rate, and computational efficiency while effectively filtering unreliable pose estimates. The results confirm the effectiveness of our approach for practical remote sensing applications.  
  </ol>  
</details>  
**comments**: 17 pages, 11 figures  
  
  



## Visual Localization  

### [LoopNet: A Multitasking Few-Shot Learning Approach for Loop Closure in Large Scale SLAM](http://arxiv.org/abs/2507.15109)  
Mohammad-Maher Nakshbandi, Ziad Sharawy, Sorin Grigorescu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    One of the main challenges in the Simultaneous Localization and Mapping (SLAM) loop closure problem is the recognition of previously visited places. In this work, we tackle the two main problems of real-time SLAM systems: 1) loop closure detection accuracy and 2) real-time computation constraints on the embedded hardware. Our LoopNet method is based on a multitasking variant of the classical ResNet architecture, adapted for online retraining on a dynamic visual dataset and optimized for embedded devices. The online retraining is designed using a few-shot learning approach. The architecture provides both an index into the queried visual dataset, and a measurement of the prediction quality. Moreover, by leveraging DISK (DIStinctive Keypoints) descriptors, LoopNet surpasses the limitations of handcrafted features and traditional deep learning methods, offering better performance under varying conditions. Code is available at https://github.com/RovisLab/LoopNet. Additinally, we introduce a new loop closure benchmarking dataset, coined LoopDB, which is available at https://github.com/RovisLab/LoopDB.  
  </ol>  
</details>  
  
### [Visual Place Recognition for Large-Scale UAV Applications](http://arxiv.org/abs/2507.15089)  
Ioannis Tsampikos Papapetros, Ioannis Kansizoglou, Antonios Gasteratos  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Place Recognition (vPR) plays a crucial role in Unmanned Aerial Vehicle (UAV) navigation, enabling robust localization across diverse environments. Despite significant advancements, aerial vPR faces unique challenges due to the limited availability of large-scale, high-altitude datasets, which limits model generalization, along with the inherent rotational ambiguity in UAV imagery. To address these challenges, we introduce LASED, a large-scale aerial dataset with approximately one million images, systematically sampled from 170,000 unique locations throughout Estonia over a decade, offering extensive geographic and temporal diversity. Its structured design ensures clear place separation significantly enhancing model training for aerial scenarios. Furthermore, we propose the integration of steerable Convolutional Neural Networks (CNNs) to explicitly handle rotational variance, leveraging their inherent rotational equivariance to produce robust, orientation-invariant feature representations. Our extensive benchmarking demonstrates that models trained on LASED achieve significantly higher recall compared to those trained on smaller, less diverse datasets, highlighting the benefits of extensive geographic coverage and temporal diversity. Moreover, steerable CNNs effectively address rotational ambiguity inherent in aerial imagery, consistently outperforming conventional convolutional architectures, achieving on average 12\% recall improvement over the best-performing non-steerable network. By combining structured, large-scale datasets with rotation-equivariant neural networks, our approach significantly enhances model robustness and generalization for aerial vPR.  
  </ol>  
</details>  
  
### [U-MARVEL: Unveiling Key Factors for Universal Multimodal Retrieval via Embedding Learning with MLLMs](http://arxiv.org/abs/2507.14902)  
Xiaojie Li, Chu Li, Shi-Zhe Chen, Xi Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Universal multimodal retrieval (UMR), which aims to address complex retrieval tasks where both queries and candidates span diverse modalities, has been significantly advanced by the emergence of MLLMs. While state-of-the-art MLLM-based methods in the literature predominantly adopt contrastive learning principles, they often differ in their specific training recipes. Despite their success, the mechanisms underlying their retrieval capabilities remain largely unexplored, potentially resulting in suboptimal performance and limited generalization ability. To address these issues, we present a comprehensive study aimed at uncovering the key factors that drive effective embedding learning for UMR using MLLMs. We begin by implementing a general MLLM-based embedding learning pipeline, and systematically analyze the primary contributors to high-performing universal retrieval systems. Based on this, we explore various aspects of the details in embedding generation and training strategies, including progressive transition, hard negative mining and re-ranker distillation. Notably, our findings reveal that often-overlooked factors can have a substantial impact on model performance. Building on these discoveries, we introduce a unified framework termed U-MARVEL (\textbf{U}niversal \textbf{M}ultimod\textbf{A}l \textbf{R}etrie\textbf{V}al via \textbf{E}mbedding \textbf{L}earning), which outperforms state-of-the-art competitors on the M-BEIR benchmark by a large margin in supervised settings, and also exihibits strong zero-shot performance on several tasks such as composed image retrieval and text-to-video retrieval. These results underscore the generalization potential of our framework across various embedding-based retrieval tasks. Code is available at https://github.com/chaxjli/U-MARVEL  
  </ol>  
</details>  
**comments**: Technical Report (in progress)  
  
### [OptiCorNet: Optimizing Sequence-Based Context Correlation for Visual Place Recognition](http://arxiv.org/abs/2507.14477)  
Zhenyu Li, Tianyi Shang, Pengjie Xu, Ruirui Zhang, Fanchen Kong  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Place Recognition (VPR) in dynamic and perceptually aliased environments remains a fundamental challenge for long-term localization. Existing deep learning-based solutions predominantly focus on single-frame embeddings, neglecting the temporal coherence present in image sequences. This paper presents OptiCorNet, a novel sequence modeling framework that unifies spatial feature extraction and temporal differencing into a differentiable, end-to-end trainable module. Central to our approach is a lightweight 1D convolutional encoder combined with a learnable differential temporal operator, termed Differentiable Sequence Delta (DSD), which jointly captures short-term spatial context and long-range temporal transitions. The DSD module models directional differences across sequences via a fixed-weight differencing kernel, followed by an LSTM-based refinement and optional residual projection, yielding compact, discriminative descriptors robust to viewpoint and appearance shifts. To further enhance inter-class separability, we incorporate a quadruplet loss that optimizes both positive alignment and multi-negative divergence within each batch. Unlike prior VPR methods that treat temporal aggregation as post-processing, OptiCorNet learns sequence-level embeddings directly, enabling more effective end-to-end place recognition. Comprehensive evaluations on multiple public benchmarks demonstrate that our approach outperforms state-of-the-art baselines under challenging seasonal and viewpoint variations.  
  </ol>  
</details>  
**comments**: 5 figures  
  
### [Developing an AI-Guided Assistant Device for the Deaf and Hearing Impaired](http://arxiv.org/abs/2507.14215)  
Jiayu, Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This study aims to develop a deep learning system for an accessibility device for the deaf or hearing impaired. The device will accurately localize and identify sound sources in real time. This study will fill an important gap in current research by leveraging machine learning techniques to target the underprivileged community. The system includes three main components. 1. JerryNet: A custom designed CNN architecture that determines the direction of arrival (DoA) for nine possible directions. 2. Audio Classification: This model is based on fine-tuning the Contrastive Language-Audio Pretraining (CLAP) model to identify the exact sound classes only based on audio. 3. Multimodal integration model: This is an accurate sound localization model that combines audio, visual, and text data to locate the exact sound sources in the images. The part consists of two modules, one object detection using Yolov9 to generate all the bounding boxes of the objects, and an audio visual localization model to identify the optimal bounding box using complete Intersection over Union (CIoU). The hardware consists of a four-microphone rectangular formation and a camera mounted on glasses with a wristband for displaying necessary information like direction. On a custom collected data set, JerryNet achieved a precision of 91. 1% for the sound direction, outperforming all the baseline models. The CLAP model achieved 98.5% and 95% accuracy on custom and AudioSet datasets, respectively. The audio-visual localization model within component 3 yielded a cIoU of 0.892 and an AUC of 0.658, surpassing other similar models. There are many future potentials to this study, paving the way to creating a new generation of accessibility devices.  
  </ol>  
</details>  
  
  



## NeRF  

### [DiSCO-3D : Discovering and segmenting Sub-Concepts from Open-vocabulary queries in NeRF](http://arxiv.org/abs/2507.14596)  
Doriand Petit, Steve Bourgeois, Vincent Gay-Bellile, Florian Chabot, Loïc Barthe  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D semantic segmentation provides high-level scene understanding for applications in robotics, autonomous systems, \textit{etc}. Traditional methods adapt exclusively to either task-specific goals (open-vocabulary segmentation) or scene content (unsupervised semantic segmentation). We propose DiSCO-3D, the first method addressing the broader problem of 3D Open-Vocabulary Sub-concepts Discovery, which aims to provide a 3D semantic segmentation that adapts to both the scene and user queries. We build DiSCO-3D on Neural Fields representations, combining unsupervised segmentation with weak open-vocabulary guidance. Our evaluations demonstrate that DiSCO-3D achieves effective performance in Open-Vocabulary Sub-concepts Discovery and exhibits state-of-the-art results in the edge cases of both open-vocabulary and unsupervised segmentation.  
  </ol>  
</details>  
**comments**: Published at ICCV'25  
  
### [Advances in Feed-Forward 3D Reconstruction and View Synthesis: A Survey](http://arxiv.org/abs/2507.14501)  
Jiahui Zhang, Yuelei Li, Anpei Chen, Muyu Xu, Kunhao Liu, Jianyuan Wang, Xiao-Xiao Long, Hanxue Liang, Zexiang Xu, Hao Su, Christian Theobalt, Christian Rupprecht, Andrea Vedaldi, Hanspeter Pfister, Shijian Lu, Fangneng Zhan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D reconstruction and view synthesis are foundational problems in computer vision, graphics, and immersive technologies such as augmented reality (AR), virtual reality (VR), and digital twins. Traditional methods rely on computationally intensive iterative optimization in a complex chain, limiting their applicability in real-world scenarios. Recent advances in feed-forward approaches, driven by deep learning, have revolutionized this field by enabling fast and generalizable 3D reconstruction and view synthesis. This survey offers a comprehensive review of feed-forward techniques for 3D reconstruction and view synthesis, with a taxonomy according to the underlying representation architectures including point cloud, 3D Gaussian Splatting (3DGS), Neural Radiance Fields (NeRF), etc. We examine key tasks such as pose-free reconstruction, dynamic 3D reconstruction, and 3D-aware image and video synthesis, highlighting their applications in digital humans, SLAM, robotics, and beyond. In addition, we review commonly used datasets with detailed statistics, along with evaluation protocols for various downstream tasks. We conclude by discussing open research challenges and promising directions for future work, emphasizing the potential of feed-forward approaches to advance the state of the art in 3D vision.  
  </ol>  
</details>  
**comments**: A project page associated with this survey is available at
  https://fnzhan.com/projects/Feed-Forward-3D  
  
  



