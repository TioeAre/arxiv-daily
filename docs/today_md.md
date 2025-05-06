<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#SafeNav:-Safe-Path-Navigation-using-Landmark-Based-Localization-in-a-GPS-denied-Environment>SafeNav: Safe Path Navigation using Landmark Based Localization in a GPS-denied Environment</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#AquaGS:-Fast-Underwater-Scene-Reconstruction-with-SfM-Free-Gaussian-Splatting>AquaGS: Fast Underwater Scene Reconstruction with SfM-Free Gaussian Splatting</a></li>
        <li><a href=#PosePilot:-Steering-Camera-Pose-for-Generative-World-Models-with-Self-supervised-Depth>PosePilot: Steering Camera Pose for Generative World Models with Self-supervised Depth</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#SafeNav:-Safe-Path-Navigation-using-Landmark-Based-Localization-in-a-GPS-denied-Environment>SafeNav: Safe Path Navigation using Landmark Based Localization in a GPS-denied Environment</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Unsupervised-training-of-keypoint-agnostic-descriptors-for-flexible-retinal-image-registration>Unsupervised training of keypoint-agnostic descriptors for flexible retinal image registration</a></li>
        <li><a href=#Unsupervised-Deep-Learning-based-Keypoint-Localization-Estimating-Descriptor-Matching-Performance>Unsupervised Deep Learning-based Keypoint Localization Estimating Descriptor Matching Performance</a></li>
        <li><a href=#Focus-What-Matters:-Matchability-Based-Reweighting-for-Local-Feature-Matching>Focus What Matters: Matchability-Based Reweighting for Local Feature Matching</a></li>
        <li><a href=#Enhancing-Lidar-Point-Cloud-Sampling-via-Colorization-and-Super-Resolution-of-Lidar-Imagery>Enhancing Lidar Point Cloud Sampling via Colorization and Super-Resolution of Lidar Imagery</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Focus-What-Matters:-Matchability-Based-Reweighting-for-Local-Feature-Matching>Focus What Matters: Matchability-Based Reweighting for Local Feature Matching</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#HandOcc:-NeRF-based-Hand-Rendering-with-Occupancy-Networks>HandOcc: NeRF-based Hand Rendering with Occupancy Networks</a></li>
        <li><a href=#Learning-Heterogeneous-Mixture-of-Scene-Experts-for-Large-scale-Neural-Radiance-Fields>Learning Heterogeneous Mixture of Scene Experts for Large-scale Neural Radiance Fields</a></li>
        <li><a href=#AquaGS:-Fast-Underwater-Scene-Reconstruction-with-SfM-Free-Gaussian-Splatting>AquaGS: Fast Underwater Scene Reconstruction with SfM-Free Gaussian Splatting</a></li>
        <li><a href=#Unified-Steganography-via-Implicit-Neural-Representation>Unified Steganography via Implicit Neural Representation</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [SafeNav: Safe Path Navigation using Landmark Based Localization in a GPS-denied Environment](http://arxiv.org/abs/2505.01956)  
Ganesh Sapkota, Sanjay Madria  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In battlefield environments, adversaries frequently disrupt GPS signals, requiring alternative localization and navigation methods. Traditional vision-based approaches like Simultaneous Localization and Mapping (SLAM) and Visual Odometry (VO) involve complex sensor fusion and high computational demand, whereas range-free methods like DV-HOP face accuracy and stability challenges in sparse, dynamic networks. This paper proposes LanBLoc-BMM, a navigation approach using landmark-based localization (LanBLoc) combined with a battlefield-specific motion model (BMM) and Extended Kalman Filter (EKF). Its performance is benchmarked against three state-of-the-art visual localization algorithms integrated with BMM and Bayesian filters, evaluated on synthetic and real-imitated trajectory datasets using metrics including Average Displacement Error (ADE), Final Displacement Error (FDE), and a newly introduced Average Weighted Risk Score (AWRS). LanBLoc-BMM (with EKF) demonstrates superior performance in ADE, FDE, and AWRS on real-imitated datasets. Additionally, two safe navigation methods, SafeNav-CHull and SafeNav-Centroid, are introduced by integrating LanBLoc-BMM(EKF) with a novel Risk-Aware RRT* (RAw-RRT*) algorithm for obstacle avoidance and risk exposure minimization. Simulation results in battlefield scenarios indicate SafeNav-Centroid excels in accuracy, risk exposure, and trajectory efficiency, while SafeNav-CHull provides superior computational speed.  
  </ol>  
</details>  
  
  



## SFM  

### [AquaGS: Fast Underwater Scene Reconstruction with SfM-Free Gaussian Splatting](http://arxiv.org/abs/2505.01799)  
Junhao Shi, Jisheng Xu, Jianping He, Zhiliang Lin  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Underwater scene reconstruction is a critical tech-nology for underwater operations, enabling the generation of 3D models from images captured by underwater platforms. However, the quality of underwater images is often degraded due to medium interference, which limits the effectiveness of Structure-from-Motion (SfM) pose estimation, leading to subsequent reconstruction failures. Additionally, SfM methods typically operate at slower speeds, further hindering their applicability in real-time scenarios. In this paper, we introduce AquaGS, an SfM-free underwater scene reconstruction model based on the SeaThru algorithm, which facilitates rapid and accurate separation of scene details and medium features. Our approach initializes Gaussians by integrating state-of-the-art multi-view stereo (MVS) technology, employs implicit Neural Radiance Fields (NeRF) for rendering translucent media and utilizes the latest explicit 3D Gaussian Splatting (3DGS) technique to render object surfaces, which effectively addresses the limitations of traditional methods and accurately simulates underwater optical phenomena. Experimental results on the data set and the robot platform show that our model can complete high-precision reconstruction in 30 seconds with only 3 image inputs, significantly enhancing the practical application of the algorithm in robotic platforms.  
  </ol>  
</details>  
  
### [PosePilot: Steering Camera Pose for Generative World Models with Self-supervised Depth](http://arxiv.org/abs/2505.01729)  
Bu Jin, Weize Li, Baihan Yang, Zhenxin Zhu, Junpeng Jiang, Huan-ang Gao, Haiyang Sun, Kun Zhan, Hengtong Hu, Xueyang Zhang, Peng Jia, Hao Zhao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in autonomous driving (AD) systems have highlighted the potential of world models in achieving robust and generalizable performance across both ordinary and challenging driving conditions. However, a key challenge remains: precise and flexible camera pose control, which is crucial for accurate viewpoint transformation and realistic simulation of scene dynamics. In this paper, we introduce PosePilot, a lightweight yet powerful framework that significantly enhances camera pose controllability in generative world models. Drawing inspiration from self-supervised depth estimation, PosePilot leverages structure-from-motion principles to establish a tight coupling between camera pose and video generation. Specifically, we incorporate self-supervised depth and pose readouts, allowing the model to infer depth and relative camera motion directly from video sequences. These outputs drive pose-aware frame warping, guided by a photometric warping loss that enforces geometric consistency across synthesized frames. To further refine camera pose estimation, we introduce a reverse warping step and a pose regression loss, improving viewpoint precision and adaptability. Extensive experiments on autonomous driving and general-domain video datasets demonstrate that PosePilot significantly enhances structural understanding and motion reasoning in both diffusion-based and auto-regressive world models. By steering camera pose with self-supervised depth, PosePilot sets a new benchmark for pose controllability, enabling physically consistent, reliable viewpoint synthesis in generative world models.  
  </ol>  
</details>  
**comments**: 8 pages, 3 figures  
  
  



## Visual Localization  

### [SafeNav: Safe Path Navigation using Landmark Based Localization in a GPS-denied Environment](http://arxiv.org/abs/2505.01956)  
Ganesh Sapkota, Sanjay Madria  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In battlefield environments, adversaries frequently disrupt GPS signals, requiring alternative localization and navigation methods. Traditional vision-based approaches like Simultaneous Localization and Mapping (SLAM) and Visual Odometry (VO) involve complex sensor fusion and high computational demand, whereas range-free methods like DV-HOP face accuracy and stability challenges in sparse, dynamic networks. This paper proposes LanBLoc-BMM, a navigation approach using landmark-based localization (LanBLoc) combined with a battlefield-specific motion model (BMM) and Extended Kalman Filter (EKF). Its performance is benchmarked against three state-of-the-art visual localization algorithms integrated with BMM and Bayesian filters, evaluated on synthetic and real-imitated trajectory datasets using metrics including Average Displacement Error (ADE), Final Displacement Error (FDE), and a newly introduced Average Weighted Risk Score (AWRS). LanBLoc-BMM (with EKF) demonstrates superior performance in ADE, FDE, and AWRS on real-imitated datasets. Additionally, two safe navigation methods, SafeNav-CHull and SafeNav-Centroid, are introduced by integrating LanBLoc-BMM(EKF) with a novel Risk-Aware RRT* (RAw-RRT*) algorithm for obstacle avoidance and risk exposure minimization. Simulation results in battlefield scenarios indicate SafeNav-Centroid excels in accuracy, risk exposure, and trajectory efficiency, while SafeNav-CHull provides superior computational speed.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Unsupervised training of keypoint-agnostic descriptors for flexible retinal image registration](http://arxiv.org/abs/2505.02787)  
David Rivas-Villar, Álvaro S. Hervella, José Rouco, Jorge Novo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Current color fundus image registration approaches are limited, among other things, by the lack of labeled data, which is even more significant in the medical domain, motivating the use of unsupervised learning. Therefore, in this work, we develop a novel unsupervised descriptor learning method that does not rely on keypoint detection. This enables the resulting descriptor network to be agnostic to the keypoint detector used during the registration inference.   To validate this approach, we perform an extensive and comprehensive comparison on the reference public retinal image registration dataset. Additionally, we test our method with multiple keypoint detectors of varied nature, even proposing some novel ones. Our results demonstrate that the proposed approach offers accurate registration, not incurring in any performance loss versus supervised methods. Additionally, it demonstrates accurate performance regardless of the keypoint detector used. Thus, this work represents a notable step towards leveraging unsupervised learning in the medical domain.  
  </ol>  
</details>  
  
### [Unsupervised Deep Learning-based Keypoint Localization Estimating Descriptor Matching Performance](http://arxiv.org/abs/2505.02779)  
David Rivas-Villar, Álvaro S. Hervella, José Rouco, Jorge Novo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Retinal image registration, particularly for color fundus images, is a challenging yet essential task with diverse clinical applications. Existing registration methods for color fundus images typically rely on keypoints and descriptors for alignment; however, a significant limitation is their reliance on labeled data, which is particularly scarce in the medical domain.   In this work, we present a novel unsupervised registration pipeline that entirely eliminates the need for labeled data. Our approach is based on the principle that locations with distinctive descriptors constitute reliable keypoints. This fully inverts the conventional state-of-the-art approach, conditioning the detector on the descriptor rather than the opposite.   First, we propose an innovative descriptor learning method that operates without keypoint detection or any labels, generating descriptors for arbitrary locations in retinal images. Next, we introduce a novel, label-free keypoint detector network which works by estimating descriptor performance directly from the input image.   We validate our method through a comprehensive evaluation on four hold-out datasets, demonstrating that our unsupervised descriptor outperforms state-of-the-art supervised descriptors and that our unsupervised detector significantly outperforms existing unsupervised detection methods. Finally, our full registration pipeline achieves performance comparable to the leading supervised methods, while not employing any labeled data. Additionally, the label-free nature and design of our method enable direct adaptation to other domains and modalities.  
  </ol>  
</details>  
  
### [Focus What Matters: Matchability-Based Reweighting for Local Feature Matching](http://arxiv.org/abs/2505.02161)  
Dongyue Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Since the rise of Transformers, many semi-dense matching methods have adopted attention mechanisms to extract feature descriptors. However, the attention weights, which capture dependencies between pixels or keypoints, are often learned from scratch. This approach can introduce redundancy and noisy interactions from irrelevant regions, as it treats all pixels or keypoints equally. Drawing inspiration from keypoint selection processes, we propose to first classify all pixels into two categories: matchable and non-matchable. Matchable pixels are expected to receive higher attention weights, while non-matchable ones are down-weighted. In this work, we propose a novel attention reweighting mechanism that simultaneously incorporates a learnable bias term into the attention logits and applies a matchability-informed rescaling to the input value features. The bias term, injected prior to the softmax operation, selectively adjusts attention scores based on the confidence of query-key interactions. Concurrently, the feature rescaling acts post-attention by modulating the influence of each value vector in the final output. This dual design allows the attention mechanism to dynamically adjust both its internal weighting scheme and the magnitude of its output representations. Extensive experiments conducted on three benchmark datasets validate the effectiveness of our method, consistently outperforming existing state-of-the-art approaches.  
  </ol>  
</details>  
  
### [Enhancing Lidar Point Cloud Sampling via Colorization and Super-Resolution of Lidar Imagery](http://arxiv.org/abs/2505.02049)  
Sier Ha, Honghao Du, Xianjia Yu, Tomi Westerlund  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in lidar technology have led to improved point cloud resolution as well as the generation of 360 degrees, low-resolution images by encoding depth, reflectivity, or near-infrared light within each pixel. These images enable the application of deep learning (DL) approaches, originally developed for RGB images from cameras to lidar-only systems, eliminating other efforts, such as lidar-camera calibration. Compared with conventional RGB images, lidar imagery demonstrates greater robustness in adverse environmental conditions, such as low light and foggy weather. Moreover, the imaging capability addresses the challenges in environments where the geometric information in point clouds may be degraded, such as long corridors, and dense point clouds may be misleading, potentially leading to drift errors.   Therefore, this paper proposes a novel framework that leverages DL-based colorization and super-resolution techniques on lidar imagery to extract reliable samples from lidar point clouds for odometry estimation. The enhanced lidar images, enriched with additional information, facilitate improved keypoint detection, which is subsequently employed for more effective point cloud downsampling. The proposed method enhances point cloud registration accuracy and mitigates mismatches arising from insufficient geometric information or misleading extra points. Experimental results indicate that our approach surpasses previous methods, achieving lower translation and rotation errors while using fewer points.  
  </ol>  
</details>  
**comments**: 7 pages. arXiv admin note: substantial text overlap with
  arXiv:2409.11532  
  
  



## Image Matching  

### [Focus What Matters: Matchability-Based Reweighting for Local Feature Matching](http://arxiv.org/abs/2505.02161)  
Dongyue Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Since the rise of Transformers, many semi-dense matching methods have adopted attention mechanisms to extract feature descriptors. However, the attention weights, which capture dependencies between pixels or keypoints, are often learned from scratch. This approach can introduce redundancy and noisy interactions from irrelevant regions, as it treats all pixels or keypoints equally. Drawing inspiration from keypoint selection processes, we propose to first classify all pixels into two categories: matchable and non-matchable. Matchable pixels are expected to receive higher attention weights, while non-matchable ones are down-weighted. In this work, we propose a novel attention reweighting mechanism that simultaneously incorporates a learnable bias term into the attention logits and applies a matchability-informed rescaling to the input value features. The bias term, injected prior to the softmax operation, selectively adjusts attention scores based on the confidence of query-key interactions. Concurrently, the feature rescaling acts post-attention by modulating the influence of each value vector in the final output. This dual design allows the attention mechanism to dynamically adjust both its internal weighting scheme and the magnitude of its output representations. Extensive experiments conducted on three benchmark datasets validate the effectiveness of our method, consistently outperforming existing state-of-the-art approaches.  
  </ol>  
</details>  
  
  



## NeRF  

### [HandOcc: NeRF-based Hand Rendering with Occupancy Networks](http://arxiv.org/abs/2505.02079)  
Maksym Ivashechkin, Oscar Mendez, Richard Bowden  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose HandOcc, a novel framework for hand rendering based upon occupancy. Popular rendering methods such as NeRF are often combined with parametric meshes to provide deformable hand models. However, in doing so, such approaches present a trade-off between the fidelity of the mesh and the complexity and dimensionality of the parametric model. The simplicity of parametric mesh structures is appealing, but the underlying issue is that it binds methods to mesh initialization, making it unable to generalize to objects where a parametric model does not exist. It also means that estimation is tied to mesh resolution and the accuracy of mesh fitting. This paper presents a pipeline for meshless 3D rendering, which we apply to the hands. By providing only a 3D skeleton, the desired appearance is extracted via a convolutional model. We do this by exploiting a NeRF renderer conditioned upon an occupancy-based representation. The approach uses the hand occupancy to resolve hand-to-hand interactions further improving results, allowing fast rendering, and excellent hand appearance transfer. On the benchmark InterHand2.6M dataset, we achieved state-of-the-art results.  
  </ol>  
</details>  
  
### [Learning Heterogeneous Mixture of Scene Experts for Large-scale Neural Radiance Fields](http://arxiv.org/abs/2505.02005)  
Zhenxing Mi, Ping Yin, Xue Xiao, Dan Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent NeRF methods on large-scale scenes have underlined the importance of scene decomposition for scalable NeRFs. Although achieving reasonable scalability, there are several critical problems remaining unexplored, i.e., learnable decomposition, modeling scene heterogeneity, and modeling efficiency. In this paper, we introduce Switch-NeRF++, a Heterogeneous Mixture of Hash Experts (HMoHE) network that addresses these challenges within a unified framework. It is a highly scalable NeRF that learns heterogeneous decomposition and heterogeneous NeRFs efficiently for large-scale scenes in an end-to-end manner. In our framework, a gating network learns to decomposes scenes and allocates 3D points to specialized NeRF experts. This gating network is co-optimized with the experts, by our proposed Sparsely Gated Mixture of Experts (MoE) NeRF framework. We incorporate a hash-based gating network and distinct heterogeneous hash experts. The hash-based gating efficiently learns the decomposition of the large-scale scene. The distinct heterogeneous hash experts consist of hash grids of different resolution ranges, enabling effective learning of the heterogeneous representation of different scene parts. These design choices make our framework an end-to-end and highly scalable NeRF solution for real-world large-scale scene modeling to achieve both quality and efficiency. We evaluate our accuracy and scalability on existing large-scale NeRF datasets and a new dataset with very large-scale scenes ( $>6.5km^2$ ) from UrbanBIS. Extensive experiments demonstrate that our approach can be easily scaled to various large-scale scenes and achieve state-of-the-art scene rendering accuracy. Furthermore, our method exhibits significant efficiency, with an 8x acceleration in training and a 16x acceleration in rendering compared to Switch-NeRF. Codes will be released in https://github.com/MiZhenxing/Switch-NeRF.  
  </ol>  
</details>  
**comments**: 15 pages, 9 figures  
  
### [AquaGS: Fast Underwater Scene Reconstruction with SfM-Free Gaussian Splatting](http://arxiv.org/abs/2505.01799)  
Junhao Shi, Jisheng Xu, Jianping He, Zhiliang Lin  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Underwater scene reconstruction is a critical tech-nology for underwater operations, enabling the generation of 3D models from images captured by underwater platforms. However, the quality of underwater images is often degraded due to medium interference, which limits the effectiveness of Structure-from-Motion (SfM) pose estimation, leading to subsequent reconstruction failures. Additionally, SfM methods typically operate at slower speeds, further hindering their applicability in real-time scenarios. In this paper, we introduce AquaGS, an SfM-free underwater scene reconstruction model based on the SeaThru algorithm, which facilitates rapid and accurate separation of scene details and medium features. Our approach initializes Gaussians by integrating state-of-the-art multi-view stereo (MVS) technology, employs implicit Neural Radiance Fields (NeRF) for rendering translucent media and utilizes the latest explicit 3D Gaussian Splatting (3DGS) technique to render object surfaces, which effectively addresses the limitations of traditional methods and accurately simulates underwater optical phenomena. Experimental results on the data set and the robot platform show that our model can complete high-precision reconstruction in 30 seconds with only 3 image inputs, significantly enhancing the practical application of the algorithm in robotic platforms.  
  </ol>  
</details>  
  
### [Unified Steganography via Implicit Neural Representation](http://arxiv.org/abs/2505.01749)  
Qi Song, Ziyuan Luo, Xiufeng Huang, Sheng Li, Renjie Wan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Digital steganography is the practice of concealing for encrypted data transmission. Typically, steganography methods embed secret data into cover data to create stega data that incorporates hidden secret data. However, steganography techniques often require designing specific frameworks for each data type, which restricts their generalizability. In this paper, we present U-INR, a novel method for steganography via Implicit Neural Representation (INR). Rather than using the specific framework for each data format, we directly use the neurons of the INR network to represent the secret data and cover data across different data types. To achieve this idea, a private key is shared between the data sender and receivers. Such a private key can be used to determine the position of secret data in INR networks. To effectively leverage this key, we further introduce a key-based selection strategy that can be used to determine the position within the INRs for data storage. Comprehensive experiments across multiple data types, including images, videos, audio, and SDF and NeRF, demonstrate the generalizability and effectiveness of U-INR, emphasizing its potential for improving data security and privacy in various applications.  
  </ol>  
</details>  
  
  



