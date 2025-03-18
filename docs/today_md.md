<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Gaussian-On-the-Fly-Splatting:-A-Progressive-Framework-for-Robust-Near-Real-Time-3DGS-Optimization>Gaussian On-the-Fly Splatting: A Progressive Framework for Robust Near Real-Time 3DGS Optimization</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Scale-Efficient-Training-for-Large-Datasets>Scale Efficient Training for Large Datasets</a></li>
        <li><a href=#Multi-Platform-Teach-and-Repeat-Navigation-by-Visual-Place-Recognition-Based-on-Deep-Learned-Local-Features>Multi-Platform Teach-and-Repeat Navigation by Visual Place Recognition Based on Deep-Learned Local Features</a></li>
        <li><a href=#All-You-Need-to-Know-About-Training-Image-Retrieval-Models>All You Need to Know About Training Image Retrieval Models</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Histogram-Transporter:-Learning-Rotation-Equivariant-Orientation-Histograms-for-High-Precision-Robotic-Kitting>Histogram Transporter: Learning Rotation-Equivariant Orientation Histograms for High-Precision Robotic Kitting</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Less-Biased-Noise-Scale-Estimation-for-Threshold-Robust-RANSAC>Less Biased Noise Scale Estimation for Threshold-Robust RANSAC</a></li>
        <li><a href=#SatDepth:-A-Novel-Dataset-for-Satellite-Image-Matching>SatDepth: A Novel Dataset for Satellite Image Matching</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#TriDF:-Triplane-Accelerated-Density-Fields-for-Few-Shot-Remote-Sensing-Novel-View-Synthesis>TriDF: Triplane-Accelerated Density Fields for Few-Shot Remote Sensing Novel View Synthesis</a></li>
        <li><a href=#DeGauss:-Dynamic-Static-Decomposition-with-Gaussian-Splatting-for-Distractor-free-3D-Reconstruction>DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction</a></li>
        <li><a href=#DivCon-NeRF:-Generating-Augmented-Rays-with-Diversity-and-Consistency-for-Few-shot-View-Synthesis>DivCon-NeRF: Generating Augmented Rays with Diversity and Consistency for Few-shot View Synthesis</a></li>
        <li><a href=#FA-BARF:-Frequency-Adapted-Bundle-Adjusting-Neural-Radiance-Fields>FA-BARF: Frequency Adapted Bundle-Adjusting Neural Radiance Fields</a></li>
        <li><a href=#Industrial-Grade-Sensor-Simulation-via-Gaussian-Splatting:-A-Modular-Framework-for-Scalable-Editing-and-Full-Stack-Validation>Industrial-Grade Sensor Simulation via Gaussian Splatting: A Modular Framework for Scalable Editing and Full-Stack Validation</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Gaussian On-the-Fly Splatting: A Progressive Framework for Robust Near Real-Time 3DGS Optimization](http://arxiv.org/abs/2503.13086)  
Yiwei Xu, Yifei Yu, Wentian Gan, Tengfei Wang, Zongqian Zhan, Hao Cheng, Xin Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D Gaussian Splatting (3DGS) achieves high-fidelity rendering with fast real-time performance, but existing methods rely on offline training after full Structure-from-Motion (SfM) processing. In contrast, this work introduces On-the-Fly GS, a progressive framework enabling near real-time 3DGS optimization during image capture. As each image arrives, its pose and sparse points are updated via on-the-fly SfM, and newly optimized Gaussians are immediately integrated into the 3DGS field. We propose a progressive local optimization strategy to prioritize new images and their neighbors by their corresponding overlapping relationship, allowing the new image and its overlapping images to get more training. To further stabilize training across old and new images, an adaptive learning rate schedule balances the iterations and the learning rate. Moreover, to maintain overall quality of the 3DGS field, an efficient global optimization scheme prevents overfitting to the newly added images. Experiments on multiple benchmark datasets show that our On-the-Fly GS reduces training time significantly, optimizing each new image in seconds with minimal rendering loss, offering the first practical step toward rapid, progressive 3DGS reconstruction.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Scale Efficient Training for Large Datasets](http://arxiv.org/abs/2503.13385)  
Qing Zhou, Junyu Gao, Qi Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The rapid growth of dataset scales has been a key driver in advancing deep learning research. However, as dataset scale increases, the training process becomes increasingly inefficient due to the presence of low-value samples, including excessive redundant samples, overly challenging samples, and inefficient easy samples that contribute little to model improvement.To address this challenge, we propose Scale Efficient Training (SeTa) for large datasets, a dynamic sample pruning approach that losslessly reduces training time. To remove low-value samples, SeTa first performs random pruning to eliminate redundant samples, then clusters the remaining samples according to their learning difficulty measured by loss. Building upon this clustering, a sliding window strategy is employed to progressively remove both overly challenging and inefficient easy clusters following an easy-to-hard curriculum.We conduct extensive experiments on large-scale synthetic datasets, including ToCa, SS1M, and ST+MJ, each containing over 3 million samples.SeTa reduces training costs by up to 50\% while maintaining or improving performance, with minimal degradation even at 70\% cost reduction. Furthermore, experiments on various scale real datasets across various backbones (CNNs, Transformers, and Mambas) and diverse tasks (instruction tuning, multi-view stereo, geo-localization, composed image retrieval, referring image segmentation) demonstrate the powerful effectiveness and universality of our approach. Code is available at https://github.com/mrazhou/SeTa.  
  </ol>  
</details>  
**comments**: Accepted by CVPR2025  
  
### [Multi-Platform Teach-and-Repeat Navigation by Visual Place Recognition Based on Deep-Learned Local Features](http://arxiv.org/abs/2503.13090)  
Václav Truhlařík, Tomáš Pivoňka, Michal Kasarda, Libor Přeučil  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Uniform and variable environments still remain a challenge for stable visual localization and mapping in mobile robot navigation. One of the possible approaches suitable for such environments is appearance-based teach-and-repeat navigation, relying on simplified localization and reactive robot motion control - all without a need for standard mapping. This work brings an innovative solution to such a system based on visual place recognition techniques. Here, the major contributions stand in the employment of a new visual place recognition technique, a novel horizontal shift computation approach, and a multi-platform system design for applications across various types of mobile robots. Secondly, a new public dataset for experimental testing of appearance-based navigation methods is introduced. Moreover, the work also provides real-world experimental testing and performance comparison of the introduced navigation system against other state-of-the-art methods. The results confirm that the new system outperforms existing methods in several testing scenarios, is capable of operation indoors and outdoors, and exhibits robustness to day and night scene variations.  
  </ol>  
</details>  
**comments**: 6 pages, 5 figures  
  
### [All You Need to Know About Training Image Retrieval Models](http://arxiv.org/abs/2503.13045)  
Gabriele Berton, Kevin Musgrave, Carlo Masone  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image retrieval is the task of finding images in a database that are most similar to a given query image. The performance of an image retrieval pipeline depends on many training-time factors, including the embedding model architecture, loss function, data sampler, mining function, learning rate(s), and batch size. In this work, we run tens of thousands of training runs to understand the effect each of these factors has on retrieval accuracy. We also discover best practices that hold across multiple datasets. The code is available at https://github.com/gmberton/image-retrieval  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Histogram Transporter: Learning Rotation-Equivariant Orientation Histograms for High-Precision Robotic Kitting](http://arxiv.org/abs/2503.12541)  
Jiadong Zhou, Yadan Zeng, Huixu Dong, I-Ming Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Robotic kitting is a critical task in industrial automation that requires the precise arrangement of objects into kits to support downstream production processes. However, when handling complex kitting tasks that involve fine-grained orientation alignment, existing approaches often suffer from limited accuracy and computational efficiency. To address these challenges, we propose Histogram Transporter, a novel kitting framework that learns high-precision pick-and-place actions from scratch using only a few demonstrations. First, our method extracts rotation-equivariant orientation histograms (EOHs) from visual observations using an efficient Fourier-based discretization strategy. These EOHs serve a dual purpose: improving picking efficiency by directly modeling action success probabilities over high-resolution orientations and enhancing placing accuracy by serving as local, discriminative feature descriptors for object-to-placement matching. Second, we introduce a subgroup alignment strategy in the place model that compresses the full spectrum of EOHs into a compact orientation representation, enabling efficient feature matching while preserving accuracy. Finally, we examine the proposed framework on the simulated Hand-Tool Kitting Dataset (HTKD), where it outperforms competitive baselines in both success rates and computational efficiency. Further experiments on five Raven-10 tasks exhibits the remarkable adaptability of our approach, with real-robot trials confirming its applicability for real-world deployment.  
  </ol>  
</details>  
**comments**: This manuscript is currently under review  
  
  



## Image Matching  

### [Less Biased Noise Scale Estimation for Threshold-Robust RANSAC](http://arxiv.org/abs/2503.13433)  
Johan Edstedt  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The gold-standard for robustly estimating relative pose through image matching is RANSAC. While RANSAC is powerful, it requires setting the inlier threshold that determines whether the error of a correspondence under an estimated model is sufficiently small to be included in its consensus set. Setting this threshold is typically done by hand, and is difficult to tune without a access to ground truth data. Thus, a method capable of automatically determining the optimal threshold would be desirable. In this paper we revisit inlier noise scale estimation, which is an attractive approach as the inlier noise scale is linear to the optimal threshold. We revisit the noise scale estimation method SIMFIT and find bias in the estimate of the noise scale. In particular, we fix underestimates from using the same data for fitting the model as estimating the inlier noise, and from not taking the threshold itself into account. Secondly, since the optimal threshold within a scene is approximately constant we propose a multi-pair extension of SIMFIT++, by filtering of estimates, which improves results. Our approach yields robust performance across a range of thresholds, shown in Figure 1.  
  </ol>  
</details>  
  
### [SatDepth: A Novel Dataset for Satellite Image Matching](http://arxiv.org/abs/2503.12706)  
Rahul Deshmukh, Avinash Kak  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advances in deep-learning based methods for image matching have demonstrated their superiority over traditional algorithms, enabling correspondence estimation in challenging scenes with significant differences in viewing angles, illumination and weather conditions. However, the existing datasets, learning frameworks, and evaluation metrics for the deep-learning based methods are limited to ground-based images recorded with pinhole cameras and have not been explored for satellite images. In this paper, we present ``SatDepth'', a novel dataset that provides dense ground-truth correspondences for training image matching frameworks meant specifically for satellite images. Satellites capture images from various viewing angles and tracks through multiple revisits over a region. To manage this variability, we propose a dataset balancing strategy through a novel image rotation augmentation procedure. This procedure allows for the discovery of corresponding pixels even in the presence of large rotational differences between the images. We benchmark four existing image matching frameworks using our dataset and carry out an ablation study that confirms that the models trained with our dataset with rotation augmentation outperform (up to 40% increase in precision) the models trained with other datasets, especially when there exist large rotational differences between the images.  
  </ol>  
</details>  
  
  



## NeRF  

### [TriDF: Triplane-Accelerated Density Fields for Few-Shot Remote Sensing Novel View Synthesis](http://arxiv.org/abs/2503.13347)  
Jiaming Kang, Keyan Chen, Zhengxia Zou, Zhenwei Shi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Remote sensing novel view synthesis (NVS) offers significant potential for 3D interpretation of remote sensing scenes, with important applications in urban planning and environmental monitoring. However, remote sensing scenes frequently lack sufficient multi-view images due to acquisition constraints. While existing NVS methods tend to overfit when processing limited input views, advanced few-shot NVS methods are computationally intensive and perform sub-optimally in remote sensing scenes. This paper presents TriDF, an efficient hybrid 3D representation for fast remote sensing NVS from as few as 3 input views. Our approach decouples color and volume density information, modeling them independently to reduce the computational burden on implicit radiance fields and accelerate reconstruction. We explore the potential of the triplane representation in few-shot NVS tasks by mapping high-frequency color information onto this compact structure, and the direct optimization of feature planes significantly speeds up convergence. Volume density is modeled as continuous density fields, incorporating reference features from neighboring views through image-based rendering to compensate for limited input data. Additionally, we introduce depth-guided optimization based on point clouds, which effectively mitigates the overfitting problem in few-shot NVS. Comprehensive experiments across multiple remote sensing scenes demonstrate that our hybrid representation achieves a 30x speed increase compared to NeRF-based methods, while simultaneously improving rendering quality metrics over advanced few-shot methods (7.4% increase in PSNR, 12.2% in SSIM, and 18.7% in LPIPS). The code is publicly available at https://github.com/kanehub/TriDF  
  </ol>  
</details>  
  
### [DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](http://arxiv.org/abs/2503.13176)  
Rui Wang, Quentin Lohmeyer, Mirko Meboldt, Siyu Tang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Reconstructing clean, distractor-free 3D scenes from real-world captures remains a significant challenge, particularly in highly dynamic and cluttered settings such as egocentric videos. To tackle this problem, we introduce DeGauss, a simple and robust self-supervised framework for dynamic scene reconstruction based on a decoupled dynamic-static Gaussian Splatting design. DeGauss models dynamic elements with foreground Gaussians and static content with background Gaussians, using a probabilistic mask to coordinate their composition and enable independent yet complementary optimization. DeGauss generalizes robustly across a wide range of real-world scenarios, from casual image collections to long, dynamic egocentric videos, without relying on complex heuristics or extensive supervision. Experiments on benchmarks including NeRF-on-the-go, ADT, AEA, Hot3D, and EPIC-Fields demonstrate that DeGauss consistently outperforms existing methods, establishing a strong baseline for generalizable, distractor-free 3D reconstructionin highly dynamic, interaction-rich environments.  
  </ol>  
</details>  
  
### [DivCon-NeRF: Generating Augmented Rays with Diversity and Consistency for Few-shot View Synthesis](http://arxiv.org/abs/2503.12947)  
Ingyun Lee, Jae Won Jang, Seunghyeon Seo, Nojun Kwak  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Field (NeRF) has shown remarkable performance in novel view synthesis but requires many multiview images, making it impractical for few-shot scenarios. Ray augmentation was proposed to prevent overfitting for sparse training data by generating additional rays. However, existing methods, which generate augmented rays only near the original rays, produce severe floaters and appearance distortion due to limited viewpoints and inconsistent rays obstructed by nearby obstacles and complex surfaces. To address these problems, we propose DivCon-NeRF, which significantly enhances both diversity and consistency. It employs surface-sphere augmentation, which preserves the distance between the original camera and the predicted surface point. This allows the model to compare the order of high-probability surface points and filter out inconsistent rays easily without requiring the exact depth. By introducing inner-sphere augmentation, DivCon-NeRF randomizes angles and distances for diverse viewpoints, further increasing diversity. Consequently, our method significantly reduces floaters and visual distortions, achieving state-of-the-art performance on the Blender, LLFF, and DTU datasets. Our code will be publicly available.  
  </ol>  
</details>  
**comments**: 11 pages, 6 figures  
  
### [FA-BARF: Frequency Adapted Bundle-Adjusting Neural Radiance Fields](http://arxiv.org/abs/2503.12086)  
Rui Qian, Chenyangguang Zhang, Yan Di, Guangyao Zhai, Ruida Zhang, Jiayu Guo, Benjamin Busam, Jian Pu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) have exhibited highly effective performance for photorealistic novel view synthesis recently. However, the key limitation it meets is the reliance on a hand-crafted frequency annealing strategy to recover 3D scenes with imperfect camera poses. The strategy exploits a temporal low-pass filter to guarantee convergence while decelerating the joint optimization of implicit scene reconstruction and camera registration. In this work, we introduce the Frequency Adapted Bundle Adjusting Radiance Field (FA-BARF), substituting the temporal low-pass filter for a frequency-adapted spatial low-pass filter to address the decelerating problem. We establish a theoretical framework to interpret the relationship between position encoding of NeRF and camera registration and show that our frequency-adapted filter can mitigate frequency fluctuation caused by the temporal filter. Furthermore, we show that applying a spatial low-pass filter in NeRF can optimize camera poses productively through radial uncertainty overlaps among various views. Extensive experiments show that FA-BARF can accelerate the joint optimization process under little perturbations in object-centric scenes and recover real-world scenes with unknown camera poses. This implies wider possibilities for NeRF applied in dense 3D mapping and reconstruction under real-time requirements. The code will be released upon paper acceptance.  
  </ol>  
</details>  
  
### [Industrial-Grade Sensor Simulation via Gaussian Splatting: A Modular Framework for Scalable Editing and Full-Stack Validation](http://arxiv.org/abs/2503.11731)  
Xianming Zeng, Sicong Du, Qifeng Chen, Lizhe Liu, Haoyu Shu, Jiaxuan Gao, Jiarun Liu, Jiulong Xu, Jianyun Xu, Mingxia Chen, Yiru Zhao, Peng Chen, Yapeng Xue, Chunming Zhao, Sheng Yang, Qiang Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Sensor simulation is pivotal for scalable validation of autonomous driving systems, yet existing Neural Radiance Fields (NeRF) based methods face applicability and efficiency challenges in industrial workflows. This paper introduces a Gaussian Splatting (GS) based system to address these challenges: We first break down sensor simulator components and analyze the possible advantages of GS over NeRF. Then in practice, we refactor three crucial components through GS, to leverage its explicit scene representation and real-time rendering: (1) choosing the 2D neural Gaussian representation for physics-compliant scene and sensor modeling, (2) proposing a scene editing pipeline to leverage Gaussian primitives library for data augmentation, and (3) coupling a controllable diffusion model for scene expansion and harmonization. We implement this framework on a proprietary autonomous driving dataset supporting cameras and LiDAR sensors. We demonstrate through ablation studies that our approach reduces frame-wise simulation latency, achieves better geometric and photometric consistency, and enables interpretable explicit scene editing and expansion. Furthermore, we showcase how integrating such a GS-based sensor simulator with traffic and dynamic simulators enables full-stack testing of end-to-end autonomy algorithms. Our work provides both algorithmic insights and practical validation, establishing GS as a cornerstone for industrial-grade sensor simulation.  
  </ol>  
</details>  
  
  



