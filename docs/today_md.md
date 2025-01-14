<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#SplatMAP:-Online-Dense-Monocular-SLAM-with-3D-Gaussian-Splatting>SplatMAP: Online Dense Monocular SLAM with 3D Gaussian Splatting</a></li>
        <li><a href=#CULTURE3D:-Cultural-Landmarks-and-Terrain-Dataset-for-3D-Applications>CULTURE3D: Cultural Landmarks and Terrain Dataset for 3D Applications</a></li>
        <li><a href=#Aug3D:-Augmenting-large-scale-outdoor-datasets-for-Generalizable-Novel-View-Synthesis>Aug3D: Augmenting large scale outdoor datasets for Generalizable Novel View Synthesis</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Efficiently-Closing-Loops-in-LiDAR-Based-SLAM-Using-Point-Cloud-Density-Maps>Efficiently Closing Loops in LiDAR-Based SLAM Using Point Cloud Density Maps</a></li>
        <li><a href=#Static-Segmentation-by-Tracking:-A-Frustratingly-Label-Efficient-Approach-to-Fine-Grained-Segmentation>Static Segmentation by Tracking: A Frustratingly Label-Efficient Approach to Fine-Grained Segmentation</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Empirical-Comparison-of-Four-Stereoscopic-Depth-Sensing-Cameras-for-Robotics-Applications>Empirical Comparison of Four Stereoscopic Depth Sensing Cameras for Robotics Applications</a></li>
        <li><a href=#Efficiently-Closing-Loops-in-LiDAR-Based-SLAM-Using-Point-Cloud-Density-Maps>Efficiently Closing Loops in LiDAR-Based SLAM Using Point Cloud Density Maps</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#MatchAnything:-Universal-Cross-Modality-Image-Matching-with-Large-Scale-Pre-Training>MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training</a></li>
        <li><a href=#Matching-Free-Depth-Recovery-from-Structured-Light>Matching Free Depth Recovery from Structured Light</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#SplatMAP:-Online-Dense-Monocular-SLAM-with-3D-Gaussian-Splatting>SplatMAP: Online Dense Monocular SLAM with 3D Gaussian Splatting</a></li>
        <li><a href=#CULTURE3D:-Cultural-Landmarks-and-Terrain-Dataset-for-3D-Applications>CULTURE3D: Cultural Landmarks and Terrain Dataset for 3D Applications</a></li>
        <li><a href=#ActiveGAMER:-Active-GAussian-Mapping-through-Efficient-Rendering>ActiveGAMER: Active GAussian Mapping through Efficient Rendering</a></li>
        <li><a href=#SuperNeRF-GAN:-A-Universal-3D-Consistent-Super-Resolution-Framework-for-Efficient-and-Enhanced-3D-Aware-Image-Synthesis>SuperNeRF-GAN: A Universal 3D-Consistent Super-Resolution Framework for Efficient and Enhanced 3D-Aware Image Synthesis</a></li>
        <li><a href=#NVS-SQA:-Exploring-Self-Supervised-Quality-Representation-Learning-for-Neurally-Synthesized-Scenes-without-References>NVS-SQA: Exploring Self-Supervised Quality Representation Learning for Neurally Synthesized Scenes without References</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [SplatMAP: Online Dense Monocular SLAM with 3D Gaussian Splatting](http://arxiv.org/abs/2501.07015)  
Yue Hu, Rong Liu, Meida Chen, Andrew Feng, Peter Beerel  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Achieving high-fidelity 3D reconstruction from monocular video remains challenging due to the inherent limitations of traditional methods like Structure-from-Motion (SfM) and monocular SLAM in accurately capturing scene details. While differentiable rendering techniques such as Neural Radiance Fields (NeRF) address some of these challenges, their high computational costs make them unsuitable for real-time applications. Additionally, existing 3D Gaussian Splatting (3DGS) methods often focus on photometric consistency, neglecting geometric accuracy and failing to exploit SLAM's dynamic depth and pose updates for scene refinement. We propose a framework integrating dense SLAM with 3DGS for real-time, high-fidelity dense reconstruction. Our approach introduces SLAM-Informed Adaptive Densification, which dynamically updates and densifies the Gaussian model by leveraging dense point clouds from SLAM. Additionally, we incorporate Geometry-Guided Optimization, which combines edge-aware geometric constraints and photometric consistency to jointly optimize the appearance and geometry of the 3DGS scene representation, enabling detailed and accurate SLAM mapping reconstruction. Experiments on the Replica and TUM-RGBD datasets demonstrate the effectiveness of our approach, achieving state-of-the-art results among monocular systems. Specifically, our method achieves a PSNR of 36.864, SSIM of 0.985, and LPIPS of 0.040 on Replica, representing improvements of 10.7%, 6.4%, and 49.4%, respectively, over the previous SOTA. On TUM-RGBD, our method outperforms the closest baseline by 10.2%, 6.6%, and 34.7% in the same metrics. These results highlight the potential of our framework in bridging the gap between photometric and geometric dense 3D scene representations, paving the way for practical and efficient monocular dense reconstruction.  
  </ol>  
</details>  
  
### [CULTURE3D: Cultural Landmarks and Terrain Dataset for 3D Applications](http://arxiv.org/abs/2501.06927)  
Xinyi Zheng, Steve Zhang, Weizhe Lin, Aaron Zhang, Walterio W. Mayol-Cuevas, Junxiao Shen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we present a large-scale fine-grained dataset using high-resolution images captured from locations worldwide. Compared to existing datasets, our dataset offers a significantly larger size and includes a higher level of detail, making it uniquely suited for fine-grained 3D applications. Notably, our dataset is built using drone-captured aerial imagery, which provides a more accurate perspective for capturing real-world site layouts and architectural structures. By reconstructing environments with these detailed images, our dataset supports applications such as the COLMAP format for Gaussian Splatting and the Structure-from-Motion (SfM) method. It is compatible with widely-used techniques including SLAM, Multi-View Stereo, and Neural Radiance Fields (NeRF), enabling accurate 3D reconstructions and point clouds. This makes it a benchmark for reconstruction and segmentation tasks. The dataset enables seamless integration with multi-modal data, supporting a range of 3D applications, from architectural reconstruction to virtual tourism. Its flexibility promotes innovation, facilitating breakthroughs in 3D modeling and analysis.  
  </ol>  
</details>  
  
### [Aug3D: Augmenting large scale outdoor datasets for Generalizable Novel View Synthesis](http://arxiv.org/abs/2501.06431)  
Aditya Rauniyar, Omar Alama, Silong Yong, Katia Sycara, Sebastian Scherer  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent photorealistic Novel View Synthesis (NVS) advances have increasingly gained attention. However, these approaches remain constrained to small indoor scenes. While optimization-based NVS models have attempted to address this, generalizable feed-forward methods, offering significant advantages, remain underexplored. In this work, we train PixelNeRF, a feed-forward NVS model, on the large-scale UrbanScene3D dataset. We propose four training strategies to cluster and train on this dataset, highlighting that performance is hindered by limited view overlap. To address this, we introduce Aug3D, an augmentation technique that leverages reconstructed scenes using traditional Structure-from-Motion (SfM). Aug3D generates well-conditioned novel views through grid and semantic sampling to enhance feed-forward NVS model learning. Our experiments reveal that reducing the number of views per cluster from 20 to 10 improves PSNR by 10%, but the performance remains suboptimal. Aug3D further addresses this by combining the newly generated novel views with the original dataset, demonstrating its effectiveness in improving the model's ability to predict novel views.  
  </ol>  
</details>  
**comments**: IROS 2024 Workshop, 9 Pages, 7 Figures  
  
  



## Visual Localization  

### [Efficiently Closing Loops in LiDAR-Based SLAM Using Point Cloud Density Maps](http://arxiv.org/abs/2501.07399)  
Saurabh Gupta, Tiziano Guadagnino, Benedikt Mersch, Niklas Trekel, Meher V. R. Malladi, Cyrill Stachniss  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Consistent maps are key for most autonomous mobile robots. They often use SLAM approaches to build such maps. Loop closures via place recognition help maintain accurate pose estimates by mitigating global drift. This paper presents a robust loop closure detection pipeline for outdoor SLAM with LiDAR-equipped robots. The method handles various LiDAR sensors with different scanning patterns, field of views and resolutions. It generates local maps from LiDAR scans and aligns them using a ground alignment module to handle both planar and non-planar motion of the LiDAR, ensuring applicability across platforms. The method uses density-preserving bird's eye view projections of these local maps and extracts ORB feature descriptors from them for place recognition. It stores the feature descriptors in a binary search tree for efficient retrieval, and self-similarity pruning addresses perceptual aliasing in repetitive environments. Extensive experiments on public and self-recorded datasets demonstrate accurate loop closure detection, long-term localization, and cross-platform multi-map alignment, agnostic to the LiDAR scanning patterns, fields of view, and motion profiles.  
  </ol>  
</details>  
  
### [Static Segmentation by Tracking: A Frustratingly Label-Efficient Approach to Fine-Grained Segmentation](http://arxiv.org/abs/2501.06749)  
Zhenyang Feng, Zihe Wang, Saul Ibaven Bueno, Tomasz Frelek, Advikaa Ramesh, Jingyan Bai, Lemeng Wang, Zanming Huang, Jianyang Gu, Jinsu Yoo, Tai-Yu Pan, Arpita Chowdhury, Michelle Ramirez, Elizabeth G. Campolongo, Matthew J. Thompson, Christopher G. Lawrence, Sydne Record, Neil Rosser, Anuj Karpatne, Daniel Rubenstein, Hilmar Lapp, Charles V. Stewart, Tanya Berger-Wolf, Yu Su, Wei-Lun Chao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We study image segmentation in the biological domain, particularly trait and part segmentation from specimen images (e.g., butterfly wing stripes or beetle body parts). This is a crucial, fine-grained task that aids in understanding the biology of organisms. The conventional approach involves hand-labeling masks, often for hundreds of images per species, and training a segmentation model to generalize these labels to other images, which can be exceedingly laborious. We present a label-efficient method named Static Segmentation by Tracking (SST). SST is built upon the insight: while specimens of the same species have inherent variations, the traits and parts we aim to segment show up consistently. This motivates us to concatenate specimen images into a ``pseudo-video'' and reframe trait and part segmentation as a tracking problem. Concretely, SST generates masks for unlabeled images by propagating annotated or predicted masks from the ``pseudo-preceding'' images. Powered by Segment Anything Model 2 (SAM~2) initially developed for video segmentation, we show that SST can achieve high-quality trait and part segmentation with merely one labeled image per species -- a breakthrough for analyzing specimen images. We further develop a cycle-consistent loss to fine-tune the model, again using one labeled image. Additionally, we highlight the broader potential of SST, including one-shot instance segmentation on images taken in the wild and trait-based image retrieval.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Empirical Comparison of Four Stereoscopic Depth Sensing Cameras for Robotics Applications](http://arxiv.org/abs/2501.07421)  
Lukas Rustler, Vojtech Volprecht, Matej Hoffmann  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Depth sensing is an essential technology in robotics and many other fields. Many depth sensing (or RGB-D) cameras are available on the market and selecting the best one for your application can be challenging. In this work, we tested four stereoscopic RGB-D cameras that sense the distance by using two images from slightly different views. We empirically compared four cameras (Intel RealSense D435, Intel RealSense D455, StereoLabs ZED 2, and Luxonis OAK-D Pro) in three scenarios: (i) planar surface perception, (ii) plastic doll perception, (iii) household object perception (YCB dataset). We recorded and evaluated more than 3,000 RGB-D frames for each camera. For table-top robotics scenarios with distance to objects up to one meter, the best performance is provided by the D435 camera. For longer distances, the other three models perform better, making them more suitable for some mobile robotics applications. OAK-D Pro additionally offers integrated AI modules (e.g., object and human keypoint detection). ZED 2 is not a standalone device and requires a computer with a GPU for depth data acquisition. All data (more than 12,000 RGB-D frames) are publicly available at https://osf.io/f2seb.  
  </ol>  
</details>  
  
### [Efficiently Closing Loops in LiDAR-Based SLAM Using Point Cloud Density Maps](http://arxiv.org/abs/2501.07399)  
Saurabh Gupta, Tiziano Guadagnino, Benedikt Mersch, Niklas Trekel, Meher V. R. Malladi, Cyrill Stachniss  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Consistent maps are key for most autonomous mobile robots. They often use SLAM approaches to build such maps. Loop closures via place recognition help maintain accurate pose estimates by mitigating global drift. This paper presents a robust loop closure detection pipeline for outdoor SLAM with LiDAR-equipped robots. The method handles various LiDAR sensors with different scanning patterns, field of views and resolutions. It generates local maps from LiDAR scans and aligns them using a ground alignment module to handle both planar and non-planar motion of the LiDAR, ensuring applicability across platforms. The method uses density-preserving bird's eye view projections of these local maps and extracts ORB feature descriptors from them for place recognition. It stores the feature descriptors in a binary search tree for efficient retrieval, and self-similarity pruning addresses perceptual aliasing in repetitive environments. Extensive experiments on public and self-recorded datasets demonstrate accurate loop closure detection, long-term localization, and cross-platform multi-map alignment, agnostic to the LiDAR scanning patterns, fields of view, and motion profiles.  
  </ol>  
</details>  
  
  



## Image Matching  

### [MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training](http://arxiv.org/abs/2501.07556)  
Xingyi He, Hao Yu, Sida Peng, Dongli Tan, Zehong Shen, Hujun Bao, Xiaowei Zhou  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image matching, which aims to identify corresponding pixel locations between images, is crucial in a wide range of scientific disciplines, aiding in image registration, fusion, and analysis. In recent years, deep learning-based image matching algorithms have dramatically outperformed humans in rapidly and accurately finding large amounts of correspondences. However, when dealing with images captured under different imaging modalities that result in significant appearance changes, the performance of these algorithms often deteriorates due to the scarcity of annotated cross-modal training data. This limitation hinders applications in various fields that rely on multiple image modalities to obtain complementary information. To address this challenge, we propose a large-scale pre-training framework that utilizes synthetic cross-modal training signals, incorporating diverse data from various sources, to train models to recognize and match fundamental structures across images. This capability is transferable to real-world, unseen cross-modality image matching tasks. Our key finding is that the matching model trained with our framework achieves remarkable generalizability across more than eight unseen cross-modality registration tasks using the same network weight, substantially outperforming existing methods, whether designed for generalization or tailored for specific tasks. This advancement significantly enhances the applicability of image matching technologies across various scientific disciplines and paves the way for new applications in multi-modality human and artificial intelligence analysis and beyond.  
  </ol>  
</details>  
**comments**: Project page: https://zju3dv.github.io/MatchAnything/  
  
### [Matching Free Depth Recovery from Structured Light](http://arxiv.org/abs/2501.07113)  
Zhuohang Yu, Kai Wang, Juyong Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present a novel approach for depth estimation from images captured by structured light systems. Unlike many previous methods that rely on image matching process, our approach uses a density voxel grid to represent scene geometry, which is trained via self-supervised differentiable volume rendering. Our method leverages color fields derived from projected patterns in structured light systems during the rendering process, enabling the isolated optimization of the geometry field. This contributes to faster convergence and high-quality output. Additionally, we incorporate normalized device coordinates (NDC), a distortion loss, and a novel surface-based color loss to enhance geometric fidelity. Experimental results demonstrate that our method outperforms existing matching-based techniques in geometric performance for few-shot scenarios, achieving approximately a 60% reduction in average estimated depth errors on synthetic scenes and about 30% on real-world captured scenes. Furthermore, our approach delivers fast training, with a speed roughly three times faster than previous matching-free methods that employ implicit representations.  
  </ol>  
</details>  
**comments**: 10 pages, 8 figures  
  
  



## NeRF  

### [SplatMAP: Online Dense Monocular SLAM with 3D Gaussian Splatting](http://arxiv.org/abs/2501.07015)  
Yue Hu, Rong Liu, Meida Chen, Andrew Feng, Peter Beerel  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Achieving high-fidelity 3D reconstruction from monocular video remains challenging due to the inherent limitations of traditional methods like Structure-from-Motion (SfM) and monocular SLAM in accurately capturing scene details. While differentiable rendering techniques such as Neural Radiance Fields (NeRF) address some of these challenges, their high computational costs make them unsuitable for real-time applications. Additionally, existing 3D Gaussian Splatting (3DGS) methods often focus on photometric consistency, neglecting geometric accuracy and failing to exploit SLAM's dynamic depth and pose updates for scene refinement. We propose a framework integrating dense SLAM with 3DGS for real-time, high-fidelity dense reconstruction. Our approach introduces SLAM-Informed Adaptive Densification, which dynamically updates and densifies the Gaussian model by leveraging dense point clouds from SLAM. Additionally, we incorporate Geometry-Guided Optimization, which combines edge-aware geometric constraints and photometric consistency to jointly optimize the appearance and geometry of the 3DGS scene representation, enabling detailed and accurate SLAM mapping reconstruction. Experiments on the Replica and TUM-RGBD datasets demonstrate the effectiveness of our approach, achieving state-of-the-art results among monocular systems. Specifically, our method achieves a PSNR of 36.864, SSIM of 0.985, and LPIPS of 0.040 on Replica, representing improvements of 10.7%, 6.4%, and 49.4%, respectively, over the previous SOTA. On TUM-RGBD, our method outperforms the closest baseline by 10.2%, 6.6%, and 34.7% in the same metrics. These results highlight the potential of our framework in bridging the gap between photometric and geometric dense 3D scene representations, paving the way for practical and efficient monocular dense reconstruction.  
  </ol>  
</details>  
  
### [CULTURE3D: Cultural Landmarks and Terrain Dataset for 3D Applications](http://arxiv.org/abs/2501.06927)  
Xinyi Zheng, Steve Zhang, Weizhe Lin, Aaron Zhang, Walterio W. Mayol-Cuevas, Junxiao Shen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we present a large-scale fine-grained dataset using high-resolution images captured from locations worldwide. Compared to existing datasets, our dataset offers a significantly larger size and includes a higher level of detail, making it uniquely suited for fine-grained 3D applications. Notably, our dataset is built using drone-captured aerial imagery, which provides a more accurate perspective for capturing real-world site layouts and architectural structures. By reconstructing environments with these detailed images, our dataset supports applications such as the COLMAP format for Gaussian Splatting and the Structure-from-Motion (SfM) method. It is compatible with widely-used techniques including SLAM, Multi-View Stereo, and Neural Radiance Fields (NeRF), enabling accurate 3D reconstructions and point clouds. This makes it a benchmark for reconstruction and segmentation tasks. The dataset enables seamless integration with multi-modal data, supporting a range of 3D applications, from architectural reconstruction to virtual tourism. Its flexibility promotes innovation, facilitating breakthroughs in 3D modeling and analysis.  
  </ol>  
</details>  
  
### [ActiveGAMER: Active GAussian Mapping through Efficient Rendering](http://arxiv.org/abs/2501.06897)  
Liyan Chen, Huangying Zhan, Kevin Chen, Xiangyu Xu, Qingan Yan, Changjiang Cai, Yi Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce ActiveGAMER, an active mapping system that utilizes 3D Gaussian Splatting (3DGS) to achieve high-quality, real-time scene mapping and exploration. Unlike traditional NeRF-based methods, which are computationally demanding and restrict active mapping performance, our approach leverages the efficient rendering capabilities of 3DGS, allowing effective and efficient exploration in complex environments. The core of our system is a rendering-based information gain module that dynamically identifies the most informative viewpoints for next-best-view planning, enhancing both geometric and photometric reconstruction accuracy. ActiveGAMER also integrates a carefully balanced framework, combining coarse-to-fine exploration, post-refinement, and a global-local keyframe selection strategy to maximize reconstruction completeness and fidelity. Our system autonomously explores and reconstructs environments with state-of-the-art geometric and photometric accuracy and completeness, significantly surpassing existing approaches in both aspects. Extensive evaluations on benchmark datasets such as Replica and MP3D highlight ActiveGAMER's effectiveness in active mapping tasks.  
  </ol>  
</details>  
  
### [SuperNeRF-GAN: A Universal 3D-Consistent Super-Resolution Framework for Efficient and Enhanced 3D-Aware Image Synthesis](http://arxiv.org/abs/2501.06770)  
Peng Zheng, Linzhi Huang, Yizhou Yu, Yi Chang, Yilin Wang, Rui Ma  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural volume rendering techniques, such as NeRF, have revolutionized 3D-aware image synthesis by enabling the generation of images of a single scene or object from various camera poses. However, the high computational cost of NeRF presents challenges for synthesizing high-resolution (HR) images. Most existing methods address this issue by leveraging 2D super-resolution, which compromise 3D-consistency. Other methods propose radiance manifolds or two-stage generation to achieve 3D-consistent HR synthesis, yet they are limited to specific synthesis tasks, reducing their universality. To tackle these challenges, we propose SuperNeRF-GAN, a universal framework for 3D-consistent super-resolution. A key highlight of SuperNeRF-GAN is its seamless integration with NeRF-based 3D-aware image synthesis methods and it can simultaneously enhance the resolution of generated images while preserving 3D-consistency and reducing computational cost. Specifically, given a pre-trained generator capable of producing a NeRF representation such as tri-plane, we first perform volume rendering to obtain a low-resolution image with corresponding depth and normal map. Then, we employ a NeRF Super-Resolution module which learns a network to obtain a high-resolution NeRF. Next, we propose a novel Depth-Guided Rendering process which contains three simple yet effective steps, including the construction of a boundary-correct multi-depth map through depth aggregation, a normal-guided depth super-resolution and a depth-guided NeRF rendering. Experimental results demonstrate the superior efficiency, 3D-consistency, and quality of our approach. Additionally, ablation studies confirm the effectiveness of our proposed components.  
  </ol>  
</details>  
  
### [NVS-SQA: Exploring Self-Supervised Quality Representation Learning for Neurally Synthesized Scenes without References](http://arxiv.org/abs/2501.06488)  
[[code](https://github.com/vincentqqu/nvs-sqa)]  
Qiang Qu, Yiran Shen, Xiaoming Chen, Yuk Ying Chung, Weidong Cai, Tongliang Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural View Synthesis (NVS), such as NeRF and 3D Gaussian Splatting, effectively creates photorealistic scenes from sparse viewpoints, typically evaluated by quality assessment methods like PSNR, SSIM, and LPIPS. However, these full-reference methods, which compare synthesized views to reference views, may not fully capture the perceptual quality of neurally synthesized scenes (NSS), particularly due to the limited availability of dense reference views. Furthermore, the challenges in acquiring human perceptual labels hinder the creation of extensive labeled datasets, risking model overfitting and reduced generalizability. To address these issues, we propose NVS-SQA, a NSS quality assessment method to learn no-reference quality representations through self-supervision without reliance on human labels. Traditional self-supervised learning predominantly relies on the "same instance, similar representation" assumption and extensive datasets. However, given that these conditions do not apply in NSS quality assessment, we employ heuristic cues and quality scores as learning objectives, along with a specialized contrastive pair preparation process to improve the effectiveness and efficiency of learning. The results show that NVS-SQA outperforms 17 no-reference methods by a large margin (i.e., on average 109.5% in SRCC, 98.6% in PLCC, and 91.5% in KRCC over the second best) and even exceeds 16 full-reference methods across all evaluation metrics (i.e., 22.9% in SRCC, 19.1% in PLCC, and 18.6% in KRCC over the second best).  
  </ol>  
</details>  
  
  



