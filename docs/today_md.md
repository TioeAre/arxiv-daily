<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#3D-Modeling:-Camera-Movement-Estimation-and-path-Correction-for-SFM-Model-using-the-Combination-of-Modified-A-SIFT-and-Stereo-System>3D Modeling: Camera Movement Estimation and path Correction for SFM Model using the Combination of Modified A-SIFT and Stereo System</a></li>
        <li><a href=#ProtoGS:-Efficient-and-High-Quality-Rendering-with-3D-Gaussian-Prototypes>ProtoGS: Efficient and High-Quality Rendering with 3D Gaussian Prototypes</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#LocDiffusion:-Identifying-Locations-on-Earth-by-Diffusing-in-the-Hilbert-Space>LocDiffusion: Identifying Locations on Earth by Diffusing in the Hilbert Space</a></li>
        <li><a href=#Selecting-and-Pruning:-A-Differentiable-Causal-Sequentialized-State-Space-Model-for-Two-View-Correspondence-Learning>Selecting and Pruning: A Differentiable Causal Sequentialized State-Space Model for Two-View Correspondence Learning</a></li>
        <li><a href=#What-Time-Tells-Us?-An-Explorative-Study-of-Time-Awareness-Learned-from-Static-Images>What Time Tells Us? An Explorative Study of Time Awareness Learned from Static Images</a></li>
        <li><a href=#good4cir:-Generating-Detailed-Synthetic-Captions-for-Composed-Image-Retrieval>good4cir: Generating Detailed Synthetic Captions for Composed Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Normalized-Matching-Transformer>Normalized Matching Transformer</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#NexusGS:-Sparse-View-Synthesis-with-Epipolar-Depth-Priors-in-3D-Gaussian-Splatting>NexusGS: Sparse View Synthesis with Epipolar Depth Priors in 3D Gaussian Splatting</a></li>
        <li><a href=#LookCloser:-Frequency-aware-Radiance-Field-for-Tiny-Detail-Scene>LookCloser: Frequency-aware Radiance Field for Tiny-Detail Scene</a></li>
        <li><a href=#NeRFPrior:-Learning-Neural-Radiance-Field-as-a-Prior-for-Indoor-Scene-Reconstruction>NeRFPrior: Learning Neural Radiance Field as a Prior for Indoor Scene Reconstruction</a></li>
        <li><a href=#End-to-End-Implicit-Neural-Representations-for-Classification>End-to-End Implicit Neural Representations for Classification</a></li>
        <li><a href=#Unraveling-the-Effects-of-Synthetic-Data-on-End-to-End-Autonomous-Driving>Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving</a></li>
        <li><a href=#PanopticSplatting:-End-to-End-Panoptic-Gaussian-Splatting>PanopticSplatting: End-to-End Panoptic Gaussian Splatting</a></li>
        <li><a href=#Splat-LOAM:-Gaussian-Splatting-LiDAR-Odometry-and-Mapping>Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [3D Modeling: Camera Movement Estimation and path Correction for SFM Model using the Combination of Modified A-SIFT and Stereo System](http://arxiv.org/abs/2503.17668)  
Usha Kumari, Shuvendu Rana  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Creating accurate and efficient 3D models poses significant challenges, particularly in addressing large viewpoint variations, computational complexity, and alignment discrepancies. Efficient camera path generation can help resolve these issues. In this context, a modified version of the Affine Scale-Invariant Feature Transform (ASIFT) is proposed to extract more matching points with reduced computational overhead, ensuring an adequate number of inliers for precise camera rotation angle estimation. Additionally, a novel two-camera-based rotation correction model is introduced to mitigate small rotational errors, further enhancing accuracy. Furthermore, a stereo camera-based translation estimation and correction model is implemented to determine camera movement in 3D space by altering the Structure From Motion (SFM) model. Finally, the novel combination of ASIFT and two camera-based SFM models provides an accurate camera movement trajectory in 3D space. Experimental results show that the proposed camera movement approach achieves 99.9% accuracy compared to the actual camera movement path and outperforms state-of-the-art camera path estimation methods. By leveraging this accurate camera path, the system facilitates the creation of precise 3D models, making it a robust solution for applications requiring high fidelity and efficiency in 3D reconstruction.  
  </ol>  
</details>  
  
### [ProtoGS: Efficient and High-Quality Rendering with 3D Gaussian Prototypes](http://arxiv.org/abs/2503.17486)  
Zhengqing Gao, Dongting Hu, Jia-Wang Bian, Huan Fu, Yan Li, Tongliang Liu, Mingming Gong, Kun Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D Gaussian Splatting (3DGS) has made significant strides in novel view synthesis but is limited by the substantial number of Gaussian primitives required, posing challenges for deployment on lightweight devices. Recent methods address this issue by compressing the storage size of densified Gaussians, yet fail to preserve rendering quality and efficiency. To overcome these limitations, we propose ProtoGS to learn Gaussian prototypes to represent Gaussian primitives, significantly reducing the total Gaussian amount without sacrificing visual quality. Our method directly uses Gaussian prototypes to enable efficient rendering and leverage the resulting reconstruction loss to guide prototype learning. To further optimize memory efficiency during training, we incorporate structure-from-motion (SfM) points as anchor points to group Gaussian primitives. Gaussian prototypes are derived within each group by clustering of K-means, and both the anchor points and the prototypes are optimized jointly. Our experiments on real-world and synthetic datasets prove that we outperform existing methods, achieving a substantial reduction in the number of Gaussians, and enabling high rendering speed while maintaining or even enhancing rendering fidelity.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [LocDiffusion: Identifying Locations on Earth by Diffusing in the Hilbert Space](http://arxiv.org/abs/2503.18142)  
Zhangyu Wang, Jielu Zhang, Zhongliang Zhou, Qian Cao, Nemin Wu, Zeping Liu, Lan Mu, Yang Song, Yiqun Xie, Ni Lao, Gengchen Mai  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image geolocalization is a fundamental yet challenging task, aiming at inferring the geolocation on Earth where an image is taken. Existing methods approach it either via grid-based classification or via image retrieval. Their performance significantly suffers when the spatial distribution of test images does not align with such choices. To address these limitations, we propose to leverage diffusion as a mechanism for image geolocalization. To avoid the problematic manifold reprojection step in diffusion, we developed a novel spherical positional encoding-decoding framework, which encodes points on a spherical surface (e.g., geolocations on Earth) into a Hilbert space of Spherical Harmonics coefficients and decodes points (geolocations) by mode-seeking. We call this type of position encoding Spherical Harmonics Dirac Delta (SHDD) Representation. We also propose a novel SirenNet-based architecture called CS-UNet to learn the conditional backward process in the latent SHDD space by minimizing a latent KL-divergence loss. We train a conditional latent diffusion model called LocDiffusion that generates geolocations under the guidance of images -- to the best of our knowledge, the first generative model for image geolocalization by diffusing geolocation information in a hidden location embedding space. We evaluate our method against SOTA image geolocalization baselines. LocDiffusion achieves competitive geolocalization performance and demonstrates significantly stronger generalizability to unseen geolocations.  
  </ol>  
</details>  
  
### [Selecting and Pruning: A Differentiable Causal Sequentialized State-Space Model for Two-View Correspondence Learning](http://arxiv.org/abs/2503.17938)  
Xiang Fang, Shihua Zhang, Hao Zhang, Tao Lu, Huabing Zhou, Jiayi Ma  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Two-view correspondence learning aims to discern true and false correspondences between image pairs by recognizing their underlying different information. Previous methods either treat the information equally or require the explicit storage of the entire context, tending to be laborious in real-world scenarios. Inspired by Mamba's inherent selectivity, we propose \textbf{CorrMamba}, a \textbf{Corr}espondence filter leveraging \textbf{Mamba}'s ability to selectively mine information from true correspondences while mitigating interference from false ones, thus achieving adaptive focus at a lower cost. To prevent Mamba from being potentially impacted by unordered keypoints that obscured its ability to mine spatial information, we customize a causal sequential learning approach based on the Gumbel-Softmax technique to establish causal dependencies between features in a fully autonomous and differentiable manner. Additionally, a local-context enhancement module is designed to capture critical contextual cues essential for correspondence pruning, complementing the core framework. Extensive experiments on relative pose estimation, visual localization, and analysis demonstrate that CorrMamba achieves state-of-the-art performance. Notably, in outdoor relative pose estimation, our method surpasses the previous SOTA by $2.58$ absolute percentage points in AUC@20\textdegree, highlighting its practical superiority. Our code will be publicly available.  
  </ol>  
</details>  
  
### [What Time Tells Us? An Explorative Study of Time Awareness Learned from Static Images](http://arxiv.org/abs/2503.17899)  
Dongheng Lin, Han Hu, Jianbo Jiao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Time becomes visible through illumination changes in what we see. Inspired by this, in this paper we explore the potential to learn time awareness from static images, trying to answer: what time tells us? To this end, we first introduce a Time-Oriented Collection (TOC) dataset, which contains 130,906 images with reliable timestamps. Leveraging this dataset, we propose a Time-Image Contrastive Learning (TICL) approach to jointly model timestamps and related visual representations through cross-modal contrastive learning. We found that the proposed TICL, 1) not only achieves state-of-the-art performance on the timestamp estimation task, over various benchmark metrics, 2) but also, interestingly, though only seeing static images, the time-aware embeddings learned from TICL show strong capability in several time-aware downstream tasks such as time-based image retrieval, video scene classification, and time-aware image editing. Our findings suggest that time-related visual cues can be learned from static images and are beneficial for various vision tasks, laying a foundation for future research on understanding time-related visual context. Project page:https://rathgrith.github.io/timetells/.  
  </ol>  
</details>  
  
### [good4cir: Generating Detailed Synthetic Captions for Composed Image Retrieval](http://arxiv.org/abs/2503.17871)  
Pranavi Kolouju, Eric Xing, Robert Pless, Nathan Jacobs, Abby Stylianou  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed image retrieval (CIR) enables users to search images using a reference image combined with textual modifications. Recent advances in vision-language models have improved CIR, but dataset limitations remain a barrier. Existing datasets often rely on simplistic, ambiguous, or insufficient manual annotations, hindering fine-grained retrieval. We introduce good4cir, a structured pipeline leveraging vision-language models to generate high-quality synthetic annotations. Our method involves: (1) extracting fine-grained object descriptions from query images, (2) generating comparable descriptions for target images, and (3) synthesizing textual instructions capturing meaningful transformations between images. This reduces hallucination, enhances modification diversity, and ensures object-level consistency. Applying our method improves existing datasets and enables creating new datasets across diverse domains. Results demonstrate improved retrieval accuracy for CIR models trained on our pipeline-generated datasets. We release our dataset construction framework to support further research in CIR and multi-modal retrieval.  
  </ol>  
</details>  
  
  



## Image Matching  

### [Normalized Matching Transformer](http://arxiv.org/abs/2503.17715)  
Abtin Pourhadi, Paul Swoboda  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present a new state of the art approach for sparse keypoint matching between pairs of images. Our method consists of a fully deep learning based approach combining a visual backbone coupled with a SplineCNN graph neural network for feature processing and a normalized transformer decoder for decoding keypoint correspondences together with the Sinkhorn algorithm. Our method is trained using a contrastive and a hyperspherical loss for better feature representations. We additionally use data augmentation during training. This comparatively simple architecture combining extensive normalization and advanced losses outperforms current state of the art approaches on PascalVOC and SPair-71k datasets by $5.1\%$ and $2.2\%$ respectively compared to BBGM, ASAR, COMMON and GMTR while training for at least $1.7x$ fewer epochs.  
  </ol>  
</details>  
  
  



## NeRF  

### [NexusGS: Sparse View Synthesis with Epipolar Depth Priors in 3D Gaussian Splatting](http://arxiv.org/abs/2503.18794)  
Yulong Zheng, Zicheng Jiang, Shengfeng He, Yandu Sun, Junyu Dong, Huaidong Zhang, Yong Du  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Field (NeRF) and 3D Gaussian Splatting (3DGS) have noticeably advanced photo-realistic novel view synthesis using images from densely spaced camera viewpoints. However, these methods struggle in few-shot scenarios due to limited supervision. In this paper, we present NexusGS, a 3DGS-based approach that enhances novel view synthesis from sparse-view images by directly embedding depth information into point clouds, without relying on complex manual regularizations. Exploiting the inherent epipolar geometry of 3DGS, our method introduces a novel point cloud densification strategy that initializes 3DGS with a dense point cloud, reducing randomness in point placement while preventing over-smoothing and overfitting. Specifically, NexusGS comprises three key steps: Epipolar Depth Nexus, Flow-Resilient Depth Blending, and Flow-Filtered Depth Pruning. These steps leverage optical flow and camera poses to compute accurate depth maps, while mitigating the inaccuracies often associated with optical flow. By incorporating epipolar depth priors, NexusGS ensures reliable dense point cloud coverage and supports stable 3DGS training under sparse-view conditions. Experiments demonstrate that NexusGS significantly enhances depth accuracy and rendering quality, surpassing state-of-the-art methods by a considerable margin. Furthermore, we validate the superiority of our generated point clouds by substantially boosting the performance of competing methods. Project page: https://usmizuki.github.io/NexusGS/.  
  </ol>  
</details>  
**comments**: This paper is accepted by CVPR 2025  
  
### [LookCloser: Frequency-aware Radiance Field for Tiny-Detail Scene](http://arxiv.org/abs/2503.18513)  
Xiaoyu Zhang, Weihong Pan, Chong Bao, Xiyu Zhang, Xiaojun Xiang, Hanqing Jiang, Hujun Bao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Humans perceive and comprehend their surroundings through information spanning multiple frequencies. In immersive scenes, people naturally scan their environment to grasp its overall structure while examining fine details of objects that capture their attention. However, current NeRF frameworks primarily focus on modeling either high-frequency local views or the broad structure of scenes with low-frequency information, which is limited to balancing both. We introduce FA-NeRF, a novel frequency-aware framework for view synthesis that simultaneously captures the overall scene structure and high-definition details within a single NeRF model. To achieve this, we propose a 3D frequency quantification method that analyzes the scene's frequency distribution, enabling frequency-aware rendering. Our framework incorporates a frequency grid for fast convergence and querying, a frequency-aware feature re-weighting strategy to balance features across different frequency contents. Extensive experiments show that our method significantly outperforms existing approaches in modeling entire scenes while preserving fine details.  
  </ol>  
</details>  
**comments**: 8 pages, 6 figures. Accepted to CVPR 2025  
  
### [NeRFPrior: Learning Neural Radiance Field as a Prior for Indoor Scene Reconstruction](http://arxiv.org/abs/2503.18361)  
Wenyuan Zhang, Emily Yue-ting Jia, Junsheng Zhou, Baorui Ma, Kanle Shi, Yu-Shen Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recently, it has shown that priors are vital for neural implicit functions to reconstruct high-quality surfaces from multi-view RGB images. However, current priors require large-scale pre-training, and merely provide geometric clues without considering the importance of color. In this paper, we present NeRFPrior, which adopts a neural radiance field as a prior to learn signed distance fields using volume rendering for surface reconstruction. Our NeRF prior can provide both geometric and color clues, and also get trained fast under the same scene without additional data. Based on the NeRF prior, we are enabled to learn a signed distance function (SDF) by explicitly imposing a multi-view consistency constraint on each ray intersection for surface inference. Specifically, at each ray intersection, we use the density in the prior as a coarse geometry estimation, while using the color near the surface as a clue to check its visibility from another view angle. For the textureless areas where the multi-view consistency constraint does not work well, we further introduce a depth consistency loss with confidence weights to infer the SDF. Our experimental results outperform the state-of-the-art methods under the widely used benchmarks.  
  </ol>  
</details>  
**comments**: Accepted by CVPR 2025. Project page:
  https://wen-yuan-zhang.github.io/NeRFPrior/  
  
### [End-to-End Implicit Neural Representations for Classification](http://arxiv.org/abs/2503.18123)  
Alexander Gielisse, Jan van Gemert  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Implicit neural representations (INRs) such as NeRF and SIREN encode a signal in neural network parameters and show excellent results for signal reconstruction. Using INRs for downstream tasks, such as classification, is however not straightforward. Inherent symmetries in the parameters pose challenges and current works primarily focus on designing architectures that are equivariant to these symmetries. However, INR-based classification still significantly under-performs compared to pixel-based methods like CNNs. This work presents an end-to-end strategy for initializing SIRENs together with a learned learning-rate scheme, to yield representations that improve classification accuracy. We show that a simple, straightforward, Transformer model applied to a meta-learned SIREN, without incorporating explicit symmetry equivariances, outperforms the current state-of-the-art. On the CIFAR-10 SIREN classification task, we improve the state-of-the-art without augmentations from 38.8% to 59.6%, and from 63.4% to 64.7% with augmentations. We demonstrate scalability on the high-resolution Imagenette dataset achieving reasonable reconstruction quality with a classification accuracy of 60.8% and are the first to do INR classification on the full ImageNet-1K dataset where we achieve a SIREN classification performance of 23.6%. To the best of our knowledge, no other SIREN classification approach has managed to set a classification baseline for any high-resolution image dataset. Our code is available at https://github.com/SanderGielisse/MWT  
  </ol>  
</details>  
**comments**: Accepted to CVPR 2025. 8 pages, supplementary material included  
  
### [Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving](http://arxiv.org/abs/2503.18108)  
Junhao Ge, Zuhong Liu, Longteng Fan, Yifan Jiang, Jiaqi Su, Yiming Li, Zhejun Zhang, Siheng Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    End-to-end (E2E) autonomous driving (AD) models require diverse, high-quality data to perform well across various driving scenarios. However, collecting large-scale real-world data is expensive and time-consuming, making high-fidelity synthetic data essential for enhancing data diversity and model robustness. Existing driving simulators for synthetic data generation have significant limitations: game-engine-based simulators struggle to produce realistic sensor data, while NeRF-based and diffusion-based methods face efficiency challenges. Additionally, recent simulators designed for closed-loop evaluation provide limited interaction with other vehicles, failing to simulate complex real-world traffic dynamics. To address these issues, we introduce SceneCrafter, a realistic, interactive, and efficient AD simulator based on 3D Gaussian Splatting (3DGS). SceneCrafter not only efficiently generates realistic driving logs across diverse traffic scenarios but also enables robust closed-loop evaluation of end-to-end models. Experimental results demonstrate that SceneCrafter serves as both a reliable evaluation platform and a efficient data generator that significantly improves end-to-end model generalization.  
  </ol>  
</details>  
  
### [PanopticSplatting: End-to-End Panoptic Gaussian Splatting](http://arxiv.org/abs/2503.18073)  
Yuxuan Xie, Xuan Yu, Changjian Jiang, Sitong Mao, Shunbo Zhou, Rui Fan, Rong Xiong, Yue Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Open-vocabulary panoptic reconstruction is a challenging task for simultaneous scene reconstruction and understanding. Recently, methods have been proposed for 3D scene understanding based on Gaussian splatting. However, these methods are multi-staged, suffering from the accumulated errors and the dependence of hand-designed components. To streamline the pipeline and achieve global optimization, we propose PanopticSplatting, an end-to-end system for open-vocabulary panoptic reconstruction. Our method introduces query-guided Gaussian segmentation with local cross attention, lifting 2D instance masks without cross-frame association in an end-to-end way. The local cross attention within view frustum effectively reduces the training memory, making our model more accessible to large scenes with more Gaussians and objects. In addition, to address the challenge of noisy labels in 2D pseudo masks, we propose label blending to promote consistent 3D segmentation with less noisy floaters, as well as label warping on 2D predictions which enhances multi-view coherence and segmentation accuracy. Our method demonstrates strong performances in 3D scene panoptic reconstruction on the ScanNet-V2 and ScanNet++ datasets, compared with both NeRF-based and Gaussian-based panoptic reconstruction methods. Moreover, PanopticSplatting can be easily generalized to numerous variants of Gaussian splatting, and we demonstrate its robustness on different Gaussian base models.  
  </ol>  
</details>  
**comments**: 8 pages, 6 figures  
  
### [Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](http://arxiv.org/abs/2503.17491)  
Emanuele Giacomini, Luca Di Giammarino, Lorenzo De Rebotti, Giorgio Grisetti, Martin R. Oswald  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    LiDARs provide accurate geometric measurements, making them valuable for ego-motion estimation and reconstruction tasks. Although its success, managing an accurate and lightweight representation of the environment still poses challenges. Both classic and NeRF-based solutions have to trade off accuracy over memory and processing times. In this work, we build on recent advancements in Gaussian Splatting methods to develop a novel LiDAR odometry and mapping pipeline that exclusively relies on Gaussian primitives for its scene representation. Leveraging spherical projection, we drive the refinement of the primitives uniquely from LiDAR measurements. Experiments show that our approach matches the current registration performance, while achieving SOTA results for mapping tasks with minimal GPU requirements. This efficiency makes it a strong candidate for further exploration and potential adoption in real-time robotics estimation tasks.  
  </ol>  
</details>  
**comments**: submitted to ICCV 2025  
  
  



