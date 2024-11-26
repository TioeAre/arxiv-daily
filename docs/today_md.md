<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#ZeroGS:-Training-3D-Gaussian-Splatting-from-Unposed-Images>ZeroGS: Training 3D Gaussian Splatting from Unposed Images</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Image-Generation-Diversity-Issues-and-How-to-Tame-Them>Image Generation Diversity Issues and How to Tame Them</a></li>
        <li><a href=#PG-SLAM:-Photo-realistic-and-Geometry-aware-RGB-D-SLAM-in-Dynamic-Environments>PG-SLAM: Photo-realistic and Geometry-aware RGB-D SLAM in Dynamic Environments</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#OCDet:-Object-Center-Detection-via-Bounding-Box-Aware-Heatmap-Prediction-on-Edge-Devices-with-NPUs>OCDet: Object Center Detection via Bounding Box-Aware Heatmap Prediction on Edge Devices with NPUs</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Quadratic-Gaussian-Splatting-for-Efficient-and-Detailed-Surface-Reconstruction>Quadratic Gaussian Splatting for Efficient and Detailed Surface Reconstruction</a></li>
        <li><a href=#U2NeRF:-Unsupervised-Underwater-Image-Restoration-and-Neural-Radiance-Fields>U2NeRF: Unsupervised Underwater Image Restoration and Neural Radiance Fields</a></li>
        <li><a href=#ZeroGS:-Training-3D-Gaussian-Splatting-from-Unposed-Images>ZeroGS: Training 3D Gaussian Splatting from Unposed Images</a></li>
        <li><a href=#GSurf:-3D-Reconstruction-via-Signed-Distance-Fields-with-Direct-Gaussian-Supervision>GSurf: 3D Reconstruction via Signed Distance Fields with Direct Gaussian Supervision</a></li>
        <li><a href=#NeRF-Inpainting-with-Geometric-Diffusion-Prior-and-Balanced-Score-Distillation>NeRF Inpainting with Geometric Diffusion Prior and Balanced Score Distillation</a></li>
        <li><a href=#SplatSDF:-Boosting-Neural-Implicit-SDF-via-Gaussian-Splatting-Fusion>SplatSDF: Boosting Neural Implicit SDF via Gaussian Splatting Fusion</a></li>
        <li><a href=#Sparse-Input-View-Synthesis:-3D-Representations-and-Reliable-Priors>Sparse Input View Synthesis: 3D Representations and Reliable Priors</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [ZeroGS: Training 3D Gaussian Splatting from Unposed Images](http://arxiv.org/abs/2411.15779)  
Yu Chen, Rolandos Alexandros Potamias, Evangelos Ververas, Jifei Song, Jiankang Deng, Gim Hee Lee  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural radiance fields (NeRF) and 3D Gaussian Splatting (3DGS) are popular techniques to reconstruct and render photo-realistic images. However, the pre-requisite of running Structure-from-Motion (SfM) to get camera poses limits their completeness. While previous methods can reconstruct from a few unposed images, they are not applicable when images are unordered or densely captured. In this work, we propose ZeroGS to train 3DGS from hundreds of unposed and unordered images. Our method leverages a pretrained foundation model as the neural scene representation. Since the accuracy of the predicted pointmaps does not suffice for accurate image registration and high-fidelity image rendering, we propose to mitigate the issue by initializing and finetuning the pretrained model from a seed image. Images are then progressively registered and added to the training buffer, which is further used to train the model. We also propose to refine the camera poses and pointmaps by minimizing a point-to-camera ray consistency loss across multiple views. Experiments on the LLFF dataset, the MipNeRF360 dataset, and the Tanks-and-Temples dataset show that our method recovers more accurate camera poses than state-of-the-art pose-free NeRF/3DGS methods, and even renders higher quality images than 3DGS with COLMAP poses. Our project page is available at https://aibluefisher.github.io/ZeroGS.  
  </ol>  
</details>  
**comments**: 16 pages, 12 figures  
  
  



## Visual Localization  

### [Image Generation Diversity Issues and How to Tame Them](http://arxiv.org/abs/2411.16171)  
[[code](https://github.com/mischad/beyondfid)]  
Mischa Dombrowski, Weitong Zhang, Sarah Cechnicka, Hadrien Reynaud, Bernhard Kainz  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Generative methods now produce outputs nearly indistinguishable from real data but often fail to fully capture the data distribution. Unlike quality issues, diversity limitations in generative models are hard to detect visually, requiring specific metrics for assessment. In this paper, we draw attention to the current lack of diversity in generative models and the inability of common metrics to measure this. We achieve this by framing diversity as an image retrieval problem, where we measure how many real images can be retrieved using synthetic data as queries. This yields the Image Retrieval Score (IRS), an interpretable, hyperparameter-free metric that quantifies the diversity of a generative model's output. IRS requires only a subset of synthetic samples and provides a statistical measure of confidence. Our experiments indicate that current feature extractors commonly used in generative model assessment are inadequate for evaluating diversity effectively. Consequently, we perform an extensive search for the best feature extractors to assess diversity. Evaluation reveals that current diffusion models converge to limited subsets of the real distribution, with no current state-of-the-art models superpassing 77% of the diversity of the training data. To address this limitation, we introduce Diversity-Aware Diffusion Models (DiADM), a novel approach that improves diversity of unconditional diffusion models without loss of image quality. We do this by disentangling diversity from image quality by using a diversity aware module that uses pseudo-unconditional features as input. We provide a Python package offering unified feature extraction and metric computation to further facilitate the evaluation of generative models https://github.com/MischaD/beyondfid.  
  </ol>  
</details>  
**comments**: 17 pages, 6 tables, 12 figures  
  
### [PG-SLAM: Photo-realistic and Geometry-aware RGB-D SLAM in Dynamic Environments](http://arxiv.org/abs/2411.15800)  
Haoang Li, Xiangqi Meng, Xingxing Zuo, Zhe Liu, Hesheng Wang, Daniel Cremers  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Simultaneous localization and mapping (SLAM) has achieved impressive performance in static environments. However, SLAM in dynamic environments remains an open question. Many methods directly filter out dynamic objects, resulting in incomplete scene reconstruction and limited accuracy of camera localization. The other works express dynamic objects by point clouds, sparse joints, or coarse meshes, which fails to provide a photo-realistic representation. To overcome the above limitations, we propose a photo-realistic and geometry-aware RGB-D SLAM method by extending Gaussian splatting. Our method is composed of three main modules to 1) map the dynamic foreground including non-rigid humans and rigid items, 2) reconstruct the static background, and 3) localize the camera. To map the foreground, we focus on modeling the deformations and/or motions. We consider the shape priors of humans and exploit geometric and appearance constraints of humans and items. For background mapping, we design an optimization strategy between neighboring local maps by integrating appearance constraint into geometric alignment. As to camera localization, we leverage both static background and dynamic foreground to increase the observations for noise compensation. We explore the geometric and appearance constraints by associating 3D Gaussians with 2D optical flows and pixel patches. Experiments on various real-world datasets demonstrate that our method outperforms state-of-the-art approaches in terms of camera localization and scene representation. Source codes will be publicly available upon paper acceptance.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [OCDet: Object Center Detection via Bounding Box-Aware Heatmap Prediction on Edge Devices with NPUs](http://arxiv.org/abs/2411.15653)  
[[code](https://github.com/chen-xin-94/ocdet)]  
Chen Xin, Thomas Motz, Andreas Hartel, Enkelejda Kasneci  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Real-time object localization on edge devices is fundamental for numerous applications, ranging from surveillance to industrial automation. Traditional frameworks, such as object detection, segmentation, and keypoint detection, struggle in resource-constrained environments, often resulting in substantial target omissions. To address these challenges, we introduce OCDet, a lightweight Object Center Detection framework optimized for edge devices with NPUs. OCDet predicts heatmaps representing object center probabilities and extracts center points through peak identification. Unlike prior methods using fixed Gaussian distribution, we introduce Generalized Centerness (GC) to generate ground truth heatmaps from bounding box annotations, providing finer spatial details without additional manual labeling. Built on NPU-friendly Semantic FPN with MobileNetV4 backbones, OCDet models are trained by our Balanced Continuous Focal Loss (BCFL), which alleviates data imbalance and focuses training on hard negative examples for probability regression tasks. Leveraging the novel Center Alignment Score (CAS) with Hungarian matching, we demonstrate that OCDet consistently outperforms YOLO11 in object center detection, achieving up to 23% higher CAS while requiring 42% fewer parameters, 34% less computation, and 64% lower NPU latency. When compared to keypoint detection frameworks, OCDet achieves substantial CAS improvements up to 186% using identical models. By integrating GC, BCFL, and CAS, OCDet establishes a new paradigm for efficient and robust object center detection on edge devices with NPUs. The code is released at https://github.com/chen-xin-94/ocdet.  
  </ol>  
</details>  
  
  



## NeRF  

### [Quadratic Gaussian Splatting for Efficient and Detailed Surface Reconstruction](http://arxiv.org/abs/2411.16392)  
Ziyu Zhang, Binbin Huang, Hanqing Jiang, Liyang Zhou, Xiaojun Xiang, Shunhan Shen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recently, 3D Gaussian Splatting (3DGS) has attracted attention for its superior rendering quality and speed over Neural Radiance Fields (NeRF). To address 3DGS's limitations in surface representation, 2D Gaussian Splatting (2DGS) introduced disks as scene primitives to model and reconstruct geometries from multi-view images, offering view-consistent geometry. However, the disk's first-order linear approximation often leads to over-smoothed results. We propose Quadratic Gaussian Splatting (QGS), a novel method that replaces disks with quadric surfaces, enhancing geometric fitting, whose code will be open-sourced. QGS defines Gaussian distributions in non-Euclidean space, allowing primitives to capture more complex textures. As a second-order surface approximation, QGS also renders spatial curvature to guide the normal consistency term, to effectively reduce over-smoothing. Moreover, QGS is a generalized version of 2DGS that achieves more accurate and detailed reconstructions, as verified by experiments on DTU and TNT, demonstrating its effectiveness in surpassing current state-of-the-art methods in geometry reconstruction. Our code willbe released as open source.  
  </ol>  
</details>  
  
### [U2NeRF: Unsupervised Underwater Image Restoration and Neural Radiance Fields](http://arxiv.org/abs/2411.16172)  
Vinayak Gupta, Manoj S, Mukund Varma T, Kaushik Mitra  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Underwater images suffer from colour shifts, low contrast, and haziness due to light absorption, refraction, scattering and restoring these images has warranted much attention. In this work, we present Unsupervised Underwater Neural Radiance Field U2NeRF, a transformer-based architecture that learns to render and restore novel views conditioned on multi-view geometry simultaneously. Due to the absence of supervision, we attempt to implicitly bake restoring capabilities onto the NeRF pipeline and disentangle the predicted color into several components - scene radiance, direct transmission map, backscatter transmission map, and global background light, and when combined reconstruct the underwater image in a self-supervised manner. In addition, we release an Underwater View Synthesis UVS dataset consisting of 12 underwater scenes, containing both synthetically-generated and real-world data. Our experiments demonstrate that when optimized on a single scene, U2NeRF outperforms several baselines by as much LPIPS 11%, UIQM 5%, UCIQE 4% (on average) and showcases improved rendering and restoration capabilities. Code will be made available upon acceptance.  
  </ol>  
</details>  
**comments**: ICLR Tiny Papers 2024. arXiv admin note: text overlap with
  arXiv:2207.13298  
  
### [ZeroGS: Training 3D Gaussian Splatting from Unposed Images](http://arxiv.org/abs/2411.15779)  
Yu Chen, Rolandos Alexandros Potamias, Evangelos Ververas, Jifei Song, Jiankang Deng, Gim Hee Lee  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural radiance fields (NeRF) and 3D Gaussian Splatting (3DGS) are popular techniques to reconstruct and render photo-realistic images. However, the pre-requisite of running Structure-from-Motion (SfM) to get camera poses limits their completeness. While previous methods can reconstruct from a few unposed images, they are not applicable when images are unordered or densely captured. In this work, we propose ZeroGS to train 3DGS from hundreds of unposed and unordered images. Our method leverages a pretrained foundation model as the neural scene representation. Since the accuracy of the predicted pointmaps does not suffice for accurate image registration and high-fidelity image rendering, we propose to mitigate the issue by initializing and finetuning the pretrained model from a seed image. Images are then progressively registered and added to the training buffer, which is further used to train the model. We also propose to refine the camera poses and pointmaps by minimizing a point-to-camera ray consistency loss across multiple views. Experiments on the LLFF dataset, the MipNeRF360 dataset, and the Tanks-and-Temples dataset show that our method recovers more accurate camera poses than state-of-the-art pose-free NeRF/3DGS methods, and even renders higher quality images than 3DGS with COLMAP poses. Our project page is available at https://aibluefisher.github.io/ZeroGS.  
  </ol>  
</details>  
**comments**: 16 pages, 12 figures  
  
### [GSurf: 3D Reconstruction via Signed Distance Fields with Direct Gaussian Supervision](http://arxiv.org/abs/2411.15723)  
[[code](https://github.com/xubaixinxbx/gsurf)]  
Xu Baixin, Hu Jiangbei, Li Jiaze, He Ying  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Surface reconstruction from multi-view images is a core challenge in 3D vision. Recent studies have explored signed distance fields (SDF) within Neural Radiance Fields (NeRF) to achieve high-fidelity surface reconstructions. However, these approaches often suffer from slow training and rendering speeds compared to 3D Gaussian splatting (3DGS). Current state-of-the-art techniques attempt to fuse depth information to extract geometry from 3DGS, but frequently result in incomplete reconstructions and fragmented surfaces. In this paper, we introduce GSurf, a novel end-to-end method for learning a signed distance field directly from Gaussian primitives. The continuous and smooth nature of SDF addresses common issues in the 3DGS family, such as holes resulting from noisy or missing depth data. By using Gaussian splatting for rendering, GSurf avoids the redundant volume rendering typically required in other GS and SDF integrations. Consequently, GSurf achieves faster training and rendering speeds while delivering 3D reconstruction quality comparable to neural implicit surface methods, such as VolSDF and NeuS. Experimental results across various benchmark datasets demonstrate the effectiveness of our method in producing high-fidelity 3D reconstructions.  
  </ol>  
</details>  
**comments**: see https://github.com/xubaixinxbx/Gsurf  
  
### [NeRF Inpainting with Geometric Diffusion Prior and Balanced Score Distillation](http://arxiv.org/abs/2411.15551)  
Menglin Zhang, Xin Luo, Yunwei Lan, Chang Liu, Rui Li, Kaidong Zhang, Ganlin Yang, Dong Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advances in NeRF inpainting have leveraged pretrained diffusion models to enhance performance. However, these methods often yield suboptimal results due to their ineffective utilization of 2D diffusion priors. The limitations manifest in two critical aspects: the inadequate capture of geometric information by pretrained diffusion models and the suboptimal guidance provided by existing Score Distillation Sampling (SDS) methods. To address these problems, we introduce GB-NeRF, a novel framework that enhances NeRF inpainting through improved utilization of 2D diffusion priors. Our approach incorporates two key innovations: a fine-tuning strategy that simultaneously learns appearance and geometric priors and a specialized normal distillation loss that integrates these geometric priors into NeRF inpainting. We propose a technique called Balanced Score Distillation (BSD) that surpasses existing methods such as Score Distillation (SDS) and the improved version, Conditional Score Distillation (CSD). BSD offers improved inpainting quality in appearance and geometric aspects. Extensive experiments show that our method provides superior appearance fidelity and geometric consistency compared to existing approaches.  
  </ol>  
</details>  
  
### [SplatSDF: Boosting Neural Implicit SDF via Gaussian Splatting Fusion](http://arxiv.org/abs/2411.15468)  
Runfa Blark Li, Keito Suzuki, Bang Du, Ki Myung Brian Le, Nikolay Atanasov, Truong Nguyen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    A signed distance function (SDF) is a useful representation for continuous-space geometry and many related operations, including rendering, collision checking, and mesh generation. Hence, reconstructing SDF from image observations accurately and efficiently is a fundamental problem. Recently, neural implicit SDF (SDF-NeRF) techniques, trained using volumetric rendering, have gained a lot of attention. Compared to earlier truncated SDF (TSDF) fusion algorithms that rely on depth maps and voxelize continuous space, SDF-NeRF enables continuous-space SDF reconstruction with better geometric and photometric accuracy. However, the accuracy and convergence speed of scene-level SDF reconstruction require further improvements for many applications. With the advent of 3D Gaussian Splatting (3DGS) as an explicit representation with excellent rendering quality and speed, several works have focused on improving SDF-NeRF by introducing consistency losses on depth and surface normals between 3DGS and SDF-NeRF. However, loss-level connections alone lead to incremental improvements. We propose a novel neural implicit SDF called "SplatSDF" to fuse 3DGSandSDF-NeRF at an architecture level with significant boosts to geometric and photometric accuracy and convergence speed. Our SplatSDF relies on 3DGS as input only during training, and keeps the same complexity and efficiency as the original SDF-NeRF during inference. Our method outperforms state-of-the-art SDF-NeRF models on geometric and photometric evaluation by the time of submission.  
  </ol>  
</details>  
  
### [Sparse Input View Synthesis: 3D Representations and Reliable Priors](http://arxiv.org/abs/2411.13631)  
Nagabhushan Somraj  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Novel view synthesis refers to the problem of synthesizing novel viewpoints of a scene given the images from a few viewpoints. This is a fundamental problem in computer vision and graphics, and enables a vast variety of applications such as meta-verse, free-view watching of events, video gaming, video stabilization and video compression. Recent 3D representations such as radiance fields and multi-plane images significantly improve the quality of images rendered from novel viewpoints. However, these models require a dense sampling of input views for high quality renders. Their performance goes down significantly when only a few input views are available. In this thesis, we focus on the sparse input novel view synthesis problem for both static and dynamic scenes. In the first part of this work, we mainly focus on sparse input novel view synthesis of static scenes using neural radiance fields (NeRF). We study the design of reliable and dense priors to better regularize the NeRF in such situations. In particular, we propose a prior on the visibility of the pixels in a pair of input views. We show that this visibility prior, which is related to the relative depth of objects, is dense and more reliable than existing priors on absolute depth. We compute the visibility prior using plane sweep volumes without the need to train a neural network on large datasets. We evaluate our approach on multiple datasets and show that our model outperforms existing approaches for sparse input novel view synthesis. In the second part, we aim to further improve the regularization by learning a scene-specific prior that does not suffer from generalization issues. We achieve this by learning the prior on the given scene alone without pre-training on large datasets. In particular, we design augmented NeRFs to obtain better depth supervision in certain regions of the scene for the main NeRF. Further, we extend this framework to also apply to newer and faster radiance field models such as TensoRF and ZipNeRF. Through extensive experiments on multiple datasets, we show the superiority of our approach in sparse input novel view synthesis. The design of sparse input fast dynamic radiance fields is severely constrained by the lack of suitable representations and reliable priors for motion. We address the first challenge by designing an explicit motion model based on factorized volumes that is compact and optimizes quickly. We also introduce reliable sparse flow priors to constrain the motion field, since we find that the popularly employed dense optical flow priors are unreliable. We show the benefits of our motion representation and reliable priors on multiple datasets. In the final part of this thesis, we study the application of view synthesis for frame rate upsampling in video gaming. Specifically, we consider the problem of temporal view synthesis, where the goal is to predict the future frames given the past frames and the camera motion. The key challenge here is in predicting the future motion of the objects by estimating their past motion and extrapolating it. We explore the use of multi-plane image representations and scene depth to reliably estimate the object motion, particularly in the occluded regions. We design a new database to effectively evaluate our approach for temporal view synthesis of dynamic scenes and show that we achieve state-of-the-art performance.  
  </ol>  
</details>  
**comments**: PhD Thesis of Nagabhushan S N, Dept of ECE, Indian Institute of
  Science (IISc); Advisor: Dr. Rajiv Soundararajan; Thesis Reviewers: Dr.
  Kaushik Mitra (IIT Madras), Dr. Aniket Bera (Purdue University); Submitted:
  May 2024; Accepted and Defended: Sep 2024; Abstract condensed, please check
  the PDF for full abstract  
  
  



