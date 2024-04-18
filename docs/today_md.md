<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#VBR:-A-Vision-Benchmark-in-Rome>VBR: A Vision Benchmark in Rome</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#A-Subspace-Constrained-Tyler's-Estimator-and-its-Applications-to-Structure-from-Motion>A Subspace-Constrained Tyler's Estimator and its Applications to Structure from Motion</a></li>
        <li><a href=#DeblurGS:-Gaussian-Splatting-for-Camera-Motion-Blur>DeblurGS: Gaussian Splatting for Camera Motion Blur</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Improving-Composed-Image-Retrieval-via-Contrastive-Learning-with-Scaling-Positives-and-Negatives>Improving Composed Image Retrieval via Contrastive Learning with Scaling Positives and Negatives</a></li>
        <li><a href=#Spatial-Aware-Image-Retrieval:-A-Hyperdimensional-Computing-Approach-for-Efficient-Similarity-Hashing>Spatial-Aware Image Retrieval: A Hyperdimensional Computing Approach for Efficient Similarity Hashing</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Pixel-Wise-Symbol-Spotting-via-Progressive-Points-Location-for-Parsing-CAD-Images>Pixel-Wise Symbol Spotting via Progressive Points Location for Parsing CAD Images</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#A-Semantic-Segmentation-guided-Approach-for-Ground-to-Aerial-Image-Matching>A Semantic Segmentation-guided Approach for Ground-to-Aerial Image Matching</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#SLAIM:-Robust-Dense-Neural-SLAM-for-Online-Tracking-and-Mapping>SLAIM: Robust Dense Neural SLAM for Online Tracking and Mapping</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [VBR: A Vision Benchmark in Rome](http://arxiv.org/abs/2404.11322)  
Leonardo Brizi, Emanuele Giacomini, Luca Di Giammarino, Simone Ferrari, Omar Salem, Lorenzo De Rebotti, Giorgio Grisetti  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper presents a vision and perception research dataset collected in Rome, featuring RGB data, 3D point clouds, IMU, and GPS data. We introduce a new benchmark targeting visual odometry and SLAM, to advance the research in autonomous robotics and computer vision. This work complements existing datasets by simultaneously addressing several issues, such as environment diversity, motion patterns, and sensor frequency. It uses up-to-date devices and presents effective procedures to accurately calibrate the intrinsic and extrinsic of the sensors while addressing temporal synchronization. During recording, we cover multi-floor buildings, gardens, urban and highway scenarios. Combining handheld and car-based data collections, our setup can simulate any robot (quadrupeds, quadrotors, autonomous vehicles). The dataset includes an accurate 6-dof ground truth based on a novel methodology that refines the RTK-GPS estimate with LiDAR point clouds through Bundle Adjustment. All sequences divided in training and testing are accessible through our website.  
  </ol>  
</details>  
**comments**: Accepted at IEEE ICRA 2024 Website:
  https://rvp-group.net/datasets/slam.html  
  
  



## SFM  

### [A Subspace-Constrained Tyler's Estimator and its Applications to Structure from Motion](http://arxiv.org/abs/2404.11590)  
Feng Yu, Teng Zhang, Gilad Lerman  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present the subspace-constrained Tyler's estimator (STE) designed for recovering a low-dimensional subspace within a dataset that may be highly corrupted with outliers. STE is a fusion of the Tyler's M-estimator (TME) and a variant of the fast median subspace. Our theoretical analysis suggests that, under a common inlier-outlier model, STE can effectively recover the underlying subspace, even when it contains a smaller fraction of inliers relative to other methods in the field of robust subspace recovery. We apply STE in the context of Structure from Motion (SfM) in two ways: for robust estimation of the fundamental matrix and for the removal of outlying cameras, enhancing the robustness of the SfM pipeline. Numerical experiments confirm the state-of-the-art performance of our method in these applications. This research makes significant contributions to the field of robust subspace recovery, particularly in the context of computer vision and 3D reconstruction.  
  </ol>  
</details>  
**comments**: 23 pages, accepted by CVPR 24  
  
### [DeblurGS: Gaussian Splatting for Camera Motion Blur](http://arxiv.org/abs/2404.11358)  
Jeongtaek Oh, Jaeyoung Chung, Dongwoo Lee, Kyoung Mu Lee  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Although significant progress has been made in reconstructing sharp 3D scenes from motion-blurred images, a transition to real-world applications remains challenging. The primary obstacle stems from the severe blur which leads to inaccuracies in the acquisition of initial camera poses through Structure-from-Motion, a critical aspect often overlooked by previous approaches. To address this challenge, we propose DeblurGS, a method to optimize sharp 3D Gaussian Splatting from motion-blurred images, even with the noisy camera pose initialization. We restore a fine-grained sharp scene by leveraging the remarkable reconstruction capability of 3D Gaussian Splatting. Our approach estimates the 6-Degree-of-Freedom camera motion for each blurry observation and synthesizes corresponding blurry renderings for the optimization process. Furthermore, we propose Gaussian Densification Annealing strategy to prevent the generation of inaccurate Gaussians at erroneous locations during the early training stages when camera motion is still imprecise. Comprehensive experiments demonstrate that our DeblurGS achieves state-of-the-art performance in deblurring and novel view synthesis for real-world and synthetic benchmark datasets, as well as field-captured blurry smartphone videos.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Improving Composed Image Retrieval via Contrastive Learning with Scaling Positives and Negatives](http://arxiv.org/abs/2404.11317)  
Zhangchi Feng, Richong Zhang, Zhijie Nie  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The Composed Image Retrieval (CIR) task aims to retrieve target images using a composed query consisting of a reference image and a modified text. Advanced methods often utilize contrastive learning as the optimization objective, which benefits from adequate positive and negative examples. However, the triplet for CIR incurs high manual annotation costs, resulting in limited positive examples. Furthermore, existing methods commonly use in-batch negative sampling, which reduces the negative number available for the model. To address the problem of lack of positives, we propose a data generation method by leveraging a multi-modal large language model to construct triplets for CIR. To introduce more negatives during fine-tuning, we design a two-stage fine-tuning framework for CIR, whose second stage introduces plenty of static representations of negatives to optimize the representation space rapidly. The above two improvements can be effectively stacked and designed to be plug-and-play, easily applied to existing CIR models without changing their original architectures. Extensive experiments and ablation analysis demonstrate that our method effectively scales positives and negatives and achieves state-of-the-art results on both FashionIQ and CIRR datasets. In addition, our methods also perform well in zero-shot composed image retrieval, providing a new CIR solution for the low-resources scenario.  
  </ol>  
</details>  
**comments**: 12 pages, 11 figures  
  
### [Spatial-Aware Image Retrieval: A Hyperdimensional Computing Approach for Efficient Similarity Hashing](http://arxiv.org/abs/2404.11025)  
Sanggeon Yun, Ryozo Masukawa, SungHeon Jeong, Mohsen Imani  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In the face of burgeoning image data, efficiently retrieving similar images poses a formidable challenge. Past research has focused on refining hash functions to distill images into compact indicators of resemblance. Initial attempts used shallow models, evolving to attention mechanism-based architectures from Convolutional Neural Networks (CNNs) to advanced models. Recognizing limitations in gradient-based models for spatial information embedding, we propose an innovative image hashing method, NeuroHash leveraging Hyperdimensional Computing (HDC). HDC symbolically encodes spatial information into high-dimensional vectors, reshaping image representation. Our approach combines pre-trained large vision models with HDC operations, enabling spatially encoded feature representations. Hashing with locality-sensitive hashing (LSH) ensures swift and efficient image retrieval. Notably, our framework allows dynamic hash manipulation for conditional image retrieval. Our work introduces a transformative image hashing framework enabling spatial-aware conditional retrieval. By seamlessly combining DNN-based neural and HDC-based symbolic models, our methodology breaks from traditional training, offering flexible and conditional image retrieval. Performance evaluations signify a paradigm shift in image-hashing methodologies, demonstrating enhanced retrieval accuracy.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Pixel-Wise Symbol Spotting via Progressive Points Location for Parsing CAD Images](http://arxiv.org/abs/2404.10985)  
Junbiao Pang, Zailin Dong, Jiaxin Deng, Mengyuan Zhu, Yunwei Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Parsing Computer-Aided Design (CAD) drawings is a fundamental step for CAD revision, semantic-based management, and the generation of 3D prototypes in both the architecture and engineering industries. Labeling symbols from a CAD drawing is a challenging yet notorious task from a practical point of view. In this work, we propose to label and spot symbols from CAD images that are converted from CAD drawings. The advantage of spotting symbols from CAD images lies in the low requirement of labelers and the low-cost annotation. However, pixel-wise spotting symbols from CAD images is challenging work. We propose a pixel-wise point location via Progressive Gaussian Kernels (PGK) to balance between training efficiency and location accuracy. Besides, we introduce a local offset to the heatmap-based point location method. Based on the keypoints detection, we propose a symbol grouping method to redraw the rectangle symbols in CAD images. We have released a dataset containing CAD images of equipment rooms from telecommunication industrial CAD drawings. Extensive experiments on this real-world dataset show that the proposed method has good generalization ability.  
  </ol>  
</details>  
**comments**: 10 pages, 10 figures,6 tables  
  
  



## Image Matching  

### [A Semantic Segmentation-guided Approach for Ground-to-Aerial Image Matching](http://arxiv.org/abs/2404.11302)  
[[code](https://github.com/pro1944191/semanticalignnet)]  
Francesco Pro, Nikolaos Dionelis, Luca Maiano, Bertrand Le Saux, Irene Amerini  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Nowadays the accurate geo-localization of ground-view images has an important role across domains as diverse as journalism, forensics analysis, transports, and Earth Observation. This work addresses the problem of matching a query ground-view image with the corresponding satellite image without GPS data. This is done by comparing the features from a ground-view image and a satellite one, innovatively leveraging the corresponding latter's segmentation mask through a three-stream Siamese-like network. The proposed method, Semantic Align Net (SAN), focuses on limited Field-of-View (FoV) and ground panorama images (images with a FoV of 360{\deg}). The novelty lies in the fusion of satellite images in combination with their semantic segmentation masks, aimed at ensuring that the model can extract useful features and focus on the significant parts of the images. This work shows how SAN through semantic analysis of images improves the performance on the unlabelled CVUSA dataset for all the tested FoVs.  
  </ol>  
</details>  
**comments**: 6 pages, 2 figures, 2 tables, Submitted to IGARSS 2024  
  
  



## NeRF  

### [SLAIM: Robust Dense Neural SLAM for Online Tracking and Mapping](http://arxiv.org/abs/2404.11419)  
Vincent Cartillier, Grant Schindler, Irfan Essa  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present SLAIM - Simultaneous Localization and Implicit Mapping. We propose a novel coarse-to-fine tracking model tailored for Neural Radiance Field SLAM (NeRF-SLAM) to achieve state-of-the-art tracking performance. Notably, existing NeRF-SLAM systems consistently exhibit inferior tracking performance compared to traditional SLAM algorithms. NeRF-SLAM methods solve camera tracking via image alignment and photometric bundle-adjustment. Such optimization processes are difficult to optimize due to the narrow basin of attraction of the optimization loss in image space (local minima) and the lack of initial correspondences. We mitigate these limitations by implementing a Gaussian pyramid filter on top of NeRF, facilitating a coarse-to-fine tracking optimization strategy. Furthermore, NeRF systems encounter challenges in converging to the right geometry with limited input views. While prior approaches use a Signed-Distance Function (SDF)-based NeRF and directly supervise SDF values by approximating ground truth SDF through depth measurements, this often results in suboptimal geometry. In contrast, our method employs a volume density representation and introduces a novel KL regularizer on the ray termination distribution, constraining scene geometry to consist of empty space and opaque surfaces. Our solution implements both local and global bundle-adjustment to produce a robust (coarse-to-fine) and accurate (KL regularizer) SLAM solution. We conduct experiments on multiple datasets (ScanNet, TUM, Replica) showing state-of-the-art results in tracking and in reconstruction accuracy.  
  </ol>  
</details>  
  
  



