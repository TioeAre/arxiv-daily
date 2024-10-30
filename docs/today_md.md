<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#LiVisSfM:-Accurate-and-Robust-Structure-from-Motion-with-LiDAR-and-Visual-Cues>LiVisSfM: Accurate and Robust Structure-from-Motion with LiDAR and Visual Cues</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#LiVisSfM:-Accurate-and-Robust-Structure-from-Motion-with-LiDAR-and-Visual-Cues>LiVisSfM: Accurate and Robust Structure-from-Motion with LiDAR and Visual Cues</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Beyond-Text:-Optimizing-RAG-with-Multimodal-Inputs-for-Industrial-Applications>Beyond Text: Optimizing RAG with Multimodal Inputs for Industrial Applications</a></li>
        <li><a href=#NYC-Event-VPR:-A-Large-Scale-High-Resolution-Event-Based-Visual-Place-Recognition-Dataset-in-Dense-Urban-Environments>NYC-Event-VPR: A Large-Scale High-Resolution Event-Based Visual Place Recognition Dataset in Dense Urban Environments</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#MVSDet:-Multi-View-Indoor-3D-Object-Detection-via-Efficient-Plane-Sweeps>MVSDet: Multi-View Indoor 3D Object Detection via Efficient Plane Sweeps</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [LiVisSfM: Accurate and Robust Structure-from-Motion with LiDAR and Visual Cues](http://arxiv.org/abs/2410.22213)  
Hanqing Jiang, Liyang Zhou, Zhuang Zhang, Yihao Yu, Guofeng Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper presents an accurate and robust Structure-from-Motion (SfM) pipeline named LiVisSfM, which is an SfM-based reconstruction system that fully combines LiDAR and visual cues. Unlike most existing LiDAR-inertial odometry (LIO) and LiDAR-inertial-visual odometry (LIVO) methods relying heavily on LiDAR registration coupled with Inertial Measurement Unit (IMU), we propose a LiDAR-visual SfM method which innovatively carries out LiDAR frame registration to LiDAR voxel map in a Point-to-Gaussian residual metrics, combined with a LiDAR-visual BA and explicit loop closure in a bundle optimization way to achieve accurate and robust LiDAR pose estimation without dependence on IMU incorporation. Besides, we propose an incremental voxel updating strategy for efficient voxel map updating during the process of LiDAR frame registration and LiDAR-visual BA optimization. Experiments demonstrate the superior effectiveness of our LiVisSfM framework over state-of-the-art LIO and LIVO works on more accurate and robust LiDAR pose recovery and dense point cloud reconstruction of both public KITTI benchmark and a variety of self-captured dataset.  
  </ol>  
</details>  
**comments**: 18 pages, 9 figures, 2 tables  
  
  



## SFM  

### [LiVisSfM: Accurate and Robust Structure-from-Motion with LiDAR and Visual Cues](http://arxiv.org/abs/2410.22213)  
Hanqing Jiang, Liyang Zhou, Zhuang Zhang, Yihao Yu, Guofeng Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper presents an accurate and robust Structure-from-Motion (SfM) pipeline named LiVisSfM, which is an SfM-based reconstruction system that fully combines LiDAR and visual cues. Unlike most existing LiDAR-inertial odometry (LIO) and LiDAR-inertial-visual odometry (LIVO) methods relying heavily on LiDAR registration coupled with Inertial Measurement Unit (IMU), we propose a LiDAR-visual SfM method which innovatively carries out LiDAR frame registration to LiDAR voxel map in a Point-to-Gaussian residual metrics, combined with a LiDAR-visual BA and explicit loop closure in a bundle optimization way to achieve accurate and robust LiDAR pose estimation without dependence on IMU incorporation. Besides, we propose an incremental voxel updating strategy for efficient voxel map updating during the process of LiDAR frame registration and LiDAR-visual BA optimization. Experiments demonstrate the superior effectiveness of our LiVisSfM framework over state-of-the-art LIO and LIVO works on more accurate and robust LiDAR pose recovery and dense point cloud reconstruction of both public KITTI benchmark and a variety of self-captured dataset.  
  </ol>  
</details>  
**comments**: 18 pages, 9 figures, 2 tables  
  
  



## Visual Localization  

### [Beyond Text: Optimizing RAG with Multimodal Inputs for Industrial Applications](http://arxiv.org/abs/2410.21943)  
Monica Riedler, Stefan Langer  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Large Language Models (LLMs) have demonstrated impressive capabilities in answering questions, but they lack domain-specific knowledge and are prone to hallucinations. Retrieval Augmented Generation (RAG) is one approach to address these challenges, while multimodal models are emerging as promising AI assistants for processing both text and images. In this paper we describe a series of experiments aimed at determining how to best integrate multimodal models into RAG systems for the industrial domain. The purpose of the experiments is to determine whether including images alongside text from documents within the industrial domain increases RAG performance and to find the optimal configuration for such a multimodal RAG system. Our experiments include two approaches for image processing and retrieval, as well as two LLMs (GPT4-Vision and LLaVA) for answer synthesis. These image processing strategies involve the use of multimodal embeddings and the generation of textual summaries from images. We evaluate our experiments with an LLM-as-a-Judge approach. Our results reveal that multimodal RAG can outperform single-modality RAG settings, although image retrieval poses a greater challenge than text retrieval. Additionally, leveraging textual summaries from images presents a more promising approach compared to the use of multimodal embeddings, providing more opportunities for future advancements.  
  </ol>  
</details>  
  
### [NYC-Event-VPR: A Large-Scale High-Resolution Event-Based Visual Place Recognition Dataset in Dense Urban Environments](http://arxiv.org/abs/2410.21615)  
Taiyi Pan, Junyang He, Chao Chen, Yiming Li, Chen Feng  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual place recognition (VPR) enables autonomous robots to identify previously visited locations, which contributes to tasks like simultaneous localization and mapping (SLAM). VPR faces challenges such as accurate image neighbor retrieval and appearance change in scenery. Event cameras, also known as dynamic vision sensors, are a new sensor modality for VPR and offer a promising solution to the challenges with their unique attributes: high temporal resolution (1MHz clock), ultra-low latency (in {\mu}s), and high dynamic range (>120dB). These attributes make event cameras less susceptible to motion blur and more robust in variable lighting conditions, making them suitable for addressing VPR challenges. However, the scarcity of event-based VPR datasets, partly due to the novelty and cost of event cameras, hampers their adoption. To fill this data gap, our paper introduces the NYC-Event-VPR dataset to the robotics and computer vision communities, featuring the Prophesee IMX636 HD event sensor (1280x720 resolution), combined with RGB camera and GPS module. It encompasses over 13 hours of geotagged event data, spanning 260 kilometers across New York City, covering diverse lighting and weather conditions, day/night scenarios, and multiple visits to various locations. Furthermore, our paper employs three frameworks to conduct generalization performance assessments, promoting innovation in event-based VPR and its integration into robotics applications.  
  </ol>  
</details>  
  
  



## NeRF  

### [MVSDet: Multi-View Indoor 3D Object Detection via Efficient Plane Sweeps](http://arxiv.org/abs/2410.21566)  
[[code](https://github.com/pixie8888/mvsdet)]  
Yating Xu, Chen Li, Gim Hee Lee  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The key challenge of multi-view indoor 3D object detection is to infer accurate geometry information from images for precise 3D detection. Previous method relies on NeRF for geometry reasoning. However, the geometry extracted from NeRF is generally inaccurate, which leads to sub-optimal detection performance. In this paper, we propose MVSDet which utilizes plane sweep for geometry-aware 3D object detection. To circumvent the requirement for a large number of depth planes for accurate depth prediction, we design a probabilistic sampling and soft weighting mechanism to decide the placement of pixel features on the 3D volume. We select multiple locations that score top in the probability volume for each pixel and use their probability score to indicate the confidence. We further apply recent pixel-aligned Gaussian Splatting to regularize depth prediction and improve detection performance with little computation overhead. Extensive experiments on ScanNet and ARKitScenes datasets are conducted to show the superiority of our model. Our code is available at https://github.com/Pixie8888/MVSDet.  
  </ol>  
</details>  
**comments**: Accepted by NeurIPS 2024  
  
  



