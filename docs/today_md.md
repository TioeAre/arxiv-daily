<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Review-of-Feed-forward-3D-Reconstruction:-From-DUSt3R-to-VGGT>Review of Feed-forward 3D Reconstruction: From DUSt3R to VGGT</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#RadiomicsRetrieval:-A-Customizable-Framework-for-Medical-Image-Retrieval-Using-Radiomics-Features>RadiomicsRetrieval: A Customizable Framework for Medical Image Retrieval Using Radiomics Features</a></li>
        <li><a href=#LiDAR,-GNSS-and-IMU-Sensor-Alignment-through-Dynamic-Time-Warping-to-Construct-3D-City-Maps>LiDAR, GNSS and IMU Sensor Alignment through Dynamic Time Warping to Construct 3D City Maps</a></li>
        <li><a href=#Deep-Hashing-with-Semantic-Hash-Centers-for-Image-Retrieval>Deep Hashing with Semantic Hash Centers for Image Retrieval</a></li>
        <li><a href=#Unveiling-Effective-In-Context-Configurations-for-Image-Captioning:-An-External-&-Internal-Analysis>Unveiling Effective In-Context Configurations for Image Captioning: An External & Internal Analysis</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Review of Feed-forward 3D Reconstruction: From DUSt3R to VGGT](http://arxiv.org/abs/2507.08448)  
Wei Zhang, Yihang Wu, Songhua Li, Wenjie Ma, Xin Ma, Qiang Li, Qi Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D reconstruction, which aims to recover the dense three-dimensional structure of a scene, is a cornerstone technology for numerous applications, including augmented/virtual reality, autonomous driving, and robotics. While traditional pipelines like Structure from Motion (SfM) and Multi-View Stereo (MVS) achieve high precision through iterative optimization, they are limited by complex workflows, high computational cost, and poor robustness in challenging scenarios like texture-less regions. Recently, deep learning has catalyzed a paradigm shift in 3D reconstruction. A new family of models, exemplified by DUSt3R, has pioneered a feed-forward approach. These models employ a unified deep network to jointly infer camera poses and dense geometry directly from an Unconstrained set of images in a single forward pass. This survey provides a systematic review of this emerging domain. We begin by dissecting the technical framework of these feed-forward models, including their Transformer-based correspondence modeling, joint pose and geometry regression mechanisms, and strategies for scaling from two-view to multi-view scenarios. To highlight the disruptive nature of this new paradigm, we contrast it with both traditional pipelines and earlier learning-based methods like MVSNet. Furthermore, we provide an overview of relevant datasets and evaluation metrics. Finally, we discuss the technology's broad application prospects and identify key future challenges and opportunities, such as model accuracy and scalability, and handling dynamic scenes.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [RadiomicsRetrieval: A Customizable Framework for Medical Image Retrieval Using Radiomics Features](http://arxiv.org/abs/2507.08546)  
Inye Na, Nejung Rue, Jiwon Chung, Hyunjin Park  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Medical image retrieval is a valuable field for supporting clinical decision-making, yet current methods primarily support 2D images and require fully annotated queries, limiting clinical flexibility. To address this, we propose RadiomicsRetrieval, a 3D content-based retrieval framework bridging handcrafted radiomics descriptors with deep learning-based embeddings at the tumor level. Unlike existing 2D approaches, RadiomicsRetrieval fully exploits volumetric data to leverage richer spatial context in medical images. We employ a promptable segmentation model (e.g., SAM) to derive tumor-specific image embeddings, which are aligned with radiomics features extracted from the same tumor via contrastive learning. These representations are further enriched by anatomical positional embedding (APE). As a result, RadiomicsRetrieval enables flexible querying based on shape, location, or partial feature sets. Extensive experiments on both lung CT and brain MRI public datasets demonstrate that radiomics features significantly enhance retrieval specificity, while APE provides global anatomical context essential for location-based searches. Notably, our framework requires only minimal user prompts (e.g., a single point), minimizing segmentation overhead and supporting diverse clinical scenarios. The capability to query using either image embeddings or selected radiomics attributes highlights its adaptability, potentially benefiting diagnosis, treatment planning, and research on large-scale medical imaging repositories. Our code is available at https://github.com/nainye/RadiomicsRetrieval.  
  </ol>  
</details>  
**comments**: Accepted at MICCAI 2025  
  
### [LiDAR, GNSS and IMU Sensor Alignment through Dynamic Time Warping to Construct 3D City Maps](http://arxiv.org/abs/2507.08420)  
Haitian Wang, Hezam Albaqami, Xinyu Wang, Muhammad Ibrahim, Zainy M. Malakan, Abdullah M. Algamdi, Mohammed H. Alghamdi, Ajmal Mian  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    LiDAR-based 3D mapping suffers from cumulative drift causing global misalignment, particularly in GNSS-constrained environments. To address this, we propose a unified framework that fuses LiDAR, GNSS, and IMU data for high-resolution city-scale mapping. The method performs velocity-based temporal alignment using Dynamic Time Warping and refines GNSS and IMU signals via extended Kalman filtering. Local maps are built using Normal Distributions Transform-based registration and pose graph optimization with loop closure detection, while global consistency is enforced using GNSS-constrained anchors followed by fine registration of overlapping segments. We also introduce a large-scale multimodal dataset captured in Perth, Western Australia to facilitate future research in this direction. Our dataset comprises 144{,}000 frames acquired with a 128-channel Ouster LiDAR, synchronized RTK-GNSS trajectories, and MEMS-IMU measurements across 21 urban loops. To assess geometric consistency, we evaluated our method using alignment metrics based on road centerlines and intersections to capture both global and local accuracy. Our method reduces the average global alignment error from 3.32\,m to 1.24\,m, achieving a 61.4\% improvement. The constructed high-fidelity map supports a wide range of applications, including smart city planning, geospatial data integration, infrastructure monitoring, and GPS-free navigation. Our method, and dataset together establish a new benchmark for evaluating 3D city mapping in GNSS-constrained environments. The dataset and code will be released publicly.  
  </ol>  
</details>  
**comments**: Preparing to submit to International Journal of Applied Earth
  Observation and Geoinformation  
  
### [Deep Hashing with Semantic Hash Centers for Image Retrieval](http://arxiv.org/abs/2507.08404)  
Li Chen, Rui Liu, Yuxiang Zhou, Xudong Ma, Yong Chen, Dell Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Deep hashing is an effective approach for large-scale image retrieval. Current methods are typically classified by their supervision types: point-wise, pair-wise, and list-wise. Recent point-wise techniques (e.g., CSQ, MDS) have improved retrieval performance by pre-assigning a hash center to each class, enhancing the discriminability of hash codes across various datasets. However, these methods rely on data-independent algorithms to generate hash centers, which neglect the semantic relationships between classes and may degrade retrieval performance.   This paper introduces the concept of semantic hash centers, building on the idea of traditional hash centers. We hypothesize that hash centers of semantically related classes should have closer Hamming distances, while those of unrelated classes should be more distant. To this end, we propose a three-stage framework, SHC, to generate hash codes that preserve semantic structure.   First, we develop a classification network to identify semantic similarities between classes using a data-dependent similarity calculation that adapts to varying data distributions. Second, we introduce an optimization algorithm to generate semantic hash centers, preserving semantic relatedness while enforcing a minimum distance between centers to avoid excessively similar hash codes. Finally, a deep hashing network is trained using these semantic centers to convert images into binary hash codes.   Experimental results on large-scale retrieval tasks across several public datasets show that SHC significantly improves retrieval performance. Specifically, SHC achieves average improvements of +7.26%, +7.62%, and +11.71% in MAP@100, MAP@1000, and MAP@ALL metrics, respectively, over state-of-the-art methods.  
  </ol>  
</details>  
  
### [Unveiling Effective In-Context Configurations for Image Captioning: An External & Internal Analysis](http://arxiv.org/abs/2507.08021)  
Li Li, Yongliang Wu, Jingze Zhu, Jiawei Peng, Jianfei Cai, Xu Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The evolution of large models has witnessed the emergence of In-Context Learning (ICL) capabilities. In Natural Language Processing (NLP), numerous studies have demonstrated the effectiveness of ICL. Inspired by the success of Large Language Models (LLMs), researchers have developed Large Multimodal Models (LMMs) with ICL capabilities. However, explorations of demonstration configuration for multimodal ICL remain preliminary. Additionally, the controllability of In-Context Examples (ICEs) provides an efficient and cost-effective means to observe and analyze the inference characteristics of LMMs under varying inputs. This paper conducts a comprehensive external and internal investigation of multimodal in-context learning on the image captioning task. Externally, we explore demonstration configuration strategies through three dimensions: shot number, image retrieval, and caption assignment. We employ multiple metrics to systematically and thoroughly evaluate and summarize key findings. Internally, we analyze typical LMM attention characteristics and develop attention-based metrics to quantify model behaviors. We also conduct auxiliary experiments to explore the feasibility of attention-driven model acceleration and compression. We further compare performance variations between LMMs with identical model design and pretraining strategies and explain the differences from the angles of pre-training data features. Our study reveals both how ICEs configuration strategies impact model performance through external experiments and characteristic typical patterns through internal inspection, providing dual perspectives for understanding multimodal ICL in LMMs. Our method of combining external and internal analysis to investigate large models, along with our newly proposed metrics, can be applied to broader research areas.  
  </ol>  
</details>  
**comments**: 16 pages, 11 figures  
  
  



