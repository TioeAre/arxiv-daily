<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#4D-Radar-Inertial-Odometry-based-on-Gaussian-Modeling-and-Multi-Hypothesis-Scan-Matching>4D Radar-Inertial Odometry based on Gaussian Modeling and Multi-Hypothesis Scan Matching</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Foundation-Models-Meet-Low-Cost-Sensors:-Test-Time-Adaptation-for-Rescaling-Disparity-for-Zero-Shot-Metric-Depth-Estimation>Foundation Models Meet Low-Cost Sensors: Test-Time Adaptation for Rescaling Disparity for Zero-Shot Metric Depth Estimation</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Adversarial-Hubness-in-Multi-Modal-Retrieval>Adversarial Hubness in Multi-Modal Retrieval</a></li>
        <li><a href=#Maybe-you-are-looking-for-CroQS:-Cross-modal-Query-Suggestion-for-Text-to-Image-Retrieval>Maybe you are looking for CroQS: Cross-modal Query Suggestion for Text-to-Image Retrieval</a></li>
        <li><a href=#ConDo:-Continual-Domain-Expansion-for-Absolute-Pose-Regression>ConDo: Continual Domain Expansion for Absolute Pose Regression</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Bringing-Multimodality-to-Amazon-Visual-Search-System>Bringing Multimodality to Amazon Visual Search System</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#GraphAvatar:-Compact-Head-Avatars-with-GNN-Generated-3D-Gaussians>GraphAvatar: Compact Head Avatars with GNN-Generated 3D Gaussians</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [4D Radar-Inertial Odometry based on Gaussian Modeling and Multi-Hypothesis Scan Matching](http://arxiv.org/abs/2412.13639)  
[[code](https://github.com/robotics-upo/gaussian-rio)]  
Fernando Amodeo, Luis Merino, Fernando Caballero  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    4D millimeter-wave (mmWave) radars are sensors that provide robustness against adverse weather conditions (rain, snow, fog, etc.), and as such they are increasingly being used for odometry and SLAM applications. However, the noisy and sparse nature of the returned scan data proves to be a challenging obstacle for existing point cloud matching based solutions, especially those originally intended for more accurate sensors such as LiDAR. Inspired by visual odometry research around 3D Gaussian Splatting, in this paper we propose using freely positioned 3D Gaussians to create a summarized representation of a radar point cloud tolerant to sensor noise, and subsequently leverage its inherent probability distribution function for registration (similar to NDT). Moreover, we propose simultaneously optimizing multiple scan matching hypotheses in order to further increase the robustness of the system against local optima of the function. Finally, we fuse our Gaussian modeling and scan matching algorithms into an EKF radar-inertial odometry system designed after current best practices. Experiments show that our Gaussian-based odometry is able to outperform current baselines on a well-known 4D radar dataset used for evaluation.  
  </ol>  
</details>  
**comments**: Our code and results can be publicly accessed at:
  https://github.com/robotics-upo/gaussian-rio  
  
  



## SFM  

### [Foundation Models Meet Low-Cost Sensors: Test-Time Adaptation for Rescaling Disparity for Zero-Shot Metric Depth Estimation](http://arxiv.org/abs/2412.14103)  
Rémi Marsal, Alexandre Chapoutot, Philippe Xu, David Filliat  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The recent development of foundation models for monocular depth estimation such as Depth Anything paved the way to zero-shot monocular depth estimation. Since it returns an affine-invariant disparity map, the favored technique to recover the metric depth consists in fine-tuning the model. However, this stage is costly to perform because of the training but also due to the creation of the dataset. It must contain images captured by the camera that will be used at test time and the corresponding ground truth. Moreover, the fine-tuning may also degrade the generalizing capacity of the original model. Instead, we propose in this paper a new method to rescale Depth Anything predictions using 3D points provided by low-cost sensors or techniques such as low-resolution LiDAR, stereo camera, structure-from-motion where poses are given by an IMU. Thus, this approach avoids fine-tuning and preserves the generalizing power of the original depth estimation model while being robust to the noise of the sensor or of the depth model. Our experiments highlight improvements relative to other metric depth estimation methods and competitive results compared to fine-tuned approaches. Code available at https://gitlab.ensta.fr/ssh/monocular-depth-rescaling.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Adversarial Hubness in Multi-Modal Retrieval](http://arxiv.org/abs/2412.14113)  
[[code](https://github.com/tingwei-zhang/adv_hub)]  
Tingwei Zhang, Fnu Suya, Rishi Jha, Collin Zhang, Vitaly Shmatikov  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Hubness is a phenomenon in high-dimensional vector spaces where a single point from the natural distribution is unusually close to many other points. This is a well-known problem in information retrieval that causes some items to accidentally (and incorrectly) appear relevant to many queries. In this paper, we investigate how attackers can exploit hubness to turn any image or audio input in a multi-modal retrieval system into an adversarial hub. Adversarial hubs can be used to inject universal adversarial content (e.g., spam) that will be retrieved in response to thousands of different queries, as well as for targeted attacks on queries related to specific, attacker-chosen concepts. We present a method for creating adversarial hubs and evaluate the resulting hubs on benchmark multi-modal retrieval datasets and an image-to-image retrieval system based on a tutorial from Pinecone, a popular vector database. For example, in text-caption-to-image retrieval, a single adversarial hub is retrieved as the top-1 most relevant image for more than 21,000 out of 25,000 test queries (by contrast, the most common natural hub is the top-1 response to only 102 queries). We also investigate whether techniques for mitigating natural hubness are an effective defense against adversarial hubs, and show that they are not effective against hubs that target queries related to specific concepts.  
  </ol>  
</details>  
  
### [Maybe you are looking for CroQS: Cross-modal Query Suggestion for Text-to-Image Retrieval](http://arxiv.org/abs/2412.13834)  
Giacomo Pacini, Fabio Carrara, Nicola Messina, Nicola Tonellotto, Giuseppe Amato, Fabrizio Falchi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Query suggestion, a technique widely adopted in information retrieval, enhances system interactivity and the browsing experience of document collections. In cross-modal retrieval, many works have focused on retrieving relevant items from natural language queries, while few have explored query suggestion solutions. In this work, we address query suggestion in cross-modal retrieval, introducing a novel task that focuses on suggesting minimal textual modifications needed to explore visually consistent subsets of the collection, following the premise of ''Maybe you are looking for''. To facilitate the evaluation and development of methods, we present a tailored benchmark named CroQS. This dataset comprises initial queries, grouped result sets, and human-defined suggested queries for each group. We establish dedicated metrics to rigorously evaluate the performance of various methods on this task, measuring representativeness, cluster specificity, and similarity of the suggested queries to the original ones. Baseline methods from related fields, such as image captioning and content summarization, are adapted for this task to provide reference performance scores. Although relatively far from human performance, our experiments reveal that both LLM-based and captioning-based methods achieve competitive results on CroQS, improving the recall on cluster specificity by more than 115% and representativeness mAP by more than 52% with respect to the initial query. The dataset, the implementation of the baseline methods and the notebooks containing our experiments are available here: https://paciosoft.com/CroQS-benchmark/  
  </ol>  
</details>  
**comments**: 15 pages, 5 figures. To be published as full paper in the Proceedings
  of the European Conference on Information Retrieval (ECIR) 2025  
  
### [ConDo: Continual Domain Expansion for Absolute Pose Regression](http://arxiv.org/abs/2412.13452)  
Zijun Li, Zhipeng Cai, Bochun Yang, Xuelun Shen, Siqi Shen, Xiaoliang Fan, Michael Paulitsch, Cheng Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual localization is a fundamental machine learning problem. Absolute Pose Regression (APR) trains a scene-dependent model to efficiently map an input image to the camera pose in a pre-defined scene. However, many applications have continually changing environments, where inference data at novel poses or scene conditions (weather, geometry) appear after deployment. Training APR on a fixed dataset leads to overfitting, making it fail catastrophically on challenging novel data. This work proposes Continual Domain Expansion (ConDo), which continually collects unlabeled inference data to update the deployed APR. Instead of applying standard unsupervised domain adaptation methods which are ineffective for APR, ConDo effectively learns from unlabeled data by distilling knowledge from scene-agnostic localization methods. By sampling data uniformly from historical and newly collected data, ConDo can effectively expand the generalization domain of APR. Large-scale benchmarks with various scene types are constructed to evaluate models under practical (long-term) data changes. ConDo consistently and significantly outperforms baselines across architectures, scene types, and data changes. On challenging scenes (Fig.1), it reduces the localization error by >7x (14.8m vs 1.7m). Analysis shows the robustness of ConDo against compute budgets, replay buffer sizes and teacher prediction noise. Comparing to model re-training, ConDo achieves similar performance up to 25x faster.  
  </ol>  
</details>  
**comments**: AAAI2025  
  
  



## Image Matching  

### [Bringing Multimodality to Amazon Visual Search System](http://arxiv.org/abs/2412.13364)  
Xinliang Zhu, Michael Huang, Han Ding, Jinyu Yang, Kelvin Chen, Tao Zhou, Tal Neiman, Ouye Xie, Son Tran, Benjamin Yao, Doug Gray, Anuj Bindal, Arnab Dhua  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image to image matching has been well studied in the computer vision community. Previous studies mainly focus on training a deep metric learning model matching visual patterns between the query image and gallery images. In this study, we show that pure image-to-image matching suffers from false positives caused by matching to local visual patterns. To alleviate this issue, we propose to leverage recent advances in vision-language pretraining research. Specifically, we introduce additional image-text alignment losses into deep metric learning, which serve as constraints to the image-to-image matching loss. With additional alignments between the text (e.g., product title) and image pairs, the model can learn concepts from both modalities explicitly, which avoids matching low-level visual features. We progressively develop two variants, a 3-tower and a 4-tower model, where the latter takes one more short text query input. Through extensive experiments, we show that this change leads to a substantial improvement to the image to image matching problem. We further leveraged this model for multimodal search, which takes both image and reformulation text queries to improve search quality. Both offline and online experiments show strong improvements on the main metrics. Specifically, we see 4.95% relative improvement on image matching click through rate with the 3-tower model and 1.13% further improvement from the 4-tower model.  
  </ol>  
</details>  
  
  



## NeRF  

### [GraphAvatar: Compact Head Avatars with GNN-Generated 3D Gaussians](http://arxiv.org/abs/2412.13983)  
[[code](https://github.com/ucwxb/graphavatar)]  
Xiaobao Wei, Peng Chen, Ming Lu, Hui Chen, Feng Tian  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Rendering photorealistic head avatars from arbitrary viewpoints is crucial for various applications like virtual reality. Although previous methods based on Neural Radiance Fields (NeRF) can achieve impressive results, they lack fidelity and efficiency. Recent methods using 3D Gaussian Splatting (3DGS) have improved rendering quality and real-time performance but still require significant storage overhead. In this paper, we introduce a method called GraphAvatar that utilizes Graph Neural Networks (GNN) to generate 3D Gaussians for the head avatar. Specifically, GraphAvatar trains a geometric GNN and an appearance GNN to generate the attributes of the 3D Gaussians from the tracked mesh. Therefore, our method can store the GNN models instead of the 3D Gaussians, significantly reducing the storage overhead to just 10MB. To reduce the impact of face-tracking errors, we also present a novel graph-guided optimization module to refine face-tracking parameters during training. Finally, we introduce a 3D-aware enhancer for post-processing to enhance the rendering quality. We conduct comprehensive experiments to demonstrate the advantages of GraphAvatar, surpassing existing methods in visual fidelity and storage consumption. The ablation study sheds light on the trade-offs between rendering quality and model size. The code will be released at: https://github.com/ucwxb/GraphAvatar  
  </ol>  
</details>  
**comments**: accepted by AAAI2025  
  
  



