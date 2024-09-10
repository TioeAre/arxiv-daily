<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#KRONC:-Keypoint-based-Robust-Camera-Optimization-for-3D-Car-Reconstruction>KRONC: Keypoint-based Robust Camera Optimization for 3D Car Reconstruction</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Referring-Expression-Generation-in-Visually-Grounded-Dialogue-with-Discourse-aware-Comprehension-Guiding>Referring Expression Generation in Visually Grounded Dialogue with Discourse-aware Comprehension Guiding</a></li>
        <li><a href=#Open-World-Dynamic-Prompt-and-Continual-Visual-Representation-Learning>Open-World Dynamic Prompt and Continual Visual Representation Learning</a></li>
        <li><a href=#Training-free-ZS-CIR-via-Weighted-Modality-Fusion-and-Similarity>Training-free ZS-CIR via Weighted Modality Fusion and Similarity</a></li>
        <li><a href=#Zero-Shot-Whole-Slide-Image-Retrieval-in-Histopathology-Using-Embeddings-of-Foundation-Models>Zero-Shot Whole Slide Image Retrieval in Histopathology Using Embeddings of Foundation Models</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#G-NeLF:-Memory--and-Data-Efficient-Hybrid-Neural-Light-Field-for-Novel-View-Synthesis>G-NeLF: Memory- and Data-Efficient Hybrid Neural Light Field for Novel View Synthesis</a></li>
        <li><a href=#From-Words-to-Poses:-Enhancing-Novel-Object-Pose-Estimation-with-Vision-Language-Models>From Words to Poses: Enhancing Novel Object Pose Estimation with Vision Language Models</a></li>
        <li><a href=#KRONC:-Keypoint-based-Robust-Camera-Optimization-for-3D-Car-Reconstruction>KRONC: Keypoint-based Robust Camera Optimization for 3D Car Reconstruction</a></li>
        <li><a href=#Lagrangian-Hashing-for-Compressed-Neural-Field-Representations>Lagrangian Hashing for Compressed Neural Field Representations</a></li>
        <li><a href=#Neural-Surface-Reconstruction-and-Rendering-for-LiDAR-Visual-Systems>Neural Surface Reconstruction and Rendering for LiDAR-Visual Systems</a></li>
        <li><a href=#SCARF:-Scalable-Continual-Learning-Framework-for-Memory-efficient-Multiple-Neural-Radiance-Fields>SCARF: Scalable Continual Learning Framework for Memory-efficient Multiple Neural Radiance Fields</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [KRONC: Keypoint-based Robust Camera Optimization for 3D Car Reconstruction](http://arxiv.org/abs/2409.05407)  
Davide Di Nucci, Alessandro Simoni, Matteo Tomei, Luca Ciuffreda, Roberto Vezzani, Rita Cucchiara  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The three-dimensional representation of objects or scenes starting from a set of images has been a widely discussed topic for years and has gained additional attention after the diffusion of NeRF-based approaches. However, an underestimated prerequisite is the knowledge of camera poses or, more specifically, the estimation of the extrinsic calibration parameters. Although excellent general-purpose Structure-from-Motion methods are available as a pre-processing step, their computational load is high and they require a lot of frames to guarantee sufficient overlapping among the views. This paper introduces KRONC, a novel approach aimed at inferring view poses by leveraging prior knowledge about the object to reconstruct and its representation through semantic keypoints. With a focus on vehicle scenes, KRONC is able to estimate the position of the views as a solution to a light optimization problem targeting the convergence of keypoints' back-projections to a singular point. To validate the method, a specific dataset of real-world car scenes has been collected. Experiments confirm KRONC's ability to generate excellent estimates of camera poses starting from very coarse initialization. Results are comparable with Structure-from-Motion methods with huge savings in computation. Code and data will be made publicly available.  
  </ol>  
</details>  
**comments**: Accepted at ECCVW  
  
  



## Visual Localization  

### [Referring Expression Generation in Visually Grounded Dialogue with Discourse-aware Comprehension Guiding](http://arxiv.org/abs/2409.05721)  
Bram Willemsen, Gabriel Skantze  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose an approach to referring expression generation (REG) in visually grounded dialogue that is meant to produce referring expressions (REs) that are both discriminative and discourse-appropriate. Our method constitutes a two-stage process. First, we model REG as a text- and image-conditioned next-token prediction task. REs are autoregressively generated based on their preceding linguistic context and a visual representation of the referent. Second, we propose the use of discourse-aware comprehension guiding as part of a generate-and-rerank strategy through which candidate REs generated with our REG model are reranked based on their discourse-dependent discriminatory power. Results from our human evaluation indicate that our proposed two-stage approach is effective in producing discriminative REs, with higher performance in terms of text-image retrieval accuracy for reranked REs compared to those generated using greedy decoding.  
  </ol>  
</details>  
**comments**: Accepted for publication at INLG 2024  
  
### [Open-World Dynamic Prompt and Continual Visual Representation Learning](http://arxiv.org/abs/2409.05312)  
Youngeun Kim, Jun Fang, Qin Zhang, Zhaowei Cai, Yantao Shen, Rahul Duggal, Dripta S. Raychaudhuri, Zhuowen Tu, Yifan Xing, Onkar Dabeer  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The open world is inherently dynamic, characterized by ever-evolving concepts and distributions. Continual learning (CL) in this dynamic open-world environment presents a significant challenge in effectively generalizing to unseen test-time classes. To address this challenge, we introduce a new practical CL setting tailored for open-world visual representation learning. In this setting, subsequent data streams systematically introduce novel classes that are disjoint from those seen in previous training phases, while also remaining distinct from the unseen test classes. In response, we present Dynamic Prompt and Representation Learner (DPaRL), a simple yet effective Prompt-based CL (PCL) method. Our DPaRL learns to generate dynamic prompts for inference, as opposed to relying on a static prompt pool in previous PCL methods. In addition, DPaRL jointly learns dynamic prompt generation and discriminative representation at each training stage whereas prior PCL methods only refine the prompt learning throughout the process. Our experimental results demonstrate the superiority of our approach, surpassing state-of-the-art methods on well-established open-world image retrieval benchmarks by an average of 4.7\% improvement in Recall@1 performance.  
  </ol>  
</details>  
**comments**: ECCV 2024  
  
### [Training-free ZS-CIR via Weighted Modality Fusion and Similarity](http://arxiv.org/abs/2409.04918)  
Ren-Di Wu, Yu-Yen Lin, Huei-Fang Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed image retrieval (CIR), which formulates the query as a combination of a reference image and modified text, has emerged as a new form of image search due to its enhanced ability to capture users' intentions. However, training a CIR model in a supervised manner typically requires labor-intensive collection of (reference image, text modifier, target image) triplets. While existing zero-shot CIR (ZS-CIR) methods eliminate the need for training on specific downstream datasets, they still require additional pretraining with large-scale image-text pairs. In this paper, we introduce a training-free approach for ZS-CIR. Our approach, \textbf{Wei}ghted \textbf{Mo}dality fusion and similarity for \textbf{CIR} (WeiMoCIR), operates under the assumption that image and text modalities can be effectively combined using a simple weighted average. This allows the query representation to be constructed directly from the reference image and text modifier. To further enhance retrieval performance, we employ multimodal large language models (MLLMs) to generate image captions for the database images and incorporate these textual captions into the similarity computation by combining them with image information using a weighted average. Our approach is simple, easy to implement, and its effectiveness is validated through experiments on the FashionIQ and CIRR datasets.  
  </ol>  
</details>  
**comments**: 13 pages, 3 figures  
  
### [Zero-Shot Whole Slide Image Retrieval in Histopathology Using Embeddings of Foundation Models](http://arxiv.org/abs/2409.04631)  
Saghir Alfasly, Peyman Nejat, Ghazal Alabtah, Sobhan Hemati, Krishna Rani Kalari, H. R. Tizhoosh  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We have tested recently published foundation models for histopathology for image retrieval. We report macro average of F1 score for top-1 retrieval, majority of top-3 retrievals, and majority of top-5 retrievals. We perform zero-shot retrievals, i.e., we do not alter embeddings and we do not train any classifier. As test data, we used diagnostic slides of TCGA, The Cancer Genome Atlas, consisting of 23 organs and 117 cancer subtypes. As a search platform we used Yottixel that enabled us to perform WSI search using patches. Achieved F1 scores show low performance, e.g., for top-5 retrievals, 27% +/- 13% (Yottixel-DenseNet), 42% +/- 14% (Yottixel-UNI), 40%+/-13% (Yottixel-Virchow), and 41%+/-13% (Yottixel-GigaPath). The results for GigaPath WSI will be delayed due to the significant computational resources required for processing  
  </ol>  
</details>  
**comments**: This paper will be updated with more results  
  
  



## NeRF  

### [G-NeLF: Memory- and Data-Efficient Hybrid Neural Light Field for Novel View Synthesis](http://arxiv.org/abs/2409.05617)  
Lutao Jiang, Lin Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Following the burgeoning interest in implicit neural representation, Neural Light Field (NeLF) has been introduced to predict the color of a ray directly. Unlike Neural Radiance Field (NeRF), NeLF does not create a point-wise representation by predicting color and volume density for each point in space. However, the current NeLF methods face a challenge as they need to train a NeRF model first and then synthesize over 10K views to train NeLF for improved performance. Additionally, the rendering quality of NeLF methods is lower compared to NeRF methods. In this paper, we propose G-NeLF, a versatile grid-based NeLF approach that utilizes spatial-aware features to unleash the potential of the neural network's inference capability, and consequently overcome the difficulties of NeLF training. Specifically, we employ a spatial-aware feature sequence derived from a meticulously crafted grid as the ray's representation. Drawing from our empirical studies on the adaptability of multi-resolution hash tables, we introduce a novel grid-based ray representation for NeLF that can represent the entire space with a very limited number of parameters. To better utilize the sequence feature, we design a lightweight ray color decoder that simulates the ray propagation process, enabling a more efficient inference of the ray's color. G-NeLF can be trained without necessitating significant storage overhead and with the model size of only 0.95 MB to surpass previous state-of-the-art NeLF. Moreover, compared with grid-based NeRF methods, e.g., Instant-NGP, we only utilize one-tenth of its parameters to achieve higher performance. Our code will be released upon acceptance.  
  </ol>  
</details>  
  
### [From Words to Poses: Enhancing Novel Object Pose Estimation with Vision Language Models](http://arxiv.org/abs/2409.05413)  
Tessa Pulli, Stefan Thalhammer, Simon Schwaiger, Markus Vincze  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Robots are increasingly envisioned to interact in real-world scenarios, where they must continuously adapt to new situations. To detect and grasp novel objects, zero-shot pose estimators determine poses without prior knowledge. Recently, vision language models (VLMs) have shown considerable advances in robotics applications by establishing an understanding between language input and image input. In our work, we take advantage of VLMs zero-shot capabilities and translate this ability to 6D object pose estimation. We propose a novel framework for promptable zero-shot 6D object pose estimation using language embeddings. The idea is to derive a coarse location of an object based on the relevancy map of a language-embedded NeRF reconstruction and to compute the pose estimate with a point cloud registration method. Additionally, we provide an analysis of LERF's suitability for open-set object pose estimation. We examine hyperparameters, such as activation thresholds for relevancy maps and investigate the zero-shot capabilities on an instance- and category-level. Furthermore, we plan to conduct robotic grasping experiments in a real-world setting.  
  </ol>  
</details>  
  
### [KRONC: Keypoint-based Robust Camera Optimization for 3D Car Reconstruction](http://arxiv.org/abs/2409.05407)  
Davide Di Nucci, Alessandro Simoni, Matteo Tomei, Luca Ciuffreda, Roberto Vezzani, Rita Cucchiara  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The three-dimensional representation of objects or scenes starting from a set of images has been a widely discussed topic for years and has gained additional attention after the diffusion of NeRF-based approaches. However, an underestimated prerequisite is the knowledge of camera poses or, more specifically, the estimation of the extrinsic calibration parameters. Although excellent general-purpose Structure-from-Motion methods are available as a pre-processing step, their computational load is high and they require a lot of frames to guarantee sufficient overlapping among the views. This paper introduces KRONC, a novel approach aimed at inferring view poses by leveraging prior knowledge about the object to reconstruct and its representation through semantic keypoints. With a focus on vehicle scenes, KRONC is able to estimate the position of the views as a solution to a light optimization problem targeting the convergence of keypoints' back-projections to a singular point. To validate the method, a specific dataset of real-world car scenes has been collected. Experiments confirm KRONC's ability to generate excellent estimates of camera poses starting from very coarse initialization. Results are comparable with Structure-from-Motion methods with huge savings in computation. Code and data will be made publicly available.  
  </ol>  
</details>  
**comments**: Accepted at ECCVW  
  
### [Lagrangian Hashing for Compressed Neural Field Representations](http://arxiv.org/abs/2409.05334)  
Shrisudhan Govindarajan, Zeno Sambugaro, Akhmedkhan, Shabanov, Towaki Takikawa, Daniel Rebain, Weiwei Sun, Nicola Conci, Kwang Moo Yi, Andrea Tagliasacchi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present Lagrangian Hashing, a representation for neural fields combining the characteristics of fast training NeRF methods that rely on Eulerian grids (i.e.~InstantNGP), with those that employ points equipped with features as a way to represent information (e.g. 3D Gaussian Splatting or PointNeRF). We achieve this by incorporating a point-based representation into the high-resolution layers of the hierarchical hash tables of an InstantNGP representation. As our points are equipped with a field of influence, our representation can be interpreted as a mixture of Gaussians stored within the hash table. We propose a loss that encourages the movement of our Gaussians towards regions that require more representation budget to be sufficiently well represented. Our main finding is that our representation allows the reconstruction of signals using a more compact representation without compromising quality.  
  </ol>  
</details>  
**comments**: Project page: https://theialab.github.io/laghashes/  
  
### [Neural Surface Reconstruction and Rendering for LiDAR-Visual Systems](http://arxiv.org/abs/2409.05310)  
Jianheng Liu, Chunran Zheng, Yunfei Wan, Bowen Wang, Yixi Cai, Fu Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper presents a unified surface reconstruction and rendering framework for LiDAR-visual systems, integrating Neural Radiance Fields (NeRF) and Neural Distance Fields (NDF) to recover both appearance and structural information from posed images and point clouds. We address the structural visible gap between NeRF and NDF by utilizing a visible-aware occupancy map to classify space into the free, occupied, visible unknown, and background regions. This classification facilitates the recovery of a complete appearance and structure of the scene. We unify the training of the NDF and NeRF using a spatial-varying scale SDF-to-density transformation for levels of detail for both structure and appearance. The proposed method leverages the learned NDF for structure-aware NeRF training by an adaptive sphere tracing sampling strategy for accurate structure rendering. In return, NeRF further refines structural in recovering missing or fuzzy structures in the NDF. Extensive experiments demonstrate the superior quality and versatility of the proposed method across various scenarios. To benefit the community, the codes will be released at \url{https://github.com/hku-mars/M2Mapping}.  
  </ol>  
</details>  
  
### [SCARF: Scalable Continual Learning Framework for Memory-efficient Multiple Neural Radiance Fields](http://arxiv.org/abs/2409.04482)  
Yuze Wang, Junyi Wang, Chen Wang, Wantong Duan, Yongtang Bao, Yue Qi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper introduces a novel continual learning framework for synthesising novel views of multiple scenes, learning multiple 3D scenes incrementally, and updating the network parameters only with the training data of the upcoming new scene. We build on Neural Radiance Fields (NeRF), which uses multi-layer perceptron to model the density and radiance field of a scene as the implicit function. While NeRF and its extensions have shown a powerful capability of rendering photo-realistic novel views in a single 3D scene, managing these growing 3D NeRF assets efficiently is a new scientific problem. Very few works focus on the efficient representation or continuous learning capability of multiple scenes, which is crucial for the practical applications of NeRF. To achieve these goals, our key idea is to represent multiple scenes as the linear combination of a cross-scene weight matrix and a set of scene-specific weight matrices generated from a global parameter generator. Furthermore, we propose an uncertain surface knowledge distillation strategy to transfer the radiance field knowledge of previous scenes to the new model. Representing multiple 3D scenes with such weight matrices significantly reduces memory requirements. At the same time, the uncertain surface distillation strategy greatly overcomes the catastrophic forgetting problem and maintains the photo-realistic rendering quality of previous scenes. Experiments show that the proposed approach achieves state-of-the-art rendering quality of continual learning NeRF on NeRF-Synthetic, LLFF, and TanksAndTemples datasets while preserving extra low storage cost.  
  </ol>  
</details>  
  
  



