<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#From-Monocular-Vision-to-Autonomous-Action:-Guiding-Tumor-Resection-via-3D-Reconstruction>From Monocular Vision to Autonomous Action: Guiding Tumor Resection via 3D Reconstruction</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#PromptHash:-Affinity-Prompted-Collaborative-Cross-Modal-Learning-for-Adaptive-Hashing-Retrieval>PromptHash: Affinity-Prompted Collaborative Cross-Modal Learning for Adaptive Hashing Retrieval</a></li>
        <li><a href=#Automating-3D-Dataset-Generation-with-Neural-Radiance-Fields>Automating 3D Dataset Generation with Neural Radiance Fields</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Loop-Closure-from-Two-Views:-Revisiting-PGO-for-Scalable-Trajectory-Estimation-through-Monocular-Priors>Loop Closure from Two Views: Revisiting PGO for Scalable Trajectory Estimation through Monocular Priors</a></li>
        <li><a href=#MapGlue:-Multimodal-Remote-Sensing-Image-Matching>MapGlue: Multimodal Remote Sensing Image Matching</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Enhancing-Close-up-Novel-View-Synthesis-via-Pseudo-labeling>Enhancing Close-up Novel View Synthesis via Pseudo-labeling</a></li>
        <li><a href=#SPNeRF:-Open-Vocabulary-3D-Neural-Scene-Segmentation-with-Superpoints>SPNeRF: Open Vocabulary 3D Neural Scene Segmentation with Superpoints</a></li>
        <li><a href=#DiffPortrait360:-Consistent-Portrait-Diffusion-for-360-View-Synthesis>DiffPortrait360: Consistent Portrait Diffusion for 360 View Synthesis</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [From Monocular Vision to Autonomous Action: Guiding Tumor Resection via 3D Reconstruction](http://arxiv.org/abs/2503.16263)  
Ayberk Acar, Mariana Smith, Lidia Al-Zogbi, Tanner Watts, Fangjie Li, Hao Li, Nural Yilmaz, Paul Maria Scheikl, Jesse F. d'Almeida, Susheela Sharma, Lauren Branscombe, Tayfun Efe Ertop, Robert J. Webster III, Ipek Oguz, Alan Kuntz, Axel Krieger, Jie Ying Wu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Surgical automation requires precise guidance and understanding of the scene. Current methods in the literature rely on bulky depth cameras to create maps of the anatomy, however this does not translate well to space-limited clinical applications. Monocular cameras are small and allow minimally invasive surgeries in tight spaces but additional processing is required to generate 3D scene understanding. We propose a 3D mapping pipeline that uses only RGB images to create segmented point clouds of the target anatomy. To ensure the most precise reconstruction, we compare different structure from motion algorithms' performance on mapping the central airway obstructions, and test the pipeline on a downstream task of tumor resection. In several metrics, including post-procedure tissue model evaluation, our pipeline performs comparably to RGB-D cameras and, in some cases, even surpasses their performance. These promising results demonstrate that automation guidance can be achieved in minimally invasive procedures with monocular cameras. This study is a step toward the complete autonomy of surgical robots.  
  </ol>  
</details>  
**comments**: 7 Pages, 8 Figures, 1 Table. This work has been submitted IEEE/RSJ
  International Conference on Intelligent Robots and Systems (IROS) for
  possible publication  
  
  



## Visual Localization  

### [PromptHash: Affinity-Prompted Collaborative Cross-Modal Learning for Adaptive Hashing Retrieval](http://arxiv.org/abs/2503.16064)  
[[code](https://github.com/ShiShuMo/PromptHash)]  
Qiang Zou, Shuli Cheng, Jiayi Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Cross-modal hashing is a promising approach for efficient data retrieval and storage optimization. However, contemporary methods exhibit significant limitations in semantic preservation, contextual integrity, and information redundancy, which constrains retrieval efficacy. We present PromptHash, an innovative framework leveraging affinity prompt-aware collaborative learning for adaptive cross-modal hashing. We propose an end-to-end framework for affinity-prompted collaborative hashing, with the following fundamental technical contributions: (i) a text affinity prompt learning mechanism that preserves contextual information while maintaining parameter efficiency, (ii) an adaptive gated selection fusion architecture that synthesizes State Space Model with Transformer network for precise cross-modal feature integration, and (iii) a prompt affinity alignment strategy that bridges modal heterogeneity through hierarchical contrastive learning. To the best of our knowledge, this study presents the first investigation into affinity prompt awareness within collaborative cross-modal adaptive hash learning, establishing a paradigm for enhanced semantic consistency across modalities. Through comprehensive evaluation on three benchmark multi-label datasets, PromptHash demonstrates substantial performance improvements over existing approaches. Notably, on the NUS-WIDE dataset, our method achieves significant gains of 18.22% and 18.65% in image-to-text and text-to-image retrieval tasks, respectively. The code is publicly available at https://github.com/ShiShuMo/PromptHash.  
  </ol>  
</details>  
**comments**: Accepted by CVPR2025  
  
### [Automating 3D Dataset Generation with Neural Radiance Fields](http://arxiv.org/abs/2503.15997)  
P. Schulz, T. Hempel, A. Al-Hamadi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D detection is a critical task to understand spatial characteristics of the environment and is used in a variety of applications including robotics, augmented reality, and image retrieval. Training performant detection models require diverse, precisely annotated, and large scale datasets that involve complex and expensive creation processes. Hence, there are only few public 3D datasets that are additionally limited in their range of classes. In this work, we propose a pipeline for automatic generation of 3D datasets for arbitrary objects. By utilizing the universal 3D representation and rendering capabilities of Radiance Fields, our pipeline generates high quality 3D models for arbitrary objects. These 3D models serve as input for a synthetic dataset generator. Our pipeline is fast, easy to use and has a high degree of automation. Our experiments demonstrate, that 3D pose estimation networks, trained with our generated datasets, archive strong performance in typical application scenarios.  
  </ol>  
</details>  
**comments**: Accepted and presented at ROBOVIS 2025 (5th International Conference
  on Robotics, Computer Vision and Intelligent Systems)  
  
  



## Image Matching  

### [Loop Closure from Two Views: Revisiting PGO for Scalable Trajectory Estimation through Monocular Priors](http://arxiv.org/abs/2503.16275)  
Tian Yi Lim, Boyang Sun, Marc Pollefeys, Hermann Blum  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    (Visual) Simultaneous Localization and Mapping (SLAM) remains a fundamental challenge in enabling autonomous systems to navigate and understand large-scale environments. Traditional SLAM approaches struggle to balance efficiency and accuracy, particularly in large-scale settings where extensive computational resources are required for scene reconstruction and Bundle Adjustment (BA). However, this scene reconstruction, in the form of sparse pointclouds of visual landmarks, is often only used within the SLAM system because navigation and planning methods require different map representations. In this work, we therefore investigate a more scalable Visual SLAM (VSLAM) approach without reconstruction, mainly based on approaches for two-view loop closures. By restricting the map to a sparse keyframed pose graph without dense geometry representations, our '2GO' system achieves efficient optimization with competitive absolute trajectory accuracy. In particular, we find that recent advancements in image matching and monocular depth priors enable very accurate trajectory optimization from two-view edges. We conduct extensive experiments on diverse datasets, including large-scale scenarios, and provide a detailed analysis of the trade-offs between runtime, accuracy, and map size. Our results demonstrate that this streamlined approach supports real-time performance, scales well in map size and trajectory duration, and effectively broadens the capabilities of VSLAM for long-duration deployments to large environments.  
  </ol>  
</details>  
  
### [MapGlue: Multimodal Remote Sensing Image Matching](http://arxiv.org/abs/2503.16185)  
Peihao Wu, Yongxiang Yao, Wenfei Zhang, Dong Wei, Yi Wan, Yansheng Li, Yongjun Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Multimodal remote sensing image (MRSI) matching is pivotal for cross-modal fusion, localization, and object detection, but it faces severe challenges due to geometric, radiometric, and viewpoint discrepancies across imaging modalities. Existing unimodal datasets lack scale and diversity, limiting deep learning solutions. This paper proposes MapGlue, a universal MRSI matching framework, and MapData, a large-scale multimodal dataset addressing these gaps. Our contributions are twofold. MapData, a globally diverse dataset spanning 233 sampling points, offers original images (7,000x5,000 to 20,000x15,000 pixels). After rigorous cleaning, it provides 121,781 aligned electronic map-visible image pairs (512x512 pixels) with hybrid manual-automated ground truth, addressing the scarcity of scalable multimodal benchmarks. MapGlue integrates semantic context with a dual graph-guided mechanism to extract cross-modal invariant features. This structure enables global-to-local interaction, enhancing descriptor robustness against modality-specific distortions. Extensive evaluations on MapData and five public datasets demonstrate MapGlue's superiority in matching accuracy under complex conditions, outperforming state-of-the-art methods. Notably, MapGlue generalizes effectively to unseen modalities without retraining, highlighting its adaptability. This work addresses longstanding challenges in MRSI matching by combining scalable dataset construction with a robust, semantics-driven framework. Furthermore, MapGlue shows strong generalization capabilities on other modality matching tasks for which it was not specifically trained. The dataset and code are available at https://github.com/PeihaoWu/MapGlue.  
  </ol>  
</details>  
**comments**: The dataset and code are available at
  https://github.com/PeihaoWu/MapGlue  
  
  



## NeRF  

### [Enhancing Close-up Novel View Synthesis via Pseudo-labeling](http://arxiv.org/abs/2503.15908)  
Jiatong Xia, Libo Sun, Lingqiao Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent methods, such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), have demonstrated remarkable capabilities in novel view synthesis. However, despite their success in producing high-quality images for viewpoints similar to those seen during training, they struggle when generating detailed images from viewpoints that significantly deviate from the training set, particularly in close-up views. The primary challenge stems from the lack of specific training data for close-up views, leading to the inability of current methods to render these views accurately. To address this issue, we introduce a novel pseudo-label-based learning strategy. This approach leverages pseudo-labels derived from existing training data to provide targeted supervision across a wide range of close-up viewpoints. Recognizing the absence of benchmarks for this specific challenge, we also present a new dataset designed to assess the effectiveness of both current and future methods in this area. Our extensive experiments demonstrate the efficacy of our approach.  
  </ol>  
</details>  
**comments**: Accepted by AAAI 2025  
  
### [SPNeRF: Open Vocabulary 3D Neural Scene Segmentation with Superpoints](http://arxiv.org/abs/2503.15712)  
Weiwen Hu, Niccolò Parodi, Marcus Zepp, Ingo Feldmann, Oliver Schreer, Peter Eisert  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Open-vocabulary segmentation, powered by large visual-language models like CLIP, has expanded 2D segmentation capabilities beyond fixed classes predefined by the dataset, enabling zero-shot understanding across diverse scenes. Extending these capabilities to 3D segmentation introduces challenges, as CLIP's image-based embeddings often lack the geometric detail necessary for 3D scene segmentation. Recent methods tend to address this by introducing additional segmentation models or replacing CLIP with variations trained on segmentation data, which lead to redundancy or loss on CLIP's general language capabilities. To overcome this limitation, we introduce SPNeRF, a NeRF based zero-shot 3D segmentation approach that leverages geometric priors. We integrate geometric primitives derived from the 3D scene into NeRF training to produce primitive-wise CLIP features, avoiding the ambiguity of point-wise features. Additionally, we propose a primitive-based merging mechanism enhanced with affinity scores. Without relying on additional segmentation models, our method further explores CLIP's capability for 3D segmentation and achieves notable improvements over original LERF.  
  </ol>  
</details>  
**comments**: In Proceedings of the 20th International Joint Conference on Computer
  Vision, Imaging and Computer Graphics Theory and Applications (2025)  
  
### [DiffPortrait360: Consistent Portrait Diffusion for 360 View Synthesis](http://arxiv.org/abs/2503.15667)  
Yuming Gu, Phong Tran, Yujian Zheng, Hongyi Xu, Heyuan Li, Adilbek Karmanov, Hao Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Generating high-quality 360-degree views of human heads from single-view images is essential for enabling accessible immersive telepresence applications and scalable personalized content creation. While cutting-edge methods for full head generation are limited to modeling realistic human heads, the latest diffusion-based approaches for style-omniscient head synthesis can produce only frontal views and struggle with view consistency, preventing their conversion into true 3D models for rendering from arbitrary angles. We introduce a novel approach that generates fully consistent 360-degree head views, accommodating human, stylized, and anthropomorphic forms, including accessories like glasses and hats. Our method builds on the DiffPortrait3D framework, incorporating a custom ControlNet for back-of-head detail generation and a dual appearance module to ensure global front-back consistency. By training on continuous view sequences and integrating a back reference image, our approach achieves robust, locally continuous view synthesis. Our model can be used to produce high-quality neural radiance fields (NeRFs) for real-time, free-viewpoint rendering, outperforming state-of-the-art methods in object synthesis and 360-degree head generation for very challenging input portraits.  
  </ol>  
</details>  
**comments**: Page:https://freedomgu.github.io/DiffPortrait360
  Code:https://github.com/FreedomGu/DiffPortrait360/  
  
  



