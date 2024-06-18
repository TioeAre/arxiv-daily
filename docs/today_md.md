<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Self-supervised-Pretraining-and-Finetuning-for-Monocular-Depth-and-Visual-Odometry>Self-supervised Pretraining and Finetuning for Monocular Depth and Visual Odometry</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#MegaScenes:-Scene-Level-View-Synthesis-at-Scale>MegaScenes: Scene-Level View Synthesis at Scale</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Matching-Query-Image-Against-Selected-NeRF-Feature-for-Efficient-and-Scalable-Localization>Matching Query Image Against Selected NeRF Feature for Efficient and Scalable Localization</a></li>
        <li><a href=#Simple-Yet-Efficient:-Towards-Self-Supervised-FG-SBIR-with-Unified-Sample-Feature-Alignment>Simple Yet Efficient: Towards Self-Supervised FG-SBIR with Unified Sample Feature Alignment</a></li>
        <li><a href=#They're-All-Doctors:-Synthesizing-Diverse-Counterfactuals-to-Mitigate-Associative-Bias>They're All Doctors: Synthesizing Diverse Counterfactuals to Mitigate Associative Bias</a></li>
        <li><a href=#Accurate-and-Fast-Pixel-Retrieval-with-Spatial-and-Uncertainty-Aware-Hypergraph-Diffusion>Accurate and Fast Pixel Retrieval with Spatial and Uncertainty Aware Hypergraph Diffusion</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Light-Up-the-Shadows:-Enhance-Long-Tailed-Entity-Grounding-with-Concept-Guided-Vision-Language-Models>Light Up the Shadows: Enhance Long-Tailed Entity Grounding with Concept-Guided Vision-Language Models</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#LLaNA:-Large-Language-and-NeRF-Assistant>LLaNA: Large Language and NeRF Assistant</a></li>
        <li><a href=#Matching-Query-Image-Against-Selected-NeRF-Feature-for-Efficient-and-Scalable-Localization>Matching Query Image Against Selected NeRF Feature for Efficient and Scalable Localization</a></li>
        <li><a href=#InterNeRF:-Scaling-Radiance-Fields-via-Parameter-Interpolation>InterNeRF: Scaling Radiance Fields via Parameter Interpolation</a></li>
        <li><a href=#NLDF:-Neural-Light-Dynamic-Fields-for-Efficient-3D-Talking-Head-Generation>NLDF: Neural Light Dynamic Fields for Efficient 3D Talking Head Generation</a></li>
        <li><a href=#NeRFDeformer:-NeRF-Transformation-from-a-Single-View-via-3D-Scene-Flows>NeRFDeformer: NeRF Transformation from a Single View via 3D Scene Flows</a></li>
        <li><a href=#Federated-Neural-Radiance-Field-for-Distributed-Intelligence>Federated Neural Radiance Field for Distributed Intelligence</a></li>
        <li><a href=#Wild-GS:-Real-Time-Novel-View-Synthesis-from-Unconstrained-Photo-Collections>Wild-GS: Real-Time Novel View Synthesis from Unconstrained Photo Collections</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Self-supervised Pretraining and Finetuning for Monocular Depth and Visual Odometry](http://arxiv.org/abs/2406.11019)  
Boris Chidlovskii, Leonid Antsfeld  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    For the task of simultaneous monocular depth and visual odometry estimation, we propose learning self-supervised transformer-based models in two steps. Our first step consists in a generic pretraining to learn 3D geometry, using cross-view completion objective (CroCo), followed by self-supervised finetuning on non-annotated videos. We show that our self-supervised models can reach state-of-the-art performance 'without bells and whistles' using standard components such as visual transformers, dense prediction transformers and adapters. We demonstrate the effectiveness of our proposed method by running evaluations on six benchmark datasets, both static and dynamic, indoor and outdoor, with synthetic and real images. For all datasets, our method outperforms state-of-the-art methods, in particular for depth prediction task.  
  </ol>  
</details>  
**comments**: 8 pages, to appear in ICRA'24  
  
  



## SFM  

### [MegaScenes: Scene-Level View Synthesis at Scale](http://arxiv.org/abs/2406.11819)  
Joseph Tung, Gene Chou, Ruojin Cai, Guandao Yang, Kai Zhang, Gordon Wetzstein, Bharath Hariharan, Noah Snavely  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Scene-level novel view synthesis (NVS) is fundamental to many vision and graphics applications. Recently, pose-conditioned diffusion models have led to significant progress by extracting 3D information from 2D foundation models, but these methods are limited by the lack of scene-level training data. Common dataset choices either consist of isolated objects (Objaverse), or of object-centric scenes with limited pose distributions (DTU, CO3D). In this paper, we create a large-scale scene-level dataset from Internet photo collections, called MegaScenes, which contains over 100K structure from motion (SfM) reconstructions from around the world. Internet photos represent a scalable data source but come with challenges such as lighting and transient objects. We address these issues to further create a subset suitable for the task of NVS. Additionally, we analyze failure cases of state-of-the-art NVS methods and significantly improve generation consistency. Through extensive experiments, we validate the effectiveness of both our dataset and method on generating in-the-wild scenes. For details on the dataset and code, see our project page at https://megascenes.github.io .  
  </ol>  
</details>  
**comments**: Our project page is at https://megascenes.github.io  
  
  



## Visual Localization  

### [Matching Query Image Against Selected NeRF Feature for Efficient and Scalable Localization](http://arxiv.org/abs/2406.11766)  
Huaiji Zhou, Bing Wang, Changhao Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural implicit representations such as NeRF have revolutionized 3D scene representation with photo-realistic quality. However, existing methods for visual localization within NeRF representations suffer from inefficiency and scalability issues, particularly in large-scale environments. This work proposes MatLoc-NeRF, a novel matching-based localization framework using selected NeRF features. It addresses efficiency by employing a learnable feature selection mechanism that identifies informative NeRF features for matching with query images. This eliminates the need for all NeRF features or additional descriptors, leading to faster and more accurate pose estimation. To tackle large-scale scenes, MatLoc-NeRF utilizes a pose-aware scene partitioning strategy. It ensures that only the most relevant NeRF sub-block generates key features for a specific pose. Additionally, scene segmentation and a place predictor provide fast coarse initial pose estimation. Evaluations on public large-scale datasets demonstrate that MatLoc-NeRF achieves superior efficiency and accuracy compared to existing NeRF-based localization methods.  
  </ol>  
</details>  
**comments**: 12 pages, 2 figures  
  
### [Simple Yet Efficient: Towards Self-Supervised FG-SBIR with Unified Sample Feature Alignment](http://arxiv.org/abs/2406.11551)  
Jianan Jiang, Di Wu, Zhilin Jiang, Weiren Yu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Fine-Grained Sketch-Based Image Retrieval (FG-SBIR) aims to minimize the distance between sketches and corresponding images in the embedding space. However, scalability is hindered by the growing complexity of solutions, mainly due to the abstract nature of fine-grained sketches. In this paper, we propose a simple yet efficient approach to narrow the gap between the two modes. It mainly facilitates unified mutual information sharing both intra- and inter-samples, rather than treating them as a single feature alignment problem between modalities. Specifically, our approach includes: (i) Employing dual weight-sharing networks to optimize alignment within sketch and image domain, which also effectively mitigates model learning saturation issues. (ii) Introducing an objective optimization function based on contrastive loss to enhance the model's ability to align features intra- and inter-samples. (iii) Presenting a learnable TRSM combined of self-attention and cross-attention to promote feature representations among tokens, further enhancing sample alignment in the embedding space. Our framework achieves excellent results on CNN- and ViT-based backbones. Extensive experiments demonstrate its superiority over existing methods. We also introduce Cloths-V1, the first professional fashion sketches and images dataset, utilized to validate our method and will be beneficial for other applications.  
  </ol>  
</details>  
**comments**: 10 pages,8 figures, 4 tables  
  
### [They're All Doctors: Synthesizing Diverse Counterfactuals to Mitigate Associative Bias](http://arxiv.org/abs/2406.11331)  
Salma Abdel Magid, Jui-Hsien Wang, Kushal Kafle, Hanspeter Pfister  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Vision Language Models (VLMs) such as CLIP are powerful models; however they can exhibit unwanted biases, making them less safe when deployed directly in applications such as text-to-image, text-to-video retrievals, reverse search, or classification tasks. In this work, we propose a novel framework to generate synthetic counterfactual images to create a diverse and balanced dataset that can be used to fine-tune CLIP. Given a set of diverse synthetic base images from text-to-image models, we leverage off-the-shelf segmentation and inpainting models to place humans with diverse visual appearances in context. We show that CLIP trained on such datasets learns to disentangle the human appearance from the context of an image, i.e., what makes a doctor is not correlated to the person's visual appearance, like skin color or body type, but to the context, such as background, the attire they are wearing, or the objects they are holding. We demonstrate that our fine-tuned CLIP model, $CF_\alpha$ , improves key fairness metrics such as MaxSkew, MinSkew, and NDKL by 40-66\% for image retrieval tasks, while still achieving similar levels of performance in downstream tasks. We show that, by design, our model retains maximal compatibility with the original CLIP models, and can be easily controlled to support different accuracy versus fairness trade-offs in a plug-n-play fashion.  
  </ol>  
</details>  
  
### [Accurate and Fast Pixel Retrieval with Spatial and Uncertainty Aware Hypergraph Diffusion](http://arxiv.org/abs/2406.11242)  
Guoyuan An, Yuchi Huo, Sung-Eui Yoon  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper presents a novel method designed to enhance the efficiency and accuracy of both image retrieval and pixel retrieval. Traditional diffusion methods struggle to propagate spatial information effectively in conventional graphs due to their reliance on scalar edge weights. To overcome this limitation, we introduce a hypergraph-based framework, uniquely capable of efficiently propagating spatial information using local features during query time, thereby accurately retrieving and localizing objects within a database.   Additionally, we innovatively utilize the structural information of the image graph through a technique we term "community selection". This approach allows for the assessment of the initial search result's uncertainty and facilitates an optimal balance between accuracy and speed. This is particularly crucial in real-world applications where such trade-offs are often necessary.   Our experimental results, conducted on the (P)ROxford and (P)RParis datasets, demonstrate the significant superiority of our method over existing diffusion techniques. We achieve state-of-the-art (SOTA) accuracy in both image-level and pixel-level retrieval, while also maintaining impressive processing speed. This dual achievement underscores the effectiveness of our hypergraph-based framework and community selection technique, marking a notable advancement in the field of content-based image retrieval.  
  </ol>  
</details>  
  
  



## Image Matching  

### [Light Up the Shadows: Enhance Long-Tailed Entity Grounding with Concept-Guided Vision-Language Models](http://arxiv.org/abs/2406.10902)  
Yikai Zhang, Qianyu He, Xintao Wang, Siyu Yuan, Jiaqing Liang, Yanghua Xiao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Multi-Modal Knowledge Graphs (MMKGs) have proven valuable for various downstream tasks. However, scaling them up is challenging because building large-scale MMKGs often introduces mismatched images (i.e., noise). Most entities in KGs belong to the long tail, meaning there are few images of them available online. This scarcity makes it difficult to determine whether a found image matches the entity. To address this, we draw on the Triangle of Reference Theory and suggest enhancing vision-language models with concept guidance. Specifically, we introduce COG, a two-stage framework with COncept-Guided vision-language models. The framework comprises a Concept Integration module, which effectively identifies image-text pairs of long-tailed entities, and an Evidence Fusion module, which offers explainability and enables human verification. To demonstrate the effectiveness of COG, we create a dataset of 25k image-text pairs of long-tailed entities. Our comprehensive experiments show that COG not only improves the accuracy of recognizing long-tailed image-text pairs compared to baselines but also offers flexibility and explainability.  
  </ol>  
</details>  
  
  



## NeRF  

### [LLaNA: Large Language and NeRF Assistant](http://arxiv.org/abs/2406.11840)  
Andrea Amaduzzi, Pierluigi Zama Ramirez, Giuseppe Lisanti, Samuele Salti, Luigi Di Stefano  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Multimodal Large Language Models (MLLMs) have demonstrated an excellent understanding of images and 3D data. However, both modalities have shortcomings in holistically capturing the appearance and geometry of objects. Meanwhile, Neural Radiance Fields (NeRFs), which encode information within the weights of a simple Multi-Layer Perceptron (MLP), have emerged as an increasingly widespread modality that simultaneously encodes the geometry and photorealistic appearance of objects. This paper investigates the feasibility and effectiveness of ingesting NeRF into MLLM. We create LLaNA, the first general-purpose NeRF-language assistant capable of performing new tasks such as NeRF captioning and Q\&A. Notably, our method directly processes the weights of the NeRF's MLP to extract information about the represented objects without the need to render images or materialize 3D data structures. Moreover, we build a dataset of NeRFs with text annotations for various NeRF-language tasks with no human intervention. Based on this dataset, we develop a benchmark to evaluate the NeRF understanding capability of our method. Results show that processing NeRF weights performs favourably against extracting 2D or 3D representations from NeRFs.  
  </ol>  
</details>  
**comments**: Under review. Project page: https://andreamaduzzi.github.io/llana/  
  
### [Matching Query Image Against Selected NeRF Feature for Efficient and Scalable Localization](http://arxiv.org/abs/2406.11766)  
Huaiji Zhou, Bing Wang, Changhao Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural implicit representations such as NeRF have revolutionized 3D scene representation with photo-realistic quality. However, existing methods for visual localization within NeRF representations suffer from inefficiency and scalability issues, particularly in large-scale environments. This work proposes MatLoc-NeRF, a novel matching-based localization framework using selected NeRF features. It addresses efficiency by employing a learnable feature selection mechanism that identifies informative NeRF features for matching with query images. This eliminates the need for all NeRF features or additional descriptors, leading to faster and more accurate pose estimation. To tackle large-scale scenes, MatLoc-NeRF utilizes a pose-aware scene partitioning strategy. It ensures that only the most relevant NeRF sub-block generates key features for a specific pose. Additionally, scene segmentation and a place predictor provide fast coarse initial pose estimation. Evaluations on public large-scale datasets demonstrate that MatLoc-NeRF achieves superior efficiency and accuracy compared to existing NeRF-based localization methods.  
  </ol>  
</details>  
**comments**: 12 pages, 2 figures  
  
### [InterNeRF: Scaling Radiance Fields via Parameter Interpolation](http://arxiv.org/abs/2406.11737)  
Clinton Wang, Peter Hedman, Polina Golland, Jonathan T. Barron, Daniel Duckworth  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRFs) have unmatched fidelity on large, real-world scenes. A common approach for scaling NeRFs is to partition the scene into regions, each of which is assigned its own parameters. When implemented naively, such an approach is limited by poor test-time scaling and inconsistent appearance and geometry. We instead propose InterNeRF, a novel architecture for rendering a target view using a subset of the model's parameters. Our approach enables out-of-core training and rendering, increasing total model capacity with only a modest increase to training time. We demonstrate significant improvements in multi-room scenes while remaining competitive on standard benchmarks.  
  </ol>  
</details>  
**comments**: Presented at CVPR 2024 Neural Rendering Intelligence Workshop  
  
### [NLDF: Neural Light Dynamic Fields for Efficient 3D Talking Head Generation](http://arxiv.org/abs/2406.11259)  
Niu Guanchen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Talking head generation based on the neural radiation fields model has shown promising visual effects. However, the slow rendering speed of NeRF seriously limits its application, due to the burdensome calculation process over hundreds of sampled points to synthesize one pixel. In this work, a novel Neural Light Dynamic Fields model is proposed aiming to achieve generating high quality 3D talking face with significant speedup. The NLDF represents light fields based on light segments, and a deep network is used to learn the entire light beam's information at once. In learning the knowledge distillation is applied and the NeRF based synthesized result is used to guide the correct coloration of light segments in NLDF. Furthermore, a novel active pool training strategy is proposed to focus on high frequency movements, particularly on the speaker mouth and eyebrows. The propose method effectively represents the facial light dynamics in 3D talking video generation, and it achieves approximately 30 times faster speed compared to state of the art NeRF based method, with comparable generation visual quality.  
  </ol>  
</details>  
  
### [NeRFDeformer: NeRF Transformation from a Single View via 3D Scene Flows](http://arxiv.org/abs/2406.10543)  
[[code](https://github.com/nerfdeformer/nerfdeformer)]  
Zhenggang Tang, Zhongzheng Ren, Xiaoming Zhao, Bowen Wen, Jonathan Tremblay, Stan Birchfield, Alexander Schwing  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present a method for automatically modifying a NeRF representation based on a single observation of a non-rigid transformed version of the original scene. Our method defines the transformation as a 3D flow, specifically as a weighted linear blending of rigid transformations of 3D anchor points that are defined on the surface of the scene. In order to identify anchor points, we introduce a novel correspondence algorithm that first matches RGB-based pairs, then leverages multi-view information and 3D reprojection to robustly filter false positives in two steps. We also introduce a new dataset for exploring the problem of modifying a NeRF scene through a single observation. Our dataset ( https://github.com/nerfdeformer/nerfdeformer ) contains 113 synthetic scenes leveraging 47 3D assets. We show that our proposed method outperforms NeRF editing methods as well as diffusion-based methods, and we also explore different methods for filtering correspondences.  
  </ol>  
</details>  
**comments**: 8 pages of main paper, CVPR 2024. Proceedings of the IEEE/CVF
  Conference on Computer Vision and Pattern Recognition. 2024  
  
### [Federated Neural Radiance Field for Distributed Intelligence](http://arxiv.org/abs/2406.10474)  
Yintian Zhang, Ziyu Shao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Novel view synthesis (NVS) is an important technology for many AR and VR applications. The recently proposed Neural Radiance Field (NeRF) approach has demonstrated superior performance on NVS tasks, and has been applied to other related fields. However, certain application scenarios with distributed data storage may pose challenges on acquiring training images for the NeRF approach, due to strict regulations and privacy concerns. In order to overcome this challenge, we focus on FedNeRF, a federated learning (FL) based NeRF approach that utilizes images available at different data owners while preserving data privacy.   In this paper, we first construct a resource-rich and functionally diverse federated learning testbed. Then, we deploy FedNeRF algorithm in such a practical FL system, and conduct FedNeRF experiments with partial client selection. It is expected that the studies of the FedNeRF approach presented in this paper will be helpful to facilitate future applications of NeRF approach in distributed data storage scenarios.  
  </ol>  
</details>  
  
### [Wild-GS: Real-Time Novel View Synthesis from Unconstrained Photo Collections](http://arxiv.org/abs/2406.10373)  
Jiacong Xu, Yiqun Mei, Vishal M. Patel  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Photographs captured in unstructured tourist environments frequently exhibit variable appearances and transient occlusions, challenging accurate scene reconstruction and inducing artifacts in novel view synthesis. Although prior approaches have integrated the Neural Radiance Field (NeRF) with additional learnable modules to handle the dynamic appearances and eliminate transient objects, their extensive training demands and slow rendering speeds limit practical deployments. Recently, 3D Gaussian Splatting (3DGS) has emerged as a promising alternative to NeRF, offering superior training and inference efficiency along with better rendering quality. This paper presents Wild-GS, an innovative adaptation of 3DGS optimized for unconstrained photo collections while preserving its efficiency benefits. Wild-GS determines the appearance of each 3D Gaussian by their inherent material attributes, global illumination and camera properties per image, and point-level local variance of reflectance. Unlike previous methods that model reference features in image space, Wild-GS explicitly aligns the pixel appearance features to the corresponding local Gaussians by sampling the triplane extracted from the reference image. This novel design effectively transfers the high-frequency detailed appearance of the reference view to 3D space and significantly expedites the training process. Furthermore, 2D visibility maps and depth regularization are leveraged to mitigate the transient effects and constrain the geometry, respectively. Extensive experiments demonstrate that Wild-GS achieves state-of-the-art rendering performance and the highest efficiency in both training and inference among all the existing techniques.  
  </ol>  
</details>  
**comments**: 15 pages, 7 figures  
  
  



