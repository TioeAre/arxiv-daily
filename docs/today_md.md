<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#pySLAM:-An-Open-Source,-Modular,-and-Extensible-Framework-for-SLAM>pySLAM: An Open-Source, Modular, and Extensible Framework for SLAM</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Descriminative-Generative-Custom-Tokens-for-Vision-Language-Models>Descriminative-Generative Custom Tokens for Vision-Language Models</a></li>
        <li><a href=#ILIAS:-Instance-Level-Image-retrieval-At-Scale>ILIAS: Instance-Level Image retrieval At Scale</a></li>
        <li><a href=#Range-and-Bird's-Eye-View-Fused-Cross-Modal-Visual-Place-Recognition>Range and Bird's Eye View Fused Cross-Modal Visual Place Recognition</a></li>
        <li><a href=#Adversarially-Robust-CLIP-Models-Can-Induce-Better-(Robust)-Perceptual-Metrics>Adversarially Robust CLIP Models Can Induce Better (Robust) Perceptual Metrics</a></li>
        <li><a href=#Precise-GPS-Denied-UAV-Self-Positioning-via-Context-Enhanced-Cross-View-Geo-Localization>Precise GPS-Denied UAV Self-Positioning via Context-Enhanced Cross-View Geo-Localization</a></li>
        <li><a href=#E2LVLM:Evidence-Enhanced-Large-Vision-Language-Model-for-Multimodal-Out-of-Context-Misinformation-Detection>E2LVLM:Evidence-Enhanced Large Vision-Language Model for Multimodal Out-of-Context Misinformation Detection</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#FeaKM:-Robust-Collaborative-Perception-under-Noisy-Pose-Conditions>FeaKM: Robust Collaborative Perception under Noisy Pose Conditions</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#HumanGif:-Single-View-Human-Diffusion-with-Generative-Prior>HumanGif: Single-View Human Diffusion with Generative Prior</a></li>
        <li><a href=#3D-Gaussian-Inpainting-with-Depth-Guided-Cross-View-Consistency>3D Gaussian Inpainting with Depth-Guided Cross-View Consistency</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [pySLAM: An Open-Source, Modular, and Extensible Framework for SLAM](http://arxiv.org/abs/2502.11955)  
Luigi Freda  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    pySLAM is an open-source Python framework for Visual SLAM, supporting monocular, stereo, and RGB-D cameras. It provides a flexible interface for integrating both classical and modern local features, making it adaptable to various SLAM tasks. The framework includes different loop closure methods, a volumetric reconstruction pipeline, and support for depth prediction models. Additionally, it offers a suite of tools for visual odometry and SLAM applications. Designed for both beginners and experienced researchers, pySLAM encourages community contributions, fostering collaborative development in the field of Visual SLAM.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Descriminative-Generative Custom Tokens for Vision-Language Models](http://arxiv.org/abs/2502.12095)  
Pramuditha Perera, Matthew Trager, Luca Zancato, Alessandro Achille, Stefano Soatto  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper explores the possibility of learning custom tokens for representing new concepts in Vision-Language Models (VLMs). Our aim is to learn tokens that can be effective for both discriminative and generative tasks while composing well with words to form new input queries. The targeted concept is specified in terms of a small set of images and a parent concept described using text. We operate on CLIP text features and propose to use a combination of a textual inversion loss and a classification loss to ensure that text features of the learned token are aligned with image features of the concept in the CLIP embedding space. We restrict the learned token to a low-dimensional subspace spanned by tokens for attributes that are appropriate for the given super-class. These modifications improve the quality of compositions of the learned token with natural language for generating new scenes. Further, we show that learned custom tokens can be used to form queries for text-to-image retrieval task, and also have the important benefit that composite queries can be visualized to ensure that the desired concept is faithfully encoded. Based on this, we introduce the method of Generation Aided Image Retrieval, where the query is modified at inference time to better suit the search intent. On the DeepFashion2 dataset, our method improves Mean Reciprocal Retrieval (MRR) over relevant baselines by 7%.  
  </ol>  
</details>  
  
### [ILIAS: Instance-Level Image retrieval At Scale](http://arxiv.org/abs/2502.11748)  
Giorgos Kordopatis-Zilos, Vladan Stojnić, Anna Manko, Pavel Šuma, Nikolaos-Antonios Ypsilantis, Nikos Efthymiadis, Zakaria Laskar, Jiří Matas, Ondřej Chum, Giorgos Tolias  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This work introduces ILIAS, a new test dataset for Instance-Level Image retrieval At Scale. It is designed to evaluate the ability of current and future foundation models and retrieval techniques to recognize particular objects. The key benefits over existing datasets include large scale, domain diversity, accurate ground truth, and a performance that is far from saturated. ILIAS includes query and positive images for 1,000 object instances, manually collected to capture challenging conditions and diverse domains. Large-scale retrieval is conducted against 100 million distractor images from YFCC100M. To avoid false negatives without extra annotation effort, we include only query objects confirmed to have emerged after 2014, i.e. the compilation date of YFCC100M. An extensive benchmarking is performed with the following observations: i) models fine-tuned on specific domains, such as landmarks or products, excel in that domain but fail on ILIAS ii) learning a linear adaptation layer using multi-domain class supervision results in performance improvements, especially for vision-language models iii) local descriptors in retrieval re-ranking are still a key ingredient, especially in the presence of severe background clutter iv) the text-to-image performance of the vision-language foundation models is surprisingly close to the corresponding image-to-image case. website: https://vrg.fel.cvut.cz/ilias/  
  </ol>  
</details>  
  
### [Range and Bird's Eye View Fused Cross-Modal Visual Place Recognition](http://arxiv.org/abs/2502.11742)  
Jianyi Peng, Fan Lu, Bin Li, Yuan Huang, Sanqing Qu, Guang Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image-to-point cloud cross-modal Visual Place Recognition (VPR) is a challenging task where the query is an RGB image, and the database samples are LiDAR point clouds. Compared to single-modal VPR, this approach benefits from the widespread availability of RGB cameras and the robustness of point clouds in providing accurate spatial geometry and distance information. However, current methods rely on intermediate modalities that capture either the vertical or horizontal field of view, limiting their ability to fully exploit the complementary information from both sensors. In this work, we propose an innovative initial retrieval + re-rank method that effectively combines information from range (or RGB) images and Bird's Eye View (BEV) images. Our approach relies solely on a computationally efficient global descriptor similarity search process to achieve re-ranking. Additionally, we introduce a novel similarity label supervision technique to maximize the utility of limited training data. Specifically, we employ points average distance to approximate appearance similarity and incorporate an adaptive margin, based on similarity differences, into the vanilla triplet loss. Experimental results on the KITTI dataset demonstrate that our method significantly outperforms state-of-the-art approaches.  
  </ol>  
</details>  
**comments**: Submmitted to IEEE IV 2025  
  
### [Adversarially Robust CLIP Models Can Induce Better (Robust) Perceptual Metrics](http://arxiv.org/abs/2502.11725)  
Francesco Croce, Christian Schlarmann, Naman Deep Singh, Matthias Hein  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Measuring perceptual similarity is a key tool in computer vision. In recent years perceptual metrics based on features extracted from neural networks with large and diverse training sets, e.g. CLIP, have become popular. At the same time, the metrics extracted from features of neural networks are not adversarially robust. In this paper we show that adversarially robust CLIP models, called R-CLIP $_\textrm{F}$ , obtained by unsupervised adversarial fine-tuning induce a better and adversarially robust perceptual metric that outperforms existing metrics in a zero-shot setting, and further matches the performance of state-of-the-art metrics while being robust after fine-tuning. Moreover, our perceptual metric achieves strong performance on related tasks such as robust image-to-image retrieval, which becomes especially relevant when applied to "Not Safe for Work" (NSFW) content detection and dataset filtering. While standard perceptual metrics can be easily attacked by a small perturbation completely degrading NSFW detection, our robust perceptual metric maintains high accuracy under an attack while having similar performance for unperturbed images. Finally, perceptual metrics induced by robust CLIP models have higher interpretability: feature inversion can show which images are considered similar, while text inversion can find what images are associated to a given prompt. This also allows us to visualize the very rich visual concepts learned by a CLIP model, including memorized persons, paintings and complex queries.  
  </ol>  
</details>  
**comments**: This work has been accepted for publication in the IEEE Conference on
  Secure and Trustworthy Machine Learning (SaTML). The final version will be
  available on IEEE Xplore  
  
### [Precise GPS-Denied UAV Self-Positioning via Context-Enhanced Cross-View Geo-Localization](http://arxiv.org/abs/2502.11408)  
Yuanze Xu, Ming Dai, Wenxiao Cai, Wankou Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image retrieval has been employed as a robust complementary technique to address the challenge of Unmanned Aerial Vehicles (UAVs) self-positioning. However, most existing methods primarily focus on localizing objects captured by UAVs through complex part-based representations, often overlooking the unique challenges associated with UAV self-positioning, such as fine-grained spatial discrimination requirements and dynamic scene variations. To address the above issues, we propose the Context-Enhanced method for precise UAV Self-Positioning (CEUSP), specifically designed for UAV self-positioning tasks. CEUSP integrates a Dynamic Sampling Strategy (DSS) to efficiently select optimal negative samples, while the Rubik's Cube Attention (RCA) module, combined with the Context-Aware Channel Integration (CACI) module, enhances feature representation and discrimination by exploiting interdimensional interactions, inspired by the rotational mechanics of a Rubik's Cube. Extensive experimental validate the effectiveness of the proposed method, demonstrating notable improvements in feature representation and UAV self-positioning accuracy within complex urban environments. Our approach achieves state-of-the-art performance on the DenseUAV dataset, which is specifically designed for dense urban contexts, and also delivers competitive results on the widely recognized University-1652 benchmark.  
  </ol>  
</details>  
**comments**: 11 pages  
  
### [E2LVLM:Evidence-Enhanced Large Vision-Language Model for Multimodal Out-of-Context Misinformation Detection](http://arxiv.org/abs/2502.10455)  
Junjie Wu, Yumeng Fu, Nan Yu, Guohong Fu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent studies in Large Vision-Language Models (LVLMs) have demonstrated impressive advancements in multimodal Out-of-Context (OOC) misinformation detection, discerning whether an authentic image is wrongly used in a claim. Despite their success, the textual evidence of authentic images retrieved from the inverse search is directly transmitted to LVLMs, leading to inaccurate or false information in the decision-making phase. To this end, we present E2LVLM, a novel evidence-enhanced large vision-language model by adapting textual evidence in two levels. First, motivated by the fact that textual evidence provided by external tools struggles to align with LVLMs inputs, we devise a reranking and rewriting strategy for generating coherent and contextually attuned content, thereby driving the aligned and effective behavior of LVLMs pertinent to authentic images. Second, to address the scarcity of news domain datasets with both judgment and explanation, we generate a novel OOC multimodal instruction-following dataset by prompting LVLMs with informative content to acquire plausible explanations. Further, we develop a multimodal instruction-tuning strategy with convincing explanations for beyond detection. This scheme contributes to E2LVLM for multimodal OOC misinformation detection and explanation. A multitude of experiments demonstrate that E2LVLM achieves superior performance than state-of-the-art methods, and also provides compelling rationales for judgments.  
  </ol>  
</details>  
  
  



## Image Matching  

### [FeaKM: Robust Collaborative Perception under Noisy Pose Conditions](http://arxiv.org/abs/2502.11003)  
Jiuwu Hao, Liguo Sun, Ti Xiang, Yuting Wan, Haolin Song, Pin Lv  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Collaborative perception is essential for networks of agents with limited sensing capabilities, enabling them to work together by exchanging information to achieve a robust and comprehensive understanding of their environment. However, localization inaccuracies often lead to significant spatial message displacement, which undermines the effectiveness of these collaborative efforts. To tackle this challenge, we introduce FeaKM, a novel method that employs Feature-level Keypoints Matching to effectively correct pose discrepancies among collaborating agents. Our approach begins by utilizing a confidence map to identify and extract salient points from intermediate feature representations, allowing for the computation of their descriptors. This step ensures that the system can focus on the most relevant information, enhancing the matching process. We then implement a target-matching strategy that generates an assignment matrix, correlating the keypoints identified by different agents. This is critical for establishing accurate correspondences, which are essential for effective collaboration. Finally, we employ a fine-grained transformation matrix to synchronize the features of all agents and ascertain their relative statuses, ensuring coherent communication among them. Our experimental results demonstrate that FeaKM significantly outperforms existing methods on the DAIR-V2X dataset, confirming its robustness even under severe noise conditions. The code and implementation details are available at https://github.com/uestchjw/FeaKM.  
  </ol>  
</details>  
**comments**: Accepted by JCRAI 2024  
  
  



## NeRF  

### [HumanGif: Single-View Human Diffusion with Generative Prior](http://arxiv.org/abs/2502.12080)  
Shoukang Hu, Takuya Narihira, Kazumi Fukuda, Ryosuke Sawata, Takashi Shibuya, Yuki Mitsufuji  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    While previous single-view-based 3D human reconstruction methods made significant progress in novel view synthesis, it remains a challenge to synthesize both view-consistent and pose-consistent results for animatable human avatars from a single image input. Motivated by the success of 2D character animation, we propose <strong>HumanGif</strong>, a single-view human diffusion model with generative prior. Specifically, we formulate the single-view-based 3D human novel view and pose synthesis as a single-view-conditioned human diffusion process, utilizing generative priors from foundational diffusion models. To ensure fine-grained and consistent novel view and pose synthesis, we introduce a Human NeRF module in HumanGif to learn spatially aligned features from the input image, implicitly capturing the relative camera and human pose transformation. Furthermore, we introduce an image-level loss during optimization to bridge the gap between latent and image spaces in diffusion models. Extensive experiments on RenderPeople and DNA-Rendering datasets demonstrate that HumanGif achieves the best perceptual performance, with better generalizability for novel view and pose synthesis.  
  </ol>  
</details>  
**comments**: Project page: https://skhu101.github.io/HumanGif/  
  
### [3D Gaussian Inpainting with Depth-Guided Cross-View Consistency](http://arxiv.org/abs/2502.11801)  
Sheng-Yu Huang, Zi-Ting Chou, Yu-Chiang Frank Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    When performing 3D inpainting using novel-view rendering methods like Neural Radiance Field (NeRF) or 3D Gaussian Splatting (3DGS), how to achieve texture and geometry consistency across camera views has been a challenge. In this paper, we propose a framework of 3D Gaussian Inpainting with Depth-Guided Cross-View Consistency (3DGIC) for cross-view consistent 3D inpainting. Guided by the rendered depth information from each training view, our 3DGIC exploits background pixels visible across different views for updating the inpainting mask, allowing us to refine the 3DGS for inpainting purposes.Through extensive experiments on benchmark datasets, we confirm that our 3DGIC outperforms current state-of-the-art 3D inpainting methods quantitatively and qualitatively.  
  </ol>  
</details>  
  
  



