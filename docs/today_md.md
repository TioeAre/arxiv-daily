<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Evaluating-Attribute-Confusion-in-Fashion-Text-to-Image-Generation>Evaluating Attribute Confusion in Fashion Text-to-Image Generation</a></li>
        <li><a href=#MS-DPPs:-Multi-Source-Determinantal-Point-Processes-for-Contextual-Diversity-Refinement-of-Composite-Attributes-in-Text-to-Image-Retrieval>MS-DPPs: Multi-Source Determinantal Point Processes for Contextual Diversity Refinement of Composite Attributes in Text to Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Reading-a-Ruler-in-the-Wild>Reading a Ruler in the Wild</a></li>
        <li><a href=#MK-Pose:-Category-Level-Object-Pose-Estimation-via-Multimodal-Based-Keypoint-Learning>MK-Pose: Category-Level Object Pose Estimation via Multimodal-Based Keypoint Learning</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Dual-Granularity-Cross-Modal-Identity-Association-for-Weakly-Supervised-Text-to-Person-Image-Matching>Dual-Granularity Cross-Modal Identity Association for Weakly-Supervised Text-to-Person Image Matching</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#A-Probabilistic-Approach-to-Uncertainty-Quantification-Leveraging-3D-Geometry>A Probabilistic Approach to Uncertainty Quantification Leveraging 3D Geometry</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Evaluating Attribute Confusion in Fashion Text-to-Image Generation](http://arxiv.org/abs/2507.07079)  
Ziyue Liu, Federico Girella, Yiming Wang, Davide Talon  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Despite the rapid advances in Text-to-Image (T2I) generation models, their evaluation remains challenging in domains like fashion, involving complex compositional generation. Recent automated T2I evaluation methods leverage pre-trained vision-language models to measure cross-modal alignment. However, our preliminary study reveals that they are still limited in assessing rich entity-attribute semantics, facing challenges in attribute confusion, i.e., when attributes are correctly depicted but associated to the wrong entities. To address this, we build on a Visual Question Answering (VQA) localization strategy targeting one single entity at a time across both visual and textual modalities. We propose a localized human evaluation protocol and introduce a novel automatic metric, Localized VQAScore (L-VQAScore), that combines visual localization with VQA probing both correct (reflection) and miss-localized (leakage) attribute generation. On a newly curated dataset featuring challenging compositional alignment scenarios, L-VQAScore outperforms state-of-the-art T2I evaluation methods in terms of correlation with human judgments, demonstrating its strength in capturing fine-grained entity-attribute associations. We believe L-VQAScore can be a reliable and scalable alternative to subjective evaluations.  
  </ol>  
</details>  
**comments**: Accepted to ICIAP25. Project page: site
  [https://intelligolabs.github.io/L-VQAScore/\  
  
### [MS-DPPs: Multi-Source Determinantal Point Processes for Contextual Diversity Refinement of Composite Attributes in Text to Image Retrieval](http://arxiv.org/abs/2507.06654)  
Naoya Sogi, Takashi Shibata, Makoto Terao, Masanori Suganuma, Takayuki Okatani  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Result diversification (RD) is a crucial technique in Text-to-Image Retrieval for enhancing the efficiency of a practical application. Conventional methods focus solely on increasing the diversity metric of image appearances. However, the diversity metric and its desired value vary depending on the application, which limits the applications of RD. This paper proposes a novel task called CDR-CA (Contextual Diversity Refinement of Composite Attributes). CDR-CA aims to refine the diversities of multiple attributes, according to the application's context. To address this task, we propose Multi-Source DPPs, a simple yet strong baseline that extends the Determinantal Point Process (DPP) to multi-sources. We model MS-DPP as a single DPP model with a unified similarity matrix based on a manifold representation. We also introduce Tangent Normalization to reflect contexts. Extensive experiments demonstrate the effectiveness of the proposed method. Our code is publicly available at https://github.com/NEC-N-SOGI/msdpp.  
  </ol>  
</details>  
**comments**: IJCAI 2025. Code: https://github.com/NEC-N-SOGI/msdpp  
  
  



## Keypoint Detection  

### [Reading a Ruler in the Wild](http://arxiv.org/abs/2507.07077)  
Yimu Pan, Manas Mehta, Gwen Sincerbeaux, Jeffery A. Goldstein, Alison D. Gernand, James Z. Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Accurately converting pixel measurements into absolute real-world dimensions remains a fundamental challenge in computer vision and limits progress in key applications such as biomedicine, forensics, nutritional analysis, and e-commerce. We introduce RulerNet, a deep learning framework that robustly infers scale "in the wild" by reformulating ruler reading as a unified keypoint-detection problem and by representing the ruler with geometric-progression parameters that are invariant to perspective transformations. Unlike traditional methods that rely on handcrafted thresholds or rigid, ruler-specific pipelines, RulerNet directly localizes centimeter marks using a distortion-invariant annotation and training strategy, enabling strong generalization across diverse ruler types and imaging conditions while mitigating data scarcity. We also present a scalable synthetic-data pipeline that combines graphics-based ruler generation with ControlNet to add photorealistic context, greatly increasing training diversity and improving performance. To further enhance robustness and efficiency, we propose DeepGP, a lightweight feed-forward network that regresses geometric-progression parameters from noisy marks and eliminates iterative optimization, enabling real-time scale estimation on mobile or edge devices. Experiments show that RulerNet delivers accurate, consistent, and efficient scale estimates under challenging real-world conditions. These results underscore its utility as a generalizable measurement tool and its potential for integration with other vision components for automated, scale-aware analysis in high-impact domains. A live demo is available at https://huggingface.co/spaces/ymp5078/RulerNet-Demo.  
  </ol>  
</details>  
  
### [MK-Pose: Category-Level Object Pose Estimation via Multimodal-Based Keypoint Learning](http://arxiv.org/abs/2507.06662)  
Yifan Yang, Peili Song, Enfan Lan, Dong Liu, Jingtai Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Category-level object pose estimation, which predicts the pose of objects within a known category without prior knowledge of individual instances, is essential in applications like warehouse automation and manufacturing. Existing methods relying on RGB images or point cloud data often struggle with object occlusion and generalization across different instances and categories. This paper proposes a multimodal-based keypoint learning framework (MK-Pose) that integrates RGB images, point clouds, and category-level textual descriptions. The model uses a self-supervised keypoint detection module enhanced with attention-based query generation, soft heatmap matching and graph-based relational modeling. Additionally, a graph-enhanced feature fusion module is designed to integrate local geometric information and global context. MK-Pose is evaluated on CAMERA25 and REAL275 dataset, and is further tested for cross-dataset capability on HouseCat6D dataset. The results demonstrate that MK-Pose outperforms existing state-of-the-art methods in both IoU and average precision without shape priors. Codes will be released at \href{https://github.com/yangyifanYYF/MK-Pose}{https://github.com/yangyifanYYF/MK-Pose}.  
  </ol>  
</details>  
  
  



## Image Matching  

### [Dual-Granularity Cross-Modal Identity Association for Weakly-Supervised Text-to-Person Image Matching](http://arxiv.org/abs/2507.06744)  
Yafei Zhang, Yongle Shang, Huafeng Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Weakly supervised text-to-person image matching, as a crucial approach to reducing models' reliance on large-scale manually labeled samples, holds significant research value. However, existing methods struggle to predict complex one-to-many identity relationships, severely limiting performance improvements. To address this challenge, we propose a local-and-global dual-granularity identity association mechanism. Specifically, at the local level, we explicitly establish cross-modal identity relationships within a batch, reinforcing identity constraints across different modalities and enabling the model to better capture subtle differences and correlations. At the global level, we construct a dynamic cross-modal identity association network with the visual modality as the anchor and introduce a confidence-based dynamic adjustment mechanism, effectively enhancing the model's ability to identify weakly associated samples while improving overall sensitivity. Additionally, we propose an information-asymmetric sample pair construction method combined with consistency learning to tackle hard sample mining and enhance model robustness. Experimental results demonstrate that the proposed method substantially boosts cross-modal matching accuracy, providing an efficient and practical solution for text-to-person image matching.  
  </ol>  
</details>  
  
  



## NeRF  

### [A Probabilistic Approach to Uncertainty Quantification Leveraging 3D Geometry](http://arxiv.org/abs/2507.06269)  
Rushil Desai, Frederik Warburg, Trevor Darrell, Marissa Ramirez de Chanlatte  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Quantifying uncertainty in neural implicit 3D representations, particularly those utilizing Signed Distance Functions (SDFs), remains a substantial challenge due to computational inefficiencies, scalability issues, and geometric inconsistencies. Existing methods typically neglect direct geometric integration, leading to poorly calibrated uncertainty maps. We introduce BayesSDF, a novel probabilistic framework for uncertainty quantification in neural implicit SDF models, motivated by scientific simulation applications with 3D environments (e.g., forests) such as modeling fluid flow through forests, where precise surface geometry and awareness of fidelity surface geometric uncertainty are essential. Unlike radiance-based models such as NeRF or 3D Gaussian splatting, which lack explicit surface formulations, SDFs define continuous and differentiable geometry, making them better suited for physical modeling and analysis. BayesSDF leverages a Laplace approximation to quantify local surface instability via Hessian-based metrics, enabling computationally efficient, surface-aware uncertainty estimation. Our method shows that uncertainty predictions correspond closely with poorly reconstructed geometry, providing actionable confidence measures for downstream use. Extensive evaluations on synthetic and real-world datasets demonstrate that BayesSDF outperforms existing methods in both calibration and geometric consistency, establishing a strong foundation for uncertainty-aware 3D scene reconstruction, simulation, and robotic decision-making.  
  </ol>  
</details>  
**comments**: ICCV 2025 Workshops (8 Pages, 6 Figures, 2 Tables)  
  
  



