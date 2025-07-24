<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#VLM-Guided-Visual-Place-Recognition-for-Planet-Scale-Geo-Localization>VLM-Guided Visual Place Recognition for Planet-Scale Geo-Localization</a></li>
        <li><a href=#Content-based-3D-Image-Retrieval-and-a-ColBERT-inspired-Re-ranking-for-Tumor-Flagging-and-Staging>Content-based 3D Image Retrieval and a ColBERT-inspired Re-ranking for Tumor Flagging and Staging</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#CartoonAlive:-Towards-Expressive-Live2D-Modeling-from-Single-Portraits>CartoonAlive: Towards Expressive Live2D Modeling from Single Portraits</a></li>
        <li><a href=#Toward-a-Real-Time-Framework-for-Accurate-Monocular-3D-Human-Pose-Estimation-with-Geometric-Priors>Toward a Real-Time Framework for Accurate Monocular 3D Human Pose Estimation with Geometric Priors</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Exploring-Active-Learning-for-Label-Efficient-Training-of-Semantic-Neural-Radiance-Field>Exploring Active Learning for Label-Efficient Training of Semantic Neural Radiance Field</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [VLM-Guided Visual Place Recognition for Planet-Scale Geo-Localization](http://arxiv.org/abs/2507.17455)  
Sania Waheed, Na Min An, Michael Milford, Sarvapali D. Ramchurn, Shoaib Ehsan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Geo-localization from a single image at planet scale (essentially an advanced or extreme version of the kidnapped robot problem) is a fundamental and challenging task in applications such as navigation, autonomous driving and disaster response due to the vast diversity of locations, environmental conditions, and scene variations. Traditional retrieval-based methods for geo-localization struggle with scalability and perceptual aliasing, while classification-based approaches lack generalization and require extensive training data. Recent advances in vision-language models (VLMs) offer a promising alternative by leveraging contextual understanding and reasoning. However, while VLMs achieve high accuracy, they are often prone to hallucinations and lack interpretability, making them unreliable as standalone solutions. In this work, we propose a novel hybrid geo-localization framework that combines the strengths of VLMs with retrieval-based visual place recognition (VPR) methods. Our approach first leverages a VLM to generate a prior, effectively guiding and constraining the retrieval search space. We then employ a retrieval step, followed by a re-ranking mechanism that selects the most geographically plausible matches based on feature similarity and proximity to the initially estimated coordinates. We evaluate our approach on multiple geo-localization benchmarks and show that it consistently outperforms prior state-of-the-art methods, particularly at street (up to 4.51%) and city level (up to 13.52%). Our results demonstrate that VLM-generated geographic priors in combination with VPR lead to scalable, robust, and accurate geo-localization systems.  
  </ol>  
</details>  
  
### [Content-based 3D Image Retrieval and a ColBERT-inspired Re-ranking for Tumor Flagging and Staging](http://arxiv.org/abs/2507.17412)  
Farnaz Khun Jush, Steffen Vogler, Matthias Lenga  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The increasing volume of medical images poses challenges for radiologists in retrieving relevant cases. Content-based image retrieval (CBIR) systems offer potential for efficient access to similar cases, yet lack standardized evaluation and comprehensive studies. Building on prior studies for tumor characterization via CBIR, this study advances CBIR research for volumetric medical images through three key contributions: (1) a framework eliminating reliance on pre-segmented data and organ-specific datasets, aligning with large and unstructured image archiving systems, i.e. PACS in clinical practice; (2) introduction of C-MIR, a novel volumetric re-ranking method adapting ColBERT's contextualized late interaction mechanism for 3D medical imaging; (3) comprehensive evaluation across four tumor sites using three feature extractors and three database configurations. Our evaluations highlight the significant advantages of C-MIR. We demonstrate the successful adaptation of the late interaction principle to volumetric medical images, enabling effective context-aware re-ranking. A key finding is C-MIR's ability to effectively localize the region of interest, eliminating the need for pre-segmentation of datasets and offering a computationally efficient alternative to systems relying on expensive data enrichment steps. C-MIR demonstrates promising improvements in tumor flagging, achieving improved performance, particularly for colon and lung tumors (p<0.05). C-MIR also shows potential for improving tumor staging, warranting further exploration of its capabilities. Ultimately, our work seeks to bridge the gap between advanced retrieval techniques and their practical applications in healthcare, paving the way for improved diagnostic processes.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [CartoonAlive: Towards Expressive Live2D Modeling from Single Portraits](http://arxiv.org/abs/2507.17327)  
Chao He, Jianqiang Ren, Jianjing Xiang, Xiejie Shen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    With the rapid advancement of large foundation models, AIGC, cloud rendering, and real-time motion capture technologies, digital humans are now capable of achieving synchronized facial expressions and body movements, engaging in intelligent dialogues driven by natural language, and enabling the fast creation of personalized avatars. While current mainstream approaches to digital humans primarily focus on 3D models and 2D video-based representations, interactive 2D cartoon-style digital humans have received relatively less attention. Compared to 3D digital humans that require complex modeling and high rendering costs, and 2D video-based solutions that lack flexibility and real-time interactivity, 2D cartoon-style Live2D models offer a more efficient and expressive alternative. By simulating 3D-like motion through layered segmentation without the need for traditional 3D modeling, Live2D enables dynamic and real-time manipulation. In this technical report, we present CartoonAlive, an innovative method for generating high-quality Live2D digital humans from a single input portrait image. CartoonAlive leverages the shape basis concept commonly used in 3D face modeling to construct facial blendshapes suitable for Live2D. It then infers the corresponding blendshape weights based on facial keypoints detected from the input image. This approach allows for the rapid generation of a highly expressive and visually accurate Live2D model that closely resembles the input portrait, within less than half a minute. Our work provides a practical and scalable solution for creating interactive 2D cartoon characters, opening new possibilities in digital content creation and virtual character animation. The project homepage is https://human3daigc.github.io/CartoonAlive_webpage/.  
  </ol>  
</details>  
  
### [Toward a Real-Time Framework for Accurate Monocular 3D Human Pose Estimation with Geometric Priors](http://arxiv.org/abs/2507.16850)  
Mohamed Adjel  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Monocular 3D human pose estimation remains a challenging and ill-posed problem, particularly in real-time settings and unconstrained environments. While direct imageto-3D approaches require large annotated datasets and heavy models, 2D-to-3D lifting offers a more lightweight and flexible alternative-especially when enhanced with prior knowledge. In this work, we propose a framework that combines real-time 2D keypoint detection with geometry-aware 2D-to-3D lifting, explicitly leveraging known camera intrinsics and subject-specific anatomical priors. Our approach builds on recent advances in self-calibration and biomechanically-constrained inverse kinematics to generate large-scale, plausible 2D-3D training pairs from MoCap and synthetic datasets. We discuss how these ingredients can enable fast, personalized, and accurate 3D pose estimation from monocular images without requiring specialized hardware. This proposal aims to foster discussion on bridging data-driven learning and model-based priors to improve accuracy, interpretability, and deployability of 3D human motion capture on edge devices in the wild.  
  </ol>  
</details>  
**comments**: IEEE ICRA 2025 (workshop: Enhancing Human Mobility: From Computer
  Vision-Based Motion Tracking to Wearable Assistive Robot Control), May 2025,
  Atlanta (Georgia), United States  
  
  



## NeRF  

### [Exploring Active Learning for Label-Efficient Training of Semantic Neural Radiance Field](http://arxiv.org/abs/2507.17351)  
Yuzhe Zhu, Lile Cai, Kangkang Lu, Fayao Liu, Xulei Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Field (NeRF) models are implicit neural scene representation methods that offer unprecedented capabilities in novel view synthesis. Semantically-aware NeRFs not only capture the shape and radiance of a scene, but also encode semantic information of the scene. The training of semantically-aware NeRFs typically requires pixel-level class labels, which can be prohibitively expensive to collect. In this work, we explore active learning as a potential solution to alleviate the annotation burden. We investigate various design choices for active learning of semantically-aware NeRF, including selection granularity and selection strategies. We further propose a novel active learning strategy that takes into account 3D geometric constraints in sample selection. Our experiments demonstrate that active learning can effectively reduce the annotation cost of training semantically-aware NeRF, achieving more than 2X reduction in annotation cost compared to random sampling.  
  </ol>  
</details>  
**comments**: Accepted to ICME 2025  
  
  



