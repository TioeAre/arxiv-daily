<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Best-Foot-Forward:-Robust-Foot-Reconstruction-in-the-wild>Best Foot Forward: Robust Foot Reconstruction in-the-wild</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#CoTMR:-Chain-of-Thought-Multi-Scale-Reasoning-for-Training-Free-Zero-Shot-Composed-Image-Retrieval>CoTMR: Chain-of-Thought Multi-Scale Reasoning for Training-Free Zero-Shot Composed Image Retrieval</a></li>
        <li><a href=#SciceVPR:-Stable-Cross-Image-Correlation-Enhanced-Model-for-Visual-Place-Recognition>SciceVPR: Stable Cross-Image Correlation Enhanced Model for Visual Place Recognition</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Best Foot Forward: Robust Foot Reconstruction in-the-wild](http://arxiv.org/abs/2502.20511)  
Kyle Fogarty, Jing Yang, Chayan Kumar Patodi, Aadi Bhanti, Steven Chacko, Cengiz Oztireli, Ujwal Bonde  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Accurate 3D foot reconstruction is crucial for personalized orthotics, digital healthcare, and virtual fittings. However, existing methods struggle with incomplete scans and anatomical variations, particularly in self-scanning scenarios where user mobility is limited, making it difficult to capture areas like the arch and heel. We present a novel end-to-end pipeline that refines Structure-from-Motion (SfM) reconstruction. It first resolves scan alignment ambiguities using SE(3) canonicalization with a viewpoint prediction module, then completes missing geometry through an attention-based network trained on synthetically augmented point clouds. Our approach achieves state-of-the-art performance on reconstruction metrics while preserving clinically validated anatomical fidelity. By combining synthetic training data with learned geometric priors, we enable robust foot reconstruction under real-world capture conditions, unlocking new opportunities for mobile-based 3D scanning in healthcare and retail.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [CoTMR: Chain-of-Thought Multi-Scale Reasoning for Training-Free Zero-Shot Composed Image Retrieval](http://arxiv.org/abs/2502.20826)  
Zelong Sun, Dong Jing, Zhiwu Lu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Zero-Shot Composed Image Retrieval (ZS-CIR) aims to retrieve target images by integrating information from a composed query (reference image and modification text) without training samples. Existing methods primarily combine caption models and large language models (LLMs) to generate target captions based on composed queries but face various issues such as incompatibility, visual information loss, and insufficient reasoning. In this work, we propose CoTMR, a training-free framework crafted for ZS-CIR with novel Chain-of-thought (CoT) and Multi-scale Reasoning. Instead of relying on caption models for modality transformation, CoTMR employs the Large Vision-Language Model (LVLM) to achieve unified understanding and reasoning for composed queries. To enhance the reasoning reliability, we devise CIRCoT, which guides the LVLM through a step-by-step inference process using predefined subtasks. Considering that existing approaches focus solely on global-level reasoning, our CoTMR incorporates multi-scale reasoning to achieve more comprehensive inference via fine-grained predictions about the presence or absence of key elements at the object scale. Further, we design a Multi-Grained Scoring (MGS) mechanism, which integrates CLIP similarity scores of the above reasoning outputs with candidate images to realize precise retrieval. Extensive experiments demonstrate that our CoTMR not only drastically outperforms previous methods across four prominent benchmarks but also offers appealing interpretability.  
  </ol>  
</details>  
  
### [SciceVPR: Stable Cross-Image Correlation Enhanced Model for Visual Place Recognition](http://arxiv.org/abs/2502.20676)  
Shanshan Wan, Yingmei Wei, Lai Kang, Tianrui Shen, Haixuan Wang, Yee-Hong Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Place Recognition (VPR) is a major challenge for robotics and autonomous systems, with the goal of predicting the location of an image based solely on its visual features. State-of-the-art (SOTA) models extract global descriptors using the powerful foundation model DINOv2 as backbone. These models either explore the cross-image correlation or propose a time-consuming two-stage re-ranking strategy to achieve better performance. However, existing works only utilize the final output of DINOv2, and the current cross-image correlation causes unstable retrieval results. To produce both discriminative and constant global descriptors, this paper proposes stable cross-image correlation enhanced model for VPR called SciceVPR. This model explores the full potential of DINOv2 in providing useful feature representations that implicitly encode valuable contextual knowledge. Specifically, SciceVPR first uses a multi-layer feature fusion module to capture increasingly detailed task-relevant channel and spatial information from the multi-layer output of DINOv2. Secondly, SciceVPR considers the invariant correlation between images within a batch as valuable knowledge to be distilled into the proposed self-enhanced encoder. In this way, SciceVPR can acquire fairly robust global features regardless of domain shifts (e.g., changes in illumination, weather and viewpoint between pictures taken in the same place). Experimental results demonstrate that the base variant, SciceVPR-B, outperforms SOTA one-stage methods with single input on multiple datasets with varying domain conditions. The large variant, SciceVPR-L, performs on par with SOTA two-stage models, scoring over 3% higher in Recall@1 compared to existing models on the challenging Tokyo24/7 dataset. Our code will be released at https://github.com/shuimushan/SciceVPR.  
  </ol>  
</details>  
  
  



