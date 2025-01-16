<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#BRIGHT-VO:-Brightness-Guided-Hybrid-Transformer-for-Visual-Odometry-with-Multi-modality-Refinement-Module>BRIGHT-VO: Brightness-Guided Hybrid Transformer for Visual Odometry with Multi-modality Refinement Module</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Vision-Foundation-Models-for-Computed-Tomography>Vision Foundation Models for Computed Tomography</a></li>
        <li><a href=#SCOT:-Self-Supervised-Contrastive-Pretraining-For-Zero-Shot-Compositional-Retrieval>SCOT: Self-Supervised Contrastive Pretraining For Zero-Shot Compositional Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#SLC$^2$-SLAM:-Semantic-guided-Loop-Closure-with-Shared-Latent-Code-for-NeRF-SLAM>SLC$^2$-SLAM: Semantic-guided Loop Closure with Shared Latent Code for NeRF SLAM</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [BRIGHT-VO: Brightness-Guided Hybrid Transformer for Visual Odometry with Multi-modality Refinement Module](http://arxiv.org/abs/2501.08659)  
Dongzhihan Wang, Yang Yang, Liang Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual odometry (VO) plays a crucial role in autonomous driving, robotic navigation, and other related tasks by estimating the position and orientation of a camera based on visual input. Significant progress has been made in data-driven VO methods, particularly those leveraging deep learning techniques to extract image features and estimate camera poses. However, these methods often struggle in low-light conditions because of the reduced visibility of features and the increased difficulty of matching keypoints. To address this limitation, we introduce BrightVO, a novel VO model based on Transformer architecture, which not only performs front-end visual feature extraction, but also incorporates a multi-modality refinement module in the back-end that integrates Inertial Measurement Unit (IMU) data. Using pose graph optimization, this module iteratively refines pose estimates to reduce errors and improve both accuracy and robustness. Furthermore, we create a synthetic low-light dataset, KiC4R, which includes a variety of lighting conditions to facilitate the training and evaluation of VO frameworks in challenging environments. Experimental results demonstrate that BrightVO achieves state-of-the-art performance on both the KiC4R dataset and the KITTI benchmarks. Specifically, it provides an average improvement of 20% in pose estimation accuracy in normal outdoor environments and 259% in low-light conditions, outperforming existing methods. For widespread use and further development, the research work is fully open-source at https://github.com/Anastasiawd/BrightVO.  
  </ol>  
</details>  
**comments**: 9 pages, 7 figures  
  
  



## Visual Localization  

### [Vision Foundation Models for Computed Tomography](http://arxiv.org/abs/2501.09001)  
Suraj Pai, Ibrahim Hadzic, Dennis Bontempi, Keno Bressem, Benjamin H. Kann, Andriy Fedorov, Raymond H. Mak, Hugo J. W. L. Aerts  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Foundation models (FMs) have shown transformative potential in radiology by performing diverse, complex tasks across imaging modalities. Here, we developed CT-FM, a large-scale 3D image-based pre-trained model designed explicitly for various radiological tasks. CT-FM was pre-trained using 148,000 computed tomography (CT) scans from the Imaging Data Commons through label-agnostic contrastive learning. We evaluated CT-FM across four categories of tasks, namely, whole-body and tumor segmentation, head CT triage, medical image retrieval, and semantic understanding, showing superior performance against state-of-the-art models. Beyond quantitative success, CT-FM demonstrated the ability to cluster regions anatomically and identify similar anatomical and structural concepts across scans. Furthermore, it remained robust across test-retest settings and indicated reasonable salient regions attached to its embeddings. This study demonstrates the value of large-scale medical imaging foundation models and by open-sourcing the model weights, code, and data, aims to support more adaptable, reliable, and interpretable AI solutions in radiology.  
  </ol>  
</details>  
**comments**: 6 figures, followed by 9 Extended Data Figures and a Supplementary
  Information document  
  
### [SCOT: Self-Supervised Contrastive Pretraining For Zero-Shot Compositional Retrieval](http://arxiv.org/abs/2501.08347)  
Bhavin Jawade, Joao V. B. Soares, Kapil Thadani, Deen Dayal Mohan, Amir Erfan Eshratifar, Benjamin Culpepper, Paloma de Juan, Srirangaraj Setlur, Venu Govindaraju  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Compositional image retrieval (CIR) is a multimodal learning task where a model combines a query image with a user-provided text modification to retrieve a target image. CIR finds applications in a variety of domains including product retrieval (e-commerce) and web search. Existing methods primarily focus on fully-supervised learning, wherein models are trained on datasets of labeled triplets such as FashionIQ and CIRR. This poses two significant challenges: (i) curating such triplet datasets is labor intensive; and (ii) models lack generalization to unseen objects and domains. In this work, we propose SCOT (Self-supervised COmpositional Training), a novel zero-shot compositional pretraining strategy that combines existing large image-text pair datasets with the generative capabilities of large language models to contrastively train an embedding composition network. Specifically, we show that the text embedding from a large-scale contrastively-pretrained vision-language model can be utilized as proxy target supervision during compositional pretraining, replacing the target image embedding. In zero-shot settings, this strategy surpasses SOTA zero-shot compositional retrieval methods as well as many fully-supervised methods on standard benchmarks such as FashionIQ and CIRR.  
  </ol>  
</details>  
**comments**: Paper accepted at WACV 2025 in round 1  
  
  



## NeRF  

### [SLC $^2$ -SLAM: Semantic-guided Loop Closure with Shared Latent Code for NeRF SLAM](http://arxiv.org/abs/2501.08880)  
Yuhang Ming, Di Ma, Weichen Dai, Han Yang, Rui Fan, Guofeng Zhang, Wanzeng Kong  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Targeting the notorious cumulative drift errors in NeRF SLAM, we propose a Semantic-guided Loop Closure with Shared Latent Code, dubbed SLC$^2$-SLAM. Especially, we argue that latent codes stored in many NeRF SLAM systems are not fully exploited, as they are only used for better reconstruction. In this paper, we propose a simple yet effective way to detect potential loops using the same latent codes as local features. To further improve the loop detection performance, we use the semantic information, which are also decoded from the same latent codes to guide the aggregation of local features. Finally, with the potential loops detected, we close them with a graph optimization followed by bundle adjustment to refine both the estimated poses and the reconstructed scene. To evaluate the performance of our SLC$^2$-SLAM, we conduct extensive experiments on Replica and ScanNet datasets. Our proposed semantic-guided loop closure significantly outperforms the pre-trained NetVLAD and ORB combined with Bag-of-Words, which are used in all the other NeRF SLAM with loop closure. As a result, our SLC$^2$-SLAM also demonstrated better tracking and reconstruction performance, especially in larger scenes with more loops, like ScanNet.  
  </ol>  
</details>  
**comments**: 8 pages, 5 figures, 4 tables  
  
  



