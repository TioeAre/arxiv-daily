<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#PathAlign:-A-vision-language-model-for-whole-slide-images-in-histopathology>PathAlign: A vision-language model for whole slide images in histopathology</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Beyond-First-Order:-A-Multi-Scale-Approach-to-Finger-Knuckle-Print-Biometrics>Beyond First-Order: A Multi-Scale Approach to Finger Knuckle Print Biometrics</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#ASSR-NeRF:-Arbitrary-Scale-Super-Resolution-on-Voxel-Grid-for-High-Quality-Radiance-Fields-Reconstruction>ASSR-NeRF: Arbitrary-Scale Super-Resolution on Voxel Grid for High-Quality Radiance Fields Reconstruction</a></li>
        <li><a href=#EgoGaussian:-Dynamic-Scene-Understanding-from-Egocentric-Video-with-3D-Gaussian-Splatting>EgoGaussian: Dynamic Scene Understanding from Egocentric Video with 3D Gaussian Splatting</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [PathAlign: A vision-language model for whole slide images in histopathology](http://arxiv.org/abs/2406.19578)  
Faruk Ahmed, Andrew Sellergren, Lin Yang, Shawn Xu, Boris Babenko, Abbi Ward, Niels Olson, Arash Mohtashamian, Yossi Matias, Greg S. Corrado, Quang Duong, Dale R. Webster, Shravya Shetty, Daniel Golden, Yun Liu, David F. Steiner, Ellery Wulczyn  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Microscopic interpretation of histopathology images underlies many important diagnostic and treatment decisions. While advances in vision-language modeling raise new opportunities for analysis of such images, the gigapixel-scale size of whole slide images (WSIs) introduces unique challenges. Additionally, pathology reports simultaneously highlight key findings from small regions while also aggregating interpretation across multiple slides, often making it difficult to create robust image-text pairs. As such, pathology reports remain a largely untapped source of supervision in computational pathology, with most efforts relying on region-of-interest annotations or self-supervision at the patch-level. In this work, we develop a vision-language model based on the BLIP-2 framework using WSIs paired with curated text from pathology reports. This enables applications utilizing a shared image-text embedding space, such as text or image retrieval for finding cases of interest, as well as integration of the WSI encoder with a frozen large language model (LLM) for WSI-based generative text capabilities such as report generation or AI-in-the-loop interactions. We utilize a de-identified dataset of over 350,000 WSIs and diagnostic text pairs, spanning a wide range of diagnoses, procedure types, and tissue types. We present pathologist evaluation of text generation and text retrieval using WSI embeddings, as well as results for WSI classification and workflow prioritization (slide-level triaging). Model-generated text for WSIs was rated by pathologists as accurate, without clinically significant error or omission, for 78% of WSIs on average. This work demonstrates exciting potential capabilities for language-aligned WSI embeddings.  
  </ol>  
</details>  
**comments**: 9 main pages and 19 pages of supplemental material; 3 main tables, 3
  main figures and 11 supplemental tables, 7 supplemental figures  
  
  



## Keypoint Detection  

### [Beyond First-Order: A Multi-Scale Approach to Finger Knuckle Print Biometrics](http://arxiv.org/abs/2406.19672)  
Chengrui Gao, Ziyuan Yang, Andrew Beng Jin Teoh, Min Zhu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recently, finger knuckle prints (FKPs) have gained attention due to their rich textural patterns, positioning them as a promising biometric for identity recognition. Prior FKP recognition methods predominantly leverage first-order feature descriptors, which capture intricate texture details but fail to account for structural information. Emerging research, however, indicates that second-order textures, which describe the curves and arcs of the textures, encompass this overlooked structural information. This paper introduces a novel FKP recognition approach, the Dual-Order Texture Competition Network (DOTCNet), designed to capture texture information in FKP images comprehensively. DOTCNet incorporates three dual-order texture competitive modules (DTCMs), each targeting textures at different scales. Each DTCM employs a learnable texture descriptor, specifically a learnable Gabor filter (LGF), to extract texture features. By leveraging LGFs, the network extracts first and second order textures to describe fine textures and structural features thoroughly. Furthermore, an attention mechanism enhances relevant features in the first-order features, thereby highlighting significant texture details. For second-order features, a competitive mechanism emphasizes structural information while reducing noise from higher-order features. Extensive experimental results reveal that DOTCNet significantly outperforms several standard algorithms on the publicly available PolyU-FKP dataset.  
  </ol>  
</details>  
  
  



## NeRF  

### [ASSR-NeRF: Arbitrary-Scale Super-Resolution on Voxel Grid for High-Quality Radiance Fields Reconstruction](http://arxiv.org/abs/2406.20066)  
Ding-Jiun Huang, Zi-Ting Chou, Yu-Chiang Frank Wang, Cheng Sun  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    NeRF-based methods reconstruct 3D scenes by building a radiance field with implicit or explicit representations. While NeRF-based methods can perform novel view synthesis (NVS) at arbitrary scale, the performance in high-resolution novel view synthesis (HRNVS) with low-resolution (LR) optimization often results in oversmoothing. On the other hand, single-image super-resolution (SR) aims to enhance LR images to HR counterparts but lacks multi-view consistency. To address these challenges, we propose Arbitrary-Scale Super-Resolution NeRF (ASSR-NeRF), a novel framework for super-resolution novel view synthesis (SRNVS). We propose an attention-based VoxelGridSR model to directly perform 3D super-resolution (SR) on the optimized volume. Our model is trained on diverse scenes to ensure generalizability. For unseen scenes trained with LR views, we then can directly apply our VoxelGridSR to further refine the volume and achieve multi-view consistent SR. We demonstrate quantitative and qualitatively that the proposed method achieves significant performance in SRNVS.  
  </ol>  
</details>  
  
### [EgoGaussian: Dynamic Scene Understanding from Egocentric Video with 3D Gaussian Splatting](http://arxiv.org/abs/2406.19811)  
Daiwei Zhang, Gengyan Li, Jiajie Li, Mickaël Bressieux, Otmar Hilliges, Marc Pollefeys, Luc Van Gool, Xi Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Human activities are inherently complex, and even simple household tasks involve numerous object interactions. To better understand these activities and behaviors, it is crucial to model their dynamic interactions with the environment. The recent availability of affordable head-mounted cameras and egocentric data offers a more accessible and efficient means to understand dynamic human-object interactions in 3D environments. However, most existing methods for human activity modeling either focus on reconstructing 3D models of hand-object or human-scene interactions or on mapping 3D scenes, neglecting dynamic interactions with objects. The few existing solutions often require inputs from multiple sources, including multi-camera setups, depth-sensing cameras, or kinesthetic sensors. To this end, we introduce EgoGaussian, the first method capable of simultaneously reconstructing 3D scenes and dynamically tracking 3D object motion from RGB egocentric input alone. We leverage the uniquely discrete nature of Gaussian Splatting and segment dynamic interactions from the background. Our approach employs a clip-level online learning pipeline that leverages the dynamic nature of human activities, allowing us to reconstruct the temporal evolution of the scene in chronological order and track rigid object motion. Additionally, our method automatically segments object and background Gaussians, providing 3D representations for both static scenes and dynamic objects. EgoGaussian outperforms previous NeRF and Dynamic Gaussian methods in challenging in-the-wild videos and we also qualitatively demonstrate the high quality of the reconstructed models.  
  </ol>  
</details>  
  
  



