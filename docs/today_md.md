<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Wild-refitting-for-black-box-prediction>Wild refitting for black box prediction</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#OracleFusion:-Assisting-the-Decipherment-of-Oracle-Bone-Script-with-Structurally-Constrained-Semantic-Typography>OracleFusion: Assisting the Decipherment of Oracle Bone Script with Structurally Constrained Semantic Typography</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#PanSt3R:-Multi-view-Consistent-Panoptic-Segmentation>PanSt3R: Multi-view Consistent Panoptic Segmentation</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Wild refitting for black box prediction](http://arxiv.org/abs/2506.21460)  
Martin J. Wainwright  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We describe and analyze a computionally efficient refitting procedure for computing high-probability upper bounds on the instance-wise mean-squared prediction error of penalized nonparametric estimates based on least-squares minimization. Requiring only a single dataset and black box access to the prediction method, it consists of three steps: computing suitable residuals, symmetrizing and scaling them with a pre-factor $\rho$, and using them to define and solve a modified prediction problem recentered at the current estimate. We refer to it as wild refitting, since it uses Rademacher residual symmetrization as in a wild bootstrap variant. Under relatively mild conditions allowing for noise heterogeneity, we establish a high probability guarantee on its performance, showing that the wild refit with a suitably chosen wild noise scale $\rho$ gives an upper bound on prediction error. This theoretical analysis provides guidance into the design of such procedures, including how the residuals should be formed, the amount of noise rescaling in the wild sub-problem needed for upper bounds, and the local stability properties of the block-box procedure. We illustrate the applicability of this procedure to various problems, including non-rigid structure-from-motion recovery with structured matrix penalties; plug-and-play image restoration with deep neural network priors; and randomized sketching with kernel methods.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [OracleFusion: Assisting the Decipherment of Oracle Bone Script with Structurally Constrained Semantic Typography](http://arxiv.org/abs/2506.21101)  
Caoshuo Li, Zengmao Ding, Xiaobin Hu, Bang Li, Donghao Luo, AndyPian Wu, Chaoyang Wang, Chengjie Wang, Taisong Jin, SevenShu, Yunsheng Wu, Yongge Liu, Rongrong Ji  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    As one of the earliest ancient languages, Oracle Bone Script (OBS) encapsulates the cultural records and intellectual expressions of ancient civilizations. Despite the discovery of approximately 4,500 OBS characters, only about 1,600 have been deciphered. The remaining undeciphered ones, with their complex structure and abstract imagery, pose significant challenges for interpretation. To address these challenges, this paper proposes a novel two-stage semantic typography framework, named OracleFusion. In the first stage, this approach leverages the Multimodal Large Language Model (MLLM) with enhanced Spatial Awareness Reasoning (SAR) to analyze the glyph structure of the OBS character and perform visual localization of key components. In the second stage, we introduce Oracle Structural Vector Fusion (OSVF), incorporating glyph structure constraints and glyph maintenance constraints to ensure the accurate generation of semantically enriched vector fonts. This approach preserves the objective integrity of the glyph structure, offering visually enhanced representations that assist experts in deciphering OBS. Extensive qualitative and quantitative experiments demonstrate that OracleFusion outperforms state-of-the-art baseline models in terms of semantics, visual appeal, and glyph maintenance, significantly enhancing both readability and aesthetic quality. Furthermore, OracleFusion provides expert-like insights on unseen oracle characters, making it a valuable tool for advancing the decipherment of OBS.  
  </ol>  
</details>  
**comments**: Accepted to ICCV 2025  
  
  



## NeRF  

### [PanSt3R: Multi-view Consistent Panoptic Segmentation](http://arxiv.org/abs/2506.21348)  
Lojze Zust, Yohann Cabon, Juliette Marrie, Leonid Antsfeld, Boris Chidlovskii, Jerome Revaud, Gabriela Csurka  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Panoptic segmentation of 3D scenes, involving the segmentation and classification of object instances in a dense 3D reconstruction of a scene, is a challenging problem, especially when relying solely on unposed 2D images. Existing approaches typically leverage off-the-shelf models to extract per-frame 2D panoptic segmentations, before optimizing an implicit geometric representation (often based on NeRF) to integrate and fuse the 2D predictions. We argue that relying on 2D panoptic segmentation for a problem inherently 3D and multi-view is likely suboptimal as it fails to leverage the full potential of spatial relationships across views. In addition to requiring camera parameters, these approaches also necessitate computationally expensive test-time optimization for each scene. Instead, in this work, we propose a unified and integrated approach PanSt3R, which eliminates the need for test-time optimization by jointly predicting 3D geometry and multi-view panoptic segmentation in a single forward pass. Our approach builds upon recent advances in 3D reconstruction, specifically upon MUSt3R, a scalable multi-view version of DUSt3R, and enhances it with semantic awareness and multi-view panoptic segmentation capabilities. We additionally revisit the standard post-processing mask merging procedure and introduce a more principled approach for multi-view segmentation. We also introduce a simple method for generating novel-view predictions based on the predictions of PanSt3R and vanilla 3DGS. Overall, the proposed PanSt3R is conceptually simple, yet fast and scalable, and achieves state-of-the-art performance on several benchmarks, while being orders of magnitude faster than existing methods.  
  </ol>  
</details>  
**comments**: Accepted at ICCV 2025  
  
  



