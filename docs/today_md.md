<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#BRUM:-Robust-3D-Vehicle-Reconstruction-from-360-Sparse-Images>BRUM: Robust 3D Vehicle Reconstruction from 360 Sparse Images</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#QuRe:-Query-Relevant-Retrieval-through-Hard-Negative-Sampling-in-Composed-Image-Retrieval>QuRe: Query-Relevant Retrieval through Hard Negative Sampling in Composed Image Retrieval</a></li>
        <li><a href=#CorrMoE:-Mixture-of-Experts-with-De-stylization-Learning-for-Cross-Scene-and-Cross-Domain-Correspondence-Pruning>CorrMoE: Mixture of Experts with De-stylization Learning for Cross-Scene and Cross-Domain Correspondence Pruning</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#DoRF:-Doppler-Radiance-Fields-for-Robust-Human-Activity-Recognition-Using-Wi-Fi>DoRF: Doppler Radiance Fields for Robust Human Activity Recognition Using Wi-Fi</a></li>
        <li><a href=#HPR3D:-Hierarchical-Proxy-Representation-for-High-Fidelity-3D-Reconstruction-and-Controllable-Editing>HPR3D: Hierarchical Proxy Representation for High-Fidelity 3D Reconstruction and Controllable Editing</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [BRUM: Robust 3D Vehicle Reconstruction from 360 Sparse Images](http://arxiv.org/abs/2507.12095)  
Davide Di Nucci, Matteo Tomei, Guido Borghi, Luca Ciuffreda, Roberto Vezzani, Rita Cucchiara  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Accurate 3D reconstruction of vehicles is vital for applications such as vehicle inspection, predictive maintenance, and urban planning. Existing methods like Neural Radiance Fields and Gaussian Splatting have shown impressive results but remain limited by their reliance on dense input views, which hinders real-world applicability. This paper addresses the challenge of reconstructing vehicles from sparse-view inputs, leveraging depth maps and a robust pose estimation architecture to synthesize novel views and augment training data. Specifically, we enhance Gaussian Splatting by integrating a selective photometric loss, applied only to high-confidence pixels, and replacing standard Structure-from-Motion pipelines with the DUSt3R architecture to improve camera pose estimation. Furthermore, we present a novel dataset featuring both synthetic and real-world public transportation vehicles, enabling extensive evaluation of our approach. Experimental results demonstrate state-of-the-art performance across multiple benchmarks, showcasing the method's ability to achieve high-quality reconstructions even under constrained input conditions.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [QuRe: Query-Relevant Retrieval through Hard Negative Sampling in Composed Image Retrieval](http://arxiv.org/abs/2507.12416)  
Jaehyun Kwak, Ramahdani Muhammad Izaaz Inhar, Se-Young Yun, Sung-Ju Lee  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed Image Retrieval (CIR) retrieves relevant images based on a reference image and accompanying text describing desired modifications. However, existing CIR methods only focus on retrieving the target image and disregard the relevance of other images. This limitation arises because most methods employing contrastive learning-which treats the target image as positive and all other images in the batch as negatives-can inadvertently include false negatives. This may result in retrieving irrelevant images, reducing user satisfaction even when the target image is retrieved. To address this issue, we propose Query-Relevant Retrieval through Hard Negative Sampling (QuRe), which optimizes a reward model objective to reduce false negatives. Additionally, we introduce a hard negative sampling strategy that selects images positioned between two steep drops in relevance scores following the target image, to effectively filter false negatives. In order to evaluate CIR models on their alignment with human satisfaction, we create Human-Preference FashionIQ (HP-FashionIQ), a new dataset that explicitly captures user preferences beyond target retrieval. Extensive experiments demonstrate that QuRe achieves state-of-the-art performance on FashionIQ and CIRR datasets while exhibiting the strongest alignment with human preferences on the HP-FashionIQ dataset. The source code is available at https://github.com/jackwaky/QuRe.  
  </ol>  
</details>  
**comments**: Accepted to ICML 2025  
  
### [CorrMoE: Mixture of Experts with De-stylization Learning for Cross-Scene and Cross-Domain Correspondence Pruning](http://arxiv.org/abs/2507.11834)  
Peiwen Xia, Tangfei Liao, Wei Zhu, Danhuai Zhao, Jianjun Ke, Kaihao Zhang, Tong Lu, Tao Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Establishing reliable correspondences between image pairs is a fundamental task in computer vision, underpinning applications such as 3D reconstruction and visual localization. Although recent methods have made progress in pruning outliers from dense correspondence sets, they often hypothesize consistent visual domains and overlook the challenges posed by diverse scene structures. In this paper, we propose CorrMoE, a novel correspondence pruning framework that enhances robustness under cross-domain and cross-scene variations. To address domain shift, we introduce a De-stylization Dual Branch, performing style mixing on both implicit and explicit graph features to mitigate the adverse influence of domain-specific representations. For scene diversity, we design a Bi-Fusion Mixture of Experts module that adaptively integrates multi-perspective features through linear-complexity attention and dynamic expert routing. Extensive experiments on benchmark datasets demonstrate that CorrMoE achieves superior accuracy and generalization compared to state-of-the-art methods. The code and pre-trained models are available at https://github.com/peiwenxia/CorrMoE.  
  </ol>  
</details>  
**comments**: Accepted by ECAI 2025  
  
  



## NeRF  

### [DoRF: Doppler Radiance Fields for Robust Human Activity Recognition Using Wi-Fi](http://arxiv.org/abs/2507.12132)  
Navid Hasanzadeh, Shahrokh Valaee  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Wi-Fi Channel State Information (CSI) has gained increasing interest for remote sensing applications. Recent studies show that Doppler velocity projections extracted from CSI can enable human activity recognition (HAR) that is robust to environmental changes and generalizes to new users. However, despite these advances, generalizability still remains insufficient for practical deployment. Inspired by neural radiance fields (NeRF), which learn a volumetric representation of a 3D scene from 2D images, this work proposes a novel approach to reconstruct an informative 3D latent motion representation from one-dimensional Doppler velocity projections extracted from Wi-Fi CSI. The resulting latent representation is then used to construct a uniform Doppler radiance field (DoRF) of the motion, providing a comprehensive view of the performed activity and improving the robustness to environmental variability. The results show that the proposed approach noticeably enhances the generalization accuracy of Wi-Fi-based HAR, highlighting the strong potential of DoRFs for practical sensing applications.  
  </ol>  
</details>  
  
### [HPR3D: Hierarchical Proxy Representation for High-Fidelity 3D Reconstruction and Controllable Editing](http://arxiv.org/abs/2507.11971)  
Tielong Wang, Yuxuan Xiong, Jinfan Liu, Zhifan Zhang, Ye Chen, Yue Shi, Bingbing Ni  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Current 3D representations like meshes, voxels, point clouds, and NeRF-based neural implicit fields exhibit significant limitations: they are often task-specific, lacking universal applicability across reconstruction, generation, editing, and driving. While meshes offer high precision, their dense vertex data complicates editing; NeRFs deliver excellent rendering but suffer from structural ambiguity, hindering animation and manipulation; all representations inherently struggle with the trade-off between data complexity and fidelity. To overcome these issues, we introduce a novel 3D Hierarchical Proxy Node representation. Its core innovation lies in representing an object's shape and texture via a sparse set of hierarchically organized (tree-structured) proxy nodes distributed on its surface and interior. Each node stores local shape and texture information (implicitly encoded by a small MLP) within its neighborhood. Querying any 3D coordinate's properties involves efficient neural interpolation and lightweight decoding from relevant nearby and parent nodes. This framework yields a highly compact representation where nodes align with local semantics, enabling direct drag-and-edit manipulation, and offers scalable quality-complexity control. Extensive experiments across 3D reconstruction and editing demonstrate our method's expressive efficiency, high-fidelity rendering quality, and superior editability.  
  </ol>  
</details>  
  
  



