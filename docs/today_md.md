<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#UFM:-A-Simple-Path-towards-Unified-Dense-Correspondence-with-Flow>UFM: A Simple Path towards Unified Dense Correspondence with Flow</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Hierarchical-Image-Matching-for-UAV-Absolute-Visual-Localization-via-Semantic-and-Structural-Constraints>Hierarchical Image Matching for UAV Absolute Visual Localization via Semantic and Structural Constraints</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Hierarchical-Image-Matching-for-UAV-Absolute-Visual-Localization-via-Semantic-and-Structural-Constraints>Hierarchical Image Matching for UAV Absolute Visual Localization via Semantic and Structural Constraints</a></li>
        <li><a href=#ScaleLSD:-Scalable-Deep-Line-Segment-Detection-Streamlined>ScaleLSD: Scalable Deep Line Segment Detection Streamlined</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#The-Less-You-Depend,-The-More-You-Learn:-Synthesizing-Novel-Views-from-Sparse,-Unposed-Images-without-Any-3D-Knowledge>The Less You Depend, The More You Learn: Synthesizing Novel Views from Sparse, Unposed Images without Any 3D Knowledge</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [UFM: A Simple Path towards Unified Dense Correspondence with Flow](http://arxiv.org/abs/2506.09278)  
Yuchen Zhang, Nikhil Keetha, Chenwei Lyu, Bhuvan Jhamb, Yutian Chen, Yuheng Qiu, Jay Karhade, Shreyas Jha, Yaoyu Hu, Deva Ramanan, Sebastian Scherer, Wenshan Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Dense image correspondence is central to many applications, such as visual odometry, 3D reconstruction, object association, and re-identification. Historically, dense correspondence has been tackled separately for wide-baseline scenarios and optical flow estimation, despite the common goal of matching content between two images. In this paper, we develop a Unified Flow & Matching model (UFM), which is trained on unified data for pixels that are co-visible in both source and target images. UFM uses a simple, generic transformer architecture that directly regresses the (u,v) flow. It is easier to train and more accurate for large flows compared to the typical coarse-to-fine cost volumes in prior work. UFM is 28% more accurate than state-of-the-art flow methods (Unimatch), while also having 62% less error and 6.7x faster than dense wide-baseline matchers (RoMa). UFM is the first to demonstrate that unified training can outperform specialized approaches across both domains. This result enables fast, general-purpose correspondence and opens new directions for multi-modal, long-range, and real-time correspondence tasks.  
  </ol>  
</details>  
**comments**: Project Page: https://uniflowmatch.github.io/  
  
  



## Visual Localization  

### [Hierarchical Image Matching for UAV Absolute Visual Localization via Semantic and Structural Constraints](http://arxiv.org/abs/2506.09748)  
Xiangkai Zhang, Xiang Zhou, Mao Chen, Yuchen Lu, Xu Yang, Zhiyong Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Absolute localization, aiming to determine an agent's location with respect to a global reference, is crucial for unmanned aerial vehicles (UAVs) in various applications, but it becomes challenging when global navigation satellite system (GNSS) signals are unavailable. Vision-based absolute localization methods, which locate the current view of the UAV in a reference satellite map to estimate its position, have become popular in GNSS-denied scenarios. However, existing methods mostly rely on traditional and low-level image matching, suffering from difficulties due to significant differences introduced by cross-source discrepancies and temporal variations. To overcome these limitations, in this paper, we introduce a hierarchical cross-source image matching method designed for UAV absolute localization, which integrates a semantic-aware and structure-constrained coarse matching module with a lightweight fine-grained matching module. Specifically, in the coarse matching module, semantic features derived from a vision foundation model first establish region-level correspondences under semantic and structural constraints. Then, the fine-grained matching module is applied to extract fine features and establish pixel-level correspondences. Building upon this, a UAV absolute visual localization pipeline is constructed without any reliance on relative localization techniques, mainly by employing an image retrieval module before the proposed hierarchical image matching modules. Experimental evaluations on public benchmark datasets and a newly introduced CS-UAV dataset demonstrate superior accuracy and robustness of the proposed method under various challenging conditions, confirming its effectiveness.  
  </ol>  
</details>  
**comments**: 8 pages, 6 figures  
  
  



## Image Matching  

### [Hierarchical Image Matching for UAV Absolute Visual Localization via Semantic and Structural Constraints](http://arxiv.org/abs/2506.09748)  
Xiangkai Zhang, Xiang Zhou, Mao Chen, Yuchen Lu, Xu Yang, Zhiyong Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Absolute localization, aiming to determine an agent's location with respect to a global reference, is crucial for unmanned aerial vehicles (UAVs) in various applications, but it becomes challenging when global navigation satellite system (GNSS) signals are unavailable. Vision-based absolute localization methods, which locate the current view of the UAV in a reference satellite map to estimate its position, have become popular in GNSS-denied scenarios. However, existing methods mostly rely on traditional and low-level image matching, suffering from difficulties due to significant differences introduced by cross-source discrepancies and temporal variations. To overcome these limitations, in this paper, we introduce a hierarchical cross-source image matching method designed for UAV absolute localization, which integrates a semantic-aware and structure-constrained coarse matching module with a lightweight fine-grained matching module. Specifically, in the coarse matching module, semantic features derived from a vision foundation model first establish region-level correspondences under semantic and structural constraints. Then, the fine-grained matching module is applied to extract fine features and establish pixel-level correspondences. Building upon this, a UAV absolute visual localization pipeline is constructed without any reliance on relative localization techniques, mainly by employing an image retrieval module before the proposed hierarchical image matching modules. Experimental evaluations on public benchmark datasets and a newly introduced CS-UAV dataset demonstrate superior accuracy and robustness of the proposed method under various challenging conditions, confirming its effectiveness.  
  </ol>  
</details>  
**comments**: 8 pages, 6 figures  
  
### [ScaleLSD: Scalable Deep Line Segment Detection Streamlined](http://arxiv.org/abs/2506.09369)  
Zeran Ke, Bin Tan, Xianwei Zheng, Yujun Shen, Tianfu Wu, Nan Xue  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper studies the problem of Line Segment Detection (LSD) for the characterization of line geometry in images, with the aim of learning a domain-agnostic robust LSD model that works well for any natural images. With the focus of scalable self-supervised learning of LSD, we revisit and streamline the fundamental designs of (deep and non-deep) LSD approaches to have a high-performing and efficient LSD learner, dubbed as ScaleLSD, for the curation of line geometry at scale from over 10M unlabeled real-world images. Our ScaleLSD works very well to detect much more number of line segments from any natural images even than the pioneered non-deep LSD approach, having a more complete and accurate geometric characterization of images using line segments. Experimentally, our proposed ScaleLSD is comprehensively testified under zero-shot protocols in detection performance, single-view 3D geometry estimation, two-view line segment matching, and multiview 3D line mapping, all with excellent performance obtained. Based on the thorough evaluation, our ScaleLSD is observed to be the first deep approach that outperforms the pioneered non-deep LSD in all aspects we have tested, significantly expanding and reinforcing the versatility of the line geometry of images. Code and Models are available at https://github.com/ant-research/scalelsd  
  </ol>  
</details>  
**comments**: accepted to CVPR 2025; 17 pages, appendices included  
  
  



## NeRF  

### [The Less You Depend, The More You Learn: Synthesizing Novel Views from Sparse, Unposed Images without Any 3D Knowledge](http://arxiv.org/abs/2506.09885)  
Haoru Wang, Kai Ye, Yangyan Li, Wenzheng Chen, Baoquan Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We consider the problem of generalizable novel view synthesis (NVS), which aims to generate photorealistic novel views from sparse or even unposed 2D images without per-scene optimization. This task remains fundamentally challenging, as it requires inferring 3D structure from incomplete and ambiguous 2D observations. Early approaches typically rely on strong 3D knowledge, including architectural 3D inductive biases (e.g., embedding explicit 3D representations, such as NeRF or 3DGS, into network design) and ground-truth camera poses for both input and target views. While recent efforts have sought to reduce the 3D inductive bias or the dependence on known camera poses of input views, critical questions regarding the role of 3D knowledge and the necessity of circumventing its use remain under-explored. In this work, we conduct a systematic analysis on the 3D knowledge and uncover a critical trend: the performance of methods that requires less 3D knowledge accelerates more as data scales, eventually achieving performance on par with their 3D knowledge-driven counterparts, which highlights the increasing importance of reducing dependence on 3D knowledge in the era of large-scale data. Motivated by and following this trend, we propose a novel NVS framework that minimizes 3D inductive bias and pose dependence for both input and target views. By eliminating this 3D knowledge, our method fully leverages data scaling and learns implicit 3D awareness directly from sparse 2D images, without any 3D inductive bias or pose annotation during training. Extensive experiments demonstrate that our model generates photorealistic and 3D-consistent novel views, achieving even comparable performance with methods that rely on posed inputs, thereby validating the feasibility and effectiveness of our data-centric paradigm. Project page: https://pku-vcl-geometry.github.io/Less3Depend/ .  
  </ol>  
</details>  
  
  



