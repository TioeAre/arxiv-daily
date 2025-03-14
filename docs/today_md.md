<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#ImageScope:-Unifying-Language-Guided-Image-Retrieval-via-Large-Multimodal-Model-Collective-Reasoning>ImageScope: Unifying Language-Guided Image Retrieval via Large Multimodal Model Collective Reasoning</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Speedy-MASt3R>Speedy MASt3R</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Flow-NeRF:-Joint-Learning-of-Geometry,-Poses,-and-Dense-Flow-within-Unified-Neural-Representations>Flow-NeRF: Joint Learning of Geometry, Poses, and Dense Flow within Unified Neural Representations</a></li>
        <li><a href=#AI-assisted-3D-Preservation-and-Reconstruction-of-Temple-Arts>AI-assisted 3D Preservation and Reconstruction of Temple Arts</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [ImageScope: Unifying Language-Guided Image Retrieval via Large Multimodal Model Collective Reasoning](http://arxiv.org/abs/2503.10166)  
Pengfei Luo, Jingbo Zhou, Tong Xu, Yuan Xia, Linli Xu, Enhong Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    With the proliferation of images in online content, language-guided image retrieval (LGIR) has emerged as a research hotspot over the past decade, encompassing a variety of subtasks with diverse input forms. While the development of large multimodal models (LMMs) has significantly facilitated these tasks, existing approaches often address them in isolation, requiring the construction of separate systems for each task. This not only increases system complexity and maintenance costs, but also exacerbates challenges stemming from language ambiguity and complex image content, making it difficult for retrieval systems to provide accurate and reliable results. To this end, we propose ImageScope, a training-free, three-stage framework that leverages collective reasoning to unify LGIR tasks. The key insight behind the unification lies in the compositional nature of language, which transforms diverse LGIR tasks into a generalized text-to-image retrieval process, along with the reasoning of LMMs serving as a universal verification to refine the results. To be specific, in the first stage, we improve the robustness of the framework by synthesizing search intents across varying levels of semantic granularity using chain-of-thought (CoT) reasoning. In the second and third stages, we then reflect on retrieval results by verifying predicate propositions locally, and performing pairwise evaluations globally. Experiments conducted on six LGIR datasets demonstrate that ImageScope outperforms competitive baselines. Comprehensive evaluations and ablation studies further confirm the effectiveness of our design.  
  </ol>  
</details>  
**comments**: WWW 2025  
  
  



## Image Matching  

### [Speedy MASt3R](http://arxiv.org/abs/2503.10017)  
Jingxing Li, Yongjae Lee, Abhay Kumar Yadav, Cheng Peng, Rama Chellappa, Deliang Fan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image matching is a key component of modern 3D vision algorithms, essential for accurate scene reconstruction and localization. MASt3R redefines image matching as a 3D task by leveraging DUSt3R and introducing a fast reciprocal matching scheme that accelerates matching by orders of magnitude while preserving theoretical guarantees. This approach has gained strong traction, with DUSt3R and MASt3R collectively cited over 250 times in a short span, underscoring their impact. However, despite its accuracy, MASt3R's inference speed remains a bottleneck. On an A40 GPU, latency per image pair is 198.16 ms, mainly due to computational overhead from the ViT encoder-decoder and Fast Reciprocal Nearest Neighbor (FastNN) matching.   To address this, we introduce Speedy MASt3R, a post-training optimization framework that enhances inference efficiency while maintaining accuracy. It integrates multiple optimization techniques, including FlashMatch-an approach leveraging FlashAttention v2 with tiling strategies for improved efficiency, computation graph optimization via layer and tensor fusion having kernel auto-tuning with TensorRT (GraphFusion), and a streamlined FastNN pipeline that reduces memory access time from quadratic to linear while accelerating block-wise correlation scoring through vectorized computation (FastNN-Lite). Additionally, it employs mixed-precision inference with FP16/FP32 hybrid computations (HybridCast), achieving speedup while preserving numerical precision. Evaluated on Aachen Day-Night, InLoc, 7-Scenes, ScanNet1500, and MegaDepth1500, Speedy MASt3R achieves a 54% reduction in inference time (198 ms to 91 ms per image pair) without sacrificing accuracy. This advancement enables real-time 3D understanding, benefiting applications like mixed reality navigation and large-scale 3D scene reconstruction.  
  </ol>  
</details>  
  
  



## NeRF  

### [Flow-NeRF: Joint Learning of Geometry, Poses, and Dense Flow within Unified Neural Representations](http://arxiv.org/abs/2503.10464)  
Xunzhi Zheng, Dan Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Learning accurate scene reconstruction without pose priors in neural radiance fields is challenging due to inherent geometric ambiguity. Recent development either relies on correspondence priors for regularization or uses off-the-shelf flow estimators to derive analytical poses. However, the potential for jointly learning scene geometry, camera poses, and dense flow within a unified neural representation remains largely unexplored. In this paper, we present Flow-NeRF, a unified framework that simultaneously optimizes scene geometry, camera poses, and dense optical flow all on-the-fly. To enable the learning of dense flow within the neural radiance field, we design and build a bijective mapping for flow estimation, conditioned on pose. To make the scene reconstruction benefit from the flow estimation, we develop an effective feature enhancement mechanism to pass canonical space features to world space representations, significantly enhancing scene geometry. We validate our model across four important tasks, i.e., novel view synthesis, depth estimation, camera pose prediction, and dense optical flow estimation, using several datasets. Our approach surpasses previous methods in almost all metrics for novel-view view synthesis and depth estimation and yields both qualitatively sound and quantitatively accurate novel-view flow. Our project page is https://zhengxunzhi.github.io/flownerf/.  
  </ol>  
</details>  
  
### [AI-assisted 3D Preservation and Reconstruction of Temple Arts](http://arxiv.org/abs/2503.10031)  
Naai-Jung Shih  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    How does AI connect to the past in conservation? What can 17 years old photos be helpful in a renewed effort of preservation? This research aims to use AI to connect both in a seamless 3D reconstruction of heritage from imagery data taken from Gongfan Palace, Yunlin Taiwan. AI-assisted 3D modeling was used to reconstruct correspondent details across different 3D platforms of 3DGS or NeRF models generated by Postshot or KIRI Engine. Polygon or point models by Zephyr were referred to and assessed in two sets. The results also include AI-assist modeling outcomes in Stable Diffusion and Postshot-based animation. The evolved documenta-tion and interpretation in AI presents a novel arrangement of working processes contributed by new structure and management of resources, formats, and interfaces, as a continuous preservation effort.  
  </ol>  
</details>  
**comments**: 13 pages, 9 figures, 1 table  
  
  



