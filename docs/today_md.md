<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#A-Semantically-Aware-Relevance-Measure-for-Content-Based-Medical-Image-Retrieval-Evaluation>A Semantically-Aware Relevance Measure for Content-Based Medical Image Retrieval Evaluation</a></li>
        <li><a href=#Hierarchical-Multi-Positive-Contrastive-Learning-for-Patent-Image-Retrieval>Hierarchical Multi-Positive Contrastive Learning for Patent Image Retrieval</a></li>
        <li><a href=#EmbodiedPlace:-Learning-Mixture-of-Features-with-Embodied-Constraints-for-Visual-Place-Recognition>EmbodiedPlace: Learning Mixture-of-Features with Embodied Constraints for Visual Place Recognition</a></li>
        <li><a href=#SuperPlace:-The-Renaissance-of-Classical-Feature-Aggregation-for-Visual-Place-Recognition-in-the-Era-of-Foundation-Models>SuperPlace: The Renaissance of Classical Feature Aggregation for Visual Place Recognition in the Era of Foundation Models</a></li>
        <li><a href=#Feature-Complementation-Architecture-for-Visual-Place-Recognition>Feature Complementation Architecture for Visual Place Recognition</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#EmbodiedPlace:-Learning-Mixture-of-Features-with-Embodied-Constraints-for-Visual-Place-Recognition>EmbodiedPlace: Learning Mixture-of-Features with Embodied Constraints for Visual Place Recognition</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Rasterizing-Wireless-Radiance-Field-via-Deformable-2D-Gaussian-Splatting>Rasterizing Wireless Radiance Field via Deformable 2D Gaussian Splatting</a></li>
        <li><a href=#Efficient-multi-view-training-for-3D-Gaussian-Splatting>Efficient multi-view training for 3D Gaussian Splatting</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [A Semantically-Aware Relevance Measure for Content-Based Medical Image Retrieval Evaluation](http://arxiv.org/abs/2506.13509)  
Xiaoyang Wei, Camille Kurtz, Florence Cloppet  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Performance evaluation for Content-Based Image Retrieval (CBIR) remains a crucial but unsolved problem today especially in the medical domain. Various evaluation metrics have been discussed in the literature to solve this problem. Most of the existing metrics (e.g., precision, recall) are adapted from classification tasks which require manual labels as ground truth. However, such labels are often expensive and unavailable in specific thematic domains. Furthermore, medical images are usually associated with (radiological) case reports or annotated with descriptive captions in literature figures, such text contains information that can help to assess CBIR.Several researchers have argued that the medical concepts hidden in the text can serve as the basis for CBIR evaluation purpose. However, these works often consider these medical concepts as independent and isolated labels while in fact the subtle relationships between various concepts are neglected. In this work, we introduce the use of knowledge graphs to measure the distance between various medical concepts and propose a novel relevance measure for the evaluation of CBIR by defining an approximate matching-based relevance score between two sets of medical concepts which allows us to indirectly measure the similarity between medical images.We quantitatively demonstrate the effectiveness and feasibility of our relevance measure using a public dataset.  
  </ol>  
</details>  
**comments**: This paper has been accepted by the International Conference on Image
  Analysis and Processing 2025  
  
### [Hierarchical Multi-Positive Contrastive Learning for Patent Image Retrieval](http://arxiv.org/abs/2506.13496)  
Kshitij Kavimandan, Angelos Nalmpantis, Emma Beauxis-Aussalet, Robert-Jan Sips  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Patent images are technical drawings that convey information about a patent's innovation. Patent image retrieval systems aim to search in vast collections and retrieve the most relevant images. Despite recent advances in information retrieval, patent images still pose significant challenges due to their technical intricacies and complex semantic information, requiring efficient fine-tuning for domain adaptation. Current methods neglect patents' hierarchical relationships, such as those defined by the Locarno International Classification (LIC) system, which groups broad categories (e.g., "furnishing") into subclasses (e.g., "seats" and "beds") and further into specific patent designs. In this work, we introduce a hierarchical multi-positive contrastive loss that leverages the LIC's taxonomy to induce such relations in the retrieval process. Our approach assigns multiple positive pairs to each patent image within a batch, with varying similarity scores based on the hierarchical taxonomy. Our experimental analysis with various vision and multimodal models on the DeepPatent2 dataset shows that the proposed method enhances the retrieval results. Notably, our method is effective with low-parameter models, which require fewer computational resources and can be deployed on environments with limited hardware.  
  </ol>  
</details>  
**comments**: 5 pages, 3 figures, Accepted as a short paper at the 6th Workshop on
  Patent Text Mining and Semantic Technologies (PatentSemTech 2025), co-located
  with SIGIR 2025  
  
### [EmbodiedPlace: Learning Mixture-of-Features with Embodied Constraints for Visual Place Recognition](http://arxiv.org/abs/2506.13133)  
Bingxi Liu, Hao Chen, Shiyi Guo, Yihong Wu, Jinqiang Cui, Hong Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Place Recognition (VPR) is a scene-oriented image retrieval problem in computer vision in which re-ranking based on local features is commonly employed to improve performance. In robotics, VPR is also referred to as Loop Closure Detection, which emphasizes spatial-temporal verification within a sequence. However, designing local features specifically for VPR is impractical, and relying on motion sequences imposes limitations. Inspired by these observations, we propose a novel, simple re-ranking method that refines global features through a Mixture-of-Features (MoF) approach under embodied constraints. First, we analyze the practical feasibility of embodied constraints in VPR and categorize them according to existing datasets, which include GPS tags, sequential timestamps, local feature matching, and self-similarity matrices. We then propose a learning-based MoF weight-computation approach, utilizing a multi-metric loss function. Experiments demonstrate that our method improves the state-of-the-art (SOTA) performance on public datasets with minimal additional computational overhead. For instance, with only 25 KB of additional parameters and a processing time of 10 microseconds per frame, our method achieves a 0.9\% improvement over a DINOv2-based baseline performance on the Pitts-30k test set.  
  </ol>  
</details>  
**comments**: 17 Pages  
  
### [SuperPlace: The Renaissance of Classical Feature Aggregation for Visual Place Recognition in the Era of Foundation Models](http://arxiv.org/abs/2506.13073)  
Bingxi Liu, Pengju Zhang, Li He, Hao Chen, Shiyi Guo, Yihong Wu, Jinqiang Cui, Hong Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent visual place recognition (VPR) approaches have leveraged foundation models (FM) and introduced novel aggregation techniques. However, these methods have failed to fully exploit key concepts of FM, such as the effective utilization of extensive training sets, and they have overlooked the potential of classical aggregation methods, such as GeM and NetVLAD. Building on these insights, we revive classical feature aggregation methods and develop more fundamental VPR models, collectively termed SuperPlace. First, we introduce a supervised label alignment method that enables training across various VPR datasets within a unified framework. Second, we propose G $^2$M, a compact feature aggregation method utilizing two GeMs, where one GeM learns the principal components of feature maps along the channel dimension and calibrates the output of the other. Third, we propose the secondary fine-tuning (FT$^2$) strategy for NetVLAD-Linear (NVL). NetVLAD first learns feature vectors in a high-dimensional space and then compresses them into a lower-dimensional space via a single linear layer. Extensive experiments highlight our contributions and demonstrate the superiority of SuperPlace. Specifically, G$^2$M achieves promising results with only one-tenth of the feature dimensions compared to recent methods. Moreover, NVL-FT$^2$ ranks first on the MSLS leaderboard.  
  </ol>  
</details>  
**comments**: 11 pages  
  
### [Feature Complementation Architecture for Visual Place Recognition](http://arxiv.org/abs/2506.12401)  
Weiwei Wang, Meijia Wang, Haoyi Wang, Wenqiang Guo, Jiapan Guo, Changming Sun, Lingkun Ma, Weichuan Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual place recognition (VPR) plays a crucial role in robotic localization and navigation. The key challenge lies in constructing feature representations that are robust to environmental changes. Existing methods typically adopt convolutional neural networks (CNNs) or vision Transformers (ViTs) as feature extractors. However, these architectures excel in different aspects -- CNNs are effective at capturing local details. At the same time, ViTs are better suited for modeling global context, making it difficult to leverage the strengths of both. To address this issue, we propose a local-global feature complementation network (LGCN) for VPR which integrates a parallel CNN-ViT hybrid architecture with a dynamic feature fusion module (DFM). The DFM performs dynamic feature fusion through joint modeling of spatial and channel-wise dependencies. Furthermore, to enhance the expressiveness and adaptability of the ViT branch for VPR tasks, we introduce lightweight frequency-to-spatial fusion adapters into the frozen ViT backbone. These adapters enable task-specific adaptation with controlled parameter overhead. Extensive experiments on multiple VPR benchmark datasets demonstrate that the proposed LGCN consistently outperforms existing approaches in terms of localization accuracy and robustness, validating its effectiveness and generalizability.  
  </ol>  
</details>  
  
  



## Image Matching  

### [EmbodiedPlace: Learning Mixture-of-Features with Embodied Constraints for Visual Place Recognition](http://arxiv.org/abs/2506.13133)  
Bingxi Liu, Hao Chen, Shiyi Guo, Yihong Wu, Jinqiang Cui, Hong Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Place Recognition (VPR) is a scene-oriented image retrieval problem in computer vision in which re-ranking based on local features is commonly employed to improve performance. In robotics, VPR is also referred to as Loop Closure Detection, which emphasizes spatial-temporal verification within a sequence. However, designing local features specifically for VPR is impractical, and relying on motion sequences imposes limitations. Inspired by these observations, we propose a novel, simple re-ranking method that refines global features through a Mixture-of-Features (MoF) approach under embodied constraints. First, we analyze the practical feasibility of embodied constraints in VPR and categorize them according to existing datasets, which include GPS tags, sequential timestamps, local feature matching, and self-similarity matrices. We then propose a learning-based MoF weight-computation approach, utilizing a multi-metric loss function. Experiments demonstrate that our method improves the state-of-the-art (SOTA) performance on public datasets with minimal additional computational overhead. For instance, with only 25 KB of additional parameters and a processing time of 10 microseconds per frame, our method achieves a 0.9\% improvement over a DINOv2-based baseline performance on the Pitts-30k test set.  
  </ol>  
</details>  
**comments**: 17 Pages  
  
  



## NeRF  

### [Rasterizing Wireless Radiance Field via Deformable 2D Gaussian Splatting](http://arxiv.org/abs/2506.12787)  
Mufan Liu, Cixiao Zhang, Qi Yang, Yujie Cao, Yiling Xu, Yin Xu, Shu Sun, Mingzeng Dai, Yunfeng Guan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Modeling the wireless radiance field (WRF) is fundamental to modern communication systems, enabling key tasks such as localization, sensing, and channel estimation. Traditional approaches, which rely on empirical formulas or physical simulations, often suffer from limited accuracy or require strong scene priors. Recent neural radiance field (NeRF-based) methods improve reconstruction fidelity through differentiable volumetric rendering, but their reliance on computationally expensive multilayer perceptron (MLP) queries hinders real-time deployment. To overcome these challenges, we introduce Gaussian splatting (GS) to the wireless domain, leveraging its efficiency in modeling optical radiance fields to enable compact and accurate WRF reconstruction. Specifically, we propose SwiftWRF, a deformable 2D Gaussian splatting framework that synthesizes WRF spectra at arbitrary positions under single-sided transceiver mobility. SwiftWRF employs CUDA-accelerated rasterization to render spectra at over 100000 fps and uses a lightweight MLP to model the deformation of 2D Gaussians, effectively capturing mobility-induced WRF variations. In addition to novel spectrum synthesis, the efficacy of SwiftWRF is further underscored in its applications in angle-of-arrival (AoA) and received signal strength indicator (RSSI) prediction. Experiments conducted on both real-world and synthetic indoor scenes demonstrate that SwiftWRF can reconstruct WRF spectra up to 500x faster than existing state-of-the-art methods, while significantly enhancing its signal quality. Code and datasets will be released.  
  </ol>  
</details>  
  
### [Efficient multi-view training for 3D Gaussian Splatting](http://arxiv.org/abs/2506.12727)  
Minhyuk Choi, Injae Kim, Hyunwoo J. Kim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D Gaussian Splatting (3DGS) has emerged as a preferred choice alongside Neural Radiance Fields (NeRF) in inverse rendering due to its superior rendering speed. Currently, the common approach in 3DGS is to utilize "single-view" mini-batch training, where only one image is processed per iteration, in contrast to NeRF's "multi-view" mini-batch training, which leverages multiple images. We observe that such single-view training can lead to suboptimal optimization due to increased variance in mini-batch stochastic gradients, highlighting the necessity for multi-view training. However, implementing multi-view training in 3DGS poses challenges. Simply rendering multiple images per iteration incurs considerable overhead and may result in suboptimal Gaussian densification due to its reliance on single-view assumptions. To address these issues, we modify the rasterization process to minimize the overhead associated with multi-view training and propose a 3D distance-aware D-SSIM loss and multi-view adaptive density control that better suits multi-view scenarios. Our experiments demonstrate that the proposed methods significantly enhance the performance of 3DGS and its variants, freeing 3DGS from the constraints of single-view training.  
  </ol>  
</details>  
  
  



