<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#DINO-VO:-A-Feature-based-Visual-Odometry-Leveraging-a-Visual-Foundation-Model>DINO-VO: A Feature-based Visual Odometry Leveraging a Visual Foundation Model</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#FAR-Net:-Multi-Stage-Fusion-Network-with-Enhanced-Semantic-Alignment-and-Adaptive-Reconciliation-for-Composed-Image-Retrieval>FAR-Net: Multi-Stage Fusion Network with Enhanced Semantic Alignment and Adaptive Reconciliation for Composed Image Retrieval</a></li>
        <li><a href=#MCoT-RE:-Multi-Faceted-Chain-of-Thought-and-Re-Ranking-for-Training-Free-Zero-Shot-Composed-Image-Retrieval>MCoT-RE: Multi-Faceted Chain-of-Thought and Re-Ranking for Training-Free Zero-Shot Composed Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#DINO-VO:-A-Feature-based-Visual-Odometry-Leveraging-a-Visual-Foundation-Model>DINO-VO: A Feature-based Visual Odometry Leveraging a Visual Foundation Model</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [DINO-VO: A Feature-based Visual Odometry Leveraging a Visual Foundation Model](http://arxiv.org/abs/2507.13145)  
Maulana Bisyir Azhari, David Hyunchul Shim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Learning-based monocular visual odometry (VO) poses robustness, generalization, and efficiency challenges in robotics. Recent advances in visual foundation models, such as DINOv2, have improved robustness and generalization in various vision tasks, yet their integration in VO remains limited due to coarse feature granularity. In this paper, we present DINO-VO, a feature-based VO system leveraging DINOv2 visual foundation model for its sparse feature matching. To address the integration challenge, we propose a salient keypoints detector tailored to DINOv2's coarse features. Furthermore, we complement DINOv2's robust-semantic features with fine-grained geometric features, resulting in more localizable representations. Finally, a transformer-based matcher and differentiable pose estimation layer enable precise camera motion estimation by learning good matches. Against prior detector-descriptor networks like SuperPoint, DINO-VO demonstrates greater robustness in challenging environments. Furthermore, we show superior accuracy and generalization of the proposed feature descriptors against standalone DINOv2 coarse features. DINO-VO outperforms prior frame-to-frame VO methods on the TartanAir and KITTI datasets and is competitive on EuRoC dataset, while running efficiently at 72 FPS with less than 1GB of memory usage on a single GPU. Moreover, it performs competitively against Visual SLAM systems on outdoor driving scenarios, showcasing its generalization capabilities.  
  </ol>  
</details>  
**comments**: 8 pages, 6 figures. Accepted for publication in IEEE Robotics and
  Automation Letters (RA-L), July 2025  
  
  



## Visual Localization  

### [FAR-Net: Multi-Stage Fusion Network with Enhanced Semantic Alignment and Adaptive Reconciliation for Composed Image Retrieval](http://arxiv.org/abs/2507.12823)  
Jeong-Woo Park, Young-Eun Kim, Seong-Whan Lee  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed image retrieval (CIR) is a vision language task that retrieves a target image using a reference image and modification text, enabling intuitive specification of desired changes. While effectively fusing visual and textual modalities is crucial, existing methods typically adopt either early or late fusion. Early fusion tends to excessively focus on explicitly mentioned textual details and neglect visual context, whereas late fusion struggles to capture fine-grained semantic alignments between image regions and textual tokens. To address these issues, we propose FAR-Net, a multi-stage fusion framework designed with enhanced semantic alignment and adaptive reconciliation, integrating two complementary modules. The enhanced semantic alignment module (ESAM) employs late fusion with cross-attention to capture fine-grained semantic relationships, while the adaptive reconciliation module (ARM) applies early fusion with uncertainty embeddings to enhance robustness and adaptability. Experiments on CIRR and FashionIQ show consistent performance gains, improving Recall@1 by up to 2.4% and Recall@50 by 1.04% over existing state-of-the-art methods, empirically demonstrating that FAR Net provides a robust and scalable solution to CIR tasks.  
  </ol>  
</details>  
**comments**: 6 pages, 3 figures, 3 tables  
  
### [MCoT-RE: Multi-Faceted Chain-of-Thought and Re-Ranking for Training-Free Zero-Shot Composed Image Retrieval](http://arxiv.org/abs/2507.12819)  
Jeong-Woo Park, Seong-Whan Lee  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed Image Retrieval (CIR) is the task of retrieving a target image from a gallery using a composed query consisting of a reference image and a modification text. Among various CIR approaches, training-free zero-shot methods based on pre-trained models are cost-effective but still face notable limitations. For example, sequential VLM-LLM pipelines process each modality independently, which often results in information loss and limits cross-modal interaction. In contrast, methods based on multimodal large language models (MLLMs) often focus exclusively on applying changes indicated by the text, without fully utilizing the contextual visual information from the reference image. To address these issues, we propose multi-faceted Chain-of-Thought with re-ranking (MCoT-RE), a training-free zero-shot CIR framework. MCoT-RE utilizes multi-faceted Chain-of-Thought to guide the MLLM to balance explicit modifications and contextual visual cues, generating two distinct captions: one focused on modification and the other integrating comprehensive visual-textual context. The first caption is used to filter candidate images. Subsequently, we combine these two captions and the reference image to perform multi-grained re-ranking. This two-stage approach facilitates precise retrieval by aligning with the textual modification instructions while preserving the visual context of the reference image. Through extensive experiments, MCoT-RE achieves state-of-the-art results among training-free methods, yielding improvements of up to 6.24% in Recall@10 on FashionIQ and 8.58% in Recall@1 on CIRR.  
  </ol>  
</details>  
**comments**: 6 pages, 4 figures, 2025 IEEE International Conference on Systems,
  Man, and Cybernetics  
  
  



## Keypoint Detection  

### [DINO-VO: A Feature-based Visual Odometry Leveraging a Visual Foundation Model](http://arxiv.org/abs/2507.13145)  
Maulana Bisyir Azhari, David Hyunchul Shim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Learning-based monocular visual odometry (VO) poses robustness, generalization, and efficiency challenges in robotics. Recent advances in visual foundation models, such as DINOv2, have improved robustness and generalization in various vision tasks, yet their integration in VO remains limited due to coarse feature granularity. In this paper, we present DINO-VO, a feature-based VO system leveraging DINOv2 visual foundation model for its sparse feature matching. To address the integration challenge, we propose a salient keypoints detector tailored to DINOv2's coarse features. Furthermore, we complement DINOv2's robust-semantic features with fine-grained geometric features, resulting in more localizable representations. Finally, a transformer-based matcher and differentiable pose estimation layer enable precise camera motion estimation by learning good matches. Against prior detector-descriptor networks like SuperPoint, DINO-VO demonstrates greater robustness in challenging environments. Furthermore, we show superior accuracy and generalization of the proposed feature descriptors against standalone DINOv2 coarse features. DINO-VO outperforms prior frame-to-frame VO methods on the TartanAir and KITTI datasets and is competitive on EuRoC dataset, while running efficiently at 72 FPS with less than 1GB of memory usage on a single GPU. Moreover, it performs competitively against Visual SLAM systems on outdoor driving scenarios, showcasing its generalization capabilities.  
  </ol>  
</details>  
**comments**: 8 pages, 6 figures. Accepted for publication in IEEE Robotics and
  Automation Letters (RA-L), July 2025  
  
  



