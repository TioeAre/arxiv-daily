<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#VIPeR:-Visual-Incremental-Place-Recognition-with-Adaptive-Mining-and-Lifelong-Learning>VIPeR: Visual Incremental Place Recognition with Adaptive Mining and Lifelong Learning</a></li>
        <li><a href=#SuperVINS:-A-visual-inertial-SLAM-framework-integrated-deep-learning-features>SuperVINS: A visual-inertial SLAM framework integrated deep learning features</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#PAV:-Personalized-Head-Avatar-from-Unstructured-Video-Collection>PAV: Personalized Head Avatar from Unstructured Video Collection</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [VIPeR: Visual Incremental Place Recognition with Adaptive Mining and Lifelong Learning](http://arxiv.org/abs/2407.21416)  
Yuhang Ming, Minyang Xu, Xingrui Yang, Weicai Ye, Weihan Wang, Yong Peng, Weichen Dai, Wanzeng Kong  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual place recognition (VPR) is an essential component of many autonomous and augmented/virtual reality systems. It enables the systems to robustly localize themselves in large-scale environments. Existing VPR methods demonstrate attractive performance at the cost of heavy pre-training and limited generalizability. When deployed in unseen environments, these methods exhibit significant performance drops. Targeting this issue, we present VIPeR, a novel approach for visual incremental place recognition with the ability to adapt to new environments while retaining the performance of previous environments. We first introduce an adaptive mining strategy that balances the performance within a single environment and the generalizability across multiple environments. Then, to prevent catastrophic forgetting in lifelong learning, we draw inspiration from human memory systems and design a novel memory bank for our VIPeR. Our memory bank contains a sensory memory, a working memory and a long-term memory, with the first two focusing on the current environment and the last one for all previously visited environments. Additionally, we propose a probabilistic knowledge distillation to explicitly safeguard the previously learned knowledge. We evaluate our proposed VIPeR on three large-scale datasets, namely Oxford Robotcar, Nordland, and TartanAir. For comparison, we first set a baseline performance with naive finetuning. Then, several more recent lifelong learning methods are compared. Our VIPeR achieves better performance in almost all aspects with the biggest improvement of 13.65% in average performance.  
  </ol>  
</details>  
**comments**: 8 pages, 4 figures  
  
### [SuperVINS: A visual-inertial SLAM framework integrated deep learning features](http://arxiv.org/abs/2407.21348)  
Hongkun Luo, Chi Guo, Yang Liu, Zengke Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this article, we propose enhancements to VINS-Fusion by incorporating deep learning features and deep learning matching methods. We implemented the training of deep learning feature bag of words and utilized these features for loop closure detection. Additionally, we introduce the RANSAC algorithm in the deep learning feature matching module to optimize matching. SuperVINS, an improved version of VINS-Fusion, outperforms it in terms of positioning accuracy, robustness, and more. Particularly in challenging scenarios like low illumination and rapid jitter, traditional geometric features fail to fully exploit image information, whereas deep learning features excel at capturing image features.To validate our proposed improvement scheme, we conducted experiments using open source datasets. We performed a comprehensive analysis of the experimental results from both qualitative and quantitative perspectives. The results demonstrate the feasibility and effectiveness of this deep learning-based approach for SLAM systems.To foster knowledge exchange in this field, we have made the code for this article publicly available. You can find the code at this link: https://github.com/luohongk/SuperVINS.  
  </ol>  
</details>  
  
  



## NeRF  

### [PAV: Personalized Head Avatar from Unstructured Video Collection](http://arxiv.org/abs/2407.21047)  
Akin Caliskan, Berkay Kicanaoglu, Hyeongwoo Kim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose PAV, Personalized Head Avatar for the synthesis of human faces under arbitrary viewpoints and facial expressions. PAV introduces a method that learns a dynamic deformable neural radiance field (NeRF), in particular from a collection of monocular talking face videos of the same character under various appearance and shape changes. Unlike existing head NeRF methods that are limited to modeling such input videos on a per-appearance basis, our method allows for learning multi-appearance NeRFs, introducing appearance embedding for each input video via learnable latent neural features attached to the underlying geometry. Furthermore, the proposed appearance-conditioned density formulation facilitates the shape variation of the character, such as facial hair and soft tissues, in the radiance field prediction. To the best of our knowledge, our approach is the first dynamic deformable NeRF framework to model appearance and shape variations in a single unified network for multi-appearances of the same subject. We demonstrate experimentally that PAV outperforms the baseline method in terms of visual rendering quality in our quantitative and qualitative studies on various subjects.  
  </ol>  
</details>  
**comments**: Accepted to ECCV24. Project page:
  https://akincaliskan3d.github.io/PAV  
  
  



