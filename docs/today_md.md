<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#On-Unifying-Video-Generation-and-Camera-Pose-Estimation>On Unifying Video Generation and Camera Pose Estimation</a></li>
        <li><a href=#EasySplat:-View-Adaptive-Learning-makes-3D-Gaussian-Splatting-Easy>EasySplat: View-Adaptive Learning makes 3D Gaussian Splatting Easy</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#R-SCoRe:-Revisiting-Scene-Coordinate-Regression-for-Robust-Large-Scale-Visual-Localization>R-SCoRe: Revisiting Scene Coordinate Regression for Robust Large-Scale Visual Localization</a></li>
        <li><a href=#Training-Medical-Large-Vision-Language-Models-with-Abnormal-Aware-Feedback>Training Medical Large Vision-Language Models with Abnormal-Aware Feedback</a></li>
        <li><a href=#Domain-invariant-feature-learning-in-brain-MR-imaging-for-content-based-image-retrieval>Domain-invariant feature learning in brain MR imaging for content-based image retrieval</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Sparis:-Neural-Implicit-Surface-Reconstruction-of-Indoor-Scenes-from-Sparse-Views>Sparis: Neural Implicit Surface Reconstruction of Indoor Scenes from Sparse Views</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [On Unifying Video Generation and Camera Pose Estimation](http://arxiv.org/abs/2501.01409)  
Chun-Hao Paul Huang, Jae Shin Yoon, Hyeonho Jeong, Niloy Mitra, Duygu Ceylan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Inspired by the emergent 3D capabilities in image generators, we explore whether video generators similarly exhibit 3D awareness. Using structure-from-motion (SfM) as a benchmark for 3D tasks, we investigate if intermediate features from OpenSora, a video generation model, can support camera pose estimation. We first examine native 3D awareness in video generation features by routing raw intermediate outputs to SfM-prediction modules like DUSt3R. Then, we explore the impact of fine-tuning on camera pose estimation to enhance 3D awareness. Results indicate that while video generator features have limited inherent 3D awareness, task-specific supervision significantly boosts their accuracy for camera pose estimation, resulting in competitive performance. The proposed unified model, named JOG3R, produces camera pose estimates with competitive quality without degrading video generation quality.  
  </ol>  
</details>  
  
### [EasySplat: View-Adaptive Learning makes 3D Gaussian Splatting Easy](http://arxiv.org/abs/2501.01003)  
Ao Gao, Luosong Guo, Tao Chen, Zhao Wang, Ying Tai, Jian Yang, Zhenyu Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D Gaussian Splatting (3DGS) techniques have achieved satisfactory 3D scene representation. Despite their impressive performance, they confront challenges due to the limitation of structure-from-motion (SfM) methods on acquiring accurate scene initialization, or the inefficiency of densification strategy. In this paper, we introduce a novel framework EasySplat to achieve high-quality 3DGS modeling. Instead of using SfM for scene initialization, we employ a novel method to release the power of large-scale pointmap approaches. Specifically, we propose an efficient grouping strategy based on view similarity, and use robust pointmap priors to obtain high-quality point clouds and camera poses for 3D scene initialization. After obtaining a reliable scene structure, we propose a novel densification approach that adaptively splits Gaussian primitives based on the average shape of neighboring Gaussian ellipsoids, utilizing KNN scheme. In this way, the proposed method tackles the limitation on initialization and optimization, leading to an efficient and accurate 3DGS modeling. Extensive experiments demonstrate that EasySplat outperforms the current state-of-the-art (SOTA) in handling novel view synthesis.  
  </ol>  
</details>  
**comments**: 6 pages, 5figures  
  
  



## Visual Localization  

### [R-SCoRe: Revisiting Scene Coordinate Regression for Robust Large-Scale Visual Localization](http://arxiv.org/abs/2501.01421)  
Xudong Jiang, Fangjinhua Wang, Silvano Galliani, Christoph Vogel, Marc Pollefeys  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Learning-based visual localization methods that use scene coordinate regression (SCR) offer the advantage of smaller map sizes. However, on datasets with complex illumination changes or image-level ambiguities, it remains a less robust alternative to feature matching methods. This work aims to close the gap. We introduce a covisibility graph-based global encoding learning and data augmentation strategy, along with a depth-adjusted reprojection loss to facilitate implicit triangulation. Additionally, we revisit the network architecture and local feature extraction module. Our method achieves state-of-the-art on challenging large-scale datasets without relying on network ensembles or 3D supervision. On Aachen Day-Night, we are 10 $\times$ more accurate than previous SCR methods with similar map sizes and require at least 5$\times$ smaller map sizes than any other SCR method while still delivering superior accuracy. Code will be available at: https://github.com/cvg/scrstudio .  
  </ol>  
</details>  
**comments**: Code: https://github.com/cvg/scrstudio  
  
### [Training Medical Large Vision-Language Models with Abnormal-Aware Feedback](http://arxiv.org/abs/2501.01377)  
Yucheng Zhou, Lingran Song, Jianbing Shen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Existing Medical Large Vision-Language Models (Med-LVLMs), which encapsulate extensive medical knowledge, demonstrate excellent capabilities in understanding medical images and responding to human queries based on these images. However, there remain challenges in visual localization in medical images, which is crucial for abnormality detection and interpretation. To address these issues, we propose a novel UMed-LVLM designed with Unveiling Medical abnormalities. Specifically, we collect a Medical Abnormalities Unveiling (MAU) dataset and propose a two-stage training method for UMed-LVLM training. To collect MAU dataset, we propose a prompt method utilizing the GPT-4V to generate diagnoses based on identified abnormal areas in medical images. Moreover, the two-stage training method includes Abnormal-Aware Instruction Tuning and Abnormal-Aware Rewarding, comprising Abnormal Localization Rewarding and Vision Relevance Rewarding. Experimental results demonstrate that our UMed-LVLM surpasses existing Med-LVLMs in identifying and understanding medical abnormality. In addition, this work shows that enhancing the abnormality detection capabilities of Med-LVLMs significantly improves their understanding of medical images and generalization capability.  
  </ol>  
</details>  
**comments**: 16 pages  
  
### [Domain-invariant feature learning in brain MR imaging for content-based image retrieval](http://arxiv.org/abs/2501.01326)  
Shuya Tobari, Shuhei Tomoshige, Hayato Muraki, Kenichi Oishi, Hitoshi Iyatomi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    When conducting large-scale studies that collect brain MR images from multiple facilities, the impact of differences in imaging equipment and protocols at each site cannot be ignored, and this domain gap has become a significant issue in recent years. In this study, we propose a new low-dimensional representation (LDR) acquisition method called style encoder adversarial domain adaptation (SE-ADA) to realize content-based image retrieval (CBIR) of brain MR images. SE-ADA reduces domain differences while preserving pathological features by separating domain-specific information from LDR and minimizing domain differences using adversarial learning.   In evaluation experiments comparing SE-ADA with recent domain harmonization methods on eight public brain MR datasets (ADNI1/2/3, OASIS1/2/3/4, PPMI), SE-ADA effectively removed domain information while preserving key aspects of the original brain structure and demonstrated the highest disease search accuracy.  
  </ol>  
</details>  
**comments**: 6 pages, 1 figures. Accepted at the SPIE Medical Imaging 2025  
  
  



## Image Matching  

### [Sparis: Neural Implicit Surface Reconstruction of Indoor Scenes from Sparse Views](http://arxiv.org/abs/2501.01196)  
Yulun Wu, Han Huang, Wenyuan Zhang, Chao Deng, Ge Gao, Ming Gu, Yu-Shen Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In recent years, reconstructing indoor scene geometry from multi-view images has achieved encouraging accomplishments. Current methods incorporate monocular priors into neural implicit surface models to achieve high-quality reconstructions. However, these methods require hundreds of images for scene reconstruction. When only a limited number of views are available as input, the performance of monocular priors deteriorates due to scale ambiguity, leading to the collapse of the reconstructed scene geometry. In this paper, we propose a new method, named Sparis, for indoor surface reconstruction from sparse views. Specifically, we investigate the impact of monocular priors on sparse scene reconstruction, introducing a novel prior based on inter-image matching information. Our prior offers more accurate depth information while ensuring cross-view matching consistency. Additionally, we employ an angular filter strategy and an epipolar matching weight function, aiming to reduce errors due to view matching inaccuracies, thereby refining the inter-image prior for improved reconstruction accuracy. The experiments conducted on widely used benchmarks demonstrate superior performance in sparse-view scene reconstruction.  
  </ol>  
</details>  
**comments**: Accepted by AAAI 2025. Project page:
  https://yulunwu0108.github.io/Sparis/  
  
  



