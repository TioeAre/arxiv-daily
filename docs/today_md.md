<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#MonoGS++:-Fast-and-Accurate-Monocular-RGB-Gaussian-SLAM>MonoGS++: Fast and Accurate Monocular RGB Gaussian SLAM</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Re-thinking-Temporal-Search-for-Long-Form-Video-Understanding>Re-thinking Temporal Search for Long-Form Video Understanding</a></li>
        <li><a href=#A-Chefs-KISS----Utilizing-semantic-information-in-both-ICP-and-SLAM-framework>A Chefs KISS -- Utilizing semantic information in both ICP and SLAM framework</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#MultiNeRF:-Multiple-Watermark-Embedding-for-Neural-Radiance-Fields>MultiNeRF: Multiple Watermark Embedding for Neural Radiance Fields</a></li>
        <li><a href=#LPA3D:-3D-Room-Level-Scene-Generation-from-In-the-Wild-Images>LPA3D: 3D Room-Level Scene Generation from In-the-Wild Images</a></li>
        <li><a href=#OccludeNeRF:-Geometric-aware-3D-Scene-Inpainting-with-Collaborative-Score-Distillation-in-NeRF>OccludeNeRF: Geometric-aware 3D Scene Inpainting with Collaborative Score Distillation in NeRF</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [MonoGS++: Fast and Accurate Monocular RGB Gaussian SLAM](http://arxiv.org/abs/2504.02437)  
Renwu Li, Wenjing Ke, Dong Li, Lu Tian, Emad Barsoum  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present MonoGS++, a novel fast and accurate Simultaneous Localization and Mapping (SLAM) method that leverages 3D Gaussian representations and operates solely on RGB inputs. While previous 3D Gaussian Splatting (GS)-based methods largely depended on depth sensors, our approach reduces the hardware dependency and only requires RGB input, leveraging online visual odometry (VO) to generate sparse point clouds in real-time. To reduce redundancy and enhance the quality of 3D scene reconstruction, we implemented a series of methodological enhancements in 3D Gaussian mapping. Firstly, we introduced dynamic 3D Gaussian insertion to avoid adding redundant Gaussians in previously well-reconstructed areas. Secondly, we introduced clarity-enhancing Gaussian densification module and planar regularization to handle texture-less areas and flat surfaces better. We achieved precise camera tracking results both on the synthetic Replica and real-world TUM-RGBD datasets, comparable to those of the state-of-the-art. Additionally, our method realized a significant 5.57x improvement in frames per second (fps) over the previous state-of-the-art, MonoGS.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Re-thinking Temporal Search for Long-Form Video Understanding](http://arxiv.org/abs/2504.02259)  
Jinhui Ye, Zihan Wang, Haosen Sun, Keshigeyan Chandrasegaran, Zane Durante, Cristobal Eyzaguirre, Yonatan Bisk, Juan Carlos Niebles, Ehsan Adeli, Li Fei-Fei, Jiajun Wu, Manling Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Efficient understanding of long-form videos remains a significant challenge in computer vision. In this work, we revisit temporal search paradigms for long-form video understanding, studying a fundamental issue pertaining to all state-of-the-art (SOTA) long-context vision-language models (VLMs). In particular, our contributions are two-fold: First, we formulate temporal search as a Long Video Haystack problem, i.e., finding a minimal set of relevant frames (typically one to five) among tens of thousands of frames from real-world long videos given specific queries. To validate our formulation, we create LV-Haystack, the first benchmark containing 3,874 human-annotated instances with fine-grained evaluation metrics for assessing keyframe search quality and computational efficiency. Experimental results on LV-Haystack highlight a significant research gap in temporal search capabilities, with SOTA keyframe selection methods achieving only 2.1% temporal F1 score on the LVBench subset.   Next, inspired by visual search in images, we re-think temporal searching and propose a lightweight keyframe searching framework, T*, which casts the expensive temporal search as a spatial search problem. T* leverages superior visual localization capabilities typically used in images and introduces an adaptive zooming-in mechanism that operates across both temporal and spatial dimensions. Our extensive experiments show that when integrated with existing methods, T* significantly improves SOTA long-form video understanding performance. Specifically, under an inference budget of 32 frames, T* improves GPT-4o's performance from 50.5% to 53.1% and LLaVA-OneVision-72B's performance from 56.5% to 62.4% on LongVideoBench XL subset. Our PyTorch code, benchmark dataset and models are included in the Supplementary material.  
  </ol>  
</details>  
**comments**: Accepted by CVPR 2025; A real-world long video needle-in-haystack
  benchmark; long-video QA with human ref frames  
  
### [A Chefs KISS -- Utilizing semantic information in both ICP and SLAM framework](http://arxiv.org/abs/2504.02086)  
Sven Ochs, Marc Heinrich, Philip Schörner, Marc René Zofka, J. Marius Zöllner  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    For utilizing autonomous vehicle in urban areas a reliable localization is needed. Especially when HD maps are used, a precise and repeatable method has to be chosen. Therefore accurate map generation but also re-localization against these maps is necessary. Due to best 3D reconstruction of the surrounding, LiDAR has become a reliable modality for localization. The latest LiDAR odometry estimation are based on iterative closest point (ICP) approaches, namely KISS-ICP and SAGE-ICP. We extend the capabilities of KISS-ICP by incorporating semantic information into the point alignment process using a generalizable approach with minimal parameter tuning. This enhancement allows us to surpass KISS-ICP in terms of absolute trajectory error (ATE), the primary metric for map accuracy. Additionally, we improve the Cartographer mapping framework to handle semantic information. Cartographer facilitates loop closure detection over larger areas, mitigating odometry drift and further enhancing ATE accuracy. By integrating semantic information into the mapping process, we enable the filtering of specific classes, such as parked vehicles, from the resulting map. This filtering improves relocalization quality by addressing temporal changes, such as vehicles being moved.  
  </ol>  
</details>  
  
  



## NeRF  

### [MultiNeRF: Multiple Watermark Embedding for Neural Radiance Fields](http://arxiv.org/abs/2504.02517)  
Yash Kulthe, Andrew Gilbert, John Collomosse  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present MultiNeRF, a 3D watermarking method that embeds multiple uniquely keyed watermarks within images rendered by a single Neural Radiance Field (NeRF) model, whilst maintaining high visual quality. Our approach extends the TensoRF NeRF model by incorporating a dedicated watermark grid alongside the existing geometry and appearance grids. This extension ensures higher watermark capacity without entangling watermark signals with scene content. We propose a FiLM-based conditional modulation mechanism that dynamically activates watermarks based on input identifiers, allowing multiple independent watermarks to be embedded and extracted without requiring model retraining. MultiNeRF is validated on the NeRF-Synthetic and LLFF datasets, with statistically significant improvements in robust capacity without compromising rendering quality. By generalizing single-watermark NeRF methods into a flexible multi-watermarking framework, MultiNeRF provides a scalable solution for 3D content. attribution.  
  </ol>  
</details>  
  
### [LPA3D: 3D Room-Level Scene Generation from In-the-Wild Images](http://arxiv.org/abs/2504.02337)  
Ming-Jia Yang, Yu-Xiao Guo, Yang Liu, Bin Zhou, Xin Tong  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Generating realistic, room-level indoor scenes with semantically plausible and detailed appearances from in-the-wild images is crucial for various applications in VR, AR, and robotics. The success of NeRF-based generative methods indicates a promising direction to address this challenge. However, unlike their success at the object level, existing scene-level generative methods require additional information, such as multiple views, depth images, or semantic guidance, rather than relying solely on RGB images. This is because NeRF-based methods necessitate prior knowledge of camera poses, which is challenging to approximate for indoor scenes due to the complexity of defining alignment and the difficulty of globally estimating poses from a single image, given the unseen parts behind the camera. To address this challenge, we redefine global poses within the framework of Local-Pose-Alignment (LPA) -- an anchor-based multi-local-coordinate system that uses a selected number of anchors as the roots of these coordinates. Building on this foundation, we introduce LPA-GAN, a novel NeRF-based generative approach that incorporates specific modifications to estimate the priors of camera poses under LPA. It also co-optimizes the pose predictor and scene generation processes. Our ablation study and comparisons with straightforward extensions of NeRF-based object generative methods demonstrate the effectiveness of our approach. Furthermore, visual comparisons with other techniques reveal that our method achieves superior view-to-view consistency and semantic normality.  
  </ol>  
</details>  
  
### [OccludeNeRF: Geometric-aware 3D Scene Inpainting with Collaborative Score Distillation in NeRF](http://arxiv.org/abs/2504.02007)  
Jingyu Shi, Achleshwar Luthra, Jiazhi Li, Xiang Gao, Xiyun Song, Zongfang Lin, David Gu, Heather Yu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    With Neural Radiance Fields (NeRFs) arising as a powerful 3D representation, research has investigated its various downstream tasks, including inpainting NeRFs with 2D images. Despite successful efforts addressing the view consistency and geometry quality, prior methods yet suffer from occlusion in NeRF inpainting tasks, where 2D prior is severely limited in forming a faithful reconstruction of the scene to inpaint.   To address this, we propose a novel approach that enables cross-view information sharing during knowledge distillation from a diffusion model, effectively propagating occluded information across limited views. Additionally, to align the distillation direction across multiple sampled views, we apply a grid-based denoising strategy and incorporate additional rendered views to enhance cross-view consistency. To assess our approach's capability of handling occlusion cases, we construct a dataset consisting of challenging scenes with severe occlusion, in addition to existing datasets. Compared with baseline methods, our method demonstrates better performance in cross-view consistency and faithfulness in reconstruction, while preserving high rendering quality and fidelity.  
  </ol>  
</details>  
**comments**: CVPR 2025 CV4Metaverse  
  
  



