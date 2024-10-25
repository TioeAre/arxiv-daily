<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Large-Spatial-Model:-End-to-end-Unposed-Images-to-Semantic-3D>Large Spatial Model: End-to-end Unposed Images to Semantic 3D</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#ChatSearch:-a-Dataset-and-a-Generative-Retrieval-Model-for-General-Conversational-Image-Retrieval>ChatSearch: a Dataset and a Generative Retrieval Model for General Conversational Image Retrieval</a></li>
        <li><a href=#On-Model-Free-Re-ranking-for-Visual-Place-Recognition-with-Deep-Learned-Local-Features>On Model-Free Re-ranking for Visual Place Recognition with Deep Learned Local Features</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Binocular-Guided-3D-Gaussian-Splatting-with-View-Consistency-for-Sparse-View-Synthesis>Binocular-Guided 3D Gaussian Splatting with View Consistency for Sparse View Synthesis</a></li>
        <li><a href=#Real-time-3D-aware-Portrait-Video-Relighting>Real-time 3D-aware Portrait Video Relighting</a></li>
        <li><a href=#Advancing-Super-Resolution-in-Neural-Radiance-Fields-via-Variational-Diffusion-Strategies>Advancing Super-Resolution in Neural Radiance Fields via Variational Diffusion Strategies</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Large Spatial Model: End-to-end Unposed Images to Semantic 3D](http://arxiv.org/abs/2410.18956)  
Zhiwen Fan, Jian Zhang, Wenyan Cong, Peihao Wang, Renjie Li, Kairun Wen, Shijie Zhou, Achuta Kadambi, Zhangyang Wang, Danfei Xu, Boris Ivanovic, Marco Pavone, Yue Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Reconstructing and understanding 3D structures from a limited number of images is a well-established problem in computer vision. Traditional methods usually break this task into multiple subtasks, each requiring complex transformations between different data representations. For instance, dense reconstruction through Structure-from-Motion (SfM) involves converting images into key points, optimizing camera parameters, and estimating structures. Afterward, accurate sparse reconstructions are required for further dense modeling, which is subsequently fed into task-specific neural networks. This multi-step process results in considerable processing time and increased engineering complexity.   In this work, we present the Large Spatial Model (LSM), which processes unposed RGB images directly into semantic radiance fields. LSM simultaneously estimates geometry, appearance, and semantics in a single feed-forward operation, and it can generate versatile label maps by interacting with language at novel viewpoints. Leveraging a Transformer-based architecture, LSM integrates global geometry through pixel-aligned point maps. To enhance spatial attribute regression, we incorporate local context aggregation with multi-scale fusion, improving the accuracy of fine local details. To tackle the scarcity of labeled 3D semantic data and enable natural language-driven scene manipulation, we incorporate a pre-trained 2D language-based segmentation model into a 3D-consistent semantic feature field. An efficient decoder then parameterizes a set of semantic anisotropic Gaussians, facilitating supervised end-to-end learning. Extensive experiments across various tasks show that LSM unifies multiple 3D vision tasks directly from unposed images, achieving real-time semantic 3D reconstruction for the first time.  
  </ol>  
</details>  
**comments**: Project Website: https://largespatialmodel.github.io  
  
  



## Visual Localization  

### [ChatSearch: a Dataset and a Generative Retrieval Model for General Conversational Image Retrieval](http://arxiv.org/abs/2410.18715)  
[[code](https://github.com/joez17/chatsearch)]  
Zijia Zhao, Longteng Guo, Tongtian Yue, Erdong Hu, Shuai Shao, Zehuan Yuan, Hua Huang, Jing Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we investigate the task of general conversational image retrieval on open-domain images. The objective is to search for images based on interactive conversations between humans and computers. To advance this task, we curate a dataset called ChatSearch. This dataset includes a multi-round multimodal conversational context query for each target image, thereby requiring the retrieval system to find the accurate image from database. Simultaneously, we propose a generative retrieval model named ChatSearcher, which is trained end-to-end to accept/produce interleaved image-text inputs/outputs. ChatSearcher exhibits strong capability in reasoning with multimodal context and can leverage world knowledge to yield visual retrieval results. It demonstrates superior performance on the ChatSearch dataset and also achieves competitive results on other image retrieval tasks and visual conversation tasks. We anticipate that this work will inspire further research on interactive multimodal retrieval systems. Our dataset will be available at https://github.com/joez17/ChatSearch.  
  </ol>  
</details>  
  
### [On Model-Free Re-ranking for Visual Place Recognition with Deep Learned Local Features](http://arxiv.org/abs/2410.18573)  
Tomáš Pivoňka, Libor Přeučil  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Re-ranking is the second stage of a visual place recognition task, in which the system chooses the best-matching images from a pre-selected subset of candidates. Model-free approaches compute the image pair similarity based on a spatial comparison of corresponding local visual features, eliminating the need for computationally expensive estimation of a model describing transformation between images. The article focuses on model-free re-ranking based on standard local visual features and their applicability in long-term autonomy systems. It introduces three new model-free re-ranking methods that were designed primarily for deep-learned local visual features. These features evince high robustness to various appearance changes, which stands as a crucial property for use with long-term autonomy systems. All the introduced methods were employed in a new visual place recognition system together with the D2-net feature detector (Dusmanu, 2019) and experimentally tested with diverse, challenging public datasets. The obtained results are on par with current state-of-the-art methods, affirming that model-free approaches are a viable and worthwhile path for long-term visual place recognition.  
  </ol>  
</details>  
**comments**: 12 pages, 9 figures  
  
  



## NeRF  

### [Binocular-Guided 3D Gaussian Splatting with View Consistency for Sparse View Synthesis](http://arxiv.org/abs/2410.18822)  
Liang Han, Junsheng Zhou, Yu-Shen Liu, Zhizhong Han  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Novel view synthesis from sparse inputs is a vital yet challenging task in 3D computer vision. Previous methods explore 3D Gaussian Splatting with neural priors (e.g. depth priors) as an additional supervision, demonstrating promising quality and efficiency compared to the NeRF based methods. However, the neural priors from 2D pretrained models are often noisy and blurry, which struggle to precisely guide the learning of radiance fields. In this paper, We propose a novel method for synthesizing novel views from sparse views with Gaussian Splatting that does not require external prior as supervision. Our key idea lies in exploring the self-supervisions inherent in the binocular stereo consistency between each pair of binocular images constructed with disparity-guided image warping. To this end, we additionally introduce a Gaussian opacity constraint which regularizes the Gaussian locations and avoids Gaussian redundancy for improving the robustness and efficiency of inferring 3D Gaussians from sparse views. Extensive experiments on the LLFF, DTU, and Blender datasets demonstrate that our method significantly outperforms the state-of-the-art methods.  
  </ol>  
</details>  
**comments**: Accepted by NeurIPS 2024. Project page:
  https://hanl2010.github.io/Binocular3DGS/  
  
### [Real-time 3D-aware Portrait Video Relighting](http://arxiv.org/abs/2410.18355)  
[[code](https://github.com/GhostCai/PortraitRelighting)]  
Ziqi Cai, Kaiwen Jiang, Shu-Yu Chen, Yu-Kun Lai, Hongbo Fu, Boxin Shi, Lin Gao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Synthesizing realistic videos of talking faces under custom lighting conditions and viewing angles benefits various downstream applications like video conferencing. However, most existing relighting methods are either time-consuming or unable to adjust the viewpoints. In this paper, we present the first real-time 3D-aware method for relighting in-the-wild videos of talking faces based on Neural Radiance Fields (NeRF). Given an input portrait video, our method can synthesize talking faces under both novel views and novel lighting conditions with a photo-realistic and disentangled 3D representation. Specifically, we infer an albedo tri-plane, as well as a shading tri-plane based on a desired lighting condition for each video frame with fast dual-encoders. We also leverage a temporal consistency network to ensure smooth transitions and reduce flickering artifacts. Our method runs at 32.98 fps on consumer-level hardware and achieves state-of-the-art results in terms of reconstruction quality, lighting error, lighting instability, temporal consistency and inference speed. We demonstrate the effectiveness and interactivity of our method on various portrait videos with diverse lighting and viewing conditions.  
  </ol>  
</details>  
**comments**: Accepted to CVPR 2024 (Highlight). Project page:
  http://geometrylearning.com/VideoRelighting  
  
### [Advancing Super-Resolution in Neural Radiance Fields via Variational Diffusion Strategies](http://arxiv.org/abs/2410.18137)  
[[code](https://github.com/shreyvish5678/SR-NeRF-with-Variational-Diffusion-Strategies)]  
Shrey Vishen, Jatin Sarabu, Chinmay Bharathulwar, Rithwick Lakshmanan, Vishnu Srinivas  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present a novel method for diffusion-guided frameworks for view-consistent super-resolution (SR) in neural rendering. Our approach leverages existing 2D SR models in conjunction with advanced techniques such as Variational Score Distilling (VSD) and a LoRA fine-tuning helper, with spatial training to significantly boost the quality and consistency of upscaled 2D images compared to the previous methods in the literature, such as Renoised Score Distillation (RSD) proposed in DiSR-NeRF (1), or SDS proposed in DreamFusion. The VSD score facilitates precise fine-tuning of SR models, resulting in high-quality, view-consistent images. To address the common challenge of inconsistencies among independent SR 2D images, we integrate Iterative 3D Synchronization (I3DS) from the DiSR-NeRF framework. Our quantitative benchmarks and qualitative results on the LLFF dataset demonstrate the superior performance of our system compared to existing methods such as DiSR-NeRF.  
  </ol>  
</details>  
**comments**: All our code is available at
  https://github.com/shreyvish5678/Advancing-Super-Resolution-in-Neural-Radiance-Fields-via-Variational-Diffusion-Strategies  
  
  



