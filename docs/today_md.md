<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Sparse2DGS:-Sparse-View-Surface-Reconstruction-using-2D-Gaussian-Splatting-with-Dense-Point-Cloud>Sparse2DGS: Sparse-View Surface Reconstruction using 2D Gaussian Splatting with Dense Point Cloud</a></li>
        <li><a href=#Improving-Novel-view-synthesis-of-360$^\circ$-Scenes-in-Extremely-Sparse-Views-by-Jointly-Training-Hemisphere-Sampled-Synthetic-Images>Improving Novel view synthesis of 360$^\circ$ Scenes in Extremely Sparse Views by Jointly Training Hemisphere Sampled Synthetic Images</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Multimodal-Reasoning-Agent-for-Zero-Shot-Composed-Image-Retrieval>Multimodal Reasoning Agent for Zero-Shot Composed Image Retrieval</a></li>
        <li><a href=#Can-Visual-Encoder-Learn-to-See-Arrows?>Can Visual Encoder Learn to See Arrows?</a></li>
        <li><a href=#MLLM-Guided-VLM-Fine-Tuning-with-Joint-Inference-for-Zero-Shot-Composed-Image-Retrieval>MLLM-Guided VLM Fine-Tuning with Joint Inference for Zero-Shot Composed Image Retrieval</a></li>
        <li><a href=#Why-Not-Replace?-Sustaining-Long-Term-Visual-Localization-via-Handcrafted-Learned-Feature-Collaboration-on-CPU>Why Not Replace? Sustaining Long-Term Visual Localization via Handcrafted-Learned Feature Collaboration on CPU</a></li>
        <li><a href=#TNG-CLIP:Training-Time-Negation-Data-Generation-for-Negation-Awareness-of-CLIP>TNG-CLIP:Training-Time Negation Data Generation for Negation Awareness of CLIP</a></li>
        <li><a href=#ImLPR:-Image-based-LiDAR-Place-Recognition-using-Vision-Foundation-Models>ImLPR: Image-based LiDAR Place Recognition using Vision Foundation Models</a></li>
        <li><a href=#DART$^3$:-Leveraging-Distance-for-Test-Time-Adaptation-in-Person-Re-Identification>DART$^3$: Leveraging Distance for Test Time Adaptation in Person Re-Identification</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Why-Not-Replace?-Sustaining-Long-Term-Visual-Localization-via-Handcrafted-Learned-Feature-Collaboration-on-CPU>Why Not Replace? Sustaining Long-Term Visual Localization via Handcrafted-Learned Feature Collaboration on CPU</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#OB3D:-A-New-Dataset-for-Benchmarking-Omnidirectional-3D-Reconstruction-Using-Blender>OB3D: A New Dataset for Benchmarking Omnidirectional 3D Reconstruction Using Blender</a></li>
        <li><a href=#ErpGS:-Equirectangular-Image-Rendering-enhanced-with-3D-Gaussian-Regularization>ErpGS: Equirectangular Image Rendering enhanced with 3D Gaussian Regularization</a></li>
        <li><a href=#GoLF-NRT:-Integrating-Global-Context-and-Local-Geometry-for-Few-Shot-View-Synthesis>GoLF-NRT: Integrating Global Context and Local Geometry for Few-Shot View Synthesis</a></li>
        <li><a href=#Depth-Guided-Bundle-Sampling-for-Efficient-Generalizable-Neural-Radiance-Field-Reconstruction>Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction</a></li>
        <li><a href=#ADD-SLAM:-Adaptive-Dynamic-Dense-SLAM-with-Gaussian-Splatting>ADD-SLAM: Adaptive Dynamic Dense SLAM with Gaussian Splatting</a></li>
        <li><a href=#Triangle-Splatting-for-Real-Time-Radiance-Field-Rendering>Triangle Splatting for Real-Time Radiance Field Rendering</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Sparse2DGS: Sparse-View Surface Reconstruction using 2D Gaussian Splatting with Dense Point Cloud](http://arxiv.org/abs/2505.19854)  
Natsuki Takama, Shintaro Ito, Koichi Ito, Hwann-Tzong Chen, Takafumi Aoki  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Gaussian Splatting (GS) has gained attention as a fast and effective method for novel view synthesis. It has also been applied to 3D reconstruction using multi-view images and can achieve fast and accurate 3D reconstruction. However, GS assumes that the input contains a large number of multi-view images, and therefore, the reconstruction accuracy significantly decreases when only a limited number of input images are available. One of the main reasons is the insufficient number of 3D points in the sparse point cloud obtained through Structure from Motion (SfM), which results in a poor initialization for optimizing the Gaussian primitives. We propose a new 3D reconstruction method, called Sparse2DGS, to enhance 2DGS in reconstructing objects using only three images. Sparse2DGS employs DUSt3R, a fundamental model for stereo images, along with COLMAP MVS to generate highly accurate and dense 3D point clouds, which are then used to initialize 2D Gaussians. Through experiments on the DTU dataset, we show that Sparse2DGS can accurately reconstruct the 3D shapes of objects using just three images.  
  </ol>  
</details>  
**comments**: Accepted to ICIP 2025  
  
### [Improving Novel view synthesis of 360 $^\circ$ Scenes in Extremely Sparse Views by Jointly Training Hemisphere Sampled Synthetic Images](http://arxiv.org/abs/2505.19264)  
Guangan Chen, Anh Minh Truong, Hanhe Lin, Michiel Vlaminck, Wilfried Philips, Hiep Luong  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Novel view synthesis in 360$^\circ$ scenes from extremely sparse input views is essential for applications like virtual reality and augmented reality. This paper presents a novel framework for novel view synthesis in extremely sparse-view cases. As typical structure-from-motion methods are unable to estimate camera poses in extremely sparse-view cases, we apply DUSt3R to estimate camera poses and generate a dense point cloud. Using the poses of estimated cameras, we densely sample additional views from the upper hemisphere space of the scenes, from which we render synthetic images together with the point cloud. Training 3D Gaussian Splatting model on a combination of reference images from sparse views and densely sampled synthetic images allows a larger scene coverage in 3D space, addressing the overfitting challenge due to the limited input in sparse-view cases. Retraining a diffusion-based image enhancement model on our created dataset, we further improve the quality of the point-cloud-rendered images by removing artifacts. We compare our framework with benchmark methods in cases of only four input views, demonstrating significant improvement in novel view synthesis under extremely sparse-view conditions for 360$^\circ$ scenes.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Multimodal Reasoning Agent for Zero-Shot Composed Image Retrieval](http://arxiv.org/abs/2505.19952)  
Rong-Cheng Tu, Wenhao Sun, Hanzhe You, Yingjie Wang, Jiaxing Huang, Li Shen, Dacheng Tao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Zero-Shot Composed Image Retrieval (ZS-CIR) aims to retrieve target images given a compositional query, consisting of a reference image and a modifying text-without relying on annotated training data. Existing approaches often generate a synthetic target text using large language models (LLMs) to serve as an intermediate anchor between the compositional query and the target image. Models are then trained to align the compositional query with the generated text, and separately align images with their corresponding texts using contrastive learning. However, this reliance on intermediate text introduces error propagation, as inaccuracies in query-to-text and text-to-image mappings accumulate, ultimately degrading retrieval performance. To address these problems, we propose a novel framework by employing a Multimodal Reasoning Agent (MRA) for ZS-CIR. MRA eliminates the dependence on textual intermediaries by directly constructing triplets, <reference image, modification text, target image>, using only unlabeled image data. By training on these synthetic triplets, our model learns to capture the relationships between compositional queries and candidate images directly. Extensive experiments on three standard CIR benchmarks demonstrate the effectiveness of our approach. On the FashionIQ dataset, our method improves Average R@10 by at least 7.5\% over existing baselines; on CIRR, it boosts R@1 by 9.6\%; and on CIRCO, it increases mAP@5 by 9.5\%.  
  </ol>  
</details>  
  
### [Can Visual Encoder Learn to See Arrows?](http://arxiv.org/abs/2505.19944)  
Naoyuki Terashita, Yusuke Tozaki, Hideaki Omote, Congkha Nguyen, Ryosuke Nakamoto, Yuta Koreeda, Hiroaki Ozaki  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The diagram is a visual representation of a relationship illustrated with edges (lines or arrows), which is widely used in industrial and scientific communication. Although recognizing diagrams is essential for vision language models (VLMs) to comprehend domain-specific knowledge, recent studies reveal that many VLMs fail to identify edges in images. We hypothesize that these failures stem from an over-reliance on textual and positional biases, preventing VLMs from learning explicit edge features. Based on this idea, we empirically investigate whether the image encoder in VLMs can learn edge representation through training on a diagram dataset in which edges are biased neither by textual nor positional information. To this end, we conduct contrastive learning on an artificially generated diagram--caption dataset to train an image encoder and evaluate its diagram-related features on three tasks: probing, image retrieval, and captioning. Our results show that the finetuned model outperforms pretrained CLIP in all tasks and surpasses zero-shot GPT-4o and LLaVA-Mistral in the captioning task. These findings confirm that eliminating textual and positional biases fosters accurate edge recognition in VLMs, offering a promising path for advancing diagram understanding.  
  </ol>  
</details>  
**comments**: This work has been accepted for poster presentation at the Second
  Workshop on Visual Concepts in CVPR 2025  
  
### [MLLM-Guided VLM Fine-Tuning with Joint Inference for Zero-Shot Composed Image Retrieval](http://arxiv.org/abs/2505.19707)  
Rong-Cheng Tu, Zhao Jin, Jingyi Liao, Xiao Luo, Yingjie Wang, Li Shen, Dacheng Tao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Existing Zero-Shot Composed Image Retrieval (ZS-CIR) methods typically train adapters that convert reference images into pseudo-text tokens, which are concatenated with the modifying text and processed by frozen text encoders in pretrained VLMs or LLMs. While this design leverages the strengths of large pretrained models, it only supervises the adapter to produce encoder-compatible tokens that loosely preserve visual semantics. Crucially, it does not directly optimize the composed query representation to capture the full intent of the composition or to align with the target semantics, thereby limiting retrieval performance, particularly in cases involving fine-grained or complex visual transformations. To address this problem, we propose MLLM-Guided VLM Fine-Tuning with Joint Inference (MVFT-JI), a novel approach that leverages a pretrained multimodal large language model (MLLM) to construct two complementary training tasks using only unlabeled images: target text retrieval taskand text-to-image retrieval task. By jointly optimizing these tasks, our method enables the VLM to inherently acquire robust compositional retrieval capabilities, supported by the provided theoretical justifications and empirical validation. Furthermore, during inference, we further prompt the MLLM to generate target texts from composed queries and compute retrieval scores by integrating similarities between (i) the composed query and candidate images, and (ii) the MLLM-generated target text and candidate images. This strategy effectively combines the VLM's semantic alignment strengths with the MLLM's reasoning capabilities.  
  </ol>  
</details>  
  
### [Why Not Replace? Sustaining Long-Term Visual Localization via Handcrafted-Learned Feature Collaboration on CPU](http://arxiv.org/abs/2505.18652)  
Yicheng Lin, Yunlong Jiang, Xujia Jiao, Bin Han  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Robust long-term visual localization in complex industrial environments is critical for mobile robotic systems. Existing approaches face limitations: handcrafted features are illumination-sensitive, learned features are computationally intensive, and semantic- or marker-based methods are environmentally constrained. Handcrafted and learned features share similar representations but differ functionally. Handcrafted features are optimized for continuous tracking, while learned features excel in wide-baseline matching. Their complementarity calls for integration rather than replacement. Building on this, we propose a hierarchical localization framework. It leverages real-time handcrafted feature extraction for relative pose estimation. In parallel, it employs selective learned keypoint detection on optimized keyframes for absolute positioning. This design enables CPU-efficient, long-term visual localization. Experiments systematically progress through three validation phases: Initially establishing feature complementarity through comparative analysis, followed by computational latency profiling across algorithm stages on CPU platforms. Final evaluation under photometric variations (including seasonal transitions and diurnal cycles) demonstrates 47% average error reduction with significantly improved localization consistency. The code implementation is publicly available at https://github.com/linyicheng1/ORB_SLAM3_localization.  
  </ol>  
</details>  
**comments**: 8 pages, 6 gifures  
  
### [TNG-CLIP:Training-Time Negation Data Generation for Negation Awareness of CLIP](http://arxiv.org/abs/2505.18434)  
Yuliang Cai, Jesse Thomason, Mohammad Rostami  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Vision-language models (VLMs), such as CLIP, have demonstrated strong performance across a range of downstream tasks. However, CLIP is still limited in negation understanding: the ability to recognize the absence or exclusion of a concept. Existing methods address the problem by using a large language model (LLM) to generate large-scale data of image captions containing negation for further fine-tuning CLIP. However, these methods are both time- and compute-intensive, and their evaluations are typically restricted to image-text matching tasks. To expand the horizon, we (1) introduce a training-time negation data generation pipeline such that negation captions are generated during the training stage, which only increases 2.5% extra training time, and (2) we propose the first benchmark, Neg-TtoI, for evaluating text-to-image generation models on prompts containing negation, assessing model's ability to produce semantically accurate images. We show that our proposed method, TNG-CLIP, achieves SOTA performance on diverse negation benchmarks of image-to-text matching, text-to-image retrieval, and image generation.  
  </ol>  
</details>  
**comments**: 15 pages, 3 figures  
  
### [ImLPR: Image-based LiDAR Place Recognition using Vision Foundation Models](http://arxiv.org/abs/2505.18364)  
Minwoo Jung, Lanke Frank Tarimo Fu, Maurice Fallon, Ayoung Kim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    LiDAR Place Recognition (LPR) is a key component in robotic localization, enabling robots to align current scans with prior maps of their environment. While Visual Place Recognition (VPR) has embraced Vision Foundation Models (VFMs) to enhance descriptor robustness, LPR has relied on task-specific models with limited use of pre-trained foundation-level knowledge. This is due to the lack of 3D foundation models and the challenges of using VFM with LiDAR point clouds. To tackle this, we introduce ImLPR, a novel pipeline that employs a pre-trained DINOv2 VFM to generate rich descriptors for LPR. To our knowledge, ImLPR is the first method to leverage a VFM to support LPR. ImLPR converts raw point clouds into Range Image Views (RIV) to leverage VFM in the LiDAR domain. It employs MultiConv adapters and Patch-InfoNCE loss for effective feature learning. We validate ImLPR using public datasets where it outperforms state-of-the-art (SOTA) methods in intra-session and inter-session LPR with top Recall@1 and F1 scores across various LiDARs. We also demonstrate that RIV outperforms Bird's-Eye-View (BEV) as a representation choice for adapting LiDAR for VFM. We release ImLPR as open source for the robotics community.  
  </ol>  
</details>  
**comments**: Technical report, 22 Pages, 13 Figures and 12 Tables  
  
### [DART $^3$ : Leveraging Distance for Test Time Adaptation in Person Re-Identification](http://arxiv.org/abs/2505.18337)  
Rajarshi Bhattacharya, Shakeeb Murtaza, Christian Desrosiers, Jose Dolz, Maguelonne Heritier, Eric Granger  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Person re-identification (ReID) models are known to suffer from camera bias, where learned representations cluster according to camera viewpoints rather than identity, leading to significant performance degradation under (inter-camera) domain shifts in real-world surveillance systems when new cameras are added to camera networks. State-of-the-art test-time adaptation (TTA) methods, largely designed for classification tasks, rely on classification entropy-based objectives that fail to generalize well to ReID, thus making them unsuitable for tackling camera bias. In this paper, we introduce DART$^3$, a TTA framework specifically designed to mitigate camera-induced domain shifts in person ReID. DART$^3$ (Distance-Aware Retrieval Tuning at Test Time) leverages a distance-based objective that aligns better with image retrieval tasks like ReID by exploiting the correlation between nearest-neighbor distance and prediction error. Unlike prior ReID-specific domain adaptation methods, DART$^3$ requires no source data, architectural modifications, or retraining, and can be deployed in both fully black-box and hybrid settings. Empirical evaluations on multiple ReID benchmarks indicate that DART$^3$ and DART$^3$ LITE, a lightweight alternative to the approach, consistently outperforms state-of-the-art TTA baselines, making for a viable option to online learning to mitigate the adverse effects of camera bias.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Why Not Replace? Sustaining Long-Term Visual Localization via Handcrafted-Learned Feature Collaboration on CPU](http://arxiv.org/abs/2505.18652)  
Yicheng Lin, Yunlong Jiang, Xujia Jiao, Bin Han  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Robust long-term visual localization in complex industrial environments is critical for mobile robotic systems. Existing approaches face limitations: handcrafted features are illumination-sensitive, learned features are computationally intensive, and semantic- or marker-based methods are environmentally constrained. Handcrafted and learned features share similar representations but differ functionally. Handcrafted features are optimized for continuous tracking, while learned features excel in wide-baseline matching. Their complementarity calls for integration rather than replacement. Building on this, we propose a hierarchical localization framework. It leverages real-time handcrafted feature extraction for relative pose estimation. In parallel, it employs selective learned keypoint detection on optimized keyframes for absolute positioning. This design enables CPU-efficient, long-term visual localization. Experiments systematically progress through three validation phases: Initially establishing feature complementarity through comparative analysis, followed by computational latency profiling across algorithm stages on CPU platforms. Final evaluation under photometric variations (including seasonal transitions and diurnal cycles) demonstrates 47% average error reduction with significantly improved localization consistency. The code implementation is publicly available at https://github.com/linyicheng1/ORB_SLAM3_localization.  
  </ol>  
</details>  
**comments**: 8 pages, 6 gifures  
  
  



## NeRF  

### [OB3D: A New Dataset for Benchmarking Omnidirectional 3D Reconstruction Using Blender](http://arxiv.org/abs/2505.20126)  
Shintaro Ito, Natsuki Takama, Toshiki Watanabe, Koichi Ito, Hwann-Tzong Chen, Takafumi Aoki  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in radiance field rendering, exemplified by Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), have significantly progressed 3D modeling and reconstruction. The use of multiple 360-degree omnidirectional images for these tasks is increasingly favored due to advantages in data acquisition and comprehensive scene capture. However, the inherent geometric distortions in common omnidirectional representations, such as equirectangular projection (particularly severe in polar regions and varying with latitude), pose substantial challenges to achieving high-fidelity 3D reconstructions. Current datasets, while valuable, often lack the specific focus, scene composition, and ground truth granularity required to systematically benchmark and drive progress in overcoming these omnidirectional-specific challenges. To address this critical gap, we introduce Omnidirectional Blender 3D (OB3D), a new synthetic dataset curated for advancing 3D reconstruction from multiple omnidirectional images. OB3D features diverse and complex 3D scenes generated from Blender 3D projects, with a deliberate emphasis on challenging scenarios. The dataset provides comprehensive ground truth, including omnidirectional RGB images, precise omnidirectional camera parameters, and pixel-aligned equirectangular maps for depth and normals, alongside evaluation metrics. By offering a controlled yet challenging environment, OB3Daims to facilitate the rigorous evaluation of existing methods and prompt the development of new techniques to enhance the accuracy and reliability of 3D reconstruction from omnidirectional images.  
  </ol>  
</details>  
  
### [ErpGS: Equirectangular Image Rendering enhanced with 3D Gaussian Regularization](http://arxiv.org/abs/2505.19883)  
Shintaro Ito, Natsuki Takama, Koichi Ito, Hwann-Tzong Chen, Takafumi Aoki  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The use of multi-view images acquired by a 360-degree camera can reconstruct a 3D space with a wide area. There are 3D reconstruction methods from equirectangular images based on NeRF and 3DGS, as well as Novel View Synthesis (NVS) methods. On the other hand, it is necessary to overcome the large distortion caused by the projection model of a 360-degree camera when equirectangular images are used. In 3DGS-based methods, the large distortion of the 360-degree camera model generates extremely large 3D Gaussians, resulting in poor rendering accuracy. We propose ErpGS, which is Omnidirectional GS based on 3DGS to realize NVS addressing the problems. ErpGS introduce some rendering accuracy improvement techniques: geometric regularization, scale regularization, and distortion-aware weights and a mask to suppress the effects of obstacles in equirectangular images. Through experiments on public datasets, we demonstrate that ErpGS can render novel view images more accurately than conventional methods.  
  </ol>  
</details>  
**comments**: Accepted to ICIP2025  
  
### [GoLF-NRT: Integrating Global Context and Local Geometry for Few-Shot View Synthesis](http://arxiv.org/abs/2505.19813)  
You Wang, Li Fang, Hao Zhu, Fei Hu, Long Ye, Zhan Ma  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) have transformed novel view synthesis by modeling scene-specific volumetric representations directly from images. While generalizable NeRF models can generate novel views across unknown scenes by learning latent ray representations, their performance heavily depends on a large number of multi-view observations. However, with limited input views, these methods experience significant degradation in rendering quality. To address this limitation, we propose GoLF-NRT: a Global and Local feature Fusion-based Neural Rendering Transformer. GoLF-NRT enhances generalizable neural rendering from few input views by leveraging a 3D transformer with efficient sparse attention to capture global scene context. In parallel, it integrates local geometric features extracted along the epipolar line, enabling high-quality scene reconstruction from as few as 1 to 3 input views. Furthermore, we introduce an adaptive sampling strategy based on attention weights and kernel regression, improving the accuracy of transformer-based neural rendering. Extensive experiments on public datasets show that GoLF-NRT achieves state-of-the-art performance across varying numbers of input views, highlighting the effectiveness and superiority of our approach. Code is available at https://github.com/KLMAV-CUC/GoLF-NRT.  
  </ol>  
</details>  
**comments**: CVPR 2025  
  
### [Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction](http://arxiv.org/abs/2505.19793)  
Li Fang, Hao Zhu, Longlong Chen, Fei Hu, Long Ye, Zhan Ma  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in generalizable novel view synthesis have achieved impressive quality through interpolation between nearby views. However, rendering high-resolution images remains computationally intensive due to the need for dense sampling of all rays. Recognizing that natural scenes are typically piecewise smooth and sampling all rays is often redundant, we propose a novel depth-guided bundle sampling strategy to accelerate rendering. By grouping adjacent rays into a bundle and sampling them collectively, a shared representation is generated for decoding all rays within the bundle. To further optimize efficiency, our adaptive sampling strategy dynamically allocates samples based on depth confidence, concentrating more samples in complex regions while reducing them in smoother areas. When applied to ENeRF, our method achieves up to a 1.27 dB PSNR improvement and a 47% increase in FPS on the DTU dataset. Extensive experiments on synthetic and real-world datasets demonstrate state-of-the-art rendering quality and up to 2x faster rendering compared to existing generalizable methods. Code is available at https://github.com/KLMAV-CUC/GDB-NeRF.  
  </ol>  
</details>  
**comments**: CVPR 2025  
  
### [ADD-SLAM: Adaptive Dynamic Dense SLAM with Gaussian Splatting](http://arxiv.org/abs/2505.19420)  
Wenhua Wu, Chenpeng Su, Siting Zhu, Tianchen Deng, Zhe Liu, Hesheng Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in Neural Radiance Fields (NeRF) and 3D Gaussian-based Simultaneous Localization and Mapping (SLAM) methods have demonstrated exceptional localization precision and remarkable dense mapping performance. However, dynamic objects introduce critical challenges by disrupting scene consistency, leading to tracking drift and mapping artifacts. Existing methods that employ semantic segmentation or object detection for dynamic identification and filtering typically rely on predefined categorical priors, while discarding dynamic scene information crucial for robotic applications such as dynamic obstacle avoidance and environmental interaction. To overcome these challenges, we propose ADD-SLAM: an Adaptive Dynamic Dense SLAM framework based on Gaussian splitting. We design an adaptive dynamic identification mechanism grounded in scene consistency analysis, comparing geometric and textural discrepancies between real-time observations and historical maps. Ours requires no predefined semantic category priors and adaptively discovers scene dynamics. Precise dynamic object recognition effectively mitigates interference from moving targets during localization. Furthermore, we propose a dynamic-static separation mapping strategy that constructs a temporal Gaussian model to achieve online incremental dynamic modeling. Experiments conducted on multiple dynamic datasets demonstrate our method's flexible and accurate dynamic segmentation capabilities, along with state-of-the-art performance in both localization and mapping.  
  </ol>  
</details>  
  
### [Triangle Splatting for Real-Time Radiance Field Rendering](http://arxiv.org/abs/2505.19175)  
Jan Held, Renaud Vandeghen, Adrien Deliege, Abdullah Hamdi, Silvio Giancola, Anthony Cioppa, Andrea Vedaldi, Bernard Ghanem, Andrea Tagliasacchi, Marc Van Droogenbroeck  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The field of computer graphics was revolutionized by models such as Neural Radiance Fields and 3D Gaussian Splatting, displacing triangles as the dominant representation for photogrammetry. In this paper, we argue for a triangle comeback. We develop a differentiable renderer that directly optimizes triangles via end-to-end gradients. We achieve this by rendering each triangle as differentiable splats, combining the efficiency of triangles with the adaptive density of representations based on independent primitives. Compared to popular 2D and 3D Gaussian Splatting methods, our approach achieves higher visual fidelity, faster convergence, and increased rendering throughput. On the Mip-NeRF360 dataset, our method outperforms concurrent non-volumetric primitives in visual fidelity and achieves higher perceptual quality than the state-of-the-art Zip-NeRF on indoor scenes. Triangles are simple, compatible with standard graphics stacks and GPU hardware, and highly efficient: for the \textit{Garden} scene, we achieve over 2,400 FPS at 1280x720 resolution using an off-the-shelf mesh renderer. These results highlight the efficiency and effectiveness of triangle-based representations for high-quality novel view synthesis. Triangles bring us closer to mesh-based optimization by combining classical computer graphics with modern differentiable rendering frameworks. The project page is https://trianglesplatting.github.io/  
  </ol>  
</details>  
**comments**: 18 pages, 13 figures, 10 tables  
  
  



