<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#ZeroVO:-Visual-Odometry-with-Minimal-Assumptions>ZeroVO: Visual Odometry with Minimal Assumptions</a></li>
        <li><a href=#UNO:-Unified-Self-Supervised-Monocular-Odometry-for-Platform-Agnostic-Deployment>UNO: Unified Self-Supervised Monocular Odometry for Platform-Agnostic Deployment</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Interpretable-and-Reliable-Detection-of-AI-Generated-Images-via-Grounded-Reasoning-in-MLLMs>Interpretable and Reliable Detection of AI-Generated Images via Grounded Reasoning in MLLMs</a></li>
        <li><a href=#Zero-Shot-Composed-Image-Retrieval>Zero Shot Composed Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Anti-interrupted-sampling-repeater-jamming-via-linear-canonical-Wigner-distribution-lightweight-LFM-detection>Anti-interrupted sampling repeater jamming via linear canonical Wigner distribution lightweight LFM detection</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Speedy-Deformable-3D-Gaussian-Splatting:-Fast-Rendering-and-Compression-of-Dynamic-Scenes>Speedy Deformable 3D Gaussian Splatting: Fast Rendering and Compression of Dynamic Scenes</a></li>
        <li><a href=#Genesis:-Multimodal-Driving-Scene-Generation-with-Spatio-Temporal-and-Cross-Modal-Consistency>Genesis: Multimodal Driving Scene Generation with Spatio-Temporal and Cross-Modal Consistency</a></li>
        <li><a href=#SPC-to-3D:-Novel-View-Synthesis-from-Binary-SPC-via-I2I-translation>SPC to 3D: Novel View Synthesis from Binary SPC via I2I translation</a></li>
        <li><a href=#Splat-and-Replace:-3D-Reconstruction-with-Repetitive-Elements>Splat and Replace: 3D Reconstruction with Repetitive Elements</a></li>
        <li><a href=#NeurNCD:-Novel-Class-Discovery-via-Implicit-Neural-Representation>NeurNCD: Novel Class Discovery via Implicit Neural Representation</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [ZeroVO: Visual Odometry with Minimal Assumptions](http://arxiv.org/abs/2506.08005)  
Lei Lai, Zekai Yin, Eshed Ohn-Bar  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce ZeroVO, a novel visual odometry (VO) algorithm that achieves zero-shot generalization across diverse cameras and environments, overcoming limitations in existing methods that depend on predefined or static camera calibration setups. Our approach incorporates three main innovations. First, we design a calibration-free, geometry-aware network structure capable of handling noise in estimated depth and camera parameters. Second, we introduce a language-based prior that infuses semantic information to enhance robust feature extraction and generalization to previously unseen domains. Third, we develop a flexible, semi-supervised training paradigm that iteratively adapts to new scenes using unlabeled data, further boosting the models' ability to generalize across diverse real-world scenarios. We analyze complex autonomous driving contexts, demonstrating over 30% improvement against prior methods on three standard benchmarks, KITTI, nuScenes, and Argoverse 2, as well as a newly introduced, high-fidelity synthetic dataset derived from Grand Theft Auto (GTA). By not requiring fine-tuning or camera calibration, our work broadens the applicability of VO, providing a versatile solution for real-world deployment at scale.  
  </ol>  
</details>  
  
### [UNO: Unified Self-Supervised Monocular Odometry for Platform-Agnostic Deployment](http://arxiv.org/abs/2506.07013)  
Wentao Zhao, Yihe Niu, Yanbo Wang, Tianchen Deng, Shenghai Yuan, Zhenli Wang, Rui Guo, Jingchuan Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This work presents UNO, a unified monocular visual odometry framework that enables robust and adaptable pose estimation across diverse environments, platforms, and motion patterns. Unlike traditional methods that rely on deployment-specific tuning or predefined motion priors, our approach generalizes effectively across a wide range of real-world scenarios, including autonomous vehicles, aerial drones, mobile robots, and handheld devices. To this end, we introduce a Mixture-of-Experts strategy for local state estimation, with several specialized decoders that each handle a distinct class of ego-motion patterns. Moreover, we introduce a fully differentiable Gumbel-Softmax module that constructs a robust inter-frame correlation graph, selects the optimal expert decoder, and prunes erroneous estimates. These cues are then fed into a unified back-end that combines pre-trained, scale-independent depth priors with a lightweight bundling adjustment to enforce geometric consistency. We extensively evaluate our method on three major benchmark datasets: KITTI (outdoor/autonomous driving), EuRoC-MAV (indoor/aerial drones), and TUM-RGBD (indoor/handheld), demonstrating state-of-the-art performance.  
  </ol>  
</details>  
**comments**: 15pages, 8 figures  
  
  



## Visual Localization  

### [Interpretable and Reliable Detection of AI-Generated Images via Grounded Reasoning in MLLMs](http://arxiv.org/abs/2506.07045)  
Yikun Ji, Hong Yan, Jun Lan, Huijia Zhu, Weiqiang Wang, Qi Fan, Liqing Zhang, Jianfu Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The rapid advancement of image generation technologies intensifies the demand for interpretable and robust detection methods. Although existing approaches often attain high accuracy, they typically operate as black boxes without providing human-understandable justifications. Multi-modal Large Language Models (MLLMs), while not originally intended for forgery detection, exhibit strong analytical and reasoning capabilities. When properly fine-tuned, they can effectively identify AI-generated images and offer meaningful explanations. However, existing MLLMs still struggle with hallucination and often fail to align their visual interpretations with actual image content and human reasoning. To bridge this gap, we construct a dataset of AI-generated images annotated with bounding boxes and descriptive captions that highlight synthesis artifacts, establishing a foundation for human-aligned visual-textual grounded reasoning. We then finetune MLLMs through a multi-stage optimization strategy that progressively balances the objectives of accurate detection, visual localization, and coherent textual explanation. The resulting model achieves superior performance in both detecting AI-generated images and localizing visual flaws, significantly outperforming baseline methods.  
  </ol>  
</details>  
  
### [Zero Shot Composed Image Retrieval](http://arxiv.org/abs/2506.06602)  
Santhosh Kakarla, Gautama Shastry Bulusu Venkata  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed image retrieval (CIR) allows a user to locate a target image by applying a fine-grained textual edit (e.g., ``turn the dress blue'' or ``remove stripes'') to a reference image. Zero-shot CIR, which embeds the image and the text with separate pretrained vision-language encoders, reaches only 20-25\% Recall@10 on the FashionIQ benchmark. We improve this by fine-tuning BLIP-2 with a lightweight Q-Former that fuses visual and textual features into a single embedding, raising Recall@10 to 45.6\% (shirt), 40.1\% (dress), and 50.4\% (top-tee) and increasing the average Recall@50 to 67.6\%. We also examine Retrieval-DPO, which fine-tunes CLIP's text encoder with a Direct Preference Optimization loss applied to FAISS-mined hard negatives. Despite extensive tuning of the scaling factor, index, and sampling strategy, Retrieval-DPO attains only 0.02\% Recall@10 -- far below zero-shot and prompt-tuned baselines -- because it (i) lacks joint image-text fusion, (ii) uses a margin objective misaligned with top- $K$ metrics, (iii) relies on low-quality negatives, and (iv) keeps the vision and Transformer layers frozen. Our results show that effective preference-based CIR requires genuine multimodal fusion, ranking-aware objectives, and carefully curated negatives.  
  </ol>  
</details>  
**comments**: 8 pages, 3 figures  
  
  



## Image Matching  

### [Anti-interrupted sampling repeater jamming via linear canonical Wigner distribution lightweight LFM detection](http://arxiv.org/abs/2506.06302)  
Jia-Mian Li, Bing-Zhao Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Interrupted sampling repeater jamming (ISRJ) poses a serious threat to radar target detection. Traditional time-frequency (TF) domain anti-jamming methods are prone to TF aliasing in multi-component signal scenarios, and cannot effectively suppress ISRJ with energy close to the real target under low signal-to-noise ratio (SNR) conditions. To address these challenges, this paper proposes an anti-jamming method based on generalized linear canonical Wigner distribution (GLWD) line detection. By setting the parameters reasonably, the TF image of GLWD can have excellent TF resolution and energy concentration, greatly improving the signal separation and SNR. Furthermore, in order to enhance the detection capability of the target LFM signal, the existing mobile line segment detection (M-LSD) is improved and the mobile long line segment detection (M-LLSD) is proposed. M-LLSD can detect the target signal more easily and reduce the sensitivity to the jamming signal, so as to efficiently and accurately extract the TF position information of the target signal. Finally, a TF filter is constructed based on the mapping between GLWD and short-time Fourier transform (STFT), performing filtering in the STFT domain to suppress jamming. Simulations and experiments show that the method can effectively suppress such difficult-to-distinguish jamming and is suitable for real-time radar anti-jamming with good robustness.  
  </ol>  
</details>  
**comments**: 28 pages, 19 figures  
  
  



## NeRF  

### [Speedy Deformable 3D Gaussian Splatting: Fast Rendering and Compression of Dynamic Scenes](http://arxiv.org/abs/2506.07917)  
Allen Tu, Haiyang Ying, Alex Hanson, Yonghan Lee, Tom Goldstein, Matthias Zwicker  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent extensions of 3D Gaussian Splatting (3DGS) to dynamic scenes achieve high-quality novel view synthesis by using neural networks to predict the time-varying deformation of each Gaussian. However, performing per-Gaussian neural inference at every frame poses a significant bottleneck, limiting rendering speed and increasing memory and compute requirements. In this paper, we present Speedy Deformable 3D Gaussian Splatting (SpeeDe3DGS), a general pipeline for accelerating the rendering speed of dynamic 3DGS and 4DGS representations by reducing neural inference through two complementary techniques. First, we propose a temporal sensitivity pruning score that identifies and removes Gaussians with low contribution to the dynamic scene reconstruction. We also introduce an annealing smooth pruning mechanism that improves pruning robustness in real-world scenes with imprecise camera poses. Second, we propose GroupFlow, a motion analysis technique that clusters Gaussians by trajectory similarity and predicts a single rigid transformation per group instead of separate deformations for each Gaussian. Together, our techniques accelerate rendering by $10.37\times$, reduce model size by $7.71\times$, and shorten training time by $2.71\times$ on the NeRF-DS dataset. SpeeDe3DGS also improves rendering speed by $4.20\times$ and $58.23\times$ on the D-NeRF and HyperNeRF vrig datasets. Our methods are modular and can be integrated into any deformable 3DGS or 4DGS framework.  
  </ol>  
</details>  
**comments**: Project Page: https://speede3dgs.github.io/  
  
### [Genesis: Multimodal Driving Scene Generation with Spatio-Temporal and Cross-Modal Consistency](http://arxiv.org/abs/2506.07497)  
Xiangyu Guo, Zhanqian Wu, Kaixin Xiong, Ziyang Xu, Lijun Zhou, Gangwei Xu, Shaoqing Xu, Haiyang Sun, Bing Wang, Guang Chen, Hangjun Ye, Wenyu Liu, Xinggang Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present Genesis, a unified framework for joint generation of multi-view driving videos and LiDAR sequences with spatio-temporal and cross-modal consistency. Genesis employs a two-stage architecture that integrates a DiT-based video diffusion model with 3D-VAE encoding, and a BEV-aware LiDAR generator with NeRF-based rendering and adaptive sampling. Both modalities are directly coupled through a shared latent space, enabling coherent evolution across visual and geometric domains. To guide the generation with structured semantics, we introduce DataCrafter, a captioning module built on vision-language models that provides scene-level and instance-level supervision. Extensive experiments on the nuScenes benchmark demonstrate that Genesis achieves state-of-the-art performance across video and LiDAR metrics (FVD 16.95, FID 4.24, Chamfer 0.611), and benefits downstream tasks including segmentation and 3D detection, validating the semantic fidelity and practical utility of the generated data.  
  </ol>  
</details>  
  
### [SPC to 3D: Novel View Synthesis from Binary SPC via I2I translation](http://arxiv.org/abs/2506.06890)  
Sumit Sharma, Gopi Raju Matta, Kaushik Mitra  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Single Photon Avalanche Diodes (SPADs) represent a cutting-edge imaging technology, capable of detecting individual photons with remarkable timing precision. Building on this sensitivity, Single Photon Cameras (SPCs) enable image capture at exceptionally high speeds under both low and high illumination. Enabling 3D reconstruction and radiance field recovery from such SPC data holds significant promise. However, the binary nature of SPC images leads to severe information loss, particularly in texture and color, making traditional 3D synthesis techniques ineffective. To address this challenge, we propose a modular two-stage framework that converts binary SPC images into high-quality colorized novel views. The first stage performs image-to-image (I2I) translation using generative models such as Pix2PixHD, converting binary SPC inputs into plausible RGB representations. The second stage employs 3D scene reconstruction techniques like Neural Radiance Fields (NeRF) or Gaussian Splatting (3DGS) to generate novel views. We validate our two-stage pipeline (Pix2PixHD + Nerf/3DGS) through extensive qualitative and quantitative experiments, demonstrating significant improvements in perceptual quality and geometric consistency over the alternative baseline.  
  </ol>  
</details>  
**comments**: Accepted for publication at ICIP 2025  
  
### [Splat and Replace: 3D Reconstruction with Repetitive Elements](http://arxiv.org/abs/2506.06462)  
Nicolás Violante, Andreas Meuleman, Alban Gauthier, Frédo Durand, Thibault Groueix, George Drettakis  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We leverage repetitive elements in 3D scenes to improve novel view synthesis. Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have greatly improved novel view synthesis but renderings of unseen and occluded parts remain low-quality if the training views are not exhaustive enough. Our key observation is that our environment is often full of repetitive elements. We propose to leverage those repetitions to improve the reconstruction of low-quality parts of the scene due to poor coverage and occlusions. We propose a method that segments each repeated instance in a 3DGS reconstruction, registers them together, and allows information to be shared among instances. Our method improves the geometry while also accounting for appearance variations across instances. We demonstrate our method on a variety of synthetic and real scenes with typical repetitive elements, leading to a substantial improvement in the quality of novel view synthesis.  
  </ol>  
</details>  
**comments**: SIGGRAPH Conference Papers 2025. Project site:
  https://repo-sam.inria.fr/nerphys/splat-and-replace/  
  
### [NeurNCD: Novel Class Discovery via Implicit Neural Representation](http://arxiv.org/abs/2506.06412)  
Junming Wang, Yi Shi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Discovering novel classes in open-world settings is crucial for real-world applications. Traditional explicit representations, such as object descriptors or 3D segmentation maps, are constrained by their discrete, hole-prone, and noisy nature, which hinders accurate novel class discovery. To address these challenges, we introduce NeurNCD, the first versatile and data-efficient framework for novel class discovery that employs the meticulously designed Embedding-NeRF model combined with KL divergence as a substitute for traditional explicit 3D segmentation maps to aggregate semantic embedding and entropy in visual embedding space. NeurNCD also integrates several key components, including feature query, feature modulation and clustering, facilitating efficient feature augmentation and information exchange between the pre-trained semantic segmentation network and implicit neural representations. As a result, our framework achieves superior segmentation performance in both open and closed-world settings without relying on densely labelled datasets for supervised training or human interaction to generate sparse label supervision. Extensive experiments demonstrate that our method significantly outperforms state-of-the-art approaches on the NYUv2 and Replica datasets.  
  </ol>  
</details>  
**comments**: Accepted by ICMR 2024  
  
  



