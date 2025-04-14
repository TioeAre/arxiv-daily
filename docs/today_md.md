<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Stereophotoclinometry-Revisited>Stereophotoclinometry Revisited</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Hypergraph-Vision-Transformers:-Images-are-More-than-Nodes,-More-than-Edges>Hypergraph Vision Transformers: Images are More than Nodes, More than Edges</a></li>
        <li><a href=#FocalLens:-Instruction-Tuning-Enables-Zero-Shot-Conditional-Image-Representations>FocalLens: Instruction Tuning Enables Zero-Shot Conditional Image Representations</a></li>
        <li><a href=#PNE-SGAN:-Probabilistic-NDT-Enhanced-Semantic-Graph-Attention-Network-for-LiDAR-Loop-Closure-Detection>PNE-SGAN: Probabilistic NDT-Enhanced Semantic Graph Attention Network for LiDAR Loop Closure Detection</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Stereophotoclinometry-Revisited>Stereophotoclinometry Revisited</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Geometric-Consistency-Refinement-for-Single-Image-Novel-View-Synthesis-via-Test-Time-Adaptation-of-Diffusion-Models>Geometric Consistency Refinement for Single Image Novel View Synthesis via Test-Time Adaptation of Diffusion Models</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Stereophotoclinometry Revisited](http://arxiv.org/abs/2504.08252)  
Travis Driver, Andrew Vaughan, Yang Cheng, Adnan Ansar, John Christian, Panagiotis Tsiotras  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image-based surface reconstruction and characterization is crucial for missions to small celestial bodies, as it informs mission planning, navigation, and scientific analysis. However, current state-of-the-practice methods, such as stereophotoclinometry (SPC), rely heavily on human-in-the-loop verification and high-fidelity a priori information. This paper proposes Photoclinometry-from-Motion (PhoMo), a novel framework that incorporates photoclinometry techniques into a keypoint-based structure-from-motion (SfM) system to estimate the surface normal and albedo at detected landmarks to improve autonomous surface and shape characterization of small celestial bodies from in-situ imagery. In contrast to SPC, we forego the expensive maplet estimation step and instead use dense keypoint measurements and correspondences from an autonomous keypoint detection and matching method based on deep learning. Moreover, we develop a factor graph-based approach allowing for simultaneous optimization of the spacecraft's pose, landmark positions, Sun-relative direction, and surface normals and albedos via fusion of Sun vector measurements and image keypoint measurements. The proposed framework is validated on real imagery taken by the Dawn mission to the asteroid 4 Vesta and the minor planet 1 Ceres and compared against an SPC reconstruction, where we demonstrate superior rendering performance compared to an SPC solution and precise alignment to a stereophotogrammetry (SPG) solution without relying on any a priori camera pose and topography information or humans-in-the-loop.  
  </ol>  
</details>  
**comments**: arXiv admin note: substantial text overlap with arXiv:2312.06865  
  
  



## Visual Localization  

### [Hypergraph Vision Transformers: Images are More than Nodes, More than Edges](http://arxiv.org/abs/2504.08710)  
Joshua Fixelle  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in computer vision have highlighted the scalability of Vision Transformers (ViTs) across various tasks, yet challenges remain in balancing adaptability, computational efficiency, and the ability to model higher-order relationships. Vision Graph Neural Networks (ViGs) offer an alternative by leveraging graph-based methodologies but are hindered by the computational bottlenecks of clustering algorithms used for edge generation. To address these issues, we propose the Hypergraph Vision Transformer (HgVT), which incorporates a hierarchical bipartite hypergraph structure into the vision transformer framework to capture higher-order semantic relationships while maintaining computational efficiency. HgVT leverages population and diversity regularization for dynamic hypergraph construction without clustering, and expert edge pooling to enhance semantic extraction and facilitate graph-based image retrieval. Empirical results demonstrate that HgVT achieves strong performance on image classification and retrieval, positioning it as an efficient framework for semantic-based vision tasks.  
  </ol>  
</details>  
**comments**: Accepted by CVPR 2025  
  
### [FocalLens: Instruction Tuning Enables Zero-Shot Conditional Image Representations](http://arxiv.org/abs/2504.08368)  
Cheng-Yu Hsieh, Pavan Kumar Anasosalu Vasu, Fartash Faghri, Raviteja Vemulapalli, Chun-Liang Li, Ranjay Krishna, Oncel Tuzel, Hadi Pouransari  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual understanding is inherently contextual -- what we focus on in an image depends on the task at hand. For instance, given an image of a person holding a bouquet of flowers, we may focus on either the person such as their clothing, or the type of flowers, depending on the context of interest. Yet, most existing image encoding paradigms represent an image as a fixed, generic feature vector, overlooking the potential needs of prioritizing varying visual information for different downstream use cases. In this work, we introduce FocalLens, a conditional visual encoding method that produces different representations for the same image based on the context of interest, expressed flexibly through natural language. We leverage vision instruction tuning data and contrastively finetune a pretrained vision encoder to take natural language instructions as additional inputs for producing conditional image representations. Extensive experiments validate that conditional image representation from FocalLens better pronounce the visual features of interest compared to generic features produced by standard vision encoders like CLIP. In addition, we show FocalLens further leads to performance improvements on a range of downstream tasks including image-image retrieval, image classification, and image-text retrieval, with an average gain of 5 and 10 points on the challenging SugarCrepe and MMVP-VLM benchmarks, respectively.  
  </ol>  
</details>  
  
### [PNE-SGAN: Probabilistic NDT-Enhanced Semantic Graph Attention Network for LiDAR Loop Closure Detection](http://arxiv.org/abs/2504.08280)  
Xiong Li, Shulei Liu, Xingning Chen, Yisong Wu, Dong Zhu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    LiDAR loop closure detection (LCD) is crucial for consistent Simultaneous Localization and Mapping (SLAM) but faces challenges in robustness and accuracy. Existing methods, including semantic graph approaches, often suffer from coarse geometric representations and lack temporal robustness against noise, dynamics, and viewpoint changes. We introduce PNE-SGAN, a Probabilistic NDT-Enhanced Semantic Graph Attention Network, to overcome these limitations. PNE-SGAN enhances semantic graphs by using Normal Distributions Transform (NDT) covariance matrices as rich, discriminative geometric node features, processed via a Graph Attention Network (GAT). Crucially, it integrates graph similarity scores into a probabilistic temporal filtering framework (modeled as an HMM/Bayes filter), incorporating uncertain odometry for motion modeling and utilizing forward-backward smoothing to effectively handle ambiguities. Evaluations on challenging KITTI sequences (00 and 08) demonstrate state-of-the-art performance, achieving Average Precision of 96.2\% and 95.1\%, respectively. PNE-SGAN significantly outperforms existing methods, particularly in difficult bidirectional loop scenarios where others falter. By synergizing detailed NDT geometry with principled probabilistic temporal reasoning, PNE-SGAN offers a highly accurate and robust solution for LiDAR LCD, enhancing SLAM reliability in complex, large-scale environments.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Stereophotoclinometry Revisited](http://arxiv.org/abs/2504.08252)  
Travis Driver, Andrew Vaughan, Yang Cheng, Adnan Ansar, John Christian, Panagiotis Tsiotras  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image-based surface reconstruction and characterization is crucial for missions to small celestial bodies, as it informs mission planning, navigation, and scientific analysis. However, current state-of-the-practice methods, such as stereophotoclinometry (SPC), rely heavily on human-in-the-loop verification and high-fidelity a priori information. This paper proposes Photoclinometry-from-Motion (PhoMo), a novel framework that incorporates photoclinometry techniques into a keypoint-based structure-from-motion (SfM) system to estimate the surface normal and albedo at detected landmarks to improve autonomous surface and shape characterization of small celestial bodies from in-situ imagery. In contrast to SPC, we forego the expensive maplet estimation step and instead use dense keypoint measurements and correspondences from an autonomous keypoint detection and matching method based on deep learning. Moreover, we develop a factor graph-based approach allowing for simultaneous optimization of the spacecraft's pose, landmark positions, Sun-relative direction, and surface normals and albedos via fusion of Sun vector measurements and image keypoint measurements. The proposed framework is validated on real imagery taken by the Dawn mission to the asteroid 4 Vesta and the minor planet 1 Ceres and compared against an SPC reconstruction, where we demonstrate superior rendering performance compared to an SPC solution and precise alignment to a stereophotogrammetry (SPG) solution without relying on any a priori camera pose and topography information or humans-in-the-loop.  
  </ol>  
</details>  
**comments**: arXiv admin note: substantial text overlap with arXiv:2312.06865  
  
  



## Image Matching  

### [Geometric Consistency Refinement for Single Image Novel View Synthesis via Test-Time Adaptation of Diffusion Models](http://arxiv.org/abs/2504.08348)  
Josef Bengtson, David Nilsson, Fredrik Kahl  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Diffusion models for single image novel view synthesis (NVS) can generate highly realistic and plausible images, but they are limited in the geometric consistency to the given relative poses. The generated images often show significant errors with respect to the epipolar constraints that should be fulfilled, as given by the target pose. In this paper we address this issue by proposing a methodology to improve the geometric correctness of images generated by a diffusion model for single image NVS. We formulate a loss function based on image matching and epipolar constraints, and optimize the starting noise in a diffusion sampling process such that the generated image should both be a realistic image and fulfill geometric constraints derived from the given target pose. Our method does not require training data or fine-tuning of the diffusion models, and we show that we can apply it to multiple state-of-the-art models for single image NVS. The method is evaluated on the MegaScenes dataset and we show that geometric consistency is improved compared to the baseline models while retaining the quality of the generated images.  
  </ol>  
</details>  
**comments**: Accepted to CVPR 2025 EDGE Workshop. Project page:
  https://gc-ref.github.io/  
  
  



