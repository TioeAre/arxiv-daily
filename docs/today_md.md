<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Are-Minimal-Radial-Distortion-Solvers-Necessary-for-Relative-Pose-Estimation?>Are Minimal Radial Distortion Solvers Necessary for Relative Pose Estimation?</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Exploiting-Distribution-Constraints-for-Scalable-and-Efficient-Image-Retrieval>Exploiting Distribution Constraints for Scalable and Efficient Image Retrieval</a></li>
        <li><a href=#Pair-VPR:-Place-Aware-Pre-training-and-Contrastive-Pair-Classification-for-Visual-Place-Recognition-with-Vision-Transformers>Pair-VPR: Place-Aware Pre-training and Contrastive Pair Classification for Visual Place Recognition with Vision Transformers</a></li>
        <li><a href=#MedImageInsight:-An-Open-Source-Embedding-Model-for-General-Domain-Medical-Imaging>MedImageInsight: An Open-Source Embedding Model for General Domain Medical Imaging</a></li>
        <li><a href=#Temporal-Image-Caption-Retrieval-Competition----Description-and-Results>Temporal Image Caption Retrieval Competition -- Description and Results</a></li>
        <li><a href=#Monocular-Visual-Place-Recognition-in-LiDAR-Maps-via-Cross-Modal-State-Space-Model-and-Multi-View-Matching>Monocular Visual Place Recognition in LiDAR Maps via Cross-Modal State Space Model and Multi-View Matching</a></li>
        <li><a href=#GSLoc:-Visual-Localization-with-3D-Gaussian-Splatting>GSLoc: Visual Localization with 3D Gaussian Splatting</a></li>
        <li><a href=#Beyond-Captioning:-Task-Specific-Prompting-for-Improved-VLM-Performance-in-Mathematical-Reasoning>Beyond Captioning: Task-Specific Prompting for Improved VLM Performance in Mathematical Reasoning</a></li>
        <li><a href=#RNR-Nav:-A-Real-World-Visual-Navigation-System-Using-Renderable-Neural-Radiance-Maps>RNR-Nav: A Real-World Visual Navigation System Using Renderable Neural Radiance Maps</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Unsupervised-Model-Diagnosis>Unsupervised Model Diagnosis</a></li>
        <li><a href=#Equi-GSPR:-Equivariant-SE(3)-Graph-Network-Model-for-Sparse-Point-Cloud-Registration>Equi-GSPR: Equivariant SE(3) Graph Network Model for Sparse Point Cloud Registration</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#DreamMesh4D:-Video-to-4D-Generation-with-Sparse-Controlled-Gaussian-Mesh-Hybrid-Representation>DreamMesh4D: Video-to-4D Generation with Sparse-Controlled Gaussian-Mesh Hybrid Representation</a></li>
        <li><a href=#MimicTalk:-Mimicking-a-personalized-and-expressive-3D-talking-face-in-minutes>MimicTalk: Mimicking a personalized and expressive 3D talking face in minutes</a></li>
        <li><a href=#3D-Representation-Methods:-A-Survey>3D Representation Methods: A Survey</a></li>
        <li><a href=#Comparative-Analysis-of-Novel-View-Synthesis-and-Photogrammetry-for-3D-Forest-Stand-Reconstruction-and-extraction-of-individual-tree-parameters>Comparative Analysis of Novel View Synthesis and Photogrammetry for 3D Forest Stand Reconstruction and extraction of individual tree parameters</a></li>
        <li><a href=#Toward-General-Object-level-Mapping-from-Sparse-Views-with-3D-Diffusion-Priors>Toward General Object-level Mapping from Sparse Views with 3D Diffusion Priors</a></li>
        <li><a href=#PH-Dropout:-Prctical-Epistemic-Uncertainty-Quantification-for-View-Synthesis>PH-Dropout: Prctical Epistemic Uncertainty Quantification for View Synthesis</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Are Minimal Radial Distortion Solvers Necessary for Relative Pose Estimation?](http://arxiv.org/abs/2410.05984)  
[[code](https://github.com/kocurvik/rd)]  
Charalambos Tzamos, Viktor Kocur, Yaqing Ding, Torsten Sattler, Zuzana Kukelova  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Estimating the relative pose between two cameras is a fundamental step in many applications such as Structure-from-Motion. The common approach to relative pose estimation is to apply a minimal solver inside a RANSAC loop. Highly efficient solvers exist for pinhole cameras. Yet, (nearly) all cameras exhibit radial distortion. Not modeling radial distortion leads to (significantly) worse results. However, minimal radial distortion solvers are significantly more complex than pinhole solvers, both in terms of run-time and implementation efforts. This paper compares radial distortion solvers with a simple-to-implement approach that combines an efficient pinhole solver with sampled radial distortion parameters. Extensive experiments on multiple datasets and RANSAC variants show that this simple approach performs similarly or better than the most accurate minimal distortion solvers at faster run-times while being significantly more accurate than faster non-minimal solvers. We clearly show that complex radial distortion solvers are not necessary in practice. Code and benchmark are available at https://github.com/kocurvik/rd.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Exploiting Distribution Constraints for Scalable and Efficient Image Retrieval](http://arxiv.org/abs/2410.07022)  
Mohammad Omama, Po-han Li, Sandeep P. Chinchali  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image retrieval is crucial in robotics and computer vision, with downstream applications in robot place recognition and vision-based product recommendations. Modern retrieval systems face two key challenges: scalability and efficiency. State-of-the-art image retrieval systems train specific neural networks for each dataset, an approach that lacks scalability. Furthermore, since retrieval speed is directly proportional to embedding size, existing systems that use large embeddings lack efficiency. To tackle scalability, recent works propose using off-the-shelf foundation models. However, these models, though applicable across datasets, fall short in achieving performance comparable to that of dataset-specific models. Our key observation is that, while foundation models capture necessary subtleties for effective retrieval, the underlying distribution of their embedding space can negatively impact cosine similarity searches. We introduce Autoencoders with Strong Variance Constraints (AE-SVC), which, when used for projection, significantly improves the performance of foundation models. We provide an in-depth theoretical analysis of AE-SVC. Addressing efficiency, we introduce Single-shot Similarity Space Distillation ((SS) $_2$D), a novel approach to learn embeddings with adaptive sizes that offers a better trade-off between size and performance. We conducted extensive experiments on four retrieval datasets, including Stanford Online Products (SoP) and Pittsburgh30k, using four different off-the-shelf foundation models, including DinoV2 and CLIP. AE-SVC demonstrates up to a $16\%$ improvement in retrieval performance, while (SS)$_2$D shows a further $10\%$ improvement for smaller embedding sizes.  
  </ol>  
</details>  
  
### [Pair-VPR: Place-Aware Pre-training and Contrastive Pair Classification for Visual Place Recognition with Vision Transformers](http://arxiv.org/abs/2410.06614)  
Stephen Hausler, Peyman Moghadam  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this work we propose a novel joint training method for Visual Place Recognition (VPR), which simultaneously learns a global descriptor and a pair classifier for re-ranking. The pair classifier can predict whether a given pair of images are from the same place or not. The network only comprises Vision Transformer components for both the encoder and the pair classifier, and both components are trained using their respective class tokens. In existing VPR methods, typically the network is initialized using pre-trained weights from a generic image dataset such as ImageNet. In this work we propose an alternative pre-training strategy, by using Siamese Masked Image Modelling as a pre-training task. We propose a Place-aware image sampling procedure from a collection of large VPR datasets for pre-training our model, to learn visual features tuned specifically for VPR. By re-using the Mask Image Modelling encoder and decoder weights in the second stage of training, Pair-VPR can achieve state-of-the-art VPR performance across five benchmark datasets with a ViT-B encoder, along with further improvements in localization recall with larger encoders. The Pair-VPR website is: https://csiro-robotics.github.io/Pair-VPR.  
  </ol>  
</details>  
  
### [MedImageInsight: An Open-Source Embedding Model for General Domain Medical Imaging](http://arxiv.org/abs/2410.06542)  
Noel C. F. Codella, Ying Jin, Shrey Jain, Yu Gu, Ho Hin Lee, Asma Ben Abacha, Alberto Santamaria-Pang, Will Guyman, Naiteek Sangani, Sheng Zhang, Hoifung Poon, Stephanie Hyland, Shruthi Bannur, Javier Alvarez-Valle, Xue Li, John Garrett, Alan McMillan, Gaurav Rajguru, Madhu Maddi, Nilesh Vijayrania, Rehaan Bhimai, Nick Mecklenburg, Rupal Jain, Daniel Holstein, Naveen Gaur, Vijay Aski, Jenq-Neng Hwang, Thomas Lin, Ivan Tarapov, Matthew Lungren, Mu Wei  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this work, we present MedImageInsight, an open-source medical imaging embedding model. MedImageInsight is trained on medical images with associated text and labels across a diverse collection of domains, including X-Ray, CT, MRI, dermoscopy, OCT, fundus photography, ultrasound, histopathology, and mammography. Rigorous evaluations demonstrate MedImageInsight's ability to achieve state-of-the-art (SOTA) or human expert level performance across classification, image-image search, and fine-tuning tasks. Specifically, on public datasets, MedImageInsight achieves SOTA in CT 3D medical image retrieval, as well as SOTA in disease classification and search for chest X-ray, dermatology, and OCT imaging. Furthermore, MedImageInsight achieves human expert performance in bone age estimation (on both public and partner data), as well as AUC above 0.9 in most other domains. When paired with a text decoder, MedImageInsight achieves near SOTA level single image report findings generation with less than 10\% the parameters of other models. Compared to fine-tuning GPT-4o with only MIMIC-CXR data for the same task, MedImageInsight outperforms in clinical metrics, but underperforms on lexical metrics where GPT-4o sets a new SOTA. Importantly for regulatory purposes, MedImageInsight can generate ROC curves, adjust sensitivity and specificity based on clinical need, and provide evidence-based decision support through image-image search (which can also enable retrieval augmented generation). In an independent clinical evaluation of image-image search in chest X-ray, MedImageInsight outperformed every other publicly available foundation model evaluated by large margins (over 6 points AUC), and significantly outperformed other models in terms of AI fairness (across age and gender). We hope releasing MedImageInsight will help enhance collective progress in medical imaging AI research and development.  
  </ol>  
</details>  
  
### [Temporal Image Caption Retrieval Competition -- Description and Results](http://arxiv.org/abs/2410.06314)  
Jakub Pokrywka, Piotr Wierzchoń, Kornel Weryszko, Krzysztof Jassem  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Multimodal models, which combine visual and textual information, have recently gained significant recognition. This paper addresses the multimodal challenge of Text-Image retrieval and introduces a novel task that extends the modalities to include temporal data. The Temporal Image Caption Retrieval Competition (TICRC) presented in this paper is based on the Chronicling America and Challenging America projects, which offer access to an extensive collection of digitized historic American newspapers spanning 274 years. In addition to the competition results, we provide an analysis of the delivered dataset and the process of its creation.  
  </ol>  
</details>  
  
### [Monocular Visual Place Recognition in LiDAR Maps via Cross-Modal State Space Model and Multi-View Matching](http://arxiv.org/abs/2410.06285)  
Gongxin Yao, Xinyang Li, Luowei Fu, Yu Pan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Achieving monocular camera localization within pre-built LiDAR maps can bypass the simultaneous mapping process of visual SLAM systems, potentially reducing the computational overhead of autonomous localization. To this end, one of the key challenges is cross-modal place recognition, which involves retrieving 3D scenes (point clouds) from a LiDAR map according to online RGB images. In this paper, we introduce an efficient framework to learn descriptors for both RGB images and point clouds. It takes visual state space model (VMamba) as the backbone and employs a pixel-view-scene joint training strategy for cross-modal contrastive learning. To address the field-of-view differences, independent descriptors are generated from multiple evenly distributed viewpoints for point clouds. A visible 3D points overlap strategy is then designed to quantify the similarity between point cloud views and RGB images for multi-view supervision. Additionally, when generating descriptors from pixel-level features using NetVLAD, we compensate for the loss of geometric information, and introduce an efficient scheme for multi-view generation. Experimental results on the KITTI and KITTI-360 datasets demonstrate the effectiveness and generalization of our method. The code will be released upon acceptance.  
  </ol>  
</details>  
  
### [GSLoc: Visual Localization with 3D Gaussian Splatting](http://arxiv.org/abs/2410.06165)  
Kazii Botashev, Vladislav Pyatov, Gonzalo Ferrer, Stamatios Lefkimmiatis  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present GSLoc: a new visual localization method that performs dense camera alignment using 3D Gaussian Splatting as a map representation of the scene. GSLoc backpropagates pose gradients over the rendering pipeline to align the rendered and target images, while it adopts a coarse-to-fine strategy by utilizing blurring kernels to mitigate the non-convexity of the problem and improve the convergence. The results show that our approach succeeds at visual localization in challenging conditions of relatively small overlap between initial and target frames inside textureless environments when state-of-the-art neural sparse methods provide inferior results. Using the byproduct of realistic rendering from the 3DGS map representation, we show how to enhance localization results by mixing a set of observed and virtual reference keyframes when solving the image retrieval problem. We evaluate our method both on synthetic and real-world data, discussing its advantages and application potential.  
  </ol>  
</details>  
  
### [Beyond Captioning: Task-Specific Prompting for Improved VLM Performance in Mathematical Reasoning](http://arxiv.org/abs/2410.05928)  
Ayush Singh, Mansi Gupta, Shivank Garg, Abhinav Kumar, Vansh Agrawal  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Vision-Language Models (VLMs) have transformed tasks requiring visual and reasoning abilities, such as image retrieval and Visual Question Answering (VQA). Despite their success, VLMs face significant challenges with tasks involving geometric reasoning, algebraic problem-solving, and counting. These limitations stem from difficulties effectively integrating multiple modalities and accurately interpreting geometry-related tasks. Various works claim that introducing a captioning pipeline before VQA tasks enhances performance. We incorporated this pipeline for tasks involving geometry, algebra, and counting. We found that captioning results are not generalizable, specifically with larger VLMs primarily trained on downstream QnA tasks showing random performance on math-related challenges. However, we present a promising alternative: task-based prompting, enriching the prompt with task-specific guidance. This approach shows promise and proves more effective than direct captioning methods for math-heavy problems.  
  </ol>  
</details>  
  
### [RNR-Nav: A Real-World Visual Navigation System Using Renderable Neural Radiance Maps](http://arxiv.org/abs/2410.05621)  
Minsoo Kim, Obin Kwon, Howoong Jun, Songhwai Oh  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose a novel visual localization and navigation framework for real-world environments directly integrating observed visual information into the bird-eye-view map. While the renderable neural radiance map (RNR-Map) shows considerable promise in simulated settings, its deployment in real-world scenarios poses undiscovered challenges. RNR-Map utilizes projections of multiple vectors into a single latent code, resulting in information loss under suboptimal conditions. To address such issues, our enhanced RNR-Map for real-world robots, RNR-Map++, incorporates strategies to mitigate information loss, such as a weighted map and positional encoding. For robust real-time localization, we integrate a particle filter into the correlation-based localization framework using RNRMap++ without a rendering procedure. Consequently, we establish a real-world robot system for visual navigation utilizing RNR-Map++, which we call "RNR-Nav." Experimental results demonstrate that the proposed methods significantly enhance rendering quality and localization robustness compared to previous approaches. In real-world navigation tasks, RNR-Nav achieves a success rate of 84.4%, marking a 68.8% enhancement over the methods of the original RNR-Map paper.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Unsupervised Model Diagnosis](http://arxiv.org/abs/2410.06243)  
Yinong Oliver Wang, Eileen Li, Jinqi Luo, Zhaoning Wang, Fernando De la Torre  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Ensuring model explainability and robustness is essential for reliable deployment of deep vision systems. Current methods for evaluating robustness rely on collecting and annotating extensive test sets. While this is common practice, the process is labor-intensive and expensive with no guarantee of sufficient coverage across attributes of interest. Recently, model diagnosis frameworks have emerged leveraging user inputs (e.g., text) to assess the vulnerability of the model. However, such dependence on human can introduce bias and limitation given the domain knowledge of particular users. This paper proposes Unsupervised Model Diagnosis (UMO), that leverages generative models to produce semantic counterfactual explanations without any user guidance. Given a differentiable computer vision model (i.e., the target model), UMO optimizes for the most counterfactual directions in a generative latent space. Our approach identifies and visualizes changes in semantics, and then matches these changes to attributes from wide-ranging text sources, such as dictionaries or language models. We validate the framework on multiple vision tasks (e.g., classification, segmentation, keypoint detection). Extensive experiments show that our unsupervised discovery of semantic directions can correctly highlight spurious correlations and visualize the failure mode of target models without any human intervention.  
  </ol>  
</details>  
**comments**: 9 pages, 9 figures, 3 tables  
  
### [Equi-GSPR: Equivariant SE(3) Graph Network Model for Sparse Point Cloud Registration](http://arxiv.org/abs/2410.05729)  
[[code](https://github.com/alexandor91/se3-equi-graph-registration)]  
Xueyang Kang, Zhaoliang Luan, Kourosh Khoshelham, Bing Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Point cloud registration is a foundational task for 3D alignment and reconstruction applications. While both traditional and learning-based registration approaches have succeeded, leveraging the intrinsic symmetry of point cloud data, including rotation equivariance, has received insufficient attention. This prohibits the model from learning effectively, resulting in a requirement for more training data and increased model complexity. To address these challenges, we propose a graph neural network model embedded with a local Spherical Euclidean 3D equivariance property through SE(3) message passing based propagation. Our model is composed mainly of a descriptor module, equivariant graph layers, match similarity, and the final regression layers. Such modular design enables us to utilize sparsely sampled input points and initialize the descriptor by self-trained or pre-trained geometric feature descriptors easily. Experiments conducted on the 3DMatch and KITTI datasets exhibit the compelling and robust performance of our model compared to state-of-the-art approaches, while the model complexity remains relatively low at the same time.  
  </ol>  
</details>  
**comments**: 18 main body pages, and 9 pages for supplementary part  
  
  



## NeRF  

### [DreamMesh4D: Video-to-4D Generation with Sparse-Controlled Gaussian-Mesh Hybrid Representation](http://arxiv.org/abs/2410.06756)  
Zhiqi Li, Yiming Chen, Peidong Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in 2D/3D generative techniques have facilitated the generation of dynamic 3D objects from monocular videos. Previous methods mainly rely on the implicit neural radiance fields (NeRF) or explicit Gaussian Splatting as the underlying representation, and struggle to achieve satisfactory spatial-temporal consistency and surface appearance. Drawing inspiration from modern 3D animation pipelines, we introduce DreamMesh4D, a novel framework combining mesh representation with geometric skinning technique to generate high-quality 4D object from a monocular video. Instead of utilizing classical texture map for appearance, we bind Gaussian splats to triangle face of mesh for differentiable optimization of both the texture and mesh vertices. In particular, DreamMesh4D begins with a coarse mesh obtained through an image-to-3D generation procedure. Sparse points are then uniformly sampled across the mesh surface, and are used to build a deformation graph to drive the motion of the 3D object for the sake of computational efficiency and providing additional constraint. For each step, transformations of sparse control points are predicted using a deformation network, and the mesh vertices as well as the surface Gaussians are deformed via a novel geometric skinning algorithm, which is a hybrid approach combining LBS (linear blending skinning) and DQS (dual-quaternion skinning), mitigating drawbacks associated with both approaches. The static surface Gaussians and mesh vertices as well as the deformation network are learned via reference view photometric loss, score distillation loss as well as other regularizers in a two-stage manner. Extensive experiments demonstrate superior performance of our method. Furthermore, our method is compatible with modern graphic pipelines, showcasing its potential in the 3D gaming and film industry.  
  </ol>  
</details>  
**comments**: NeurIPS 2024  
  
### [MimicTalk: Mimicking a personalized and expressive 3D talking face in minutes](http://arxiv.org/abs/2410.06734)  
Zhenhui Ye, Tianyun Zhong, Yi Ren, Ziyue Jiang, Jiawei Huang, Rongjie Huang, Jinglin Liu, Jinzheng He, Chen Zhang, Zehan Wang, Xize Chen, Xiang Yin, Zhou Zhao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Talking face generation (TFG) aims to animate a target identity's face to create realistic talking videos. Personalized TFG is a variant that emphasizes the perceptual identity similarity of the synthesized result (from the perspective of appearance and talking style). While previous works typically solve this problem by learning an individual neural radiance field (NeRF) for each identity to implicitly store its static and dynamic information, we find it inefficient and non-generalized due to the per-identity-per-training framework and the limited training data. To this end, we propose MimicTalk, the first attempt that exploits the rich knowledge from a NeRF-based person-agnostic generic model for improving the efficiency and robustness of personalized TFG. To be specific, (1) we first come up with a person-agnostic 3D TFG model as the base model and propose to adapt it into a specific identity; (2) we propose a static-dynamic-hybrid adaptation pipeline to help the model learn the personalized static appearance and facial dynamic features; (3) To generate the facial motion of the personalized talking style, we propose an in-context stylized audio-to-motion model that mimics the implicit talking style provided in the reference video without information loss by an explicit style representation. The adaptation process to an unseen identity can be performed in 15 minutes, which is 47 times faster than previous person-dependent methods. Experiments show that our MimicTalk surpasses previous baselines regarding video quality, efficiency, and expressiveness. Source code and video samples are available at https://mimictalk.github.io .  
  </ol>  
</details>  
**comments**: Accepted by NeurIPS 2024  
  
### [3D Representation Methods: A Survey](http://arxiv.org/abs/2410.06475)  
Zhengren Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The field of 3D representation has experienced significant advancements, driven by the increasing demand for high-fidelity 3D models in various applications such as computer graphics, virtual reality, and autonomous systems. This review examines the development and current state of 3D representation methods, highlighting their research trajectories, innovations, strength and weakness. Key techniques such as Voxel Grid, Point Cloud, Mesh, Signed Distance Function (SDF), Neural Radiance Field (NeRF), 3D Gaussian Splatting, Tri-Plane, and Deep Marching Tetrahedra (DMTet) are reviewed. The review also introduces essential datasets that have been pivotal in advancing the field, highlighting their characteristics and impact on research progress. Finally, we explore potential research directions that hold promise for further expanding the capabilities and applications of 3D representation methods.  
  </ol>  
</details>  
**comments**: Preliminary Draft  
  
### [Comparative Analysis of Novel View Synthesis and Photogrammetry for 3D Forest Stand Reconstruction and extraction of individual tree parameters](http://arxiv.org/abs/2410.05772)  
Guoji Tian, Chongcheng Chen, Hongyu Huang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Accurate and efficient 3D reconstruction of trees is crucial for forest resource assessments and management. Close-Range Photogrammetry (CRP) is commonly used for reconstructing forest scenes but faces challenges like low efficiency and poor quality. Recently, Novel View Synthesis (NVS) technologies, including Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), have shown promise for 3D plant reconstruction with limited images. However, existing research mainly focuses on small plants in orchards or individual trees, leaving uncertainty regarding their application in larger, complex forest stands. In this study, we collected sequential images of forest plots with varying complexity and performed dense reconstruction using NeRF and 3DGS. The resulting point clouds were compared with those from photogrammetry and laser scanning. Results indicate that NVS methods significantly enhance reconstruction efficiency. Photogrammetry struggles with complex stands, leading to point clouds with excessive canopy noise and incorrectly reconstructed trees, such as duplicated trunks. NeRF, while better for canopy regions, may produce errors in ground areas with limited views. The 3DGS method generates sparser point clouds, particularly in trunk areas, affecting diameter at breast height (DBH) accuracy. All three methods can extract tree height information, with NeRF yielding the highest accuracy; however, photogrammetry remains superior for DBH accuracy. These findings suggest that NVS methods have significant potential for 3D reconstruction of forest stands, offering valuable support for complex forest resource inventory and visualization tasks.  
  </ol>  
</details>  
**comments**: 31page,15figures  
  
### [Toward General Object-level Mapping from Sparse Views with 3D Diffusion Priors](http://arxiv.org/abs/2410.05514)  
[[code](https://github.com/trailab/generalobjectmapping)]  
Ziwei Liao, Binbin Xu, Steven L. Waslander  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Object-level mapping builds a 3D map of objects in a scene with detailed shapes and poses from multi-view sensor observations. Conventional methods struggle to build complete shapes and estimate accurate poses due to partial occlusions and sensor noise. They require dense observations to cover all objects, which is challenging to achieve in robotics trajectories. Recent work introduces generative shape priors for object-level mapping from sparse views, but is limited to single-category objects. In this work, we propose a General Object-level Mapping system, GOM, which leverages a 3D diffusion model as shape prior with multi-category support and outputs Neural Radiance Fields (NeRFs) for both texture and geometry for all objects in a scene. GOM includes an effective formulation to guide a pre-trained diffusion model with extra nonlinear constraints from sensor measurements without finetuning. We also develop a probabilistic optimization formulation to fuse multi-view sensor observations and diffusion priors for joint 3D object pose and shape estimation. Our GOM system demonstrates superior multi-category mapping performance from sparse views, and achieves more accurate mapping results compared to state-of-the-art methods on the real-world benchmarks. We will release our code: https://github.com/TRAILab/GeneralObjectMapping.  
  </ol>  
</details>  
**comments**: Accepted by CoRL 2024  
  
### [PH-Dropout: Prctical Epistemic Uncertainty Quantification for View Synthesis](http://arxiv.org/abs/2410.05468)  
Chuanhao Sun, Thanos Triantafyllou, Anthos Makris, Maja Drmač, Kai Xu, Luo Mai, Mahesh K. Marina  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    View synthesis using Neural Radiance Fields (NeRF) and Gaussian Splatting (GS) has demonstrated impressive fidelity in rendering real-world scenarios. However, practical methods for accurate and efficient epistemic Uncertainty Quantification (UQ) in view synthesis are lacking. Existing approaches for NeRF either introduce significant computational overhead (e.g., ``10x increase in training time" or ``10x repeated training") or are limited to specific uncertainty conditions or models. Notably, GS models lack any systematic approach for comprehensive epistemic UQ. This capability is crucial for improving the robustness and scalability of neural view synthesis, enabling active model updates, error estimation, and scalable ensemble modeling based on uncertainty. In this paper, we revisit NeRF and GS-based methods from a function approximation perspective, identifying key differences and connections in 3D representation learning. Building on these insights, we introduce PH-Dropout (Post hoc Dropout), the first real-time and accurate method for epistemic uncertainty estimation that operates directly on pre-trained NeRF and GS models. Extensive evaluations validate our theoretical findings and demonstrate the effectiveness of PH-Dropout.  
  </ol>  
</details>  
**comments**: 21 pages, in submision  
  
  



