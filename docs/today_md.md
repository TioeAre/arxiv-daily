<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#SfM-Free-3D-Gaussian-Splatting-via-Hierarchical-Training>SfM-Free 3D Gaussian Splatting via Hierarchical Training</a></li>
        <li><a href=#MVImgNet2.0:-A-Larger-scale-Dataset-of-Multi-view-Images>MVImgNet2.0: A Larger-scale Dataset of Multi-view Images</a></li>
        <li><a href=#Look-Ma,-No-Ground-Truth!-Ground-Truth-Free-Tuning-of-Structure-from-Motion-and-Visual-SLAM>Look Ma, No Ground Truth! Ground-Truth-Free Tuning of Structure from Motion and Visual SLAM</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Active-Learning-via-Classifier-Impact-and-Greedy-Selection-for-Interactive-Image-Retrieval>Active Learning via Classifier Impact and Greedy Selection for Interactive Image Retrieval</a></li>
        <li><a href=#Mutli-View-3D-Reconstruction-using-Knowledge-Distillation>Mutli-View 3D Reconstruction using Knowledge Distillation</a></li>
        <li><a href=#Optimizing-Domain-Specific-Image-Retrieval:-A-Benchmark-of-FAISS-and-Annoy-with-Fine-Tuned-Features>Optimizing Domain-Specific Image Retrieval: A Benchmark of FAISS and Annoy with Fine-Tuned Features</a></li>
        <li><a href=#Neuron-Abandoning-Attention-Flow:-Visual-Explanation-of-Dynamics-inside-CNN-Models>Neuron Abandoning Attention Flow: Visual Explanation of Dynamics inside CNN Models</a></li>
        <li><a href=#EDTformer:-An-Efficient-Decoder-Transformer-for-Visual-Place-Recognition>EDTformer: An Efficient Decoder Transformer for Visual Place Recognition</a></li>
        <li><a href=#EFSA:-Episodic-Few-Shot-Adaptation-for-Text-to-Image-Retrieval>EFSA: Episodic Few-Shot Adaptation for Text-to-Image Retrieval</a></li>
        <li><a href=#Unleashing-the-Power-of-Data-Synthesis-in-Visual-Localization>Unleashing the Power of Data Synthesis in Visual Localization</a></li>
        <li><a href=#Relation-Aware-Meta-Learning-for-Zero-shot-Sketch-Based-Image-Retrieval>Relation-Aware Meta-Learning for Zero-shot Sketch-Based Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#MamKPD:-A-Simple-Mamba-Baseline-for-Real-Time-2D-Keypoint-Detection>MamKPD: A Simple Mamba Baseline for Real-Time 2D Keypoint Detection</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#SAGA:-Surface-Aligned-Gaussian-Avatar>SAGA: Surface-Aligned Gaussian Avatar</a></li>
        <li><a href=#CtrlNeRF:-The-Generative-Neural-Radiation-Fields-for-the-Controllable-Synthesis-of-High-fidelity-3D-Aware-Images>CtrlNeRF: The Generative Neural Radiation Fields for the Controllable Synthesis of High-fidelity 3D-Aware Images</a></li>
        <li><a href=#Speedy-Splat:-Fast-3D-Gaussian-Splatting-with-Sparse-Pixels-and-Sparse-Primitives>Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives</a></li>
        <li><a href=#Instant3dit:-Multiview-Inpainting-for-Fast-Editing-of-3D-Objects>Instant3dit: Multiview Inpainting for Fast Editing of 3D Objects</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [SfM-Free 3D Gaussian Splatting via Hierarchical Training](http://arxiv.org/abs/2412.01553)  
[[code](https://github.com/jibo27/3dgs_hierarchical_training)]  
Bo Ji, Angela Yao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Standard 3D Gaussian Splatting (3DGS) relies on known or pre-computed camera poses and a sparse point cloud, obtained from structure-from-motion (SfM) preprocessing, to initialize and grow 3D Gaussians. We propose a novel SfM-Free 3DGS (SFGS) method for video input, eliminating the need for known camera poses and SfM preprocessing. Our approach introduces a hierarchical training strategy that trains and merges multiple 3D Gaussian representations -- each optimized for specific scene regions -- into a single, unified 3DGS model representing the entire scene. To compensate for large camera motions, we leverage video frame interpolation models. Additionally, we incorporate multi-source supervision to reduce overfitting and enhance representation. Experimental results reveal that our approach significantly surpasses state-of-the-art SfM-free novel view synthesis methods. On the Tanks and Temples dataset, we improve PSNR by an average of 2.25dB, with a maximum gain of 3.72dB in the best scene. On the CO3D-V2 dataset, we achieve an average PSNR boost of 1.74dB, with a top gain of 3.90dB. The code is available at https://github.com/jibo27/3DGS_Hierarchical_Training.  
  </ol>  
</details>  
  
### [MVImgNet2.0: A Larger-scale Dataset of Multi-view Images](http://arxiv.org/abs/2412.01430)  
Xiaoguang Han, Yushuang Wu, Luyue Shi, Haolin Liu, Hongjie Liao, Lingteng Qiu, Weihao Yuan, Xiaodong Gu, Zilong Dong, Shuguang Cui  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    MVImgNet is a large-scale dataset that contains multi-view images of ~220k real-world objects in 238 classes. As a counterpart of ImageNet, it introduces 3D visual signals via multi-view shooting, making a soft bridge between 2D and 3D vision. This paper constructs the MVImgNet2.0 dataset that expands MVImgNet into a total of ~520k objects and 515 categories, which derives a 3D dataset with a larger scale that is more comparable to ones in the 2D domain. In addition to the expanded dataset scale and category range, MVImgNet2.0 is of a higher quality than MVImgNet owing to four new features: (i) most shoots capture 360-degree views of the objects, which can support the learning of object reconstruction with completeness; (ii) the segmentation manner is advanced to produce foreground object masks of higher accuracy; (iii) a more powerful structure-from-motion method is adopted to derive the camera pose for each frame of a lower estimation error; (iv) higher-quality dense point clouds are reconstructed via advanced methods for objects captured in 360-degree views, which can serve for downstream applications. Extensive experiments confirm the value of the proposed MVImgNet2.0 in boosting the performance of large 3D reconstruction models. MVImgNet2.0 will be public at luyues.github.io/mvimgnet2, including multi-view images of all 520k objects, the reconstructed high-quality point clouds, and data annotation codes, hoping to inspire the broader vision community.  
  </ol>  
</details>  
**comments**: ACM Transactions on Graphics (TOG), SIGGRAPH Asia 2024  
  
### [Look Ma, No Ground Truth! Ground-Truth-Free Tuning of Structure from Motion and Visual SLAM](http://arxiv.org/abs/2412.01116)  
Alejandro Fontan, Javier Civera, Tobias Fischer, Michael Milford  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Evaluation is critical to both developing and tuning Structure from Motion (SfM) and Visual SLAM (VSLAM) systems, but is universally reliant on high-quality geometric ground truth -- a resource that is not only costly and time-intensive but, in many cases, entirely unobtainable. This dependency on ground truth restricts SfM and SLAM applications across diverse environments and limits scalability to real-world scenarios. In this work, we propose a novel ground-truth-free (GTF) evaluation methodology that eliminates the need for geometric ground truth, instead using sensitivity estimation via sampling from both original and noisy versions of input images. Our approach shows strong correlation with traditional ground-truth-based benchmarks and supports GTF hyperparameter tuning. Removing the need for ground truth opens up new opportunities to leverage a much larger number of dataset sources, and for self-supervised and online tuning, with the potential for a data-driven breakthrough analogous to what has occurred in generative AI.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Active Learning via Classifier Impact and Greedy Selection for Interactive Image Retrieval](http://arxiv.org/abs/2412.02310)  
[[code](https://github.com/barleah/greedyal)]  
Leah Bar, Boaz Lerner, Nir Darshan, Rami Ben-Ari  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Active Learning (AL) is a user-interactive approach aimed at reducing annotation costs by selecting the most crucial examples to label. Although AL has been extensively studied for image classification tasks, the specific scenario of interactive image retrieval has received relatively little attention. This scenario presents unique characteristics, including an open-set and class-imbalanced binary classification, starting with very few labeled samples. We introduce a novel batch-mode Active Learning framework named GAL (Greedy Active Learning) that better copes with this application. It incorporates a new acquisition function for sample selection that measures the impact of each unlabeled sample on the classifier. We further embed this strategy in a greedy selection approach, better exploiting the samples within each batch. We evaluate our framework with both linear (SVM) and non-linear MLP/Gaussian Process classifiers. For the Gaussian Process case, we show a theoretical guarantee on the greedy approximation. Finally, we assess our performance for the interactive content-based image retrieval task on several benchmarks and demonstrate its superiority over existing approaches and common baselines. Code is available at https://github.com/barleah/GreedyAL.  
  </ol>  
</details>  
**comments**: Accepted to Transactions on Machine Learning Research (TMLR)  
  
### [Mutli-View 3D Reconstruction using Knowledge Distillation](http://arxiv.org/abs/2412.02039)  
Aditya Dutt, Ishikaa Lunawat, Manpreet Kaur  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Large Foundation Models like Dust3r can produce high quality outputs such as pointmaps, camera intrinsics, and depth estimation, given stereo-image pairs as input. However, the application of these outputs on tasks like Visual Localization requires a large amount of inference time and compute resources. To address these limitations, in this paper, we propose the use of a knowledge distillation pipeline, where we aim to build a student-teacher model with Dust3r as the teacher and explore multiple architectures of student models that are trained using the 3D reconstructed points output by Dust3r. Our goal is to build student models that can learn scene-specific representations and output 3D points with replicable performance such as Dust3r. The data set we used to train our models is 12Scenes. We test two main architectures of models: a CNN-based architecture and a Vision Transformer based architecture. For each architecture, we also compare the use of pre-trained models against models built from scratch. We qualitatively compare the reconstructed 3D points output by the student model against Dust3r's and discuss the various features learned by the student model. We also perform ablation studies on the models through hyperparameter tuning. Overall, we observe that the Vision Transformer presents the best performance visually and quantitatively.  
  </ol>  
</details>  
**comments**: 6 pages, 10 figures  
  
### [Optimizing Domain-Specific Image Retrieval: A Benchmark of FAISS and Annoy with Fine-Tuned Features](http://arxiv.org/abs/2412.01555)  
MD Shaikh Rahman, Syed Maudud E Rabbi, Muhammad Mahbubur Rashid  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Approximate Nearest Neighbor search is one of the keys to high-scale data retrieval performance in many applications. The work is a bridge between feature extraction and ANN indexing through fine-tuning a ResNet50 model with various ANN methods: FAISS and Annoy. We evaluate the systems with respect to indexing time, memory usage, query time, precision, recall, F1-score, and Recall@5 on a custom image dataset. FAISS's Product Quantization can achieve a precision of 98.40% with low memory usage at 0.24 MB index size, and Annoy is the fastest, with average query times of 0.00015 seconds, at a slight cost to accuracy. These results reveal trade-offs among speed, accuracy, and memory efficiency and offer actionable insights into the optimization of feature-based image retrieval systems. This study will serve as a blueprint for constructing actual retrieval pipelines and be built on fine-tuned deep learning networks and associated ANN methods.  
  </ol>  
</details>  
  
### [Neuron Abandoning Attention Flow: Visual Explanation of Dynamics inside CNN Models](http://arxiv.org/abs/2412.01202)  
Yi Liao, Yongsheng Gao, Weichuan Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we present a Neuron Abandoning Attention Flow (NAFlow) method to address the open problem of visually explaining the attention evolution dynamics inside CNNs when making their classification decisions. A novel cascading neuron abandoning back-propagation algorithm is designed to trace neurons in all layers of a CNN that involve in making its prediction to address the problem of significant interference from abandoned neurons. Firstly, a Neuron Abandoning Back-Propagation (NA-BP) module is proposed to generate Back-Propagated Feature Maps (BPFM) by using the inverse function of the intermediate layers of CNN models, on which the neurons not used for decision-making are abandoned. Meanwhile, the cascading NA-BP modules calculate the tensors of importance coefficients which are linearly combined with the tensors of BPFMs to form the NAFlow. Secondly, to be able to visualize attention flow for similarity metric-based CNN models, a new channel contribution weights module is proposed to calculate the importance coefficients via Jacobian Matrix. The effectiveness of the proposed NAFlow is validated on nine widely-used CNN models for various tasks of general image classification, contrastive learning classification, few-shot image classification, and image retrieval.  
  </ol>  
</details>  
  
### [EDTformer: An Efficient Decoder Transformer for Visual Place Recognition](http://arxiv.org/abs/2412.00784)  
Tong Jin, Feng Lu, Shuyu Hu, Chun Yuan, Yunpeng Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual place recognition (VPR) aims to determine the general geographical location of a query image by retrieving visually similar images from a large geo-tagged database. To obtain a global representation for each place image, most approaches typically focus on the aggregation of deep features extracted from a backbone through using current prominent architectures (e.g., CNNs, MLPs, pooling layer and transformer encoder), giving little attention to the transformer decoder. However, we argue that its strong capability in capturing contextual dependencies and generating accurate features holds considerable potential for the VPR task. To this end, we propose an Efficient Decoder Transformer (EDTformer) for feature aggregation, which consists of several stacked simplified decoder blocks followed by two linear layers to directly generate robust and discriminative global representations for VPR. Specifically, we do this by formulating deep features as the keys and values, as well as a set of independent learnable parameters as the queries. EDTformer can fully utilize the contextual information within deep features, then gradually decode and aggregate the effective features into the learnable queries to form the final global representations. Moreover, to provide powerful deep features for EDTformer and further facilitate the robustness, we use the foundation model DINOv2 as the backbone and propose a Low-Rank Parallel Adaptation (LoPA) method to enhance it, which can refine the intermediate features of the backbone progressively in a memory- and parameter-efficient way. As a result, our method not only outperforms single-stage VPR methods on multiple benchmark datasets, but also outperforms two-stage VPR methods which add a re-ranking with considerable cost. Code will be available at https://github.com/Tong-Jin01/EDTformer.  
  </ol>  
</details>  
**comments**: 14 pages, 6 figures  
  
### [EFSA: Episodic Few-Shot Adaptation for Text-to-Image Retrieval](http://arxiv.org/abs/2412.00139)  
Muhammad Huzaifa, Yova Kementchedjhieva  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Text-to-image retrieval is a critical task for managing diverse visual content, but common benchmarks for the task rely on small, single-domain datasets that fail to capture real-world complexity. Pre-trained vision-language models tend to perform well with easy negatives but struggle with hard negatives--visually similar yet incorrect images--especially in open-domain scenarios. To address this, we introduce Episodic Few-Shot Adaptation (EFSA), a novel test-time framework that adapts pre-trained models dynamically to a query's domain by fine-tuning on top-k retrieved candidates and synthetic captions generated for them. EFSA improves performance across diverse domains while preserving generalization, as shown in evaluations on queries from eight highly distinct visual domains and an open-domain retrieval pool of over one million images. Our work highlights the potential of episodic few-shot adaptation to enhance robustness in the critical and understudied task of open-domain text-to-image retrieval.  
  </ol>  
</details>  
  
### [Unleashing the Power of Data Synthesis in Visual Localization](http://arxiv.org/abs/2412.00138)  
Sihang Li, Siqi Tan, Bowen Chang, Jing Zhang, Chen Feng, Yiming Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual localization, which estimates a camera's pose within a known scene, is a long-standing challenge in vision and robotics. Recent end-to-end methods that directly regress camera poses from query images have gained attention for fast inference. However, existing methods often struggle to generalize to unseen views. In this work, we aim to unleash the power of data synthesis to promote the generalizability of pose regression. Specifically, we lift real 2D images into 3D Gaussian Splats with varying appearance and deblurring abilities, which are then used as a data engine to synthesize more posed images. To fully leverage the synthetic data, we build a two-branch joint training pipeline, with an adversarial discriminator to bridge the syn-to-real gap. Experiments on established benchmarks show that our method outperforms state-of-the-art end-to-end approaches, reducing translation and rotation errors by 50% and 21.6% on indoor datasets, and 35.56% and 38.7% on outdoor datasets. We also validate the effectiveness of our method in dynamic driving scenarios under varying weather conditions. Notably, as data synthesis scales up, our method exhibits a growing ability to interpolate and extrapolate training data for localizing unseen views. Project Page: https://ai4ce.github.io/RAP/  
  </ol>  
</details>  
**comments**: 24 pages, 21 figures  
  
### [Relation-Aware Meta-Learning for Zero-shot Sketch-Based Image Retrieval](http://arxiv.org/abs/2412.00120)  
Yang Liu, Jiale Du, Xinbo Gao, Jungong Han  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Sketch-based image retrieval (SBIR) relies on free-hand sketches to retrieve natural photos within the same class. However, its practical application is limited by its inability to retrieve classes absent from the training set. To address this limitation, the task has evolved into Zero-Shot Sketch-Based Image Retrieval (ZS-SBIR), where model performance is evaluated on unseen categories. Traditional SBIR primarily focuses on narrowing the domain gap between photo and sketch modalities. However, in the zero-shot setting, the model not only needs to address this cross-modal discrepancy but also requires a strong generalization capability to transfer knowledge to unseen categories. To this end, we propose a novel framework for ZS-SBIR that employs a pair-based relation-aware quadruplet loss to bridge feature gaps. By incorporating two negative samples from different modalities, the approach prevents positive features from becoming disproportionately distant from one modality while remaining close to another, thus enhancing inter-class separability. We also propose a Relation-Aware Meta-Learning Network (RAMLN) to obtain the margin, a hyper-parameter of cross-modal quadruplet loss, to improve the generalization ability of the model. RAMLN leverages external memory to store feature information, which it utilizes to assign optimal margin values. Experimental results obtained on the extended Sketchy and TU-Berlin datasets show a sharp improvement over existing state-of-the-art methods in ZS-SBIR.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [MamKPD: A Simple Mamba Baseline for Real-Time 2D Keypoint Detection](http://arxiv.org/abs/2412.01422)  
Yonghao Dang, Liyuan Liu, Hui Kang, Ping Ye, Jianqin Yin  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Real-time 2D keypoint detection plays an essential role in computer vision. Although CNN-based and Transformer-based methods have achieved breakthrough progress, they often fail to deliver superior performance and real-time speed. This paper introduces MamKPD, the first efficient yet effective mamba-based pose estimation framework for 2D keypoint detection. The conventional Mamba module exhibits limited information interaction between patches. To address this, we propose a lightweight contextual modeling module (CMM) that uses depth-wise convolutions to model inter-patch dependencies and linear layers to distill the pose cues within each patch. Subsequently, by combining Mamba for global modeling across all patches, MamKPD effectively extracts instances' pose information. We conduct extensive experiments on human and animal pose estimation datasets to validate the effectiveness of MamKPD. Our MamKPD-L achieves 77.3% AP on the COCO dataset with 1492 FPS on an NVIDIA GTX 4090 GPU. Moreover, MamKPD achieves state-of-the-art results on the MPII dataset and competitive results on the AP-10K dataset while saving 85% of the parameters compared to ViTPose. Our project page is available at https://mamkpd.github.io/.  
  </ol>  
</details>  
  
  



## NeRF  

### [SAGA: Surface-Aligned Gaussian Avatar](http://arxiv.org/abs/2412.00845)  
Ronghan Chen, Yang Cong, Jiayue Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper presents a Surface-Aligned Gaussian representation for creating animatable human avatars from monocular videos,aiming at improving the novel view and pose synthesis performance while ensuring fast training and real-time rendering. Recently,3DGS has emerged as a more efficient and expressive alternative to NeRF, and has been used for creating dynamic human avatars. However,when applied to the severely ill-posed task of monocular dynamic reconstruction, the Gaussians tend to overfit the constantly changing regions such as clothes wrinkles or shadows since these regions cannot provide consistent supervision, resulting in noisy geometry and abrupt deformation that typically fail to generalize under novel views and poses.To address these limitations, we present SAGA,i.e.,Surface-Aligned Gaussian Avatar,which aligns the Gaussians with a mesh to enforce well-defined geometry and consistent deformation, thereby improving generalization under novel views and poses. Unlike existing strict alignment methods that suffer from limited expressive power and low realism,SAGA employs a two-stage alignment strategy where the Gaussians are first adhered on while then detached from the mesh, thus facilitating both good geometry and high expressivity. In the Adhered Stage, we improve the flexibility of Adhered-on-Mesh Gaussians by allowing them to flow on the mesh, in contrast to existing methods that rigidly bind Gaussians to fixed location. In the second Detached Stage, we introduce a Gaussian-Mesh Alignment regularization, which allows us to unleash the expressivity by detaching the Gaussians but maintain the geometric alignment by minimizing their location and orientation offsets from the bound triangles. Finally, since the Gaussians may drift outside the bound triangles during optimization, an efficient Walking-on-Mesh strategy is proposed to dynamically update the bound triangles.  
  </ol>  
</details>  
**comments**: Submitted to TPAMI. Major Revision. Project page:
  https://gostinshell.github.io/SAGA/  
  
### [CtrlNeRF: The Generative Neural Radiation Fields for the Controllable Synthesis of High-fidelity 3D-Aware Images](http://arxiv.org/abs/2412.00754)  
Jian Liu, Zhen Yu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The neural radiance field (NERF) advocates learning the continuous representation of 3D geometry through a multilayer perceptron (MLP). By integrating this into a generative model, the generative neural radiance field (GRAF) is capable of producing images from random noise z without 3D supervision. In practice, the shape and appearance are modeled by z_s and z_a, respectively, to manipulate them separately during inference. However, it is challenging to represent multiple scenes using a solitary MLP and precisely control the generation of 3D geometry in terms of shape and appearance. In this paper, we introduce a controllable generative model (i.e. \textbf{CtrlNeRF}) that uses a single MLP network to represent multiple scenes with shared weights. Consequently, we manipulated the shape and appearance codes to realize the controllable generation of high-fidelity images with 3D consistency. Moreover, the model enables the synthesis of novel views that do not exist in the training sets via camera pose alteration and feature interpolation. Extensive experiments were conducted to demonstrate its superiority in 3D-aware image generation compared to its counterparts.  
  </ol>  
</details>  
  
### [Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives](http://arxiv.org/abs/2412.00578)  
Alex Hanson, Allen Tu, Geng Lin, Vasu Singla, Matthias Zwicker, Tom Goldstein  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D Gaussian Splatting (3D-GS) is a recent 3D scene reconstruction technique that enables real-time rendering of novel views by modeling scenes as parametric point clouds of differentiable 3D Gaussians. However, its rendering speed and model size still present bottlenecks, especially in resource-constrained settings. In this paper, we identify and address two key inefficiencies in 3D-GS, achieving substantial improvements in rendering speed, model size, and training time. First, we optimize the rendering pipeline to precisely localize Gaussians in the scene, boosting rendering speed without altering visual fidelity. Second, we introduce a novel pruning technique and integrate it into the training pipeline, significantly reducing model size and training time while further raising rendering speed. Our Speedy-Splat approach combines these techniques to accelerate average rendering speed by a drastic $6.71\times$ across scenes from the Mip-NeRF 360, Tanks & Temples, and Deep Blending datasets with $10.6\times$ fewer primitives than 3D-GS.  
  </ol>  
</details>  
  
### [Instant3dit: Multiview Inpainting for Fast Editing of 3D Objects](http://arxiv.org/abs/2412.00518)  
Amir Barda, Matheus Gadelha, Vladimir G. Kim, Noam Aigerman, Amit H. Bermano, Thibault Groueix  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose a generative technique to edit 3D shapes, represented as meshes, NeRFs, or Gaussian Splats, in approximately 3 seconds, without the need for running an SDS type of optimization. Our key insight is to cast 3D editing as a multiview image inpainting problem, as this representation is generic and can be mapped back to any 3D representation using the bank of available Large Reconstruction Models. We explore different fine-tuning strategies to obtain both multiview generation and inpainting capabilities within the same diffusion model. In particular, the design of the inpainting mask is an important factor of training an inpainting model, and we propose several masking strategies to mimic the types of edits a user would perform on a 3D shape. Our approach takes 3D generative editing from hours to seconds and produces higher-quality results compared to previous works.  
  </ol>  
</details>  
**comments**: project page: https://amirbarda.github.io/Instant3dit.github.io/  
  
  



