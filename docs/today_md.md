<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Neural-Active-Structure-from-Motion-in-Dark-and-Textureless-Environment>Neural Active Structure-from-Motion in Dark and Textureless Environment</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#GSSF:-Generalized-Structural-Sparse-Function-for-Deep-Cross-modal-Metric-Learning>GSSF: Generalized Structural Sparse Function for Deep Cross-modal Metric Learning</a></li>
        <li><a href=#Visual-Navigation-of-Digital-Libraries:-Retrieval-and-Classification-of-Images-in-the-National-Library-of-Norway's-Digitised-Book-Collection>Visual Navigation of Digital Libraries: Retrieval and Classification of Images in the National Library of Norway's Digitised Book Collection</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Self-Supervised-Keypoint-Detection-with-Distilled-Depth-Keypoint-Representation>Self-Supervised Keypoint Detection with Distilled Depth Keypoint Representation</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#FrugalNeRF:-Fast-Convergence-for-Few-shot-Novel-View-Synthesis-without-Learned-Priors>FrugalNeRF: Fast Convergence for Few-shot Novel View Synthesis without Learned Priors</a></li>
        <li><a href=#EF-3DGS:-Event-Aided-Free-Trajectory-3D-Gaussian-Splatting>EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting</a></li>
        <li><a href=#Neural-Radiance-Field-Image-Refinement-through-End-to-End-Sampling-Point-Optimization>Neural Radiance Field Image Refinement through End-to-End Sampling Point Optimization</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Neural Active Structure-from-Motion in Dark and Textureless Environment](http://arxiv.org/abs/2410.15378)  
Kazuto Ichimaru, Diego Thomas, Takafumi Iwaguchi, Hiroshi Kawasaki  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Active 3D measurement, especially structured light (SL) has been widely used in various fields for its robustness against textureless or equivalent surfaces by low light illumination. In addition, reconstruction of large scenes by moving the SL system has become popular, however, there have been few practical techniques to obtain the system's precise pose information only from images, since most conventional techniques are based on image features, which cannot be retrieved under textureless environments. In this paper, we propose a simultaneous shape reconstruction and pose estimation technique for SL systems from an image set where sparsely projected patterns onto the scene are observed (i.e. no scene texture information), which we call Active SfM. To achieve this, we propose a full optimization framework of the volumetric shape that employs neural signed distance fields (Neural-SDF) for SL with the goal of not only reconstructing the scene shape but also estimating the poses for each motion of the system. Experimental results show that the proposed method is able to achieve accurate shape reconstruction as well as pose estimation from images where only projected patterns are observed.  
  </ol>  
</details>  
**comments**: Accepted in Asian Conference on Computer Vision 2024  
  
  



## Visual Localization  

### [GSSF: Generalized Structural Sparse Function for Deep Cross-modal Metric Learning](http://arxiv.org/abs/2410.15266)  
[[code](https://github.com/paranioar/gssf)]  
Haiwen Diao, Ying Zhang, Shang Gao, Jiawen Zhu, Long Chen, Huchuan Lu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Cross-modal metric learning is a prominent research topic that bridges the semantic heterogeneity between vision and language. Existing methods frequently utilize simple cosine or complex distance metrics to transform the pairwise features into a similarity score, which suffers from an inadequate or inefficient capability for distance measurements. Consequently, we propose a Generalized Structural Sparse Function to dynamically capture thorough and powerful relationships across modalities for pair-wise similarity learning while remaining concise but efficient. Specifically, the distance metric delicately encapsulates two formats of diagonal and block-diagonal terms, automatically distinguishing and highlighting the cross-channel relevancy and dependency inside a structured and organized topology. Hence, it thereby empowers itself to adapt to the optimal matching patterns between the paired features and reaches a sweet spot between model complexity and capability. Extensive experiments on cross-modal and two extra uni-modal retrieval tasks (image-text retrieval, person re-identification, fine-grained image retrieval) have validated its superiority and flexibility over various popular retrieval frameworks. More importantly, we further discover that it can be seamlessly incorporated into multiple application scenarios, and demonstrates promising prospects from Attention Mechanism to Knowledge Distillation in a plug-and-play manner. Our code is publicly available at: https://github.com/Paranioar/GSSF.  
  </ol>  
</details>  
**comments**: 12 pages, 9 figures, Accepted by TIP2024  
  
### [Visual Navigation of Digital Libraries: Retrieval and Classification of Images in the National Library of Norway's Digitised Book Collection](http://arxiv.org/abs/2410.14969)  
Marie Roald, Magnus Breder Birkenes, Lars Gunnarsønn Bagøien Johnsen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Digital tools for text analysis have long been essential for the searchability and accessibility of digitised library collections. Recent computer vision advances have introduced similar capabilities for visual materials, with deep learning-based embeddings showing promise for analysing visual heritage. Given that many books feature visuals in addition to text, taking advantage of these breakthroughs is critical to making library collections open and accessible. In this work, we present a proof-of-concept image search application for exploring images in the National Library of Norway's pre-1900 books, comparing Vision Transformer (ViT), Contrastive Language-Image Pre-training (CLIP), and Sigmoid loss for Language-Image Pre-training (SigLIP) embeddings for image retrieval and classification. Our results show that the application performs well for exact image retrieval, with SigLIP embeddings slightly outperforming CLIP and ViT in both retrieval and classification tasks. Additionally, SigLIP-based image classification can aid in cleaning image datasets from a digitisation pipeline.  
  </ol>  
</details>  
**comments**: 13 pages, 2 figures, 4 tables, Accepted to the 2024 Computational
  Humanities Research Conference (CHR)  
  
  



## Keypoint Detection  

### [Self-Supervised Keypoint Detection with Distilled Depth Keypoint Representation](http://arxiv.org/abs/2410.14700)  
Aman Anand, Elyas Rashno, Amir Eskandari, Farhana Zulkernine  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Existing unsupervised keypoint detection methods apply artificial deformations to images such as masking a significant portion of images and using reconstruction of original image as a learning objective to detect keypoints. However, this approach lacks depth information in the image and often detects keypoints on the background. To address this, we propose Distill-DKP, a novel cross-modal knowledge distillation framework that leverages depth maps and RGB images for keypoint detection in a self-supervised setting. During training, Distill-DKP extracts embedding-level knowledge from a depth-based teacher model to guide an image-based student model with inference restricted to the student. Experiments show that Distill-DKP significantly outperforms previous unsupervised methods by reducing mean L2 error by 47.15% on Human3.6M, mean average error by 5.67% on Taichi, and improving keypoints accuracy by 1.3% on DeepFashion dataset. Detailed ablation studies demonstrate the sensitivity of knowledge distillation across different layers of the network. Project Page: https://23wm13.github.io/distill-dkp/  
  </ol>  
</details>  
  
  



## NeRF  

### [FrugalNeRF: Fast Convergence for Few-shot Novel View Synthesis without Learned Priors](http://arxiv.org/abs/2410.16271)  
Chin-Yang Lin, Chung-Ho Wu, Chang-Han Yeh, Shih-Han Yen, Cheng Sun, Yu-Lun Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) face significant challenges in few-shot scenarios, primarily due to overfitting and long training times for high-fidelity rendering. Existing methods, such as FreeNeRF and SparseNeRF, use frequency regularization or pre-trained priors but struggle with complex scheduling and bias. We introduce FrugalNeRF, a novel few-shot NeRF framework that leverages weight-sharing voxels across multiple scales to efficiently represent scene details. Our key contribution is a cross-scale geometric adaptation scheme that selects pseudo ground truth depth based on reprojection errors across scales. This guides training without relying on externally learned priors, enabling full utilization of the training data. It can also integrate pre-trained priors, enhancing quality without slowing convergence. Experiments on LLFF, DTU, and RealEstate-10K show that FrugalNeRF outperforms other few-shot NeRF methods while significantly reducing training time, making it a practical solution for efficient and accurate 3D scene reconstruction.  
  </ol>  
</details>  
**comments**: Project page: https://linjohnss.github.io/frugalnerf/  
  
### [EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](http://arxiv.org/abs/2410.15392)  
Bohao Liao, Wei Zhai, Zengyu Wan, Tianzhu Zhang, Yang Cao, Zheng-Jun Zha  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Scene reconstruction from casually captured videos has wide applications in real-world scenarios. With recent advancements in differentiable rendering techniques, several methods have attempted to simultaneously optimize scene representations (NeRF or 3DGS) and camera poses. Despite recent progress, existing methods relying on traditional camera input tend to fail in high-speed (or equivalently low-frame-rate) scenarios. Event cameras, inspired by biological vision, record pixel-wise intensity changes asynchronously with high temporal resolution, providing valuable scene and motion information in blind inter-frame intervals. In this paper, we introduce the event camera to aid scene construction from a casually captured video for the first time, and propose Event-Aided Free-Trajectory 3DGS, called EF-3DGS, which seamlessly integrates the advantages of event cameras into 3DGS through three key components. First, we leverage the Event Generation Model (EGM) to fuse events and frames, supervising the rendered views observed by the event stream. Second, we adopt the Contrast Maximization (CMax) framework in a piece-wise manner to extract motion information by maximizing the contrast of the Image of Warped Events (IWE), thereby calibrating the estimated poses. Besides, based on the Linear Event Generation Model (LEGM), the brightness information encoded in the IWE is also utilized to constrain the 3DGS in the gradient domain. Third, to mitigate the absence of color information of events, we introduce photometric bundle adjustment (PBA) to ensure view consistency across events and frames.We evaluate our method on the public Tanks and Temples benchmark and a newly collected real-world dataset, RealEv-DAVIS. Our project page is https://lbh666.github.io/ef-3dgs/.  
  </ol>  
</details>  
**comments**: Project Page: https://lbh666.github.io/ef-3dgs/  
  
### [Neural Radiance Field Image Refinement through End-to-End Sampling Point Optimization](http://arxiv.org/abs/2410.14958)  
Kazuhiro Ohta, Satoshi Ono  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Field (NeRF), capable of synthesizing high-quality novel viewpoint images, suffers from issues like artifact occurrence due to its fixed sampling points during rendering. This study proposes a method that optimizes sampling points to reduce artifacts and produce more detailed images.  
  </ol>  
</details>  
  
  



