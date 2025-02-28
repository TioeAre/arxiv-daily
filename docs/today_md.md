<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#BEV-DWPVO:-BEV-based-Differentiable-Weighted-Procrustes-for-Low-Scale-drift-Monocular-Visual-Odometry-on-Ground>BEV-DWPVO: BEV-based Differentiable Weighted Procrustes for Low Scale-drift Monocular Visual Odometry on Ground</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#A2-GNN:-Angle-Annular-GNN-for-Visual-Descriptor-free-Camera-Relocalization>A2-GNN: Angle-Annular GNN for Visual Descriptor-free Camera Relocalization</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Automatic-Temporal-Segmentation-for-Post-Stroke-Rehabilitation:-A-Keypoint-Detection-and-Temporal-Segmentation-Approach-for-Small-Datasets>Automatic Temporal Segmentation for Post-Stroke Rehabilitation: A Keypoint Detection and Temporal Segmentation Approach for Small Datasets</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#A2-GNN:-Angle-Annular-GNN-for-Visual-Descriptor-free-Camera-Relocalization>A2-GNN: Angle-Annular GNN for Visual Descriptor-free Camera Relocalization</a></li>
        <li><a href=#RUBIK:-A-Structured-Benchmark-for-Image-Matching-across-Geometric-Challenges>RUBIK: A Structured Benchmark for Image Matching across Geometric Challenges</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Identity-preserving-Distillation-Sampling-by-Fixed-Point-Iterator>Identity-preserving Distillation Sampling by Fixed-Point Iterator</a></li>
        <li><a href=#NeRFCom:-Feature-Transform-Coding-Meets-Neural-Radiance-Field-for-Free-View-3D-Scene-Semantic-Transmission>NeRFCom: Feature Transform Coding Meets Neural Radiance Field for Free-View 3D Scene Semantic Transmission</a></li>
        <li><a href=#Compression-in-3D-Gaussian-Splatting:-A-Survey-of-Methods,-Trends,-and-Future-Directions>Compression in 3D Gaussian Splatting: A Survey of Methods, Trends, and Future Directions</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [BEV-DWPVO: BEV-based Differentiable Weighted Procrustes for Low Scale-drift Monocular Visual Odometry on Ground](http://arxiv.org/abs/2502.20078)  
Yufei Wei, Sha Lu, Wangtao Lu, Rong Xiong, Yue Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Monocular Visual Odometry (MVO) provides a cost-effective, real-time positioning solution for autonomous vehicles. However, MVO systems face the common issue of lacking inherent scale information from monocular cameras. Traditional methods have good interpretability but can only obtain relative scale and suffer from severe scale drift in long-distance tasks. Learning-based methods under perspective view leverage large amounts of training data to acquire prior knowledge and estimate absolute scale by predicting depth values. However, their generalization ability is limited due to the need to accurately estimate the depth of each point. In contrast, we propose a novel MVO system called BEV-DWPVO. Our approach leverages the common assumption of a ground plane, using Bird's-Eye View (BEV) feature maps to represent the environment in a grid-based structure with a unified scale. This enables us to reduce the complexity of pose estimation from 6 Degrees of Freedom (DoF) to 3-DoF. Keypoints are extracted and matched within the BEV space, followed by pose estimation through a differentiable weighted Procrustes solver. The entire system is fully differentiable, supporting end-to-end training with only pose supervision and no auxiliary tasks. We validate BEV-DWPVO on the challenging long-sequence datasets NCLT, Oxford, and KITTI, achieving superior results over existing MVO methods on most evaluation metrics.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [A2-GNN: Angle-Annular GNN for Visual Descriptor-free Camera Relocalization](http://arxiv.org/abs/2502.20036)  
Yejun Zhang, Shuzhe Wang, Juho Kannala  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual localization involves estimating the 6-degree-of-freedom (6-DoF) camera pose within a known scene. A critical step in this process is identifying pixel-to-point correspondences between 2D query images and 3D models. Most advanced approaches currently rely on extensive visual descriptors to establish these correspondences, facing challenges in storage, privacy issues and model maintenance. Direct 2D-3D keypoint matching without visual descriptors is becoming popular as it can overcome those challenges. However, existing descriptor-free methods suffer from low accuracy or heavy computation. Addressing this gap, this paper introduces the Angle-Annular Graph Neural Network (A2-GNN), a simple approach that efficiently learns robust geometric structural representations with annular feature extraction. Specifically, this approach clusters neighbors and embeds each group's distance information and angle as supplementary information to capture local structures. Evaluation on matching and visual localization datasets demonstrates that our approach achieves state-of-the-art accuracy with low computational overhead among visual description-free methods. Our code will be released on https://github.com/YejunZhang/a2-gnn.  
  </ol>  
</details>  
**comments**: To be published in 2025 International Conference on 3D Vision (3DV)  
  
  



## Keypoint Detection  

### [Automatic Temporal Segmentation for Post-Stroke Rehabilitation: A Keypoint Detection and Temporal Segmentation Approach for Small Datasets](http://arxiv.org/abs/2502.19766)  
Jisoo Lee, Tamim Ahmed, Thanassis Rikakis, Pavan Turaga  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Rehabilitation is essential and critical for post-stroke patients, addressing both physical and cognitive aspects. Stroke predominantly affects older adults, with 75% of cases occurring in individuals aged 65 and older, underscoring the urgent need for tailored rehabilitation strategies in aging populations. Despite the critical role therapists play in evaluating rehabilitation progress and ensuring the effectiveness of treatment, current assessment methods can often be subjective, inconsistent, and time-consuming, leading to delays in adjusting therapy protocols.   This study aims to address these challenges by providing a solution for consistent and timely analysis. Specifically, we perform temporal segmentation of video recordings to capture detailed activities during stroke patients' rehabilitation. The main application scenario motivating this study is the clinical assessment of daily tabletop object interactions, which are crucial for post-stroke physical rehabilitation.   To achieve this, we present a framework that leverages the biomechanics of movement during therapy sessions. Our solution divides the process into two main tasks: 2D keypoint detection to track patients' physical movements, and 1D time-series temporal segmentation to analyze these movements over time. This dual approach enables automated labeling with only a limited set of real-world data, addressing the challenges of variability in patient movements and limited dataset availability. By tackling these issues, our method shows strong potential for practical deployment in physical therapy settings, enhancing the speed and accuracy of rehabilitation assessments.  
  </ol>  
</details>  
  
  



## Image Matching  

### [A2-GNN: Angle-Annular GNN for Visual Descriptor-free Camera Relocalization](http://arxiv.org/abs/2502.20036)  
Yejun Zhang, Shuzhe Wang, Juho Kannala  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual localization involves estimating the 6-degree-of-freedom (6-DoF) camera pose within a known scene. A critical step in this process is identifying pixel-to-point correspondences between 2D query images and 3D models. Most advanced approaches currently rely on extensive visual descriptors to establish these correspondences, facing challenges in storage, privacy issues and model maintenance. Direct 2D-3D keypoint matching without visual descriptors is becoming popular as it can overcome those challenges. However, existing descriptor-free methods suffer from low accuracy or heavy computation. Addressing this gap, this paper introduces the Angle-Annular Graph Neural Network (A2-GNN), a simple approach that efficiently learns robust geometric structural representations with annular feature extraction. Specifically, this approach clusters neighbors and embeds each group's distance information and angle as supplementary information to capture local structures. Evaluation on matching and visual localization datasets demonstrates that our approach achieves state-of-the-art accuracy with low computational overhead among visual description-free methods. Our code will be released on https://github.com/YejunZhang/a2-gnn.  
  </ol>  
</details>  
**comments**: To be published in 2025 International Conference on 3D Vision (3DV)  
  
### [RUBIK: A Structured Benchmark for Image Matching across Geometric Challenges](http://arxiv.org/abs/2502.19955)  
Thibaut Loiseau, Guillaume Bourmaud  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Camera pose estimation is crucial for many computer vision applications, yet existing benchmarks offer limited insight into method limitations across different geometric challenges. We introduce RUBIK, a novel benchmark that systematically evaluates image matching methods across well-defined geometric difficulty levels. Using three complementary criteria - overlap, scale ratio, and viewpoint angle - we organize 16.5K image pairs from nuScenes into 33 difficulty levels. Our comprehensive evaluation of 14 methods reveals that while recent detector-free approaches achieve the best performance (>47% success rate), they come with significant computational overhead compared to detector-based methods (150-600ms vs. 40-70ms). Even the best performing method succeeds on only 54.8% of the pairs, highlighting substantial room for improvement, particularly in challenging scenarios combining low overlap, large scale differences, and extreme viewpoint changes. Benchmark will be made publicly available.  
  </ol>  
</details>  
  
  



## NeRF  

### [Identity-preserving Distillation Sampling by Fixed-Point Iterator](http://arxiv.org/abs/2502.19930)  
SeonHwa Kim, Jiwon Kim, Soobin Park, Donghoon Ahn, Jiwon Kang, Seungryong Kim, Kyong Hwan Jin, Eunju Cha  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Score distillation sampling (SDS) demonstrates a powerful capability for text-conditioned 2D image and 3D object generation by distilling the knowledge from learned score functions. However, SDS often suffers from blurriness caused by noisy gradients. When SDS meets the image editing, such degradations can be reduced by adjusting bias shifts using reference pairs, but the de-biasing techniques are still corrupted by erroneous gradients. To this end, we introduce Identity-preserving Distillation Sampling (IDS), which compensates for the gradient leading to undesired changes in the results. Based on the analysis that these errors come from the text-conditioned scores, a new regularization technique, called fixed-point iterative regularization (FPR), is proposed to modify the score itself, driving the preservation of the identity even including poses and structures. Thanks to a self-correction by FPR, the proposed method provides clear and unambiguous representations corresponding to the given prompts in image-to-image editing and editable neural radiance field (NeRF). The structural consistency between the source and the edited data is obviously maintained compared to other state-of-the-art methods.  
  </ol>  
</details>  
  
### [NeRFCom: Feature Transform Coding Meets Neural Radiance Field for Free-View 3D Scene Semantic Transmission](http://arxiv.org/abs/2502.19873)  
Weijie Yue, Zhongwei Si, Bolin Wu, Sixian Wang, Xiaoqi Qin, Kai Niu, Jincheng Dai, Ping Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce NeRFCom, a novel communication system designed for end-to-end 3D scene transmission. Compared to traditional systems relying on handcrafted NeRF semantic feature decomposition for compression and well-adaptive channel coding for transmission error correction, our NeRFCom employs a nonlinear transform and learned probabilistic models, enabling flexible variable-rate joint source-channel coding and efficient bandwidth allocation aligned with the NeRF semantic feature's different contribution to the 3D scene synthesis fidelity. Experimental results demonstrate that NeRFCom achieves free-view 3D scene efficient transmission while maintaining robustness under adverse channel conditions.  
  </ol>  
</details>  
  
### [Compression in 3D Gaussian Splatting: A Survey of Methods, Trends, and Future Directions](http://arxiv.org/abs/2502.19457)  
Muhammad Salman Ali, Chaoning Zhang, Marco Cagnazzo, Giuseppe Valenzise, Enzo Tartaglione, Sung-Ho Bae  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D Gaussian Splatting (3DGS) has recently emerged as a pioneering approach in explicit scene rendering and computer graphics. Unlike traditional neural radiance field (NeRF) methods, which typically rely on implicit, coordinate-based models to map spatial coordinates to pixel values, 3DGS utilizes millions of learnable 3D Gaussians. Its differentiable rendering technique and inherent capability for explicit scene representation and manipulation positions 3DGS as a potential game-changer for the next generation of 3D reconstruction and representation technologies. This enables 3DGS to deliver real-time rendering speeds while offering unparalleled editability levels. However, despite its advantages, 3DGS suffers from substantial memory and storage requirements, posing challenges for deployment on resource-constrained devices. In this survey, we provide a comprehensive overview focusing on the scalability and compression of 3DGS. We begin with a detailed background overview of 3DGS, followed by a structured taxonomy of existing compression methods. Additionally, we analyze and compare current methods from the topological perspective, evaluating their strengths and limitations in terms of fidelity, compression ratios, and computational efficiency. Furthermore, we explore how advancements in efficient NeRF representations can inspire future developments in 3DGS optimization. Finally, we conclude with current research challenges and highlight key directions for future exploration.  
  </ol>  
</details>  
  
  



