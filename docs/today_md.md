<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Robust-Monocular-Visual-Odometry-using-Curriculum-Learning>Robust Monocular Visual Odometry using Curriculum Learning</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#DATAP-SfM:-Dynamic-Aware-Tracking-Any-Point-for-Robust-Structure-from-Motion-in-the-Wild>DATAP-SfM: Dynamic-Aware Tracking Any Point for Robust Structure from Motion in the Wild</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Globally-Correlation-Aware-Hard-Negative-Generation>Globally Correlation-Aware Hard Negative Generation</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#DT-LSD:-Deformable-Transformer-based-Line-Segment-Detection>DT-LSD: Deformable Transformer-based Line Segment Detection</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#GazeGaussian:-High-Fidelity-Gaze-Redirection-with-3D-Gaussian-Splatting>GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Robust Monocular Visual Odometry using Curriculum Learning](http://arxiv.org/abs/2411.13438)  
Assaf Lahiany, Oren Gal  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Curriculum Learning (CL), drawing inspiration from natural learning patterns observed in humans and animals, employs a systematic approach of gradually introducing increasingly complex training data during model development. Our work applies innovative CL methodologies to address the challenging geometric problem of monocular Visual Odometry (VO) estimation, which is essential for robot navigation in constrained environments. The primary objective of our research is to push the boundaries of current state-of-the-art (SOTA) benchmarks in monocular VO by investigating various curriculum learning strategies. We enhance the end-to-end Deep-Patch-Visual Odometry (DPVO) framework through the integration of novel CL approaches, with the goal of developing more resilient models capable of maintaining high performance across challenging environments and complex motion scenarios. Our research encompasses several distinctive CL strategies. We develop methods to evaluate sample difficulty based on trajectory motion characteristics, implement sophisticated adaptive scheduling through self-paced weighted loss mechanisms, and utilize reinforcement learning agents for dynamic adjustment of training emphasis. Through comprehensive evaluation on the real-world TartanAir dataset, our Curriculum Learning-based Deep-Patch-Visual Odometry (CL-DPVO) demonstrates superior performance compared to existing SOTA methods, including both feature-based and learning-based VO approaches. The results validate the effectiveness of integrating curriculum learning principles into visual odometry systems.  
  </ol>  
</details>  
**comments**: 8 pages  
  
  



## SFM  

### [DATAP-SfM: Dynamic-Aware Tracking Any Point for Robust Structure from Motion in the Wild](http://arxiv.org/abs/2411.13291)  
Weicai Ye, Xinyu Chen, Ruohao Zhan, Di Huang, Xiaoshui Huang, Haoyi Zhu, Hujun Bao, Wanli Ouyang, Tong He, Guofeng Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper proposes a concise, elegant, and robust pipeline to estimate smooth camera trajectories and obtain dense point clouds for casual videos in the wild. Traditional frameworks, such as ParticleSfM~\cite{zhao2022particlesfm}, address this problem by sequentially computing the optical flow between adjacent frames to obtain point trajectories. They then remove dynamic trajectories through motion segmentation and perform global bundle adjustment. However, the process of estimating optical flow between two adjacent frames and chaining the matches can introduce cumulative errors. Additionally, motion segmentation combined with single-view depth estimation often faces challenges related to scale ambiguity. To tackle these challenges, we propose a dynamic-aware tracking any point (DATAP) method that leverages consistent video depth and point tracking. Specifically, our DATAP addresses these issues by estimating dense point tracking across the video sequence and predicting the visibility and dynamics of each point. By incorporating the consistent video depth prior, the performance of motion segmentation is enhanced. With the integration of DATAP, it becomes possible to estimate and optimize all camera poses simultaneously by performing global bundle adjustments for point tracking classified as static and visible, rather than relying on incremental camera registration. Extensive experiments on dynamic sequences, e.g., Sintel and TUM RGBD dynamic sequences, and on the wild video, e.g., DAVIS, demonstrate that the proposed method achieves state-of-the-art performance in terms of camera pose estimation even in complex dynamic challenge scenes.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Globally Correlation-Aware Hard Negative Generation](http://arxiv.org/abs/2411.13145)  
[[code](https://github.com/pwenjay/gca-hng)]  
Wenjie Peng, Hongxiang Huang, Tianshui Chen, Quhui Ke, Gang Dai, Shuangping Huang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Hard negative generation aims to generate informative negative samples that help to determine the decision boundaries and thus facilitate advancing deep metric learning. Current works select pair/triplet samples, learn their correlations, and fuse them to generate hard negatives. However, these works merely consider the local correlations of selected samples, ignoring global sample correlations that would provide more significant information to generate more informative negatives. In this work, we propose a Globally Correlation-Aware Hard Negative Generation (GCA-HNG) framework, which first learns sample correlations from a global perspective and exploits these correlations to guide generating hardness-adaptive and diverse negatives. Specifically, this approach begins by constructing a structured graph to model sample correlations, where each node represents a specific sample and each edge represents the correlations between corresponding samples. Then, we introduce an iterative graph message propagation to propagate the messages of node and edge through the whole graph and thus learn the sample correlations globally. Finally, with the guidance of the learned global correlations, we propose a channel-adaptive manner to combine an anchor and multiple negatives for HNG. Compared to current methods, GCA-HNG allows perceiving sample correlations with numerous negatives from a global and comprehensive perspective and generates the negatives with better hardness and diversity. Extensive experiment results demonstrate that the proposed GCA-HNG is superior to related methods on four image retrieval benchmark datasets. Codes and trained models are available at \url{https://github.com/PWenJay/GCA-HNG}.  
  </ol>  
</details>  
**comments**: Accepted by IJCV'24  
  
  



## Image Matching  

### [DT-LSD: Deformable Transformer-based Line Segment Detection](http://arxiv.org/abs/2411.13005)  
Sebastian Janampa, Marios Pattichis  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Line segment detection is a fundamental low-level task in computer vision, and improvements in this task can impact more advanced methods that depend on it. Most new methods developed for line segment detection are based on Convolutional Neural Networks (CNNs). Our paper seeks to address challenges that prevent the wider adoption of transformer-based methods for line segment detection. More specifically, we introduce a new model called Deformable Transformer-based Line Segment Detection (DT-LSD) that supports cross-scale interactions and can be trained quickly. This work proposes a novel Deformable Transformer-based Line Segment Detector (DT-LSD) that addresses LETR's drawbacks. For faster training, we introduce Line Contrastive DeNoising (LCDN), a technique that stabilizes the one-to-one matching process and speeds up training by 34 $\times$. We show that DT-LSD is faster and more accurate than its predecessor transformer-based model (LETR) and outperforms all CNN-based models in terms of accuracy. In the Wireframe dataset, DT-LSD achieves 71.7 for $sAP^{10}$ and 73.9 for $sAP^{15}$; while 33.2 for $sAP^{10}$ and 35.1 for $sAP^{15}$ in the YorkUrban dataset.  
  </ol>  
</details>  
  
  



## NeRF  

### [GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting](http://arxiv.org/abs/2411.12981)  
Xiaobao Wei, Peng Chen, Guangyu Li, Ming Lu, Hui Chen, Feng Tian  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Gaze estimation encounters generalization challenges when dealing with out-of-distribution data. To address this problem, recent methods use neural radiance fields (NeRF) to generate augmented data. However, existing methods based on NeRF are computationally expensive and lack facial details. 3D Gaussian Splatting (3DGS) has become the prevailing representation of neural fields. While 3DGS has been extensively examined in head avatars, it faces challenges with accurate gaze control and generalization across different subjects. In this work, we propose GazeGaussian, a high-fidelity gaze redirection method that uses a two-stream 3DGS model to represent the face and eye regions separately. By leveraging the unstructured nature of 3DGS, we develop a novel eye representation for rigid eye rotation based on the target gaze direction. To enhance synthesis generalization across various subjects, we integrate an expression-conditional module to guide the neural renderer. Comprehensive experiments show that GazeGaussian outperforms existing methods in rendering speed, gaze redirection accuracy, and facial synthesis across multiple datasets. We also demonstrate that existing gaze estimation methods can leverage GazeGaussian to improve their generalization performance. The code will be available at: https://ucwxb.github.io/GazeGaussian/.  
  </ol>  
</details>  
  
  



