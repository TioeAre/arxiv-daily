<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Bias-Eliminated-PnP-for-Stereo-Visual-Odometry:-Provably-Consistent-and-Large-Scale-Localization>Bias-Eliminated PnP for Stereo Visual Odometry: Provably Consistent and Large-Scale Localization</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Dynamic-Camera-Poses-and-Where-to-Find-Them>Dynamic Camera Poses and Where to Find Them</a></li>
        <li><a href=#EdgePoint2:-Compact-Descriptors-for-Superior-Efficiency-and-Accuracy>EdgePoint2: Compact Descriptors for Superior Efficiency and Accuracy</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#A-Guide-to-Structureless-Visual-Localization>A Guide to Structureless Visual Localization</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#EdgePoint2:-Compact-Descriptors-for-Superior-Efficiency-and-Accuracy>EdgePoint2: Compact Descriptors for Superior Efficiency and Accuracy</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#CasualHDRSplat:-Robust-High-Dynamic-Range-3D-Gaussian-Splatting-from-Casually-Captured-Videos>CasualHDRSplat: Robust High Dynamic Range 3D Gaussian Splatting from Casually Captured Videos</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Bias-Eliminated PnP for Stereo Visual Odometry: Provably Consistent and Large-Scale Localization](http://arxiv.org/abs/2504.17410)  
Guangyang Zeng, Yuan Shen, Ziyang Hong, Yuze Hong, Viorela Ila, Guodong Shi, Junfeng Wu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we first present a bias-eliminated weighted (Bias-Eli-W) perspective-n-point (PnP) estimator for stereo visual odometry (VO) with provable consistency. Specifically, leveraging statistical theory, we develop an asymptotically unbiased and $\sqrt {n}$ -consistent PnP estimator that accounts for varying 3D triangulation uncertainties, ensuring that the relative pose estimate converges to the ground truth as the number of features increases. Next, on the stereo VO pipeline side, we propose a framework that continuously triangulates contemporary features for tracking new frames, effectively decoupling temporal dependencies between pose and 3D point errors. We integrate the Bias-Eli-W PnP estimator into the proposed stereo VO pipeline, creating a synergistic effect that enhances the suppression of pose estimation errors. We validate the performance of our method on the KITTI and Oxford RobotCar datasets. Experimental results demonstrate that our method: 1) achieves significant improvements in both relative pose error and absolute trajectory error in large-scale environments; 2) provides reliable localization under erratic and unpredictable robot motions. The successful implementation of the Bias-Eli-W PnP in stereo VO indicates the importance of information screening in robotic estimation tasks with high-uncertainty measurements, shedding light on diverse applications where PnP is a key ingredient.  
  </ol>  
</details>  
**comments**: 10 pages, 7 figures  
  
  



## SFM  

### [Dynamic Camera Poses and Where to Find Them](http://arxiv.org/abs/2504.17788)  
Chris Rockwell, Joseph Tung, Tsung-Yi Lin, Ming-Yu Liu, David F. Fouhey, Chen-Hsuan Lin  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Annotating camera poses on dynamic Internet videos at scale is critical for advancing fields like realistic video generation and simulation. However, collecting such a dataset is difficult, as most Internet videos are unsuitable for pose estimation. Furthermore, annotating dynamic Internet videos present significant challenges even for state-of-theart methods. In this paper, we introduce DynPose-100K, a large-scale dataset of dynamic Internet videos annotated with camera poses. Our collection pipeline addresses filtering using a carefully combined set of task-specific and generalist models. For pose estimation, we combine the latest techniques of point tracking, dynamic masking, and structure-from-motion to achieve improvements over the state-of-the-art approaches. Our analysis and experiments demonstrate that DynPose-100K is both large-scale and diverse across several key attributes, opening up avenues for advancements in various downstream applications.  
  </ol>  
</details>  
**comments**: Accepted to CVPR 2025. Project Page:
  https://research.nvidia.com/labs/dir/dynpose-100k  
  
### [EdgePoint2: Compact Descriptors for Superior Efficiency and Accuracy](http://arxiv.org/abs/2504.17280)  
Haodi Yao, Fenghua He, Ning Hao, Chen Xie  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The field of keypoint extraction, which is essential for vision applications like Structure from Motion (SfM) and Simultaneous Localization and Mapping (SLAM), has evolved from relying on handcrafted methods to leveraging deep learning techniques. While deep learning approaches have significantly improved performance, they often incur substantial computational costs, limiting their deployment in real-time edge applications. Efforts to create lightweight neural networks have seen some success, yet they often result in trade-offs between efficiency and accuracy. Additionally, the high-dimensional descriptors generated by these networks poses challenges for distributed applications requiring efficient communication and coordination, highlighting the need for compact yet competitively accurate descriptors. In this paper, we present EdgePoint2, a series of lightweight keypoint detection and description neural networks specifically tailored for edge computing applications on embedded system. The network architecture is optimized for efficiency without sacrificing accuracy. To train compact descriptors, we introduce a combination of Orthogonal Procrustes loss and similarity loss, which can serve as a general approach for hypersphere embedding distillation tasks. Additionally, we offer 14 sub-models to satisfy diverse application requirements. Our experiments demonstrate that EdgePoint2 consistently achieves state-of-the-art (SOTA) accuracy and efficiency across various challenging scenarios while employing lower-dimensional descriptors (32/48/64). Beyond its accuracy, EdgePoint2 offers significant advantages in flexibility, robustness, and versatility. Consequently, EdgePoint2 emerges as a highly competitive option for visual tasks, especially in contexts demanding adaptability to diverse computational and communication constraints.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [A Guide to Structureless Visual Localization](http://arxiv.org/abs/2504.17636)  
Vojtech Panek, Qunjie Zhou, Yaqing Ding, Sérgio Agostinho, Zuzana Kukelova, Torsten Sattler, Laura Leal-Taixé  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual localization algorithms, i.e., methods that estimate the camera pose of a query image in a known scene, are core components of many applications, including self-driving cars and augmented / mixed reality systems. State-of-the-art visual localization algorithms are structure-based, i.e., they store a 3D model of the scene and use 2D-3D correspondences between the query image and 3D points in the model for camera pose estimation. While such approaches are highly accurate, they are also rather inflexible when it comes to adjusting the underlying 3D model after changes in the scene. Structureless localization approaches represent the scene as a database of images with known poses and thus offer a much more flexible representation that can be easily updated by adding or removing images. Although there is a large amount of literature on structure-based approaches, there is significantly less work on structureless methods. Hence, this paper is dedicated to providing the, to the best of our knowledge, first comprehensive discussion and comparison of structureless methods. Extensive experiments show that approaches that use a higher degree of classical geometric reasoning generally achieve higher pose accuracy. In particular, approaches based on classical absolute or semi-generalized relative pose estimation outperform very recent methods based on pose regression by a wide margin. Compared with state-of-the-art structure-based approaches, the flexibility of structureless methods comes at the cost of (slightly) lower pose accuracy, indicating an interesting direction for future work.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [EdgePoint2: Compact Descriptors for Superior Efficiency and Accuracy](http://arxiv.org/abs/2504.17280)  
Haodi Yao, Fenghua He, Ning Hao, Chen Xie  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The field of keypoint extraction, which is essential for vision applications like Structure from Motion (SfM) and Simultaneous Localization and Mapping (SLAM), has evolved from relying on handcrafted methods to leveraging deep learning techniques. While deep learning approaches have significantly improved performance, they often incur substantial computational costs, limiting their deployment in real-time edge applications. Efforts to create lightweight neural networks have seen some success, yet they often result in trade-offs between efficiency and accuracy. Additionally, the high-dimensional descriptors generated by these networks poses challenges for distributed applications requiring efficient communication and coordination, highlighting the need for compact yet competitively accurate descriptors. In this paper, we present EdgePoint2, a series of lightweight keypoint detection and description neural networks specifically tailored for edge computing applications on embedded system. The network architecture is optimized for efficiency without sacrificing accuracy. To train compact descriptors, we introduce a combination of Orthogonal Procrustes loss and similarity loss, which can serve as a general approach for hypersphere embedding distillation tasks. Additionally, we offer 14 sub-models to satisfy diverse application requirements. Our experiments demonstrate that EdgePoint2 consistently achieves state-of-the-art (SOTA) accuracy and efficiency across various challenging scenarios while employing lower-dimensional descriptors (32/48/64). Beyond its accuracy, EdgePoint2 offers significant advantages in flexibility, robustness, and versatility. Consequently, EdgePoint2 emerges as a highly competitive option for visual tasks, especially in contexts demanding adaptability to diverse computational and communication constraints.  
  </ol>  
</details>  
  
  



## NeRF  

### [CasualHDRSplat: Robust High Dynamic Range 3D Gaussian Splatting from Casually Captured Videos](http://arxiv.org/abs/2504.17728)  
Shucheng Gong, Lingzhe Zhao, Wenpu Li, Hong Xie, Yin Zhang, Shiyu Zhao, Peidong Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recently, photo-realistic novel view synthesis from multi-view images, such as neural radiance field (NeRF) and 3D Gaussian Splatting (3DGS), have garnered widespread attention due to their superior performance. However, most works rely on low dynamic range (LDR) images, which limits the capturing of richer scene details. Some prior works have focused on high dynamic range (HDR) scene reconstruction, typically require capturing of multi-view sharp images with different exposure times at fixed camera positions during exposure times, which is time-consuming and challenging in practice. For a more flexible data acquisition, we propose a one-stage method: \textbf{CasualHDRSplat} to easily and robustly reconstruct the 3D HDR scene from casually captured videos with auto-exposure enabled, even in the presence of severe motion blur and varying unknown exposure time. \textbf{CasualHDRSplat} contains a unified differentiable physical imaging model which first applies continuous-time trajectory constraint to imaging process so that we can jointly optimize exposure time, camera response function (CRF), camera poses, and sharp 3D HDR scene. Extensive experiments demonstrate that our approach outperforms existing methods in terms of robustness and rendering quality. Our source code will be available at https://github.com/WU-CVGL/CasualHDRSplat  
  </ol>  
</details>  
**comments**: Source Code: https://github.com/WU-CVGL/CasualHDRSplat  
  
  



