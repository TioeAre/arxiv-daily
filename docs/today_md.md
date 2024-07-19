<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Attenuation-Aware-Weighted-Optical-Flow-with-Medium-Transmission-Map-for-Learning-based-Visual-Odometry-in-Underwater-terrain>Attenuation-Aware Weighted Optical Flow with Medium Transmission Map for Learning-based Visual Odometry in Underwater terrain</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Visual-Haystacks:-Answering-Harder-Questions-About-Sets-of-Images>Visual Haystacks: Answering Harder Questions About Sets of Images</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#EaDeblur-GS:-Event-assisted-3D-Deblur-Reconstruction-with-Gaussian-Splatting>EaDeblur-GS: Event assisted 3D Deblur Reconstruction with Gaussian Splatting</a></li>
        <li><a href=#GeometrySticker:-Enabling-Ownership-Claim-of-Recolorized-Neural-Radiance-Fields>GeometrySticker: Enabling Ownership Claim of Recolorized Neural Radiance Fields</a></li>
        <li><a href=#KFD-NeRF:-Rethinking-Dynamic-NeRF-with-Kalman-Filter>KFD-NeRF: Rethinking Dynamic NeRF with Kalman Filter</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Attenuation-Aware Weighted Optical Flow with Medium Transmission Map for Learning-based Visual Odometry in Underwater terrain](http://arxiv.org/abs/2407.13159)  
Bach Nguyen Gia, Chanh Minh Tran, Kamioka Eiji, Tan Phan Xuan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper addresses the challenge of improving learning-based monocular visual odometry (VO) in underwater environments by integrating principles of underwater optical imaging to manipulate optical flow estimation. Leveraging the inherent properties of underwater imaging, the novel wflow-TartanVO is introduced, enhancing the accuracy of VO systems for autonomous underwater vehicles (AUVs). The proposed method utilizes a normalized medium transmission map as a weight map to adjust the estimated optical flow for emphasizing regions with lower degradation and suppressing uncertain regions affected by underwater light scattering and absorption. wflow-TartanVO does not require fine-tuning of pre-trained VO models, thus promoting its adaptability to different environments and camera models. Evaluation of different real-world underwater datasets demonstrates the outperformance of wflow-TartanVO over baseline VO methods, as evidenced by the considerably reduced Absolute Trajectory Error (ATE). The implementation code is available at: https://github.com/bachzz/wflow-TartanVO  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Visual Haystacks: Answering Harder Questions About Sets of Images](http://arxiv.org/abs/2407.13766)  
Tsung-Han Wu, Giscard Biamby, Jerome Quenum, Ritwik Gupta, Joseph E. Gonzalez, Trevor Darrell, David M. Chan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in Large Multimodal Models (LMMs) have made significant progress in the field of single-image visual question answering. However, these models face substantial challenges when tasked with queries that span extensive collections of images, similar to real-world scenarios like searching through large photo albums, finding specific information across the internet, or monitoring environmental changes through satellite imagery. This paper explores the task of Multi-Image Visual Question Answering (MIQA): given a large set of images and a natural language query, the task is to generate a relevant and grounded response. We propose a new public benchmark, dubbed "Visual Haystacks (VHs)," specifically designed to evaluate LMMs' capabilities in visual retrieval and reasoning over sets of unrelated images, where we perform comprehensive evaluations demonstrating that even robust closed-source models struggle significantly. Towards addressing these shortcomings, we introduce MIRAGE (Multi-Image Retrieval Augmented Generation), a novel retrieval/QA framework tailored for LMMs that confronts the challenges of MIQA with marked efficiency and accuracy improvements over baseline methods. Our evaluation shows that MIRAGE surpasses closed-source GPT-4o models by up to 11% on the VHs benchmark and offers up to 3.4x improvements in efficiency over text-focused multi-stage approaches.  
  </ol>  
</details>  
**comments**: Project page: https://visual-haystacks.github.io  
  
  



## NeRF  

### [EaDeblur-GS: Event assisted 3D Deblur Reconstruction with Gaussian Splatting](http://arxiv.org/abs/2407.13520)  
Yuchen Weng, Zhengwen Shen, Ruofan Chen, Qi Wang, Jun Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D deblurring reconstruction techniques have recently seen significant advancements with the development of Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). Although these techniques can recover relatively clear 3D reconstructions from blurry image inputs, they still face limitations in handling severe blurring and complex camera motion. To address these issues, we propose Event-assisted 3D Deblur Reconstruction with Gaussian Splatting (EaDeblur-GS), which integrates event camera data to enhance the robustness of 3DGS against motion blur. By employing an Adaptive Deviation Estimator (ADE) network to estimate Gaussian center deviations and using novel loss functions, EaDeblur-GS achieves sharp 3D reconstructions in real-time, demonstrating performance comparable to state-of-the-art methods.  
  </ol>  
</details>  
  
### [GeometrySticker: Enabling Ownership Claim of Recolorized Neural Radiance Fields](http://arxiv.org/abs/2407.13390)  
Xiufeng Huang, Ka Chun Cheung, Simon See, Renjie Wan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Remarkable advancements in the recolorization of Neural Radiance Fields (NeRF) have simplified the process of modifying NeRF's color attributes. Yet, with the potential of NeRF to serve as shareable digital assets, there's a concern that malicious users might alter the color of NeRF models and falsely claim the recolorized version as their own. To safeguard against such breaches of ownership, enabling original NeRF creators to establish rights over recolorized NeRF is crucial. While approaches like CopyRNeRF have been introduced to embed binary messages into NeRF models as digital signatures for copyright protection, the process of recolorization can remove these binary messages. In our paper, we present GeometrySticker, a method for seamlessly integrating binary messages into the geometry components of radiance fields, akin to applying a sticker. GeometrySticker can embed binary messages into NeRF models while preserving the effectiveness of these messages against recolorization. Our comprehensive studies demonstrate that GeometrySticker is adaptable to prevalent NeRF architectures and maintains a commendable level of robustness against various distortions. Project page: https://kevinhuangxf.github.io/GeometrySticker/.  
  </ol>  
</details>  
  
### [KFD-NeRF: Rethinking Dynamic NeRF with Kalman Filter](http://arxiv.org/abs/2407.13185)  
Yifan Zhan, Zhuoxiao Li, Muyao Niu, Zhihang Zhong, Shohei Nobuhara, Ko Nishino, Yinqiang Zheng  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce KFD-NeRF, a novel dynamic neural radiance field integrated with an efficient and high-quality motion reconstruction framework based on Kalman filtering. Our key idea is to model the dynamic radiance field as a dynamic system whose temporally varying states are estimated based on two sources of knowledge: observations and predictions. We introduce a novel plug-in Kalman filter guided deformation field that enables accurate deformation estimation from scene observations and predictions. We use a shallow Multi-Layer Perceptron (MLP) for observations and model the motion as locally linear to calculate predictions with motion equations. To further enhance the performance of the observation MLP, we introduce regularization in the canonical space to facilitate the network's ability to learn warping for different frames. Additionally, we employ an efficient tri-plane representation for encoding the canonical space, which has been experimentally demonstrated to converge quickly with high quality. This enables us to use a shallower observation MLP, consisting of just two layers in our implementation. We conduct experiments on synthetic and real data and compare with past dynamic NeRF methods. Our KFD-NeRF demonstrates similar or even superior rendering performance within comparable computational time and achieves state-of-the-art view synthesis performance with thorough training.  
  </ol>  
</details>  
**comments**: accepted to eccv2024  
  
  



