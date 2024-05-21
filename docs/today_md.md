<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#EdgeLoc:-A-Communication-Adaptive-Parallel-System-for-Real-Time-Localization-in-Infrastructure-Assisted-Autonomous-Driving>EdgeLoc: A Communication-Adaptive Parallel System for Real-Time Localization in Infrastructure-Assisted Autonomous Driving</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#UAV-VisLoc:-A-Large-scale-Dataset-for-UAV-Visual-Localization>UAV-VisLoc: A Large-scale Dataset for UAV Visual Localization</a></li>
        <li><a href=#Register-assisted-aggregation-for-Visual-Place-Recognition>Register assisted aggregation for Visual Place Recognition</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Fast-Generalizable-Gaussian-Splatting-Reconstruction-from-Multi-View-Stereo>Fast Generalizable Gaussian Splatting Reconstruction from Multi-View Stereo</a></li>
        <li><a href=#Embracing-Radiance-Field-Rendering-in-6G:-Over-the-Air-Training-and-Inference-with-3D-Contents>Embracing Radiance Field Rendering in 6G: Over-the-Air Training and Inference with 3D Contents</a></li>
        <li><a href=#NPLMV-PS:-Neural-Point-Light-Multi-View-Photometric-Stereo>NPLMV-PS: Neural Point-Light Multi-View Photometric Stereo</a></li>
        <li><a href=#Searching-Realistic-Looking-Adversarial-Objects-For-Autonomous-Driving-Systems>Searching Realistic-Looking Adversarial Objects For Autonomous Driving Systems</a></li>
        <li><a href=#R-NeRF:-Neural-Radiance-Fields-for-Modeling-RIS-enabled-Wireless-Environments>R-NeRF: Neural Radiance Fields for Modeling RIS-enabled Wireless Environments</a></li>
        <li><a href=#MotionGS-:-Compact-Gaussian-Splatting-SLAM-by-Motion-Filter>MotionGS : Compact Gaussian Splatting SLAM by Motion Filter</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [EdgeLoc: A Communication-Adaptive Parallel System for Real-Time Localization in Infrastructure-Assisted Autonomous Driving](http://arxiv.org/abs/2405.12120)  
Boyi Liu, Jingwen Tong, Yufan Zhuang, Jiawei Shao, Jun Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper presents EdgeLoc, an infrastructure-assisted, real-time localization system for autonomous driving that addresses the incompatibility between traditional localization methods and deep learning approaches. The system is built on top of the Robot Operating System (ROS) and combines the real-time performance of traditional methods with the high accuracy of deep learning approaches. The system leverages edge computing capabilities of roadside units (RSUs) for precise localization to enhance on-vehicle localization that is based on the real-time visual odometry. EdgeLoc is a parallel processing system, utilizing a proposed uncertainty-aware pose fusion solution. It achieves communication adaptivity through online learning and addresses fluctuations via window-based detection. Moreover, it achieves optimal latency and maximum improvement by utilizing auto-splitting vehicle-infrastructure collaborative inference, as well as online distribution learning for decision-making. Even with the most basic end-to-end deep neural network for localization estimation, EdgeLoc realizes a 67.75\% reduction in the localization error for real-time local visual odometry, a 29.95\% reduction for non-real-time collaborative inference, and a 30.26\% reduction compared to Kalman filtering. Finally, accuracy-to-latency conversion was experimentally validated, and an overall experiment was conducted on a practical cellular network. The system is open sourced at https://github.com/LoganCome/EdgeAssistedLocalization.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [UAV-VisLoc: A Large-scale Dataset for UAV Visual Localization](http://arxiv.org/abs/2405.11936)  
Wenjia Xu, Yaxuan Yao, Jiaqi Cao, Zhiwei Wei, Chunbo Liu, Jiuniu Wang, Mugen Peng  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The application of unmanned aerial vehicles (UAV) has been widely extended recently. It is crucial to ensure accurate latitude and longitude coordinates for UAVs, especially when the global navigation satellite systems (GNSS) are disrupted and unreliable. Existing visual localization methods achieve autonomous visual localization without error accumulation by matching the ground-down view image of UAV with the ortho satellite maps. However, collecting UAV ground-down view images across diverse locations is costly, leading to a scarcity of large-scale datasets for real-world scenarios. Existing datasets for UAV visual localization are often limited to small geographic areas or are focused only on urban regions with distinct textures. To address this, we define the UAV visual localization task by determining the UAV's real position coordinates on a large-scale satellite map based on the captured ground-down view. In this paper, we present a large-scale dataset, UAV-VisLoc, to facilitate the UAV visual localization task. This dataset comprises images from diverse drones across 11 locations in China, capturing a range of topographical features. The dataset features images from fixed-wing drones and multi-terrain drones, captured at different altitudes and orientations. Our dataset includes 6,742 drone images and 11 satellite maps, with metadata such as latitude, longitude, altitude, and capture date. Our dataset is tailored to support both the training and testing of models by providing a diverse and extensive data.  
  </ol>  
</details>  
  
### [Register assisted aggregation for Visual Place Recognition](http://arxiv.org/abs/2405.11526)  
Xuan Yu, Zhenyong Fu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Place Recognition (VPR) refers to the process of using computer vision to recognize the position of the current query image. Due to the significant changes in appearance caused by season, lighting, and time spans between query images and database images for retrieval, these differences increase the difficulty of place recognition. Previous methods often discarded useless features (such as sky, road, vehicles) while uncontrolled discarding features that help improve recognition accuracy (such as buildings, trees). To preserve these useful features, we propose a new feature aggregation method to address this issue. Specifically, in order to obtain global and local features that contain discriminative place information, we added some registers on top of the original image tokens to assist in model training. After reallocating attention weights, these registers were discarded. The experimental results show that these registers surprisingly separate unstable features from the original image representation and outperform state-of-the-art methods.  
  </ol>  
</details>  
  
  



## NeRF  

### [Fast Generalizable Gaussian Splatting Reconstruction from Multi-View Stereo](http://arxiv.org/abs/2405.12218)  
Tianqi Liu, Guangcong Wang, Shoukang Hu, Liao Shen, Xinyi Ye, Yuhang Zang, Zhiguo Cao, Wei Li, Ziwei Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present MVSGaussian, a new generalizable 3D Gaussian representation approach derived from Multi-View Stereo (MVS) that can efficiently reconstruct unseen scenes. Specifically, 1) we leverage MVS to encode geometry-aware Gaussian representations and decode them into Gaussian parameters. 2) To further enhance performance, we propose a hybrid Gaussian rendering that integrates an efficient volume rendering design for novel view synthesis. 3) To support fast fine-tuning for specific scenes, we introduce a multi-view geometric consistent aggregation strategy to effectively aggregate the point clouds generated by the generalizable model, serving as the initialization for per-scene optimization. Compared with previous generalizable NeRF-based methods, which typically require minutes of fine-tuning and seconds of rendering per image, MVSGaussian achieves real-time rendering with better synthesis quality for each scene. Compared with the vanilla 3D-GS, MVSGaussian achieves better view synthesis with less training computational cost. Extensive experiments on DTU, Real Forward-facing, NeRF Synthetic, and Tanks and Temples datasets validate that MVSGaussian attains state-of-the-art performance with convincing generalizability, real-time rendering speed, and fast per-scene optimization.  
  </ol>  
</details>  
**comments**: Project page: https://mvsgaussian.github.io/  
  
### [Embracing Radiance Field Rendering in 6G: Over-the-Air Training and Inference with 3D Contents](http://arxiv.org/abs/2405.12155)  
Guanlin Wu, Zhonghao Lyu, Juyong Zhang, Jie Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The efficient representation, transmission, and reconstruction of three-dimensional (3D) contents are becoming increasingly important for sixth-generation (6G) networks that aim to merge virtual and physical worlds for offering immersive communication experiences. Neural radiance field (NeRF) and 3D Gaussian splatting (3D-GS) have recently emerged as two promising 3D representation techniques based on radiance field rendering, which are able to provide photorealistic rendering results for complex scenes. Therefore, embracing NeRF and 3D-GS in 6G networks is envisioned to be a prominent solution to support emerging 3D applications with enhanced quality of experience. This paper provides a comprehensive overview on the integration of NeRF and 3D-GS in 6G. First, we review the basics of the radiance field rendering techniques, and highlight their applications and implementation challenges over wireless networks. Next, we consider the over-the-air training of NeRF and 3D-GS models over wireless networks by presenting various learning techniques. We particularly focus on the federated learning design over a hierarchical device-edge-cloud architecture. Then, we discuss three practical rendering architectures of NeRF and 3D-GS models at wireless network edge. We provide model compression approaches to facilitate the transmission of radiance field models, and present rendering acceleration approaches and joint computation and communication designs to enhance the rendering efficiency. In particular, we propose a new semantic communication enabled 3D content transmission design, in which the radiance field models are exploited as the semantic knowledge base to reduce the communication overhead for distributed inference. Furthermore, we present the utilization of radiance field rendering in wireless applications like radio mapping and radio imaging.  
  </ol>  
</details>  
**comments**: 15 pages,7 figures  
  
### [NPLMV-PS: Neural Point-Light Multi-View Photometric Stereo](http://arxiv.org/abs/2405.12057)  
Fotios Logothetis, Ignas Budvytis, Roberto Cipolla  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this work we present a novel multi-view photometric stereo (PS) method. Like many works in 3D reconstruction we are leveraging neural shape representations and learnt renderers. However, our work differs from the state-of-the-art multi-view PS methods such as PS-NeRF or SuperNormal we explicity leverage per-pixel intensity renderings rather than relying mainly on estimated normals.   We model point light attenuation and explicitly raytrace cast shadows in order to best approximate each points incoming radiance. This is used as input to a fully neural material renderer that uses minimal prior assumptions and it is jointly optimised with the surface. Finally, estimated normal and segmentation maps can also incorporated in order to maximise the surface accuracy.   Our method is among the first to outperform the classical approach of DiLiGenT-MV and achieves average 0.2mm Chamfer distance for objects imaged at approx 1.5m distance away with approximate 400x400 resolution. Moreover, we show robustness to poor normals in low light count scenario, achieving 0.27mm Chamfer distance when pixel rendering is used instead of estimated normals.  
  </ol>  
</details>  
  
### [Searching Realistic-Looking Adversarial Objects For Autonomous Driving Systems](http://arxiv.org/abs/2405.11629)  
Shengxiang Sun, Shenzhe Zhu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Numerous studies on adversarial attacks targeting self-driving policies fail to incorporate realistic-looking adversarial objects, limiting real-world applicability. Building upon prior research that facilitated the transition of adversarial objects from simulations to practical applications, this paper discusses a modified gradient-based texture optimization method to discover realistic-looking adversarial objects. While retaining the core architecture and techniques of the prior research, the proposed addition involves an entity termed the 'Judge'. This agent assesses the texture of a rendered object, assigning a probability score reflecting its realism. This score is integrated into the loss function to encourage the NeRF object renderer to concurrently learn realistic and adversarial textures. The paper analyzes four strategies for developing a robust 'Judge': 1) Leveraging cutting-edge vision-language models. 2) Fine-tuning open-sourced vision-language models. 3) Pretraining neurosymbolic systems. 4) Utilizing traditional image processing techniques. Our findings indicate that strategies 1) and 4) yield less reliable outcomes, pointing towards strategies 2) or 3) as more promising directions for future research.  
  </ol>  
</details>  
  
### [R-NeRF: Neural Radiance Fields for Modeling RIS-enabled Wireless Environments](http://arxiv.org/abs/2405.11541)  
Huiying Yang, Zihan Jin, Chenhao Wu, Rujing Xiong, Robert Caiming Qiu, Zenan Ling  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recently, ray tracing has gained renewed interest with the advent of Reflective Intelligent Surfaces (RIS) technology, a key enabler of 6G wireless communications due to its capability of intelligent manipulation of electromagnetic waves. However, accurately modeling RIS-enabled wireless environments poses significant challenges due to the complex variations caused by various environmental factors and the mobility of RISs. In this paper, we propose a novel modeling approach using Neural Radiance Fields (NeRF) to characterize the dynamics of electromagnetic fields in such environments. Our method utilizes NeRF-based ray tracing to intuitively capture and visualize the complex dynamics of signal propagation, effectively modeling the complete signal pathways from the transmitter to the RIS, and from the RIS to the receiver. This two-stage process accurately characterizes multiple complex transmission paths, enhancing our understanding of signal behavior in real-world scenarios. Our approach predicts the signal field for any specified RIS placement and receiver location, facilitating efficient RIS deployment. Experimental evaluations using both simulated and real-world data validate the significant benefits of our methodology.  
  </ol>  
</details>  
  
### [MotionGS : Compact Gaussian Splatting SLAM by Motion Filter](http://arxiv.org/abs/2405.11129)  
[[code](https://github.com/antonio521/motiongs)]  
Xinli Guo, Peng Han, Weidong Zhang, Hongtian Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    With their high-fidelity scene representation capability, the attention of SLAM field is deeply attracted by the Neural Radiation Field (NeRF) and 3D Gaussian Splatting (3DGS). Recently, there has been a Surge in NeRF-based SLAM, while 3DGS-based SLAM is sparse. A novel 3DGS-based SLAM approach with a fusion of deep visual feature, dual keyframe selection and 3DGS is presented in this paper. Compared with the existing methods, the proposed selectively tracking is achieved by feature extraction and motion filter on each frame. The joint optimization of pose and 3D Gaussian runs through the entire mapping process. Additionally, the coarse-to-fine pose estimation and compact Gaussian scene representation are implemented by dual keyfeature selection and novel loss functions. Experimental results demonstrate that the proposed algorithm not only outperforms the existing methods in tracking and mapping, but also has less memory usage.  
  </ol>  
</details>  
  
  



