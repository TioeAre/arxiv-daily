<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#ESVO2:-Direct-Visual-Inertial-Odometry-with-Stereo-Event-Cameras>ESVO2: Direct Visual-Inertial Odometry with Stereo Event Cameras</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Leveraging-Semantic-Cues-from-Foundation-Vision-Models-for-Enhanced-Local-Feature-Correspondence>Leveraging Semantic Cues from Foundation Vision Models for Enhanced Local Feature Correspondence</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Leveraging-Semantic-Cues-from-Foundation-Vision-Models-for-Enhanced-Local-Feature-Correspondence>Leveraging Semantic Cues from Foundation Vision Models for Enhanced Local Feature Correspondence</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Facial-Chick-Sexing:-An-Automated-Chick-Sexing-System-From-Chick-Facial-Image>Facial Chick Sexing: An Automated Chick Sexing System From Chick Facial Image</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Leveraging-Semantic-Cues-from-Foundation-Vision-Models-for-Enhanced-Local-Feature-Correspondence>Leveraging Semantic Cues from Foundation Vision Models for Enhanced Local Feature Correspondence</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#3DArticCyclists:-Generating-Simulated-Dynamic-3D-Cyclists-for-Human-Object-Interaction-(HOI)-and-Autonomous-Driving-Applications>3DArticCyclists: Generating Simulated Dynamic 3D Cyclists for Human-Object Interaction (HOI) and Autonomous Driving Applications</a></li>
        <li><a href=#NeRF-enabled-Analysis-Through-Synthesis-for-ISAR-Imaging-of-Small-Everyday-Objects-with-Sparse-and-Noisy-UWB-Radar-Data>NeRF-enabled Analysis-Through-Synthesis for ISAR Imaging of Small Everyday Objects with Sparse and Noisy UWB Radar Data</a></li>
        <li><a href=#Magnituder-Layers-for-Implicit-Neural-Representations-in-3D>Magnituder Layers for Implicit Neural Representations in 3D</a></li>
        <li><a href=#Improving-3D-Finger-Traits-Recognition-via-Generalizable-Neural-Rendering>Improving 3D Finger Traits Recognition via Generalizable Neural Rendering</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [ESVO2: Direct Visual-Inertial Odometry with Stereo Event Cameras](http://arxiv.org/abs/2410.09374)  
[[code](https://github.com/nail-hnu/esvo2)]  
Junkai Niu, Sheng Zhong, Xiuyuan Lu, Shaojie Shen, Guillermo Gallego, Yi Zhou  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Event-based visual odometry is a specific branch of visual Simultaneous Localization and Mapping (SLAM) techniques, which aims at solving tracking and mapping sub-problems in parallel by exploiting the special working principles of neuromorphic (ie, event-based) cameras. Due to the motion-dependent nature of event data, explicit data association ie, feature matching under large-baseline view-point changes is hardly established, making direct methods a more rational choice. However, state-of-the-art direct methods are limited by the high computational complexity of the mapping sub-problem and the degeneracy of camera pose tracking in certain degrees of freedom (DoF) in rotation. In this paper, we resolve these issues by building an event-based stereo visual-inertial odometry system on top of our previous direct pipeline Event-based Stereo Visual Odometry. Specifically, to speed up the mapping operation, we propose an efficient strategy for sampling contour points according to the local dynamics of events. The mapping performance is also improved in terms of structure completeness and local smoothness by merging the temporal stereo and static stereo results. To circumvent the degeneracy of camera pose tracking in recovering the pitch and yaw components of general six-DoF motion, we introduce IMU measurements as motion priors via pre-integration. To this end, a compact back-end is proposed for continuously updating the IMU bias and predicting the linear velocity, enabling an accurate motion prediction for camera pose tracking. The resulting system scales well with modern high-resolution event cameras and leads to better global positioning accuracy in large-scale outdoor environments. Extensive evaluations on five publicly available datasets featuring different resolutions and scenarios justify the superior performance of the proposed system against five state-of-the-art methods.  
  </ol>  
</details>  
  
  



## SFM  

### [Leveraging Semantic Cues from Foundation Vision Models for Enhanced Local Feature Correspondence](http://arxiv.org/abs/2410.09533)  
Felipe Cadar, Guilherme Potje, Renato Martins, Cédric Demonceaux, Erickson R. Nascimento  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual correspondence is a crucial step in key computer vision tasks, including camera localization, image registration, and structure from motion. The most effective techniques for matching keypoints currently involve using learned sparse or dense matchers, which need pairs of images. These neural networks have a good general understanding of features from both images, but they often struggle to match points from different semantic areas. This paper presents a new method that uses semantic cues from foundation vision model features (like DINOv2) to enhance local feature matching by incorporating semantic reasoning into existing descriptors. Therefore, the learned descriptors do not require image pairs at inference time, allowing feature caching and fast matching using similarity search, unlike learned matchers. We present adapted versions of six existing descriptors, with an average increase in performance of 29% in camera localization, with comparable accuracy to existing matchers as LightGlue and LoFTR in two existing benchmarks. Both code and trained models are available at https://www.verlab.dcc.ufmg.br/descriptors/reasoning_accv24  
  </ol>  
</details>  
**comments**: Accepted in ACCV 2024  
  
  



## Visual Localization  

### [Leveraging Semantic Cues from Foundation Vision Models for Enhanced Local Feature Correspondence](http://arxiv.org/abs/2410.09533)  
Felipe Cadar, Guilherme Potje, Renato Martins, Cédric Demonceaux, Erickson R. Nascimento  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual correspondence is a crucial step in key computer vision tasks, including camera localization, image registration, and structure from motion. The most effective techniques for matching keypoints currently involve using learned sparse or dense matchers, which need pairs of images. These neural networks have a good general understanding of features from both images, but they often struggle to match points from different semantic areas. This paper presents a new method that uses semantic cues from foundation vision model features (like DINOv2) to enhance local feature matching by incorporating semantic reasoning into existing descriptors. Therefore, the learned descriptors do not require image pairs at inference time, allowing feature caching and fast matching using similarity search, unlike learned matchers. We present adapted versions of six existing descriptors, with an average increase in performance of 29% in camera localization, with comparable accuracy to existing matchers as LightGlue and LoFTR in two existing benchmarks. Both code and trained models are available at https://www.verlab.dcc.ufmg.br/descriptors/reasoning_accv24  
  </ol>  
</details>  
**comments**: Accepted in ACCV 2024  
  
  



## Keypoint Detection  

### [Facial Chick Sexing: An Automated Chick Sexing System From Chick Facial Image](http://arxiv.org/abs/2410.09155)  
Marta Veganzones Rodriguez, Thinh Phan, Arthur F. A. Fernandes, Vivian Breen, Jesus Arango, Michael T. Kidd, Ngan Le  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Chick sexing, the process of determining the gender of day-old chicks, is a critical task in the poultry industry due to the distinct roles that each gender plays in production. While effective traditional methods achieve high accuracy, color, and wing feather sexing is exclusive to specific breeds, and vent sexing is invasive and requires trained experts. To address these challenges, we propose a novel approach inspired by facial gender classification techniques in humans: facial chick sexing. This new method does not require expert knowledge and aims to reduce training time while enhancing animal welfare by minimizing chick manipulation. We develop a comprehensive system for training and inference that includes data collection, facial and keypoint detection, facial alignment, and classification. We evaluate our model on two sets of images: Cropped Full Face and Cropped Middle Face, both of which maintain essential facial features of the chick for further analysis. Our experiment demonstrates the promising viability, with a final accuracy of 81.89%, of this approach for future practices in chick sexing by making them more universally applicable.  
  </ol>  
</details>  
  
  



## Image Matching  

### [Leveraging Semantic Cues from Foundation Vision Models for Enhanced Local Feature Correspondence](http://arxiv.org/abs/2410.09533)  
Felipe Cadar, Guilherme Potje, Renato Martins, Cédric Demonceaux, Erickson R. Nascimento  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual correspondence is a crucial step in key computer vision tasks, including camera localization, image registration, and structure from motion. The most effective techniques for matching keypoints currently involve using learned sparse or dense matchers, which need pairs of images. These neural networks have a good general understanding of features from both images, but they often struggle to match points from different semantic areas. This paper presents a new method that uses semantic cues from foundation vision model features (like DINOv2) to enhance local feature matching by incorporating semantic reasoning into existing descriptors. Therefore, the learned descriptors do not require image pairs at inference time, allowing feature caching and fast matching using similarity search, unlike learned matchers. We present adapted versions of six existing descriptors, with an average increase in performance of 29% in camera localization, with comparable accuracy to existing matchers as LightGlue and LoFTR in two existing benchmarks. Both code and trained models are available at https://www.verlab.dcc.ufmg.br/descriptors/reasoning_accv24  
  </ol>  
</details>  
**comments**: Accepted in ACCV 2024  
  
  



## NeRF  

### [3DArticCyclists: Generating Simulated Dynamic 3D Cyclists for Human-Object Interaction (HOI) and Autonomous Driving Applications](http://arxiv.org/abs/2410.10782)  
Eduardo R. Corral-Soto, Yang Liu, Tongtong Cao, Yuan Ren, Liu Bingbing  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Human-object interaction (HOI) and human-scene interaction (HSI) are crucial for human-centric scene understanding applications in Embodied Artificial Intelligence (EAI), robotics, and augmented reality (AR). A common limitation faced in these research areas is the data scarcity problem: insufficient labeled human-scene object pairs on the input images, and limited interaction complexity and granularity between them. Recent HOI and HSI methods have addressed this issue by generating dynamic interactions with rigid objects. But more complex dynamic interactions such as a human rider pedaling an articulated bicycle have been unexplored. To address this limitation, and to enable research on complex dynamic human-articulated object interactions, in this paper we propose a method to generate simulated 3D dynamic cyclist assets and interactions. We designed a methodology for creating a new part-based multi-view articulated synthetic 3D bicycle dataset that we call 3DArticBikes that can be used to train NeRF and 3DGS-based 3D reconstruction methods. We then propose a 3DGS-based parametric bicycle composition model to assemble 8-DoF pose-controllable 3D bicycles. Finally, using dynamic information from cyclist videos, we build a complete synthetic dynamic 3D cyclist (rider pedaling a bicycle) by re-posing a selectable synthetic 3D person while automatically placing the rider onto one of our new articulated 3D bicycles using a proposed 3D Keypoint optimization-based Inverse Kinematics pose refinement. We present both, qualitative and quantitative results where we compare our generated cyclists against those from a recent stable diffusion-based method.  
  </ol>  
</details>  
  
### [NeRF-enabled Analysis-Through-Synthesis for ISAR Imaging of Small Everyday Objects with Sparse and Noisy UWB Radar Data](http://arxiv.org/abs/2410.10085)  
Md Farhan Tasnim Oshim, Albert Reed, Suren Jayasuriya, Tauhidur Rahman  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Inverse Synthetic Aperture Radar (ISAR) imaging presents a formidable challenge when it comes to small everyday objects due to their limited Radar Cross-Section (RCS) and the inherent resolution constraints of radar systems. Existing ISAR reconstruction methods including backprojection (BP) often require complex setups and controlled environments, rendering them impractical for many real-world noisy scenarios. In this paper, we propose a novel Analysis-through-Synthesis (ATS) framework enabled by Neural Radiance Fields (NeRF) for high-resolution coherent ISAR imaging of small objects using sparse and noisy Ultra-Wideband (UWB) radar data with an inexpensive and portable setup. Our end-to-end framework integrates ultra-wideband radar wave propagation, reflection characteristics, and scene priors, enabling efficient 2D scene reconstruction without the need for costly anechoic chambers or complex measurement test beds. With qualitative and quantitative comparisons, we demonstrate that the proposed method outperforms traditional techniques and generates ISAR images of complex scenes with multiple targets and complex structures in Non-Line-of-Sight (NLOS) and noisy scenarios, particularly with limited number of views and sparse UWB radar scans. This work represents a significant step towards practical, cost-effective ISAR imaging of small everyday objects, with broad implications for robotics and mobile sensing applications.  
  </ol>  
</details>  
  
### [Magnituder Layers for Implicit Neural Representations in 3D](http://arxiv.org/abs/2410.09771)  
Sang Min Kim, Byeongchan Kim, Arijit Sehanobish, Krzysztof Choromanski, Dongseok Shim, Avinava Dubey, Min-hwan Oh  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Improving the efficiency and performance of implicit neural representations in 3D, particularly Neural Radiance Fields (NeRF) and Signed Distance Fields (SDF) is crucial for enabling their use in real-time applications. These models, while capable of generating photo-realistic novel views and detailed 3D reconstructions, often suffer from high computational costs and slow inference times. To address this, we introduce a novel neural network layer called the "magnituder", designed to reduce the number of training parameters in these models without sacrificing their expressive power. By integrating magnituders into standard feed-forward layer stacks, we achieve improved inference speed and adaptability. Furthermore, our approach enables a zero-shot performance boost in trained implicit neural representation models through layer-wise knowledge transfer without backpropagation, leading to more efficient scene reconstruction in dynamic environments.  
  </ol>  
</details>  
  
### [Improving 3D Finger Traits Recognition via Generalizable Neural Rendering](http://arxiv.org/abs/2410.09582)  
Hongbin Xu, Junduan Huang, Yuer Ma, Zifeng Li, Wenxiong Kang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D biometric techniques on finger traits have become a new trend and have demonstrated a powerful ability for recognition and anti-counterfeiting. Existing methods follow an explicit 3D pipeline that reconstructs the models first and then extracts features from 3D models. However, these explicit 3D methods suffer from the following problems: 1) Inevitable information dropping during 3D reconstruction; 2) Tight coupling between specific hardware and algorithm for 3D reconstruction. It leads us to a question: Is it indispensable to reconstruct 3D information explicitly in recognition tasks? Hence, we consider this problem in an implicit manner, leaving the nerve-wracking 3D reconstruction problem for learnable neural networks with the help of neural radiance fields (NeRFs). We propose FingerNeRF, a novel generalizable NeRF for 3D finger biometrics. To handle the shape-radiance ambiguity problem that may result in incorrect 3D geometry, we aim to involve extra geometric priors based on the correspondence of binary finger traits like fingerprints or finger veins. First, we propose a novel Trait Guided Transformer (TGT) module to enhance the feature correspondence with the guidance of finger traits. Second, we involve extra geometric constraints on the volume rendering loss with the proposed Depth Distillation Loss and Trait Guided Rendering Loss. To evaluate the performance of the proposed method on different modalities, we collect two new datasets: SCUT-Finger-3D with finger images and SCUT-FingerVein-3D with finger vein images. Moreover, we also utilize the UNSW-3D dataset with fingerprint images for evaluation. In experiments, our FingerNeRF can achieve 4.37% EER on SCUT-Finger-3D dataset, 8.12% EER on SCUT-FingerVein-3D dataset, and 2.90% EER on UNSW-3D dataset, showing the superiority of the proposed implicit method in 3D finger biometrics.  
  </ol>  
</details>  
**comments**: This paper is accepted in IJCV. For further information and access to
  the code, please visit our project page:
  https://scut-bip-lab.github.io/fingernerf/  
  
  



