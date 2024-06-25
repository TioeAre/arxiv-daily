<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Imperative-Learning:-A-Self-supervised-Neural-Symbolic-Learning-Framework-for-Robot-Autonomy>Imperative Learning: A Self-supervised Neural-Symbolic Learning Framework for Robot Autonomy</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Crowd-Sourced-NeRF:-Collecting-Data-from-Production-Vehicles-for-3D-Street-View-Reconstruction>Crowd-Sourced NeRF: Collecting Data from Production Vehicles for 3D Street View Reconstruction</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Breaking-the-Frame:-Image-Retrieval-by-Visual-Overlap-Prediction>Breaking the Frame: Image Retrieval by Visual Overlap Prediction</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#A-Certifiable-Algorithm-for-Simultaneous-Shape-Estimation-and-Object-Tracking>A Certifiable Algorithm for Simultaneous Shape Estimation and Object Tracking</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#From-Perfect-to-Noisy-World-Simulation:-Customizable-Embodied-Multi-modal-Perturbations-for-SLAM-Robustness-Benchmarking>From Perfect to Noisy World Simulation: Customizable Embodied Multi-modal Perturbations for SLAM Robustness Benchmarking</a></li>
        <li><a href=#Articulate-your-NeRF:-Unsupervised-articulated-object-modeling-via-conditional-view-synthesis>Articulate your NeRF: Unsupervised articulated object modeling via conditional view synthesis</a></li>
        <li><a href=#Crowd-Sourced-NeRF:-Collecting-Data-from-Production-Vehicles-for-3D-Street-View-Reconstruction>Crowd-Sourced NeRF: Collecting Data from Production Vehicles for 3D Street View Reconstruction</a></li>
        <li><a href=#Towards-Real-Time-Neural-Volumetric-Rendering-on-Mobile-Devices:-A-Measurement-Study>Towards Real-Time Neural Volumetric Rendering on Mobile Devices: A Measurement Study</a></li>
        <li><a href=#Learning-with-Noisy-Ground-Truth:-From-2D-Classification-to-3D-Reconstruction>Learning with Noisy Ground Truth: From 2D Classification to 3D Reconstruction</a></li>
        <li><a href=#psPRF:Pansharpening-Planar-Neural-Radiance-Field-for-Generalized-3D-Reconstruction-Satellite-Imagery>psPRF:Pansharpening Planar Neural Radiance Field for Generalized 3D Reconstruction Satellite Imagery</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Imperative Learning: A Self-supervised Neural-Symbolic Learning Framework for Robot Autonomy](http://arxiv.org/abs/2406.16087)  
Chen Wang, Kaiyi Ji, Junyi Geng, Zhongqiang Ren, Taimeng Fu, Fan Yang, Yifan Guo, Haonan He, Xiangyu Chen, Zitong Zhan, Qiwei Du, Shaoshu Su, Bowen Li, Yuheng Qiu, Yi Du, Qihang Li, Yifan Yang, Xiao Lin, Zhipeng Zhao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Data-driven methods such as reinforcement and imitation learning have achieved remarkable success in robot autonomy. However, their data-centric nature still hinders them from generalizing well to ever-changing environments. Moreover, collecting large datasets for robotic tasks is often impractical and expensive. To overcome these challenges, we introduce a new self-supervised neural-symbolic (NeSy) computational framework, imperative learning (IL), for robot autonomy, leveraging the generalization abilities of symbolic reasoning. The framework of IL consists of three primary components: a neural module, a reasoning engine, and a memory system. We formulate IL as a special bilevel optimization (BLO), which enables reciprocal learning over the three modules. This overcomes the label-intensive obstacles associated with data-driven approaches and takes advantage of symbolic reasoning concerning logical reasoning, physical principles, geometric analysis, etc. We discuss several optimization techniques for IL and verify their effectiveness in five distinct robot autonomy tasks including path planning, rule induction, optimal control, visual odometry, and multi-robot routing. Through various experiments, we show that IL can significantly enhance robot autonomy capabilities and we anticipate that it will catalyze further research across diverse domains.  
  </ol>  
</details>  
  
  



## SFM  

### [Crowd-Sourced NeRF: Collecting Data from Production Vehicles for 3D Street View Reconstruction](http://arxiv.org/abs/2406.16289)  
Tong Qin, Changze Li, Haoyang Ye, Shaowei Wan, Minzhen Li, Hongwei Liu, Ming Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recently, Neural Radiance Fields (NeRF) achieved impressive results in novel view synthesis. Block-NeRF showed the capability of leveraging NeRF to build large city-scale models. For large-scale modeling, a mass of image data is necessary. Collecting images from specially designed data-collection vehicles can not support large-scale applications. How to acquire massive high-quality data remains an opening problem. Noting that the automotive industry has a huge amount of image data, crowd-sourcing is a convenient way for large-scale data collection. In this paper, we present a crowd-sourced framework, which utilizes substantial data captured by production vehicles to reconstruct the scene with the NeRF model. This approach solves the key problem of large-scale reconstruction, that is where the data comes from and how to use them. Firstly, the crowd-sourced massive data is filtered to remove redundancy and keep a balanced distribution in terms of time and space. Then a structure-from-motion module is performed to refine camera poses. Finally, images, as well as poses, are used to train the NeRF model in a certain block. We highlight that we present a comprehensive framework that integrates multiple modules, including data selection, sparse 3D reconstruction, sequence appearance embedding, depth supervision of ground surface, and occlusion completion. The complete system is capable of effectively processing and reconstructing high-quality 3D scenes from crowd-sourced data. Extensive quantitative and qualitative experiments were conducted to validate the performance of our system. Moreover, we proposed an application, named first-view navigation, which leveraged the NeRF model to generate 3D street view and guide the driver with a synthesized video.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Breaking the Frame: Image Retrieval by Visual Overlap Prediction](http://arxiv.org/abs/2406.16204)  
Tong Wei, Philipp Lindenberger, Jiri Matas, Daniel Barath  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose a novel visual place recognition approach, VOP, that efficiently addresses occlusions and complex scenes by shifting from traditional reliance on global image similarities and local features to image overlap prediction. The proposed method enables the identification of visible image sections without requiring expensive feature detection and matching. By focusing on obtaining patch-level embeddings by a Vision Transformer backbone and establishing patch-to-patch correspondences, our approach uses a voting mechanism to assess overlap scores for potential database images, thereby providing a nuanced image retrieval metric in challenging scenarios. VOP leads to more accurate relative pose estimation and localization results on the retrieved image pairs than state-of-the-art baselines on a number of large-scale, real-world datasets. The code is available at https://github.com/weitong8591/vop.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [A Certifiable Algorithm for Simultaneous Shape Estimation and Object Tracking](http://arxiv.org/abs/2406.16837)  
Lorenzo Shaikewitz, Samuel Ubellacker, Luca Carlone  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Applications from manipulation to autonomous vehicles rely on robust and general object tracking to safely perform tasks in dynamic environments. We propose the first certifiably optimal category-level approach for simultaneous shape estimation and pose tracking of an object of known category (e.g. a car). Our approach uses 3D semantic keypoint measurements extracted from an RGB-D image sequence, and phrases the estimation as a fixed-lag smoothing problem. Temporal constraints enforce the object's rigidity (fixed shape) and smooth motion according to a constant-twist motion model. The solutions to this problem are the estimates of the object's state (poses, velocities) and shape (paramaterized according to the active shape model) over the smoothing horizon. Our key contribution is to show that despite the non-convexity of the fixed-lag smoothing problem, we can solve it to certifiable optimality using a small-size semidefinite relaxation. We also present a fast outlier rejection scheme that filters out incorrect keypoint detections with shape and time compatibility tests, and wrap our certifiable solver in a graduated non-convexity scheme. We evaluate the proposed approach on synthetic and real data, showcasing its performance in a table-top manipulation scenario and a drone-based vehicle tracking application.  
  </ol>  
</details>  
**comments**: 11 pages, 6 figures (with appendix). Code released at
  https://github.com/MIT-SPARK/certifiable_tracking. Video available at
  https://youtu.be/eTIlVD9pDtc  
  
  



## NeRF  

### [From Perfect to Noisy World Simulation: Customizable Embodied Multi-modal Perturbations for SLAM Robustness Benchmarking](http://arxiv.org/abs/2406.16850)  
[[code](https://github.com/xiaohao-xu/slam-under-perturbation)]  
Xiaohao Xu, Tianyi Zhang, Sibo Wang, Xiang Li, Yongqi Chen, Ye Li, Bhiksha Raj, Matthew Johnson-Roberson, Xiaonan Huang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Embodied agents require robust navigation systems to operate in unstructured environments, making the robustness of Simultaneous Localization and Mapping (SLAM) models critical to embodied agent autonomy. While real-world datasets are invaluable, simulation-based benchmarks offer a scalable approach for robustness evaluations. However, the creation of a challenging and controllable noisy world with diverse perturbations remains under-explored. To this end, we propose a novel, customizable pipeline for noisy data synthesis, aimed at assessing the resilience of multi-modal SLAM models against various perturbations. The pipeline comprises a comprehensive taxonomy of sensor and motion perturbations for embodied multi-modal (specifically RGB-D) sensing, categorized by their sources and propagation order, allowing for procedural composition. We also provide a toolbox for synthesizing these perturbations, enabling the transformation of clean environments into challenging noisy simulations. Utilizing the pipeline, we instantiate the large-scale Noisy-Replica benchmark, which includes diverse perturbation types, to evaluate the risk tolerance of existing advanced RGB-D SLAM models. Our extensive analysis uncovers the susceptibilities of both neural (NeRF and Gaussian Splatting -based) and non-neural SLAM models to disturbances, despite their demonstrated accuracy in standard benchmarks. Our code is publicly available at https://github.com/Xiaohao-Xu/SLAM-under-Perturbation.  
  </ol>  
</details>  
**comments**: 50 pages. arXiv admin note: substantial text overlap with
  arXiv:2402.08125  
  
### [Articulate your NeRF: Unsupervised articulated object modeling via conditional view synthesis](http://arxiv.org/abs/2406.16623)  
Jianning Deng, Kartic Subr, Hakan Bilen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose a novel unsupervised method to learn the pose and part-segmentation of articulated objects with rigid parts. Given two observations of an object in different articulation states, our method learns the geometry and appearance of object parts by using an implicit model from the first observation, distils the part segmentation and articulation from the second observation while rendering the latter observation. Additionally, to tackle the complexities in the joint optimization of part segmentation and articulation, we propose a voxel grid-based initialization strategy and a decoupled optimization procedure. Compared to the prior unsupervised work, our model obtains significantly better performance, and generalizes to objects with multiple parts while it can be efficiently from few views for the latter observation.  
  </ol>  
</details>  
**comments**: 9 pages for the maincontent, excluding references and supplementaries  
  
### [Crowd-Sourced NeRF: Collecting Data from Production Vehicles for 3D Street View Reconstruction](http://arxiv.org/abs/2406.16289)  
Tong Qin, Changze Li, Haoyang Ye, Shaowei Wan, Minzhen Li, Hongwei Liu, Ming Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recently, Neural Radiance Fields (NeRF) achieved impressive results in novel view synthesis. Block-NeRF showed the capability of leveraging NeRF to build large city-scale models. For large-scale modeling, a mass of image data is necessary. Collecting images from specially designed data-collection vehicles can not support large-scale applications. How to acquire massive high-quality data remains an opening problem. Noting that the automotive industry has a huge amount of image data, crowd-sourcing is a convenient way for large-scale data collection. In this paper, we present a crowd-sourced framework, which utilizes substantial data captured by production vehicles to reconstruct the scene with the NeRF model. This approach solves the key problem of large-scale reconstruction, that is where the data comes from and how to use them. Firstly, the crowd-sourced massive data is filtered to remove redundancy and keep a balanced distribution in terms of time and space. Then a structure-from-motion module is performed to refine camera poses. Finally, images, as well as poses, are used to train the NeRF model in a certain block. We highlight that we present a comprehensive framework that integrates multiple modules, including data selection, sparse 3D reconstruction, sequence appearance embedding, depth supervision of ground surface, and occlusion completion. The complete system is capable of effectively processing and reconstructing high-quality 3D scenes from crowd-sourced data. Extensive quantitative and qualitative experiments were conducted to validate the performance of our system. Moreover, we proposed an application, named first-view navigation, which leveraged the NeRF model to generate 3D street view and guide the driver with a synthesized video.  
  </ol>  
</details>  
  
### [Towards Real-Time Neural Volumetric Rendering on Mobile Devices: A Measurement Study](http://arxiv.org/abs/2406.16068)  
Zhe Wang, Yifei Zhu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) is an emerging technique to synthesize 3D objects from 2D images with a wide range of potential applications. However, rendering existing NeRF models is extremely computation intensive, making it challenging to support real-time interaction on mobile devices. In this paper, we take the first initiative to examine the state-of-the-art real-time NeRF rendering technique from a system perspective. We first define the entire working pipeline of the NeRF serving system. We then identify possible control knobs that are critical to the system from the communication, computation, and visual performance perspective. Furthermore, an extensive measurement study is conducted to reveal the effects of these control knobs on system performance. Our measurement results reveal that different control knobs contribute differently towards improving the system performance, with the mesh granularity being the most effective knob and the quantization being the least effective knob. In addition, diverse hardware device settings and network conditions have to be considered to fully unleash the benefit of operating under the appropriate knobs  
  </ol>  
</details>  
**comments**: This paper is accepted by ACM SIGCOMM Workshop on Emerging Multimedia
  Systems 2024  
  
### [Learning with Noisy Ground Truth: From 2D Classification to 3D Reconstruction](http://arxiv.org/abs/2406.15982)  
Yangdi Lu, Wenbo He  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Deep neural networks has been highly successful in data-intense computer vision applications, while such success relies heavily on the massive and clean data. In real-world scenarios, clean data sometimes is difficult to obtain. For example, in image classification and segmentation tasks, precise annotations of millions samples are generally very expensive and time-consuming. In 3D static scene reconstruction task, most NeRF related methods require the foundational assumption of the static scene (e.g. consistent lighting condition and persistent object positions), which is often violated in real-world scenarios. To address these problem, learning with noisy ground truth (LNGT) has emerged as an effective learning method and shows great potential. In this short survey, we propose a formal definition unify the analysis of LNGT LNGT in the context of different machine learning tasks (classification and regression). Based on this definition, we propose a novel taxonomy to classify the existing work according to the error decomposition with the fundamental definition of machine learning. Further, we provide in-depth analysis on memorization effect and insightful discussion about potential future research opportunities from 2D classification to 3D reconstruction, in the hope of providing guidance to follow-up research.  
  </ol>  
</details>  
**comments**: Computer vision, Noisy Labels, 3D reconstruction, 3D Gaussian Splats,
  (Work still in progress)  
  
### [psPRF:Pansharpening Planar Neural Radiance Field for Generalized 3D Reconstruction Satellite Imagery](http://arxiv.org/abs/2406.15707)  
Tongtong Zhang, Yuanxiang Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Most current NeRF variants for satellites are designed for one specific scene and fall short of generalization to new geometry. Additionally, the RGB images require pan-sharpening as an independent preprocessing step. This paper introduces psPRF, a Planar Neural Radiance Field designed for paired low-resolution RGB (LR-RGB) and high-resolution panchromatic (HR-PAN) images from satellite sensors with Rational Polynomial Cameras (RPC). To capture the cross-modal prior from both of the LR-RGB and HR-PAN images, for the Unet-shaped architecture, we adapt the encoder with explicit spectral-to-spatial convolution (SSConv) to enhance the multimodal representation ability. To support the generalization ability of psRPF across scenes, we adopt projection loss to ensure strong geometry self-supervision. The proposed method is evaluated with the multi-scene WorldView-3 LR-RGB and HR-PAN pairs, and achieves state-of-the-art performance.  
  </ol>  
</details>  
  
  



