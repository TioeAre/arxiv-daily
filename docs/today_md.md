<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Boosting-Self-Supervision-for-Single-View-Scene-Completion-via-Knowledge-Distillation>Boosting Self-Supervision for Single-View Scene Completion via Knowledge Distillation</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#PRAM:-Place-Recognition-Anywhere-Model-for-Efficient-Visual-Localization>PRAM: Place Recognition Anywhere Model for Efficient Visual Localization</a></li>
        <li><a href=#2DLIW-SLAM:2D-LiDAR-Inertial-Wheel-Odometry-with-Real-Time-Loop-Closure>2DLIW-SLAM:2D LiDAR-Inertial-Wheel Odometry with Real-Time Loop Closure</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Connecting-NeRFs,-Images,-and-Text>Connecting NeRFs, Images, and Text</a></li>
        <li><a href=#Boosting-Self-Supervision-for-Single-View-Scene-Completion-via-Knowledge-Distillation>Boosting Self-Supervision for Single-View Scene Completion via Knowledge Distillation</a></li>
        <li><a href=#NeuroNCAP:-Photorealistic-Closed-loop-Safety-Testing-for-Autonomous-Driving>NeuroNCAP: Photorealistic Closed-loop Safety Testing for Autonomous Driving</a></li>
        <li><a href=#G-NeRF:-Geometry-enhanced-Novel-View-Synthesis-from-Single-View-Images>G-NeRF: Geometry-enhanced Novel View Synthesis from Single-View Images</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Boosting Self-Supervision for Single-View Scene Completion via Knowledge Distillation](http://arxiv.org/abs/2404.07933)  
Keonhee Han, Dominik Muhle, Felix Wimbauer, Daniel Cremers  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Inferring scene geometry from images via Structure from Motion is a long-standing and fundamental problem in computer vision. While classical approaches and, more recently, depth map predictions only focus on the visible parts of a scene, the task of scene completion aims to reason about geometry even in occluded regions. With the popularity of neural radiance fields (NeRFs), implicit representations also became popular for scene completion by predicting so-called density fields. Unlike explicit approaches. e.g. voxel-based methods, density fields also allow for accurate depth prediction and novel-view synthesis via image-based rendering. In this work, we propose to fuse the scene reconstruction from multiple images and distill this knowledge into a more accurate single-view scene reconstruction. To this end, we propose Multi-View Behind the Scenes (MVBTS) to fuse density fields from multiple posed images, trained fully self-supervised only from image data. Using knowledge distillation, we use MVBTS to train a single-view scene completion network via direct supervision called KDBTS. It achieves state-of-the-art performance on occupancy prediction, especially in occluded regions.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [PRAM: Place Recognition Anywhere Model for Efficient Visual Localization](http://arxiv.org/abs/2404.07785)  
Fei Xue, Ignas Budvytis, Roberto Cipolla  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Humans localize themselves efficiently in known environments by first recognizing landmarks defined on certain objects and their spatial relationships, and then verifying the location by aligning detailed structures of recognized objects with those in the memory. Inspired by this, we propose the place recognition anywhere model (PRAM) to perform visual localization as efficiently as humans do. PRAM consists of two main components - recognition and registration. In detail, first of all, a self-supervised map-centric landmark definition strategy is adopted, making places in either indoor or outdoor scenes act as unique landmarks. Then, sparse keypoints extracted from images, are utilized as the input to a transformer-based deep neural network for landmark recognition; these keypoints enable PRAM to recognize hundreds of landmarks with high time and memory efficiency. Keypoints along with recognized landmark labels are further used for registration between query images and the 3D landmark map. Different from previous hierarchical methods, PRAM discards global and local descriptors, and reduces over 90% storage. Since PRAM utilizes recognition and landmark-wise verification to replace global reference search and exhaustive matching respectively, it runs 2.4 times faster than prior state-of-the-art approaches. Moreover, PRAM opens new directions for visual localization including multi-modality localization, map-centric feature learning, and hierarchical scene coordinate regression.  
  </ol>  
</details>  
**comments**: project page: https://feixue94.github.io/pram-project/  
  
### [2DLIW-SLAM:2D LiDAR-Inertial-Wheel Odometry with Real-Time Loop Closure](http://arxiv.org/abs/2404.07644)  
Bin Zhang, Zexin Peng, Bi Zeng, Junjie Lu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Due to budgetary constraints, indoor navigation typically employs 2D LiDAR rather than 3D LiDAR. However, the utilization of 2D LiDAR in Simultaneous Localization And Mapping (SLAM) frequently encounters challenges related to motion degeneracy, particularly in geometrically similar environments. To address this problem, this paper proposes a robust, accurate, and multi-sensor-fused 2D LiDAR SLAM system specifically designed for indoor mobile robots. To commence, the original LiDAR data undergoes meticulous processing through point and line extraction. Leveraging the distinctive characteristics of indoor environments, line-line constraints are established to complement other sensor data effectively, thereby augmenting the overall robustness and precision of the system. Concurrently, a tightly-coupled front-end is created, integrating data from the 2D LiDAR, IMU, and wheel odometry, thus enabling real-time state estimation. Building upon this solid foundation, a novel global feature point matching-based loop closure detection algorithm is proposed. This algorithm proves highly effective in mitigating front-end accumulated errors and ultimately constructs a globally consistent map. The experimental results indicate that our system fully meets real-time requirements. When compared to Cartographer, our system not only exhibits lower trajectory errors but also demonstrates stronger robustness, particularly in degeneracy problem.  
  </ol>  
</details>  
  
  



## NeRF  

### [Connecting NeRFs, Images, and Text](http://arxiv.org/abs/2404.07993)  
Francesco Ballerini, Pierluigi Zama Ramirez, Roberto Mirabella, Samuele Salti, Luigi Di Stefano  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRFs) have emerged as a standard framework for representing 3D scenes and objects, introducing a novel data type for information exchange and storage. Concurrently, significant progress has been made in multimodal representation learning for text and image data. This paper explores a novel research direction that aims to connect the NeRF modality with other modalities, similar to established methodologies for images and text. To this end, we propose a simple framework that exploits pre-trained models for NeRF representations alongside multimodal models for text and image processing. Our framework learns a bidirectional mapping between NeRF embeddings and those obtained from corresponding images and text. This mapping unlocks several novel and useful applications, including NeRF zero-shot classification and NeRF retrieval from images or text.  
  </ol>  
</details>  
**comments**: Accepted at CVPRW-INRV 2024  
  
### [Boosting Self-Supervision for Single-View Scene Completion via Knowledge Distillation](http://arxiv.org/abs/2404.07933)  
Keonhee Han, Dominik Muhle, Felix Wimbauer, Daniel Cremers  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Inferring scene geometry from images via Structure from Motion is a long-standing and fundamental problem in computer vision. While classical approaches and, more recently, depth map predictions only focus on the visible parts of a scene, the task of scene completion aims to reason about geometry even in occluded regions. With the popularity of neural radiance fields (NeRFs), implicit representations also became popular for scene completion by predicting so-called density fields. Unlike explicit approaches. e.g. voxel-based methods, density fields also allow for accurate depth prediction and novel-view synthesis via image-based rendering. In this work, we propose to fuse the scene reconstruction from multiple images and distill this knowledge into a more accurate single-view scene reconstruction. To this end, we propose Multi-View Behind the Scenes (MVBTS) to fuse density fields from multiple posed images, trained fully self-supervised only from image data. Using knowledge distillation, we use MVBTS to train a single-view scene completion network via direct supervision called KDBTS. It achieves state-of-the-art performance on occupancy prediction, especially in occluded regions.  
  </ol>  
</details>  
  
### [NeuroNCAP: Photorealistic Closed-loop Safety Testing for Autonomous Driving](http://arxiv.org/abs/2404.07762)  
[[code](https://github.com/wljungbergh/neuroncap)]  
William Ljungbergh, Adam Tonderski, Joakim Johnander, Holger Caesar, Kalle Åström, Michael Felsberg, Christoffer Petersson  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present a versatile NeRF-based simulator for testing autonomous driving (AD) software systems, designed with a focus on sensor-realistic closed-loop evaluation and the creation of safety-critical scenarios. The simulator learns from sequences of real-world driving sensor data and enables reconfigurations and renderings of new, unseen scenarios. In this work, we use our simulator to test the responses of AD models to safety-critical scenarios inspired by the European New Car Assessment Programme (Euro NCAP). Our evaluation reveals that, while state-of-the-art end-to-end planners excel in nominal driving scenarios in an open-loop setting, they exhibit critical flaws when navigating our safety-critical scenarios in a closed-loop setting. This highlights the need for advancements in the safety and real-world usability of end-to-end planners. By publicly releasing our simulator and scenarios as an easy-to-run evaluation suite, we invite the research community to explore, refine, and validate their AD models in controlled, yet highly configurable and challenging sensor-realistic environments. Code and instructions can be found at https://github.com/wljungbergh/NeuroNCAP  
  </ol>  
</details>  
  
### [G-NeRF: Geometry-enhanced Novel View Synthesis from Single-View Images](http://arxiv.org/abs/2404.07474)  
Zixiong Huang, Qi Chen, Libo Sun, Yifan Yang, Naizhou Wang, Mingkui Tan, Qi Wu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Novel view synthesis aims to generate new view images of a given view image collection. Recent attempts address this problem relying on 3D geometry priors (e.g., shapes, sizes, and positions) learned from multi-view images. However, such methods encounter the following limitations: 1) they require a set of multi-view images as training data for a specific scene (e.g., face, car or chair), which is often unavailable in many real-world scenarios; 2) they fail to extract the geometry priors from single-view images due to the lack of multi-view supervision. In this paper, we propose a Geometry-enhanced NeRF (G-NeRF), which seeks to enhance the geometry priors by a geometry-guided multi-view synthesis approach, followed by a depth-aware training. In the synthesis process, inspired that existing 3D GAN models can unconditionally synthesize high-fidelity multi-view images, we seek to adopt off-the-shelf 3D GAN models, such as EG3D, as a free source to provide geometry priors through synthesizing multi-view data. Simultaneously, to further improve the geometry quality of the synthetic data, we introduce a truncation method to effectively sample latent codes within 3D GAN models. To tackle the absence of multi-view supervision for single-view images, we design the depth-aware training approach, incorporating a depth-aware discriminator to guide geometry priors through depth maps. Experiments demonstrate the effectiveness of our method in terms of both qualitative and quantitative results.  
  </ol>  
</details>  
**comments**: CVPR 2024 Accepted Paper  
  
  



