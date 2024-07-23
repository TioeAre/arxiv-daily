<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Reinforcement-Learning-Meets-Visual-Odometry>Reinforcement Learning Meets Visual Odometry</a></li>
        <li><a href=#Semi-Supervised-Pipe-Video-Temporal-Defect-Interval-Localization>Semi-Supervised Pipe Video Temporal Defect Interval Localization</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#RADA:-Robust-and-Accurate-Feature-Learning-with-Domain-Adaptation>RADA: Robust and Accurate Feature Learning with Domain Adaptation</a></li>
        <li><a href=#Online-Global-Loop-Closure-Detection-for-Large-Scale-Multi-Session-Graph-Based-SLAM>Online Global Loop Closure Detection for Large-Scale Multi-Session Graph-Based SLAM</a></li>
        <li><a href=#Appearance-Based-Loop-Closure-Detection-for-Online-Large-Scale-and-Long-Term-Operation>Appearance-Based Loop Closure Detection for Online Large-Scale and Long-Term Operation</a></li>
        <li><a href=#Double-Layer-Soft-Data-Fusion-for-Indoor-Robot-WiFi-Visual-Localization>Double-Layer Soft Data Fusion for Indoor Robot WiFi-Visual Localization</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#RADA:-Robust-and-Accurate-Feature-Learning-with-Domain-Adaptation>RADA: Robust and Accurate Feature Learning with Domain Adaptation</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#RADA:-Robust-and-Accurate-Feature-Learning-with-Domain-Adaptation>RADA: Robust and Accurate Feature Learning with Domain Adaptation</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#BoostMVSNeRFs:-Boosting-MVS-based-NeRFs-to-Generalizable-View-Synthesis-in-Large-scale-Scenes>BoostMVSNeRFs: Boosting MVS-based NeRFs to Generalizable View Synthesis in Large-scale Scenes</a></li>
        <li><a href=#Enhancement-of-3D-Gaussian-Splatting-using-Raw-Mesh-for-Photorealistic-Recreation-of-Architectures>Enhancement of 3D Gaussian Splatting using Raw Mesh for Photorealistic Recreation of Architectures</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Reinforcement Learning Meets Visual Odometry](http://arxiv.org/abs/2407.15626)  
Nico Messikommer, Giovanni Cioffi, Mathias Gehrig, Davide Scaramuzza  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Odometry (VO) is essential to downstream mobile robotics and augmented/virtual reality tasks. Despite recent advances, existing VO methods still rely on heuristic design choices that require several weeks of hyperparameter tuning by human experts, hindering generalizability and robustness. We address these challenges by reframing VO as a sequential decision-making task and applying Reinforcement Learning (RL) to adapt the VO process dynamically. Our approach introduces a neural network, operating as an agent within the VO pipeline, to make decisions such as keyframe and grid-size selection based on real-time conditions. Our method minimizes reliance on heuristic choices using a reward function based on pose error, runtime, and other metrics to guide the system. Our RL framework treats the VO system and the image sequence as an environment, with the agent receiving observations from keypoints, map statistics, and prior poses. Experimental results using classical VO methods and public benchmarks demonstrate improvements in accuracy and robustness, validating the generalizability of our RL-enhanced VO approach to different scenarios. We believe this paradigm shift advances VO technology by eliminating the need for time-intensive parameter tuning of heuristics.  
  </ol>  
</details>  
  
### [Semi-Supervised Pipe Video Temporal Defect Interval Localization](http://arxiv.org/abs/2407.15170)  
Zhu Huang, Gang Pan, Chao Kang, YaoZhi Lv  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In sewer pipe Closed-Circuit Television (CCTV) inspection, accurate temporal defect localization is essential for effective defect classification, detection, segmentation and quantification. Industry standards typically do not require time-interval annotations, even though they are more informative than time-point annotations for defect localization, resulting in additional annotation costs when fully supervised methods are used. Additionally, differences in scene types and camera motion patterns between pipe inspections and Temporal Action Localization (TAL) hinder the effective transfer of point-supervised TAL methods. Therefore, this study introduces a Semi-supervised multi-Prototype-based method incorporating visual Odometry for enhanced attention guidance (PipeSPO). PipeSPO fully leverages unlabeled data through unsupervised pretext tasks and utilizes time-point annotated data with a weakly supervised multi-prototype-based method, relying on visual odometry features to capture camera pose information. Experiments on real-world datasets demonstrate that PipeSPO achieves 41.89% average precision across Intersection over Union (IoU) thresholds of 0.1-0.7, improving by 8.14% over current state-of-the-art methods.  
  </ol>  
</details>  
**comments**: 13 pages, 3 figures  
  
  



## Visual Localization  

### [RADA: Robust and Accurate Feature Learning with Domain Adaptation](http://arxiv.org/abs/2407.15791)  
Jingtai He, Gehao Zhang, Tingting Liu, Songlin Du  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in keypoint detection and descriptor extraction have shown impressive performance in local feature learning tasks. However, existing methods generally exhibit suboptimal performance under extreme conditions such as significant appearance changes and domain shifts. In this study, we introduce a multi-level feature aggregation network that incorporates two pivotal components to facilitate the learning of robust and accurate features with domain adaptation. First, we employ domain adaptation supervision to align high-level feature distributions across different domains to achieve invariant domain representations. Second, we propose a Transformer-based booster that enhances descriptor robustness by integrating visual and geometric information through wave position encoding concepts, effectively handling complex conditions. To ensure the accuracy and robustness of features, we adopt a hierarchical architecture to capture comprehensive information and apply meticulous targeted supervision to keypoint detection, descriptor extraction, and their coupled processing. Extensive experiments demonstrate that our method, RADA, achieves excellent results in image matching, camera pose estimation, and visual localization tasks.  
  </ol>  
</details>  
  
### [Online Global Loop Closure Detection for Large-Scale Multi-Session Graph-Based SLAM](http://arxiv.org/abs/2407.15305)  
Mathieu Labbe, François Michaud  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    For large-scale and long-term simultaneous localization and mapping (SLAM), a robot has to deal with unknown initial positioning caused by either the kidnapped robot problem or multi-session mapping. This paper addresses these problems by tying the SLAM system with a global loop closure detection approach, which intrinsically handles these situations. However, online processing for global loop closure detection approaches is generally influenced by the size of the environment. The proposed graph-based SLAM system uses a memory management approach that only consider portions of the map to satisfy online processing requirements. The approach is tested and demonstrated using five indoor mapping sessions of a building using a robot equipped with a laser rangefinder and a Kinect.  
  </ol>  
</details>  
**comments**: 6 pages, 12 figures  
  
### [Appearance-Based Loop Closure Detection for Online Large-Scale and Long-Term Operation](http://arxiv.org/abs/2407.15304)  
Mathieu Labbé, François Michaud  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In appearance-based localization and mapping, loop closure detection is the process used to determinate if the current observation comes from a previously visited location or a new one. As the size of the internal map increases, so does the time required to compare new observations with all stored locations, eventually limiting online processing. This paper presents an online loop closure detection approach for large-scale and long-term operation. The approach is based on a memory management method, which limits the number of locations used for loop closure detection so that the computation time remains under real-time constraints. The idea consists of keeping the most recent and frequently observed locations in a Working Memory (WM) used for loop closure detection, and transferring the others into a Long-Term Memory (LTM). When a match is found between the current location and one stored in WM, associated locations stored in LTM can be updated and remembered for additional loop closure detections. Results demonstrate the approach's adaptability and scalability using ten standard data sets from other appearance-based loop closure approaches, one custom data set using real images taken over a 2 km loop of our university campus, and one custom data set (7 hours) using virtual images from the racing video game ``Need for Speed: Most Wanted''.  
  </ol>  
</details>  
**comments**: 12 pages, 11 figures  
  
### [Double-Layer Soft Data Fusion for Indoor Robot WiFi-Visual Localization](http://arxiv.org/abs/2407.14643)  
Yuehua Ding, Jean-Francois Dollinger, Vincent Vauchey, Mourad Zghal  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper presents a novel WiFi-Visual data fusion method for indoor robot (TIAGO++) localization. This method can use 10 WiFi samples and 4 low-resolution images ( $58 \times 58$ in pixels) to localize a indoor robot with an average error distance about 1.32 meters. The experiment test is 3 months after the data collection in a general teaching building, whose WiFi and visual environments are partially changed. This indirectly shows the robustness of the proposed method.   Instead of neural network design, this paper focuses on the soft data fusion to prevent unbounded errors in visual localization. A double-layer soft data fusion is proposed. The proposed soft data fusion includes the first-layer WiFi-Visual feature fusion and the second-layer decision vector fusion. Firstly, motivated by the excellent capability of neural network in image processing and recognition, the temporal-spatial features are extracted from WiFi data, these features are represented in image form. Secondly, the WiFi temporal-spatial features in image form and the visual features taken by the robot camera are combined together, and are jointly exploited by a classification neural network to produce a likelihood vector for WiFi-Visual localization. This is called first-layer WiFi-Visual fusion. Similarly, these two types of features can exploited separately by neural networks to produce another two independent likelihood vectors. Thirdly, the three likelihood vectors are fused by Hadamard product and median filtering to produce the final likelihood vector for localization. This called the second-layer decision vector fusion. The proposed soft data fusion does not apply any threshold or prioritize any data source over the other in the fusion process. It never excludes the positions of low probabilities, which can avoid the information loss due to a hard decision. The demo video is provided. The code will be open.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [RADA: Robust and Accurate Feature Learning with Domain Adaptation](http://arxiv.org/abs/2407.15791)  
Jingtai He, Gehao Zhang, Tingting Liu, Songlin Du  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in keypoint detection and descriptor extraction have shown impressive performance in local feature learning tasks. However, existing methods generally exhibit suboptimal performance under extreme conditions such as significant appearance changes and domain shifts. In this study, we introduce a multi-level feature aggregation network that incorporates two pivotal components to facilitate the learning of robust and accurate features with domain adaptation. First, we employ domain adaptation supervision to align high-level feature distributions across different domains to achieve invariant domain representations. Second, we propose a Transformer-based booster that enhances descriptor robustness by integrating visual and geometric information through wave position encoding concepts, effectively handling complex conditions. To ensure the accuracy and robustness of features, we adopt a hierarchical architecture to capture comprehensive information and apply meticulous targeted supervision to keypoint detection, descriptor extraction, and their coupled processing. Extensive experiments demonstrate that our method, RADA, achieves excellent results in image matching, camera pose estimation, and visual localization tasks.  
  </ol>  
</details>  
  
  



## Image Matching  

### [RADA: Robust and Accurate Feature Learning with Domain Adaptation](http://arxiv.org/abs/2407.15791)  
Jingtai He, Gehao Zhang, Tingting Liu, Songlin Du  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in keypoint detection and descriptor extraction have shown impressive performance in local feature learning tasks. However, existing methods generally exhibit suboptimal performance under extreme conditions such as significant appearance changes and domain shifts. In this study, we introduce a multi-level feature aggregation network that incorporates two pivotal components to facilitate the learning of robust and accurate features with domain adaptation. First, we employ domain adaptation supervision to align high-level feature distributions across different domains to achieve invariant domain representations. Second, we propose a Transformer-based booster that enhances descriptor robustness by integrating visual and geometric information through wave position encoding concepts, effectively handling complex conditions. To ensure the accuracy and robustness of features, we adopt a hierarchical architecture to capture comprehensive information and apply meticulous targeted supervision to keypoint detection, descriptor extraction, and their coupled processing. Extensive experiments demonstrate that our method, RADA, achieves excellent results in image matching, camera pose estimation, and visual localization tasks.  
  </ol>  
</details>  
  
  



## NeRF  

### [BoostMVSNeRFs: Boosting MVS-based NeRFs to Generalizable View Synthesis in Large-scale Scenes](http://arxiv.org/abs/2407.15848)  
Chih-Hai Su, Chih-Yao Hu, Shr-Ruei Tsai, Jie-Ying Lee, Chin-Yang Lin, Yu-Lun Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    While Neural Radiance Fields (NeRFs) have demonstrated exceptional quality, their protracted training duration remains a limitation. Generalizable and MVS-based NeRFs, although capable of mitigating training time, often incur tradeoffs in quality. This paper presents a novel approach called BoostMVSNeRFs to enhance the rendering quality of MVS-based NeRFs in large-scale scenes. We first identify limitations in MVS-based NeRF methods, such as restricted viewport coverage and artifacts due to limited input views. Then, we address these limitations by proposing a new method that selects and combines multiple cost volumes during volume rendering. Our method does not require training and can adapt to any MVS-based NeRF methods in a feed-forward fashion to improve rendering quality. Furthermore, our approach is also end-to-end trainable, allowing fine-tuning on specific scenes. We demonstrate the effectiveness of our method through experiments on large-scale datasets, showing significant rendering quality improvements in large-scale scenes and unbounded outdoor scenarios. We release the source code of BoostMVSNeRFs at https://su-terry.github.io/BoostMVSNeRFs/.  
  </ol>  
</details>  
**comments**: SIGGRAPH 2024 Conference Papers. Project page:
  https://su-terry.github.io/BoostMVSNeRFs/  
  
### [Enhancement of 3D Gaussian Splatting using Raw Mesh for Photorealistic Recreation of Architectures](http://arxiv.org/abs/2407.15435)  
Ruizhe Wang, Chunliang Hua, Tomakayev Shingys, Mengyuan Niu, Qingxin Yang, Lizhong Gao, Yi Zheng, Junyan Yang, Qiao Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The photorealistic reconstruction and rendering of architectural scenes have extensive applications in industries such as film, games, and transportation. It also plays an important role in urban planning, architectural design, and the city's promotion, especially in protecting historical and cultural relics. The 3D Gaussian Splatting, due to better performance over NeRF, has become a mainstream technology in 3D reconstruction. Its only input is a set of images but it relies heavily on geometric parameters computed by the SfM process. At the same time, there is an existing abundance of raw 3D models, that could inform the structural perception of certain buildings but cannot be applied. In this paper, we propose a straightforward method to harness these raw 3D models to guide 3D Gaussians in capturing the basic shape of the building and improve the visual quality of textures and details when photos are captured non-systematically. This exploration opens up new possibilities for improving the effectiveness of 3D reconstruction techniques in the field of architectural design.  
  </ol>  
</details>  
  
  



