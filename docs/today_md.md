<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Thermal-LiDAR-Fusion-for-Robust-Tunnel-Localization-in-GNSS-Denied-and-Low-Visibility-Conditions>Thermal-LiDAR Fusion for Robust Tunnel Localization in GNSS-Denied and Low-Visibility Conditions</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Estimating-the-Diameter-at-Breast-Height-of-Trees-in-a-Forest-With-a-Single-360-Camera>Estimating the Diameter at Breast Height of Trees in a Forest With a Single 360 Camera</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Thermal-LiDAR-Fusion-for-Robust-Tunnel-Localization-in-GNSS-Denied-and-Low-Visibility-Conditions>Thermal-LiDAR Fusion for Robust Tunnel Localization in GNSS-Denied and Low-Visibility Conditions</a></li>
        <li><a href=#LiftFeat:-3D-Geometry-Aware-Local-Feature-Matching>LiftFeat: 3D Geometry-Aware Local Feature Matching</a></li>
        <li><a href=#Seeing-the-Abstract:-Translating-the-Abstract-Language-for-Vision-Language-Models>Seeing the Abstract: Translating the Abstract Language for Vision Language Models</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#LiftFeat:-3D-Geometry-Aware-Local-Feature-Matching>LiftFeat: 3D Geometry-Aware Local Feature Matching</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Thermal-LiDAR Fusion for Robust Tunnel Localization in GNSS-Denied and Low-Visibility Conditions](http://arxiv.org/abs/2505.03565)  
Lukas Schichler, Karin Festl, Selim Solmaz, Daniel Watzenig  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Despite significant progress in autonomous navigation, a critical gap remains in ensuring reliable localization in hazardous environments such as tunnels, urban disaster zones, and underground structures. Tunnels present a uniquely difficult scenario: they are not only prone to GNSS signal loss, but also provide little features for visual localization due to their repetitive walls and poor lighting. These conditions degrade conventional vision-based and LiDAR-based systems, which rely on distinguishable environmental features. To address this, we propose a novel sensor fusion framework that integrates a thermal camera with a LiDAR to enable robust localization in tunnels and other perceptually degraded environments. The thermal camera provides resilience in low-light or smoke conditions, while the LiDAR delivers precise depth perception and structural awareness. By combining these sensors, our framework ensures continuous and accurate localization across diverse and dynamic environments. We use an Extended Kalman Filter (EKF) to fuse multi-sensor inputs, and leverages visual odometry and SLAM (Simultaneous Localization and Mapping) techniques to process the sensor data, enabling robust motion estimation and mapping even in GNSS-denied environments. This fusion of sensor modalities not only enhances system resilience but also provides a scalable solution for cyber-physical systems in connected and autonomous vehicles (CAVs). To validate the framework, we conduct tests in a tunnel environment, simulating sensor degradation and visibility challenges. The results demonstrate that our method sustains accurate localization where standard approaches deteriorate due to the tunnels featureless geometry. The frameworks versatility makes it a promising solution for autonomous vehicles, inspection robots, and other cyber-physical systems operating in constrained, perceptually poor environments.  
  </ol>  
</details>  
**comments**: Submitted to IAVVC 2025  
  
  



## SFM  

### [Estimating the Diameter at Breast Height of Trees in a Forest With a Single 360 Camera](http://arxiv.org/abs/2505.03093)  
Siming He, Zachary Osman, Fernando Cladera, Dexter Ong, Nitant Rai, Patrick Corey Green, Vijay Kumar, Pratik Chaudhari  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Forest inventories rely on accurate measurements of the diameter at breast height (DBH) for ecological monitoring, resource management, and carbon accounting. While LiDAR-based techniques can achieve centimeter-level precision, they are cost-prohibitive and operationally complex. We present a low-cost alternative that only needs a consumer-grade 360 video camera. Our semi-automated pipeline comprises of (i) a dense point cloud reconstruction using Structure from Motion (SfM) photogrammetry software called Agisoft Metashape, (ii) semantic trunk segmentation by projecting Grounded Segment Anything (SAM) masks onto the 3D cloud, and (iii) a robust RANSAC-based technique to estimate cross section shape and DBH. We introduce an interactive visualization tool for inspecting segmented trees and their estimated DBH. On 61 acquisitions of 43 trees under a variety of conditions, our method attains median absolute relative errors of 5-9% with respect to "ground-truth" manual measurements. This is only 2-4% higher than LiDAR-based estimates, while employing a single 360 camera that costs orders of magnitude less, requires minimal setup, and is widely available.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Thermal-LiDAR Fusion for Robust Tunnel Localization in GNSS-Denied and Low-Visibility Conditions](http://arxiv.org/abs/2505.03565)  
Lukas Schichler, Karin Festl, Selim Solmaz, Daniel Watzenig  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Despite significant progress in autonomous navigation, a critical gap remains in ensuring reliable localization in hazardous environments such as tunnels, urban disaster zones, and underground structures. Tunnels present a uniquely difficult scenario: they are not only prone to GNSS signal loss, but also provide little features for visual localization due to their repetitive walls and poor lighting. These conditions degrade conventional vision-based and LiDAR-based systems, which rely on distinguishable environmental features. To address this, we propose a novel sensor fusion framework that integrates a thermal camera with a LiDAR to enable robust localization in tunnels and other perceptually degraded environments. The thermal camera provides resilience in low-light or smoke conditions, while the LiDAR delivers precise depth perception and structural awareness. By combining these sensors, our framework ensures continuous and accurate localization across diverse and dynamic environments. We use an Extended Kalman Filter (EKF) to fuse multi-sensor inputs, and leverages visual odometry and SLAM (Simultaneous Localization and Mapping) techniques to process the sensor data, enabling robust motion estimation and mapping even in GNSS-denied environments. This fusion of sensor modalities not only enhances system resilience but also provides a scalable solution for cyber-physical systems in connected and autonomous vehicles (CAVs). To validate the framework, we conduct tests in a tunnel environment, simulating sensor degradation and visibility challenges. The results demonstrate that our method sustains accurate localization where standard approaches deteriorate due to the tunnels featureless geometry. The frameworks versatility makes it a promising solution for autonomous vehicles, inspection robots, and other cyber-physical systems operating in constrained, perceptually poor environments.  
  </ol>  
</details>  
**comments**: Submitted to IAVVC 2025  
  
### [LiftFeat: 3D Geometry-Aware Local Feature Matching](http://arxiv.org/abs/2505.03422)  
Yepeng Liu, Wenpeng Lai, Zhou Zhao, Yuxuan Xiong, Jinchi Zhu, Jun Cheng, Yongchao Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Robust and efficient local feature matching plays a crucial role in applications such as SLAM and visual localization for robotics. Despite great progress, it is still very challenging to extract robust and discriminative visual features in scenarios with drastic lighting changes, low texture areas, or repetitive patterns. In this paper, we propose a new lightweight network called \textit{LiftFeat}, which lifts the robustness of raw descriptor by aggregating 3D geometric feature. Specifically, we first adopt a pre-trained monocular depth estimation model to generate pseudo surface normal label, supervising the extraction of 3D geometric feature in terms of predicted surface normal. We then design a 3D geometry-aware feature lifting module to fuse surface normal feature with raw 2D descriptor feature. Integrating such 3D geometric feature enhances the discriminative ability of 2D feature description in extreme conditions. Extensive experimental results on relative pose estimation, homography estimation, and visual localization tasks, demonstrate that our LiftFeat outperforms some lightweight state-of-the-art methods. Code will be released at : https://github.com/lyp-deeplearning/LiftFeat.  
  </ol>  
</details>  
**comments**: Accepted at ICRA 2025  
  
### [Seeing the Abstract: Translating the Abstract Language for Vision Language Models](http://arxiv.org/abs/2505.03242)  
Davide Talon, Federico Girella, Ziyue Liu, Marco Cristani, Yiming Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Natural language goes beyond dryly describing visual content. It contains rich abstract concepts to express feeling, creativity and properties that cannot be directly perceived. Yet, current research in Vision Language Models (VLMs) has not shed light on abstract-oriented language. Our research breaks new ground by uncovering its wide presence and under-estimated value, with extensive analysis. Particularly, we focus our investigation on the fashion domain, a highly-representative field with abstract expressions. By analyzing recent large-scale multimodal fashion datasets, we find that abstract terms have a dominant presence, rivaling the concrete ones, providing novel information, and being useful in the retrieval task. However, a critical challenge emerges: current general-purpose or fashion-specific VLMs are pre-trained with databases that lack sufficient abstract words in their text corpora, thus hindering their ability to effectively represent abstract-oriented language. We propose a training-free and model-agnostic method, Abstract-to-Concrete Translator (ACT), to shift abstract representations towards well-represented concrete ones in the VLM latent space, using pre-trained models and existing multimodal databases. On the text-to-image retrieval task, despite being training-free, ACT outperforms the fine-tuned VLMs in both same- and cross-dataset settings, exhibiting its effectiveness with a strong generalization capability. Moreover, the improvement introduced by ACT is consistent with various VLMs, making it a plug-and-play solution.  
  </ol>  
</details>  
**comments**: Accepted to CVPR25. Project page:
  https://davidetalon.github.io/fashionact-page/  
  
  



## Image Matching  

### [LiftFeat: 3D Geometry-Aware Local Feature Matching](http://arxiv.org/abs/2505.03422)  
Yepeng Liu, Wenpeng Lai, Zhou Zhao, Yuxuan Xiong, Jinchi Zhu, Jun Cheng, Yongchao Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Robust and efficient local feature matching plays a crucial role in applications such as SLAM and visual localization for robotics. Despite great progress, it is still very challenging to extract robust and discriminative visual features in scenarios with drastic lighting changes, low texture areas, or repetitive patterns. In this paper, we propose a new lightweight network called \textit{LiftFeat}, which lifts the robustness of raw descriptor by aggregating 3D geometric feature. Specifically, we first adopt a pre-trained monocular depth estimation model to generate pseudo surface normal label, supervising the extraction of 3D geometric feature in terms of predicted surface normal. We then design a 3D geometry-aware feature lifting module to fuse surface normal feature with raw 2D descriptor feature. Integrating such 3D geometric feature enhances the discriminative ability of 2D feature description in extreme conditions. Extensive experimental results on relative pose estimation, homography estimation, and visual localization tasks, demonstrate that our LiftFeat outperforms some lightweight state-of-the-art methods. Code will be released at : https://github.com/lyp-deeplearning/LiftFeat.  
  </ol>  
</details>  
**comments**: Accepted at ICRA 2025  
  
  



