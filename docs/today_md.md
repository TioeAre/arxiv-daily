<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Transfer-Learning-for-Keypoint-Detection-in-Low-Resolution-Thermal-TUG-Test-Images>Transfer Learning for Keypoint Detection in Low-Resolution Thermal TUG Test Images</a></li>
        <li><a href=#Video-based-Surgical-Tool-tip-and-Keypoint-Tracking-using-Multi-frame-Context-driven-Deep-Learning-Models>Video-based Surgical Tool-tip and Keypoint Tracking using Multi-frame Context-driven Deep Learning Models</a></li>
        <li><a href=#Lifelong-3D-Mapping-Framework-for-Hand-held-&-Robot-mounted-LiDAR-Mapping-Systems>Lifelong 3D Mapping Framework for Hand-held & Robot-mounted LiDAR Mapping Systems</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#VoD-3DGS:-View-opacity-Dependent-3D-Gaussian-Splatting>VoD-3DGS: View-opacity-Dependent 3D Gaussian Splatting</a></li>
      </ul>
    </li>
  </ol>
</details>

## Keypoint Detection  

### [Transfer Learning for Keypoint Detection in Low-Resolution Thermal TUG Test Images](http://arxiv.org/abs/2501.18453)  
Wei-Lun Chen, Chia-Yeh Hsieh, Yu-Hsiang Kao, Kai-Chun Liu, Sheng-Yu Peng, Yu Tsao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This study presents a novel approach to human keypoint detection in low-resolution thermal images using transfer learning techniques. We introduce the first application of the Timed Up and Go (TUG) test in thermal image computer vision, establishing a new paradigm for mobility assessment. Our method leverages a MobileNetV3-Small encoder and a ViTPose decoder, trained using a composite loss function that balances latent representation alignment and heatmap accuracy. The model was evaluated using the Object Keypoint Similarity (OKS) metric from the COCO Keypoint Detection Challenge. The proposed model achieves better performance with AP, AP50, and AP75 scores of 0.861, 0.942, and 0.887 respectively, outperforming traditional supervised learning approaches like Mask R-CNN and ViTPose-Base. Moreover, our model demonstrates superior computational efficiency in terms of parameter count and FLOPS. This research lays a solid foundation for future clinical applications of thermal imaging in mobility assessment and rehabilitation monitoring.  
  </ol>  
</details>  
**comments**: Accepted to AICAS 2025. This is the preprint version  
  
### [Video-based Surgical Tool-tip and Keypoint Tracking using Multi-frame Context-driven Deep Learning Models](http://arxiv.org/abs/2501.18361)  
Bhargav Ghanekar, Lianne R. Johnson, Jacob L. Laughlin, Marcia K. O'Malley, Ashok Veeraraghavan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Automated tracking of surgical tool keypoints in robotic surgery videos is an essential task for various downstream use cases such as skill assessment, expertise assessment, and the delineation of safety zones. In recent years, the explosion of deep learning for vision applications has led to many works in surgical instrument segmentation, while lesser focus has been on tracking specific tool keypoints, such as tool tips. In this work, we propose a novel, multi-frame context-driven deep learning framework to localize and track tool keypoints in surgical videos. We train and test our models on the annotated frames from the 2015 EndoVis Challenge dataset, resulting in state-of-the-art performance. By leveraging sophisticated deep learning models and multi-frame context, we achieve 90\% keypoint detection accuracy and a localization RMS error of 5.27 pixels. Results on a self-annotated JIGSAWS dataset with more challenging scenarios also show that the proposed multi-frame models can accurately track tool-tip and tool-base keypoints, with ${<}4.2$ -pixel RMS error overall. Such a framework paves the way for accurately tracking surgical instrument keypoints, enabling further downstream use cases. Project and dataset webpage: https://tinyurl.com/mfc-tracker  
  </ol>  
</details>  
  
### [Lifelong 3D Mapping Framework for Hand-held & Robot-mounted LiDAR Mapping Systems](http://arxiv.org/abs/2501.18110)  
Liudi Yang, Sai Manoj Prakhya, Senhua Zhu, Ziyuan Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose a lifelong 3D mapping framework that is modular, cloud-native by design and more importantly, works for both hand-held and robot-mounted 3D LiDAR mapping systems. Our proposed framework comprises of dynamic point removal, multi-session map alignment, map change detection and map version control. First, our sensor-setup agnostic dynamic point removal algorithm works seamlessly with both hand-held and robot-mounted setups to produce clean static 3D maps. Second, the multi-session map alignment aligns these clean static maps automatically, without manual parameter fine-tuning, into a single reference frame, using a two stage approach based on feature descriptor matching and fine registration. Third, our novel map change detection identifies positive and negative changes between two aligned maps. Finally, the map version control maintains a single base map that represents the current state of the environment, and stores the detected positive and negative changes, and boundary information. Our unique map version control system can reconstruct any of the previous clean session maps and allows users to query changes between any two random mapping sessions, all without storing any input raw session maps, making it very unique. Extensive experiments are performed using hand-held commercial LiDAR mapping devices and open-source robot-mounted LiDAR SLAM algorithms to evaluate each module and the whole 3D lifelong mapping framework.  
  </ol>  
</details>  
  
  



## NeRF  

### [VoD-3DGS: View-opacity-Dependent 3D Gaussian Splatting](http://arxiv.org/abs/2501.17978)  
Nowak Mateusz, Jarosz Wojciech, Chin Peter  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Reconstructing a 3D scene from images is challenging due to the different ways light interacts with surfaces depending on the viewer's position and the surface's material. In classical computer graphics, materials can be classified as diffuse or specular, interacting with light differently. The standard 3D Gaussian Splatting model struggles to represent view-dependent content, since it cannot differentiate an object within the scene from the light interacting with its specular surfaces, which produce highlights or reflections. In this paper, we propose to extend the 3D Gaussian Splatting model by introducing an additional symmetric matrix to enhance the opacity representation of each 3D Gaussian. This improvement allows certain Gaussians to be suppressed based on the viewer's perspective, resulting in a more accurate representation of view-dependent reflections and specular highlights without compromising the scene's integrity. By allowing the opacity to be view dependent, our enhanced model achieves state-of-the-art performance on Mip-Nerf, Tanks\&Temples, Deep Blending, and Nerf-Synthetic datasets without a significant loss in rendering speed, achieving >60FPS, and only incurring a minimal increase in memory used.  
  </ol>  
</details>  
  
  



