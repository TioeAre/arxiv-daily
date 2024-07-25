<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Revolutionizing-Text-to-Image-Retrieval-as-Autoregressive-Token-to-Voken-Generation>Revolutionizing Text-to-Image Retrieval as Autoregressive Token-to-Voken Generation</a></li>
        <li><a href=#Active-Loop-Closure-for-OSM-guided-Robotic-Mapping-in-Large-Scale-Urban-Environments>Active Loop Closure for OSM-guided Robotic Mapping in Large-Scale Urban Environments</a></li>
        <li><a href=#Pose-Estimation-from-Camera-Images-for-Underwater-Inspection>Pose Estimation from Camera Images for Underwater Inspection</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#SV4D:-Dynamic-3D-Content-Generation-with-Multi-Frame-and-Multi-View-Consistency>SV4D: Dynamic 3D Content Generation with Multi-Frame and Multi-View Consistency</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Revolutionizing Text-to-Image Retrieval as Autoregressive Token-to-Voken Generation](http://arxiv.org/abs/2407.17274)  
Yongqi Li, Hongru Cai, Wenjie Wang, Leigang Qu, Yinwei Wei, Wenjie Li, Liqiang Nie, Tat-Seng Chua  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Text-to-image retrieval is a fundamental task in multimedia processing, aiming to retrieve semantically relevant cross-modal content. Traditional studies have typically approached this task as a discriminative problem, matching the text and image via the cross-attention mechanism (one-tower framework) or in a common embedding space (two-tower framework). Recently, generative cross-modal retrieval has emerged as a new research line, which assigns images with unique string identifiers and generates the target identifier as the retrieval target. Despite its great potential, existing generative approaches are limited due to the following issues: insufficient visual information in identifiers, misalignment with high-level semantics, and learning gap towards the retrieval target. To address the above issues, we propose an autoregressive voken generation method, named AVG. AVG tokenizes images into vokens, i.e., visual tokens, and innovatively formulates the text-to-image retrieval task as a token-to-voken generation problem. AVG discretizes an image into a sequence of vokens as the identifier of the image, while maintaining the alignment with both the visual information and high-level semantics of the image. Additionally, to bridge the learning gap between generative training and the retrieval target, we incorporate discriminative training to modify the learning direction during token-to-voken training. Extensive experiments demonstrate that AVG achieves superior results in both effectiveness and efficiency.  
  </ol>  
</details>  
**comments**: Work in progress  
  
### [Active Loop Closure for OSM-guided Robotic Mapping in Large-Scale Urban Environments](http://arxiv.org/abs/2407.17078)  
Wei Gao, Zezhou Sun, Mingle Zhao, Cheng-Zhong Xu, Hui Kong  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The autonomous mapping of large-scale urban scenes presents significant challenges for autonomous robots. To mitigate the challenges, global planning, such as utilizing prior GPS trajectories from OpenStreetMap (OSM), is often used to guide the autonomous navigation of robots for mapping. However, due to factors like complex terrain, unexpected body movement, and sensor noise, the uncertainty of the robot's pose estimates inevitably increases over time, ultimately leading to the failure of robotic mapping. To address this issue, we propose a novel active loop closure procedure, enabling the robot to actively re-plan the previously planned GPS trajectory. The method can guide the robot to re-visit the previous places where the loop-closure detection can be performed to trigger the back-end optimization, effectively reducing errors and uncertainties in pose estimation. The proposed active loop closure mechanism is implemented and embedded into a real-time OSM-guided robot mapping framework. Empirical results on several large-scale outdoor scenarios demonstrate its effectiveness and promising performance.  
  </ol>  
</details>  
  
### [Pose Estimation from Camera Images for Underwater Inspection](http://arxiv.org/abs/2407.16961)  
Luyuan Peng, Hari Vishnu, Mandar Chitre, Yuen Min Too, Bharath Kalyan, Rajat Mishra, Soo Pieng Tan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    High-precision localization is pivotal in underwater reinspection missions. Traditional localization methods like inertial navigation systems, Doppler velocity loggers, and acoustic positioning face significant challenges and are not cost-effective for some applications. Visual localization is a cost-effective alternative in such cases, leveraging the cameras already equipped on inspection vehicles to estimate poses from images of the surrounding scene. Amongst these, machine learning-based pose estimation from images shows promise in underwater environments, performing efficient relocalization using models trained based on previously mapped scenes. We explore the efficacy of learning-based pose estimators in both clear and turbid water inspection missions, assessing the impact of image formats, model architectures and training data diversity. We innovate by employing novel view synthesis models to generate augmented training data, significantly enhancing pose estimation in unexplored regions. Moreover, we enhance localization accuracy by integrating pose estimator outputs with sensor data via an extended Kalman filter, demonstrating improved trajectory smoothness and accuracy.  
  </ol>  
</details>  
**comments**: Submitted to IEEE Journal of Oceanic Engineering  
  
  



## NeRF  

### [SV4D: Dynamic 3D Content Generation with Multi-Frame and Multi-View Consistency](http://arxiv.org/abs/2407.17470)  
Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, Varun Jampani  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present Stable Video 4D (SV4D), a latent video diffusion model for multi-frame and multi-view consistent dynamic 3D content generation. Unlike previous methods that rely on separately trained generative models for video generation and novel view synthesis, we design a unified diffusion model to generate novel view videos of dynamic 3D objects. Specifically, given a monocular reference video, SV4D generates novel views for each video frame that are temporally consistent. We then use the generated novel view videos to optimize an implicit 4D representation (dynamic NeRF) efficiently, without the need for cumbersome SDS-based optimization used in most prior works. To train our unified novel view video generation model, we curated a dynamic 3D object dataset from the existing Objaverse dataset. Extensive experimental results on multiple datasets and user studies demonstrate SV4D's state-of-the-art performance on novel-view video synthesis as well as 4D generation compared to prior works.  
  </ol>  
</details>  
**comments**: Project page: https://sv4d.github.io/  
  
  



