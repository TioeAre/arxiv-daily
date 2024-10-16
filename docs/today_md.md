<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#DN-4DGS:-Denoised-Deformable-Network-with-Temporal-Spatial-Aggregation-for-Dynamic-Scene-Rendering>DN-4DGS: Denoised Deformable Network with Temporal-Spatial Aggregation for Dynamic Scene Rendering</a></li>
        <li><a href=#DriveDreamer4D:-World-Models-Are-Effective-Data-Machines-for-4D-Driving-Scene-Representation>DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation</a></li>
        <li><a href=#Object-Pose-Estimation-Using-Implicit-Representation-For-Transparent-Objects>Object Pose Estimation Using Implicit Representation For Transparent Objects</a></li>
        <li><a href=#GlossyGS:-Inverse-Rendering-of-Glossy-Objects-with-3D-Gaussian-Splatting>GlossyGS: Inverse Rendering of Glossy Objects with 3D Gaussian Splatting</a></li>
      </ul>
    </li>
  </ol>
</details>

## NeRF  

### [DN-4DGS: Denoised Deformable Network with Temporal-Spatial Aggregation for Dynamic Scene Rendering](http://arxiv.org/abs/2410.13607)  
Jiahao Lu, Jiacheng Deng, Ruijie Zhu, Yanzhe Liang, Wenfei Yang, Tianzhu Zhang, Xu Zhou  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Dynamic scenes rendering is an intriguing yet challenging problem. Although current methods based on NeRF have achieved satisfactory performance, they still can not reach real-time levels. Recently, 3D Gaussian Splatting (3DGS) has gar?nered researchers attention due to their outstanding rendering quality and real?time speed. Therefore, a new paradigm has been proposed: defining a canonical 3D gaussians and deforming it to individual frames in deformable fields. How?ever, since the coordinates of canonical 3D gaussians are filled with noise, which can transfer noise into the deformable fields, and there is currently no method that adequately considers the aggregation of 4D information. Therefore, we pro?pose Denoised Deformable Network with Temporal-Spatial Aggregation for Dy?namic Scene Rendering (DN-4DGS). Specifically, a Noise Suppression Strategy is introduced to change the distribution of the coordinates of the canonical 3D gaussians and suppress noise. Additionally, a Decoupled Temporal-Spatial Ag?gregation Module is designed to aggregate information from adjacent points and frames. Extensive experiments on various real-world datasets demonstrate that our method achieves state-of-the-art rendering quality under a real-time level.  
  </ol>  
</details>  
**comments**: Accepted by NeurIPS 2024  
  
### [DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation](http://arxiv.org/abs/2410.13571)  
Guosheng Zhao, Chaojun Ni, Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Boyuan Wang, Youyi Zhang, Wenjun Mei, Xingang Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Closed-loop simulation is essential for advancing end-to-end autonomous driving systems. Contemporary sensor simulation methods, such as NeRF and 3DGS, rely predominantly on conditions closely aligned with training data distributions, which are largely confined to forward-driving scenarios. Consequently, these methods face limitations when rendering complex maneuvers (e.g., lane change, acceleration, deceleration). Recent advancements in autonomous-driving world models have demonstrated the potential to generate diverse driving videos. However, these approaches remain constrained to 2D video generation, inherently lacking the spatiotemporal coherence required to capture intricacies of dynamic driving environments. In this paper, we introduce \textit{DriveDreamer4D}, which enhances 4D driving scene representation leveraging world model priors. Specifically, we utilize the world model as a data machine to synthesize novel trajectory videos based on real-world driving data. Notably, we explicitly leverage structured conditions to control the spatial-temporal consistency of foreground and background elements, thus the generated data adheres closely to traffic constraints. To our knowledge, \textit{DriveDreamer4D} is the first to utilize video generation models for improving 4D reconstruction in driving scenarios. Experimental results reveal that \textit{DriveDreamer4D} significantly enhances generation quality under novel trajectory views, achieving a relative improvement in FID by 24.5\%, 39.0\%, and 10.5\% compared to PVG, $\text{S}^3$ Gaussian, and Deformable-GS. Moreover, \textit{DriveDreamer4D} markedly enhances the spatiotemporal coherence of driving agents, which is verified by a comprehensive user study and the relative increases of 20.3\%, 42.0\%, and 13.7\% in the NTA-IoU metric.  
  </ol>  
</details>  
**comments**: https://drivedreamer4d.github.io  
  
### [Object Pose Estimation Using Implicit Representation For Transparent Objects](http://arxiv.org/abs/2410.13465)  
Varun Burde, Artem Moroz, Vit Zeman, Pavel Burget  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Object pose estimation is a prominent task in computer vision. The object pose gives the orientation and translation of the object in real-world space, which allows various applications such as manipulation, augmented reality, etc. Various objects exhibit different properties with light, such as reflections, absorption, etc. This makes it challenging to understand the object's structure in RGB and depth channels. Recent research has been moving toward learning-based methods, which provide a more flexible and generalizable approach to object pose estimation utilizing deep learning. One such approach is the render-and-compare method, which renders the object from multiple views and compares it against the given 2D image, which often requires an object representation in the form of a CAD model. We reason that the synthetic texture of the CAD model may not be ideal for rendering and comparing operations. We showed that if the object is represented as an implicit (neural) representation in the form of Neural Radiance Field (NeRF), it exhibits a more realistic rendering of the actual scene and retains the crucial spatial features, which makes the comparison more versatile. We evaluated our NeRF implementation of the render-and-compare method on transparent datasets and found that it surpassed the current state-of-the-art results.  
  </ol>  
</details>  
  
### [GlossyGS: Inverse Rendering of Glossy Objects with 3D Gaussian Splatting](http://arxiv.org/abs/2410.13349)  
Shuichang Lai, Letian Huang, Jie Guo, Kai Cheng, Bowen Pan, Xiaoxiao Long, Jiangjing Lyu, Chengfei Lv, Yanwen Guo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Reconstructing objects from posed images is a crucial and complex task in computer graphics and computer vision. While NeRF-based neural reconstruction methods have exhibited impressive reconstruction ability, they tend to be time-comsuming. Recent strategies have adopted 3D Gaussian Splatting (3D-GS) for inverse rendering, which have led to quick and effective outcomes. However, these techniques generally have difficulty in producing believable geometries and materials for glossy objects, a challenge that stems from the inherent ambiguities of inverse rendering. To address this, we introduce GlossyGS, an innovative 3D-GS-based inverse rendering framework that aims to precisely reconstruct the geometry and materials of glossy objects by integrating material priors. The key idea is the use of micro-facet geometry segmentation prior, which helps to reduce the intrinsic ambiguities and improve the decomposition of geometries and materials. Additionally, we introduce a normal map prefiltering strategy to more accurately simulate the normal distribution of reflective surfaces. These strategies are integrated into a hybrid geometry and material representation that employs both explicit and implicit methods to depict glossy objects. We demonstrate through quantitative analysis and qualitative visualization that the proposed method is effective to reconstruct high-fidelity geometries and materials of glossy objects, and performs favorably against state-of-the-arts.  
  </ol>  
</details>  
  
  



