<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#VINGS-Mono:-Visual-Inertial-Gaussian-Splatting-Monocular-SLAM-in-Large-Scenes>VINGS-Mono: Visual-Inertial Gaussian Splatting Monocular SLAM in Large Scenes</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#VINGS-Mono:-Visual-Inertial-Gaussian-Splatting-Monocular-SLAM-in-Large-Scenes>VINGS-Mono: Visual-Inertial Gaussian Splatting Monocular SLAM in Large Scenes</a></li>
        <li><a href=#Evaluating-Human-Perception-of-Novel-View-Synthesis:-Subjective-Quality-Assessment-of-Gaussian-Splatting-and-NeRF-in-Dynamic-Scenes>Evaluating Human Perception of Novel View Synthesis: Subjective Quality Assessment of Gaussian Splatting and NeRF in Dynamic Scenes</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [VINGS-Mono: Visual-Inertial Gaussian Splatting Monocular SLAM in Large Scenes](http://arxiv.org/abs/2501.08286)  
Ke Wu, Zicheng Zhang, Muer Tie, Ziqing Ai, Zhongxue Gan, Wenchao Ding  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    VINGS-Mono is a monocular (inertial) Gaussian Splatting (GS) SLAM framework designed for large scenes. The framework comprises four main components: VIO Front End, 2D Gaussian Map, NVS Loop Closure, and Dynamic Eraser. In the VIO Front End, RGB frames are processed through dense bundle adjustment and uncertainty estimation to extract scene geometry and poses. Based on this output, the mapping module incrementally constructs and maintains a 2D Gaussian map. Key components of the 2D Gaussian Map include a Sample-based Rasterizer, Score Manager, and Pose Refinement, which collectively improve mapping speed and localization accuracy. This enables the SLAM system to handle large-scale urban environments with up to 50 million Gaussian ellipsoids. To ensure global consistency in large-scale scenes, we design a Loop Closure module, which innovatively leverages the Novel View Synthesis (NVS) capabilities of Gaussian Splatting for loop closure detection and correction of the Gaussian map. Additionally, we propose a Dynamic Eraser to address the inevitable presence of dynamic objects in real-world outdoor scenes. Extensive evaluations in indoor and outdoor environments demonstrate that our approach achieves localization performance on par with Visual-Inertial Odometry while surpassing recent GS/NeRF SLAM methods. It also significantly outperforms all existing methods in terms of mapping and rendering quality. Furthermore, we developed a mobile app and verified that our framework can generate high-quality Gaussian maps in real time using only a smartphone camera and a low-frequency IMU sensor. To the best of our knowledge, VINGS-Mono is the first monocular Gaussian SLAM method capable of operating in outdoor environments and supporting kilometer-scale large scenes.  
  </ol>  
</details>  
  
  



## NeRF  

### [VINGS-Mono: Visual-Inertial Gaussian Splatting Monocular SLAM in Large Scenes](http://arxiv.org/abs/2501.08286)  
Ke Wu, Zicheng Zhang, Muer Tie, Ziqing Ai, Zhongxue Gan, Wenchao Ding  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    VINGS-Mono is a monocular (inertial) Gaussian Splatting (GS) SLAM framework designed for large scenes. The framework comprises four main components: VIO Front End, 2D Gaussian Map, NVS Loop Closure, and Dynamic Eraser. In the VIO Front End, RGB frames are processed through dense bundle adjustment and uncertainty estimation to extract scene geometry and poses. Based on this output, the mapping module incrementally constructs and maintains a 2D Gaussian map. Key components of the 2D Gaussian Map include a Sample-based Rasterizer, Score Manager, and Pose Refinement, which collectively improve mapping speed and localization accuracy. This enables the SLAM system to handle large-scale urban environments with up to 50 million Gaussian ellipsoids. To ensure global consistency in large-scale scenes, we design a Loop Closure module, which innovatively leverages the Novel View Synthesis (NVS) capabilities of Gaussian Splatting for loop closure detection and correction of the Gaussian map. Additionally, we propose a Dynamic Eraser to address the inevitable presence of dynamic objects in real-world outdoor scenes. Extensive evaluations in indoor and outdoor environments demonstrate that our approach achieves localization performance on par with Visual-Inertial Odometry while surpassing recent GS/NeRF SLAM methods. It also significantly outperforms all existing methods in terms of mapping and rendering quality. Furthermore, we developed a mobile app and verified that our framework can generate high-quality Gaussian maps in real time using only a smartphone camera and a low-frequency IMU sensor. To the best of our knowledge, VINGS-Mono is the first monocular Gaussian SLAM method capable of operating in outdoor environments and supporting kilometer-scale large scenes.  
  </ol>  
</details>  
  
### [Evaluating Human Perception of Novel View Synthesis: Subjective Quality Assessment of Gaussian Splatting and NeRF in Dynamic Scenes](http://arxiv.org/abs/2501.08072)  
Yuhang Zhang, Joshua Maraval, Zhengyu Zhang, Nicolas Ramin, Shishun Tian, Lu Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Gaussian Splatting (GS) and Neural Radiance Fields (NeRF) are two groundbreaking technologies that have revolutionized the field of Novel View Synthesis (NVS), enabling immersive photorealistic rendering and user experiences by synthesizing multiple viewpoints from a set of images of sparse views. The potential applications of NVS, such as high-quality virtual and augmented reality, detailed 3D modeling, and realistic medical organ imaging, underscore the importance of quality assessment of NVS methods from the perspective of human perception. Although some previous studies have explored subjective quality assessments for NVS technology, they still face several challenges, especially in NVS methods selection, scenario coverage, and evaluation methodology. To address these challenges, we conducted two subjective experiments for the quality assessment of NVS technologies containing both GS-based and NeRF-based methods, focusing on dynamic and real-world scenes. This study covers 360{\deg}, front-facing, and single-viewpoint videos while providing a richer and greater number of real scenes. Meanwhile, it's the first time to explore the impact of NVS methods in dynamic scenes with moving objects. The two types of subjective experiments help to fully comprehend the influences of different viewing paths from a human perception perspective and pave the way for future development of full-reference and no-reference quality metrics. In addition, we established a comprehensive benchmark of various state-of-the-art objective metrics on the proposed database, highlighting that existing methods still struggle to accurately capture subjective quality. The results give us some insights into the limitations of existing NVS methods and may promote the development of new NVS methods.  
  </ol>  
</details>  
  
  



