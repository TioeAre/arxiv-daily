<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Sim2real-Cattle-Joint-Estimation-in-3D-point-clouds>Sim2real Cattle Joint Estimation in 3D point clouds</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Learning-autonomous-driving-from-aerial-imagery>Learning autonomous driving from aerial imagery</a></li>
        <li><a href=#DaRePlane:-Direction-aware-Representations-for-Dynamic-Scene-Reconstruction>DaRePlane: Direction-aware Representations for Dynamic Scene Reconstruction</a></li>
      </ul>
    </li>
  </ol>
</details>

## Keypoint Detection  

### [Sim2real Cattle Joint Estimation in 3D point clouds](http://arxiv.org/abs/2410.14419)  
Okour Mohammad, Falque Raphael, Alempijevic Alen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Understanding the well-being of cattle is crucial in various agricultural contexts. Cattle's body shape and joint articulation carry significant information about their welfare, yet acquiring comprehensive datasets for 3D body pose estimation presents a formidable challenge. This study delves into the construction of such a dataset specifically tailored for cattle. Leveraging the expertise of digital artists, we use a single animated 3D model to represent diverse cattle postures. To address the disparity between virtual and real-world data, we augment the 3D model's shape to encompass a range of potential body appearances, thereby narrowing the "sim2real" gap. We use these annotated models to train a deep-learning framework capable of estimating internal joints solely based on external surface curvature. Our contribution is specifically the use of geodesic distance over the surface manifold, coupled with multilateration to extract joints in a semantic keypoint detection encoder-decoder architecture. We demonstrate the robustness of joint extraction by comparing the link lengths extracted on real cattle mobbing and walking within a race. Furthermore, inspired by the established allometric relationship between bone length and the overall height of mammals, we utilise the estimated joints to predict hip height within a real cattle dataset, extending the utility of our approach to offer insights into improving cattle monitoring practices.  
  </ol>  
</details>  
  
  



## NeRF  

### [Learning autonomous driving from aerial imagery](http://arxiv.org/abs/2410.14177)  
Varun Murali, Guy Rosman, Sertac Karaman, Daniela Rus  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this work, we consider the problem of learning end to end perception to control for ground vehicles solely from aerial imagery. Photogrammetric simulators allow the synthesis of novel views through the transformation of pre-generated assets into novel views.However, they have a large setup cost, require careful collection of data and often human effort to create usable simulators. We use a Neural Radiance Field (NeRF) as an intermediate representation to synthesize novel views from the point of view of a ground vehicle. These novel viewpoints can then be used for several downstream autonomous navigation applications. In this work, we demonstrate the utility of novel view synthesis though the application of training a policy for end to end learning from images and depth data. In a traditional real to sim to real framework, the collected data would be transformed into a visual simulator which could then be used to generate novel views. In contrast, using a NeRF allows a compact representation and the ability to optimize over the parameters of the visual simulator as more data is gathered in the environment. We demonstrate the efficacy of our method in a custom built mini-city environment through the deployment of imitation policies on robotic cars. We additionally consider the task of place localization and demonstrate that our method is able to relocalize the car in the real world.  
  </ol>  
</details>  
**comments**: Presented at IROS 2024  
  
### [DaRePlane: Direction-aware Representations for Dynamic Scene Reconstruction](http://arxiv.org/abs/2410.14169)  
Ange Lou, Benjamin Planche, Zhongpai Gao, Yamin Li, Tianyu Luan, Hao Ding, Meng Zheng, Terrence Chen, Ziyan Wu, Jack Noble  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Numerous recent approaches to modeling and re-rendering dynamic scenes leverage plane-based explicit representations, addressing slow training times associated with models like neural radiance fields (NeRF) and Gaussian splatting (GS). However, merely decomposing 4D dynamic scenes into multiple 2D plane-based representations is insufficient for high-fidelity re-rendering of scenes with complex motions. In response, we present DaRePlane, a novel direction-aware representation approach that captures scene dynamics from six different directions. This learned representation undergoes an inverse dual-tree complex wavelet transformation (DTCWT) to recover plane-based information. Within NeRF pipelines, DaRePlane computes features for each space-time point by fusing vectors from these recovered planes, then passed to a tiny MLP for color regression. When applied to Gaussian splatting, DaRePlane computes the features of Gaussian points, followed by a tiny multi-head MLP for spatial-time deformation prediction. Notably, to address redundancy introduced by the six real and six imaginary direction-aware wavelet coefficients, we introduce a trainable masking approach, mitigating storage issues without significant performance decline. To demonstrate the generality and efficiency of DaRePlane, we test it on both regular and surgical dynamic scenes, for both NeRF and GS systems. Extensive experiments show that DaRePlane yields state-of-the-art performance in novel view synthesis for various complex dynamic scenes.  
  </ol>  
</details>  
**comments**: arXiv admin note: substantial text overlap with arXiv:2403.02265  
  
  



