<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Sparse-Point-Cloud-Patches-Rendering-via-Splitting-2D-Gaussians>Sparse Point Cloud Patches Rendering via Splitting 2D Gaussians</a></li>
        <li><a href=#FreeDriveRF:-Monocular-RGB-Dynamic-NeRF-without-Poses-for-Autonomous-Driving-via-Point-Level-Dynamic-Static-Decoupling>FreeDriveRF: Monocular RGB Dynamic NeRF without Poses for Autonomous Driving via Point-Level Dynamic-Static Decoupling</a></li>
        <li><a href=#TUGS:-Physics-based-Compact-Representation-of-Underwater-Scenes-by-Tensorized-Gaussian>TUGS: Physics-based Compact Representation of Underwater Scenes by Tensorized Gaussian</a></li>
      </ul>
    </li>
  </ol>
</details>

## NeRF  

### [Sparse Point Cloud Patches Rendering via Splitting 2D Gaussians](http://arxiv.org/abs/2505.09413)  
[[code](https://github.com/murcherful/gaupcrender)]  
Ma Changfeng, Bi Ran, Guo Jie, Wang Chongjun, Guo Yanwen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Current learning-based methods predict NeRF or 3D Gaussians from point clouds to achieve photo-realistic rendering but still depend on categorical priors, dense point clouds, or additional refinements. Hence, we introduce a novel point cloud rendering method by predicting 2D Gaussians from point clouds. Our method incorporates two identical modules with an entire-patch architecture enabling the network to be generalized to multiple datasets. The module normalizes and initializes the Gaussians utilizing the point cloud information including normals, colors and distances. Then, splitting decoders are employed to refine the initial Gaussians by duplicating them and predicting more accurate results, making our methodology effectively accommodate sparse point clouds as well. Once trained, our approach exhibits direct generalization to point clouds across different categories. The predicted Gaussians are employed directly for rendering without additional refinement on the rendered images, retaining the benefits of 2D Gaussians. We conduct extensive experiments on various datasets, and the results demonstrate the superiority and generalization of our method, which achieves SOTA performance. The code is available at https://github.com/murcherful/GauPCRender}{https://github.com/murcherful/GauPCRender.  
  </ol>  
</details>  
**comments**: CVPR 2025 Accepted  
  
### [FreeDriveRF: Monocular RGB Dynamic NeRF without Poses for Autonomous Driving via Point-Level Dynamic-Static Decoupling](http://arxiv.org/abs/2505.09406)  
Yue Wen, Liang Song, Yijia Liu, Siting Zhu, Yanzi Miao, Lijun Han, Hesheng Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Dynamic scene reconstruction for autonomous driving enables vehicles to perceive and interpret complex scene changes more precisely. Dynamic Neural Radiance Fields (NeRFs) have recently shown promising capability in scene modeling. However, many existing methods rely heavily on accurate poses inputs and multi-sensor data, leading to increased system complexity. To address this, we propose FreeDriveRF, which reconstructs dynamic driving scenes using only sequential RGB images without requiring poses inputs. We innovatively decouple dynamic and static parts at the early sampling level using semantic supervision, mitigating image blurring and artifacts. To overcome the challenges posed by object motion and occlusion in monocular camera, we introduce a warped ray-guided dynamic object rendering consistency loss, utilizing optical flow to better constrain the dynamic modeling process. Additionally, we incorporate estimated dynamic flow to constrain the pose optimization process, improving the stability and accuracy of unbounded scene reconstruction. Extensive experiments conducted on the KITTI and Waymo datasets demonstrate the superior performance of our method in dynamic scene modeling for autonomous driving.  
  </ol>  
</details>  
**comments**: 7 pages, 9 figures, accepted by ICRA2025  
  
### [TUGS: Physics-based Compact Representation of Underwater Scenes by Tensorized Gaussian](http://arxiv.org/abs/2505.08811)  
Shijie Lian, Ziyi Zhang, Laurence Tianruo Yang and, Mengyu Ren, Debin Liu, Hua Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Underwater 3D scene reconstruction is crucial for undewater robotic perception and navigation. However, the task is significantly challenged by the complex interplay between light propagation, water medium, and object surfaces, with existing methods unable to model their interactions accurately. Additionally, expensive training and rendering costs limit their practical application in underwater robotic systems. Therefore, we propose Tensorized Underwater Gaussian Splatting (TUGS), which can effectively solve the modeling challenges of the complex interactions between object geometries and water media while achieving significant parameter reduction. TUGS employs lightweight tensorized higher-order Gaussians with a physics-based underwater Adaptive Medium Estimation (AME) module, enabling accurate simulation of both light attenuation and backscatter effects in underwater environments. Compared to other NeRF-based and GS-based methods designed for underwater, TUGS is able to render high-quality underwater images with faster rendering speeds and less memory usage. Extensive experiments on real-world underwater datasets have demonstrated that TUGS can efficiently achieve superior reconstruction quality using a limited number of parameters, making it particularly suitable for memory-constrained underwater UAV applications  
  </ol>  
</details>  
  
  



