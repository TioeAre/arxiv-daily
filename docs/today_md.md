<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Towards-Real-Time-Gaussian-Splatting:-Accelerating-3DGS-through-Photometric-SLAM>Towards Real-Time Gaussian Splatting: Accelerating 3DGS through Photometric SLAM</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#PRISM:-PRogressive-dependency-maxImization-for-Scale-invariant-image-Matching>PRISM: PRogressive dependency maxImization for Scale-invariant image Matching</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Goal-oriented-Semantic-Communication-for-the-Metaverse-Application>Goal-oriented Semantic Communication for the Metaverse Application</a></li>
        <li><a href=#RayGauss:-Volumetric-Gaussian-Based-Ray-Casting-for-Photorealistic-Novel-View-Synthesis>RayGauss: Volumetric Gaussian-Based Ray Casting for Photorealistic Novel View Synthesis</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Towards Real-Time Gaussian Splatting: Accelerating 3DGS through Photometric SLAM](http://arxiv.org/abs/2408.03825)  
Yan Song Hu, Dayou Mao, Yuhao Chen, John Zelek  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Initial applications of 3D Gaussian Splatting (3DGS) in Visual Simultaneous Localization and Mapping (VSLAM) demonstrate the generation of high-quality volumetric reconstructions from monocular video streams. However, despite these promising advancements, current 3DGS integrations have reduced tracking performance and lower operating speeds compared to traditional VSLAM. To address these issues, we propose integrating 3DGS with Direct Sparse Odometry, a monocular photometric SLAM system. We have done preliminary experiments showing that using Direct Sparse Odometry point cloud outputs, as opposed to standard structure-from-motion methods, significantly shortens the training time needed to achieve high-quality renders. Reducing 3DGS training time enables the development of 3DGS-integrated SLAM systems that operate in real-time on mobile hardware. These promising initial findings suggest further exploration is warranted in combining traditional VSLAM systems with 3DGS.  
  </ol>  
</details>  
**comments**: This extended abstract has been submitted to be presented at an IEEE
  conference. It will be made available online by IEEE but will not be
  published in IEEE Xplore. Copyright may be transferred without notice, after
  which this version may no longer be accessible  
  
  



## Image Matching  

### [PRISM: PRogressive dependency maxImization for Scale-invariant image Matching](http://arxiv.org/abs/2408.03598)  
Xudong Cai, Yongcai Wang, Lun Luo, Minhang Wang, Deying Li, Jintao Xu, Weihao Gu, Rui Ai  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image matching aims at identifying corresponding points between a pair of images. Currently, detector-free methods have shown impressive performance in challenging scenarios, thanks to their capability of generating dense matches and global receptive field. However, performing feature interaction and proposing matches across the entire image is unnecessary, because not all image regions contribute to the matching process. Interacting and matching in unmatchable areas can introduce errors, reducing matching accuracy and efficiency. Meanwhile, the scale discrepancy issue still troubles existing methods. To address above issues, we propose PRogressive dependency maxImization for Scale-invariant image Matching (PRISM), which jointly prunes irrelevant patch features and tackles the scale discrepancy. To do this, we firstly present a Multi-scale Pruning Module (MPM) to adaptively prune irrelevant features by maximizing the dependency between the two feature sets. Moreover, we design the Scale-Aware Dynamic Pruning Attention (SADPA) to aggregate information from different scales via a hierarchical design. Our method's superior matching performance and generalization capability are confirmed by leading accuracy across various evaluation benchmarks and downstream tasks. The code is publicly available at https://github.com/Master-cai/PRISM.  
  </ol>  
</details>  
**comments**: 15 pages, 8 figures, ACM MM 2024. Supplementary materials are
  included  
  
  



## NeRF  

### [Goal-oriented Semantic Communication for the Metaverse Application](http://arxiv.org/abs/2408.03646)  
Zhe Wang, Nan Li, Yansha Deng  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    With the emergence of the metaverse and its role in enabling real-time simulation and analysis of real-world counterparts, an increasing number of personalized metaverse scenarios are being created to influence entertainment experiences and social behaviors. However, compared to traditional image and video entertainment applications, the exact transmission of the vast amount of metaverse-associated information significantly challenges the capacity of existing bit-oriented communication networks. Moreover, the current metaverse also witnesses a growing goal shift for transmitting the meaning behind custom-designed content, such as user-designed buildings and avatars, rather than exact copies of physical objects. To meet this growing goal shift and bandwidth challenge, this paper proposes a goal-oriented semantic communication framework for metaverse application (GSCM) to explore and define semantic information through the goal levels. Specifically, we first analyze the traditional image communication framework in metaverse construction and then detail our proposed semantic information along with the end-to-end wireless communication. We then describe the designed modules of the GSCM framework, including goal-oriented semantic information extraction, base knowledge definition, and neural radiance field (NeRF) based metaverse construction. Finally, numerous experiments have been conducted to demonstrate that, compared to image communication, our proposed GSCM framework decreases transmission latency by up to 92.6% and enhances the virtual object operation accuracy and metaverse construction clearance by up to 45.6% and 44.7%, respectively.  
  </ol>  
</details>  
  
### [RayGauss: Volumetric Gaussian-Based Ray Casting for Photorealistic Novel View Synthesis](http://arxiv.org/abs/2408.03356)  
Hugo Blanc, Jean-Emmanuel Deschaud, Alexis Paljic  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Differentiable volumetric rendering-based methods made significant progress in novel view synthesis. On one hand, innovative methods have replaced the Neural Radiance Fields (NeRF) network with locally parameterized structures, enabling high-quality renderings in a reasonable time. On the other hand, approaches have used differentiable splatting instead of NeRF's ray casting to optimize radiance fields rapidly using Gaussian kernels, allowing for fine adaptation to the scene. However, differentiable ray casting of irregularly spaced kernels has been scarcely explored, while splatting, despite enabling fast rendering times, is susceptible to clearly visible artifacts.   Our work closes this gap by providing a physically consistent formulation of the emitted radiance c and density {\sigma}, decomposed with Gaussian functions associated with Spherical Gaussians/Harmonics for all-frequency colorimetric representation. We also introduce a method enabling differentiable ray casting of irregularly distributed Gaussians using an algorithm that integrates radiance fields slab by slab and leverages a BVH structure. This allows our approach to finely adapt to the scene while avoiding splatting artifacts. As a result, we achieve superior rendering quality compared to the state-of-the-art while maintaining reasonable training times and achieving inference speeds of 25 FPS on the Blender dataset. Project page with videos and code: https://raygauss.github.io/  
  </ol>  
</details>  
**comments**: Project page with videos and code: https://raygauss.github.io/  
  
  



