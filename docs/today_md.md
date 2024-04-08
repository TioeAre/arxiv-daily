<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Towards-introspective-loop-closure-in-4D-radar-SLAM>Towards introspective loop closure in 4D radar SLAM</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Robust-Gaussian-Splatting>Robust Gaussian Splatting</a></li>
        <li><a href=#SC4D:-Sparse-Controlled-Video-to-4D-Generation-and-Motion-Transfer>SC4D: Sparse-Controlled Video-to-4D Generation and Motion Transfer</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Towards introspective loop closure in 4D radar SLAM](http://arxiv.org/abs/2404.03940)  
Maximilian Hilger, Vladimír Kubelka, Daniel Adolfsson, Henrik Andreasson, Achim J. Lilienthal  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Imaging radar is an emerging sensor modality in the context of Localization and Mapping (SLAM), especially suitable for vision-obstructed environments. This article investigates the use of 4D imaging radars for SLAM and analyzes the challenges in robust loop closure. Previous work indicates that 4D radars, together with inertial measurements, offer ample information for accurate odometry estimation. However, the low field of view, limited resolution, and sparse and noisy measurements render loop closure a significantly more challenging problem. Our work builds on the previous work - TBV SLAM - which was proposed for robust loop closure with 360 $^\circ$ spinning radars. This article highlights and addresses challenges inherited from a directional 4D radar, such as sparsity, noise, and reduced field of view, and discusses why the common definition of a loop closure is unsuitable. By combining multiple quality measures for accurate loop closure detection adapted to 4D radar data, significant results in trajectory estimation are achieved; the absolute trajectory error is as low as 0.46 m over a distance of 1.8 km, with consistent operation over multiple environments.  
  </ol>  
</details>  
**comments**: Submitted to the workshop "Radar in Robotics: Resilience from Signal
  to Navigation" at ICRA 2024  
  
  



## NeRF  

### [Robust Gaussian Splatting](http://arxiv.org/abs/2404.04211)  
François Darmon, Lorenzo Porzi, Samuel Rota-Bulò, Peter Kontschieder  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we address common error sources for 3D Gaussian Splatting (3DGS) including blur, imperfect camera poses, and color inconsistencies, with the goal of improving its robustness for practical applications like reconstructions from handheld phone captures. Our main contribution involves modeling motion blur as a Gaussian distribution over camera poses, allowing us to address both camera pose refinement and motion blur correction in a unified way. Additionally, we propose mechanisms for defocus blur compensation and for addressing color in-consistencies caused by ambient light, shadows, or due to camera-related factors like varying white balancing settings. Our proposed solutions integrate in a seamless way with the 3DGS formulation while maintaining its benefits in terms of training efficiency and rendering speed. We experimentally validate our contributions on relevant benchmark datasets including Scannet++ and Deblur-NeRF, obtaining state-of-the-art results and thus consistent improvements over relevant baselines.  
  </ol>  
</details>  
  
### [SC4D: Sparse-Controlled Video-to-4D Generation and Motion Transfer](http://arxiv.org/abs/2404.03736)  
Zijie Wu, Chaohui Yu, Yanqin Jiang, Chenjie Cao, Fan Wang, Xiang Bai  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advances in 2D/3D generative models enable the generation of dynamic 3D objects from a single-view video. Existing approaches utilize score distillation sampling to form the dynamic scene as dynamic NeRF or dense 3D Gaussians. However, these methods struggle to strike a balance among reference view alignment, spatio-temporal consistency, and motion fidelity under single-view conditions due to the implicit nature of NeRF or the intricate dense Gaussian motion prediction. To address these issues, this paper proposes an efficient, sparse-controlled video-to-4D framework named SC4D, that decouples motion and appearance to achieve superior video-to-4D generation. Moreover, we introduce Adaptive Gaussian (AG) initialization and Gaussian Alignment (GA) loss to mitigate shape degeneration issue, ensuring the fidelity of the learned motion and shape. Comprehensive experimental results demonstrate that our method surpasses existing methods in both quality and efficiency. In addition, facilitated by the disentangled modeling of motion and appearance of SC4D, we devise a novel application that seamlessly transfers the learned motion onto a diverse array of 4D entities according to textual descriptions.  
  </ol>  
</details>  
**comments**: Project Page: https://sc4d.github.io/  
  
  



