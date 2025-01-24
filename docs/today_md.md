<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#FAST-LIVO2-on-Resource-Constrained-Platforms:-LiDAR-Inertial-Visual-Odometry-with-Efficient-Memory-and-Computation>FAST-LIVO2 on Resource-Constrained Platforms: LiDAR-Inertial-Visual Odometry with Efficient Memory and Computation</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#VIGS-SLAM:-IMU-based-Large-Scale-3D-Gaussian-Splatting-SLAM>VIGS SLAM: IMU-based Large-Scale 3D Gaussian Splatting SLAM</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [FAST-LIVO2 on Resource-Constrained Platforms: LiDAR-Inertial-Visual Odometry with Efficient Memory and Computation](http://arxiv.org/abs/2501.13876)  
Bingyang Zhou, Chunran Zheng, Ziming Wang, Fangcheng Zhu, Yixi Cai, Fu Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper presents a lightweight LiDAR-inertial-visual odometry system optimized for resource-constrained platforms. It integrates a degeneration-aware adaptive visual frame selector into error-state iterated Kalman filter (ESIKF) with sequential updates, improving computation efficiency significantly while maintaining a similar level of robustness. Additionally, a memory-efficient mapping structure combining a locally unified visual-LiDAR map and a long-term visual map achieves a good trade-off between performance and memory usage. Extensive experiments on x86 and ARM platforms demonstrate the system's robustness and efficiency. On the Hilti dataset, our system achieves a 33% reduction in per-frame runtime and 47% lower memory usage compared to FAST-LIVO2, with only a 3 cm increase in RMSE. Despite this slight accuracy trade-off, our system remains competitive, outperforming state-of-the-art (SOTA) LIO methods such as FAST-LIO2 and most existing LIVO systems. These results validate the system's capability for scalable deployment on resource-constrained edge computing platforms.  
  </ol>  
</details>  
  
  



## NeRF  

### [VIGS SLAM: IMU-based Large-Scale 3D Gaussian Splatting SLAM](http://arxiv.org/abs/2501.13402)  
Gyuhyeon Pak, Euntai Kim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recently, map representations based on radiance fields such as 3D Gaussian Splatting and NeRF, which excellent for realistic depiction, have attracted considerable attention, leading to attempts to combine them with SLAM. While these approaches can build highly realistic maps, large-scale SLAM still remains a challenge because they require a large number of Gaussian images for mapping and adjacent images as keyframes for tracking. We propose a novel 3D Gaussian Splatting SLAM method, VIGS SLAM, that utilizes sensor fusion of RGB-D and IMU sensors for large-scale indoor environments. To reduce the computational load of 3DGS-based tracking, we adopt an ICP-based tracking framework that combines IMU preintegration to provide a good initial guess for accurate pose estimation. Our proposed method is the first to propose that Gaussian Splatting-based SLAM can be effectively performed in large-scale environments by integrating IMU sensor measurements. This proposal not only enhances the performance of Gaussian Splatting SLAM beyond room-scale scenarios but also achieves SLAM performance comparable to state-of-the-art methods in large-scale indoor environments.  
  </ol>  
</details>  
**comments**: 7 pages, 5 figures  
  
  



