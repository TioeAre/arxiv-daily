<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#MonoPatchNeRF:-Improving-Neural-Radiance-Fields-with-Patch-based-Monocular-Guidance>MonoPatchNeRF: Improving Neural Radiance Fields with Patch-based Monocular Guidance</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#OccGaussian:-3D-Gaussian-Splatting-for-Occluded-Human-Rendering>OccGaussian: 3D Gaussian Splatting for Occluded Human Rendering</a></li>
        <li><a href=#GPN:-Generative-Point-based-NeRF>GPN: Generative Point-based NeRF</a></li>
        <li><a href=#MonoPatchNeRF:-Improving-Neural-Radiance-Fields-with-Patch-based-Monocular-Guidance>MonoPatchNeRF: Improving Neural Radiance Fields with Patch-based Monocular Guidance</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [MonoPatchNeRF: Improving Neural Radiance Fields with Patch-based Monocular Guidance](http://arxiv.org/abs/2404.08252)  
Yuqun Wu, Jae Yong Lee, Chuhang Zou, Shenlong Wang, Derek Hoiem  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The latest regularized Neural Radiance Field (NeRF) approaches produce poor geometry and view extrapolation for multiview stereo (MVS) benchmarks such as ETH3D. In this paper, we aim to create 3D models that provide accurate geometry and view synthesis, partially closing the large geometric performance gap between NeRF and traditional MVS methods. We propose a patch-based approach that effectively leverages monocular surface normal and relative depth predictions. The patch-based ray sampling also enables the appearance regularization of normalized cross-correlation (NCC) and structural similarity (SSIM) between randomly sampled virtual and training views. We further show that "density restrictions" based on sparse structure-from-motion points can help greatly improve geometric accuracy with a slight drop in novel view synthesis metrics. Our experiments show 4x the performance of RegNeRF and 8x that of FreeNeRF on average F1@2cm for ETH3D MVS benchmark, suggesting a fruitful research direction to improve the geometric accuracy of NeRF-based models, and sheds light on a potential future approach to enable NeRF-based optimization to eventually outperform traditional MVS.  
  </ol>  
</details>  
**comments**: 26 pages, 15 figures  
  
  



## NeRF  

### [OccGaussian: 3D Gaussian Splatting for Occluded Human Rendering](http://arxiv.org/abs/2404.08449)  
Jingrui Ye, Zongkai Zhang, Yujiao Jiang, Qingmin Liao, Wenming Yang, Zongqing Lu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Rendering dynamic 3D human from monocular videos is crucial for various applications such as virtual reality and digital entertainment. Most methods assume the people is in an unobstructed scene, while various objects may cause the occlusion of body parts in real-life scenarios. Previous method utilizing NeRF for surface rendering to recover the occluded areas, but it requiring more than one day to train and several seconds to render, failing to meet the requirements of real-time interactive applications. To address these issues, we propose OccGaussian based on 3D Gaussian Splatting, which can be trained within 6 minutes and produces high-quality human renderings up to 160 FPS with occluded input. OccGaussian initializes 3D Gaussian distributions in the canonical space, and we perform occlusion feature query at occluded regions, the aggregated pixel-align feature is extracted to compensate for the missing information. Then we use Gaussian Feature MLP to further process the feature along with the occlusion-aware loss functions to better perceive the occluded area. Extensive experiments both in simulated and real-world occlusions, demonstrate that our method achieves comparable or even superior performance compared to the state-of-the-art method. And we improving training and inference speeds by 250x and 800x, respectively. Our code will be available for research purposes.  
  </ol>  
</details>  
**comments**: 12 April, 2024; originally announced April 2024  
  
### [GPN: Generative Point-based NeRF](http://arxiv.org/abs/2404.08312)  
Haipeng Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Scanning real-life scenes with modern registration devices typically gives incomplete point cloud representations, primarily due to the limitations of partial scanning, 3D occlusions, and dynamic light conditions. Recent works on processing incomplete point clouds have always focused on point cloud completion. However, these approaches do not ensure consistency between the completed point cloud and the captured images regarding color and geometry. We propose using Generative Point-based NeRF (GPN) to reconstruct and repair a partial cloud by fully utilizing the scanning images and the corresponding reconstructed cloud. The repaired point cloud can achieve multi-view consistency with the captured images at high spatial resolution. For the finetunes of a single scene, we optimize the global latent condition by incorporating an Auto-Decoder architecture while retaining multi-view consistency. As a result, the generated point clouds are smooth, plausible, and geometrically consistent with the partial scanning images. Extensive experiments on ShapeNet demonstrate that our works achieve competitive performances to the other state-of-the-art point cloud-based neural scene rendering and editing performances.  
  </ol>  
</details>  
  
### [MonoPatchNeRF: Improving Neural Radiance Fields with Patch-based Monocular Guidance](http://arxiv.org/abs/2404.08252)  
Yuqun Wu, Jae Yong Lee, Chuhang Zou, Shenlong Wang, Derek Hoiem  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The latest regularized Neural Radiance Field (NeRF) approaches produce poor geometry and view extrapolation for multiview stereo (MVS) benchmarks such as ETH3D. In this paper, we aim to create 3D models that provide accurate geometry and view synthesis, partially closing the large geometric performance gap between NeRF and traditional MVS methods. We propose a patch-based approach that effectively leverages monocular surface normal and relative depth predictions. The patch-based ray sampling also enables the appearance regularization of normalized cross-correlation (NCC) and structural similarity (SSIM) between randomly sampled virtual and training views. We further show that "density restrictions" based on sparse structure-from-motion points can help greatly improve geometric accuracy with a slight drop in novel view synthesis metrics. Our experiments show 4x the performance of RegNeRF and 8x that of FreeNeRF on average F1@2cm for ETH3D MVS benchmark, suggesting a fruitful research direction to improve the geometric accuracy of NeRF-based models, and sheds light on a potential future approach to enable NeRF-based optimization to eventually outperform traditional MVS.  
  </ol>  
</details>  
**comments**: 26 pages, 15 figures  
  
  



