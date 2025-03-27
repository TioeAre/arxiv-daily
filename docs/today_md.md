<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#AccidentSim:-Generating-Physically-Realistic-Vehicle-Collision-Videos-from-Real-World-Accident-Reports>AccidentSim: Generating Physically Realistic Vehicle Collision Videos from Real-World Accident Reports</a></li>
        <li><a href=#EVolSplat:-Efficient-Volume-based-Gaussian-Splatting-for-Urban-View-Synthesis>EVolSplat: Efficient Volume-based Gaussian Splatting for Urban View Synthesis</a></li>
        <li><a href=#Learning-Scene-Level-Signed-Directional-Distance-Function-with-Ellipsoidal-Priors-and-Neural-Residuals>Learning Scene-Level Signed Directional Distance Function with Ellipsoidal Priors and Neural Residuals</a></li>
      </ul>
    </li>
  </ol>
</details>

## NeRF  

### [AccidentSim: Generating Physically Realistic Vehicle Collision Videos from Real-World Accident Reports](http://arxiv.org/abs/2503.20654)  
Xiangwen Zhang, Qian Zhang, Longfei Han, Qiang Qu, Xiaoming Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Collecting real-world vehicle accident videos for autonomous driving research is challenging due to their rarity and complexity. While existing driving video generation methods may produce visually realistic videos, they often fail to deliver physically realistic simulations because they lack the capability to generate accurate post-collision trajectories. In this paper, we introduce AccidentSim, a novel framework that generates physically realistic vehicle collision videos by extracting and utilizing the physical clues and contextual information available in real-world vehicle accident reports. Specifically, AccidentSim leverages a reliable physical simulator to replicate post-collision vehicle trajectories from the physical and contextual information in the accident reports and to build a vehicle collision trajectory dataset. This dataset is then used to fine-tune a language model, enabling it to respond to user prompts and predict physically consistent post-collision trajectories across various driving scenarios based on user descriptions. Finally, we employ Neural Radiance Fields (NeRF) to render high-quality backgrounds, merging them with the foreground vehicles that exhibit physically realistic trajectories to generate vehicle collision videos. Experimental results demonstrate that the videos produced by AccidentSim excel in both visual and physical authenticity.  
  </ol>  
</details>  
  
### [EVolSplat: Efficient Volume-based Gaussian Splatting for Urban View Synthesis](http://arxiv.org/abs/2503.20168)  
Sheng Miao, Jiaxin Huang, Dongfeng Bai, Xu Yan, Hongyu Zhou, Yue Wang, Bingbing Liu, Andreas Geiger, Yiyi Liao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Novel view synthesis of urban scenes is essential for autonomous driving-related applications.Existing NeRF and 3DGS-based methods show promising results in achieving photorealistic renderings but require slow, per-scene optimization. We introduce EVolSplat, an efficient 3D Gaussian Splatting model for urban scenes that works in a feed-forward manner. Unlike existing feed-forward, pixel-aligned 3DGS methods, which often suffer from issues like multi-view inconsistencies and duplicated content, our approach predicts 3D Gaussians across multiple frames within a unified volume using a 3D convolutional network. This is achieved by initializing 3D Gaussians with noisy depth predictions, and then refining their geometric properties in 3D space and predicting color based on 2D textures. Our model also handles distant views and the sky with a flexible hemisphere background model. This enables us to perform fast, feed-forward reconstruction while achieving real-time rendering. Experimental evaluations on the KITTI-360 and Waymo datasets show that our method achieves state-of-the-art quality compared to existing feed-forward 3DGS- and NeRF-based methods.  
  </ol>  
</details>  
**comments**: CVPR2025  
  
### [Learning Scene-Level Signed Directional Distance Function with Ellipsoidal Priors and Neural Residuals](http://arxiv.org/abs/2503.20066)  
Zhirui Dai, Hojoon Shin, Yulun Tian, Ki Myung Brian Lee, Nikolay Atanasov  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Dense geometric environment representations are critical for autonomous mobile robot navigation and exploration. Recent work shows that implicit continuous representations of occupancy, signed distance, or radiance learned using neural networks offer advantages in reconstruction fidelity, efficiency, and differentiability over explicit discrete representations based on meshes, point clouds, and voxels. In this work, we explore a directional formulation of signed distance, called signed directional distance function (SDDF). Unlike signed distance function (SDF) and similar to neural radiance fields (NeRF), SDDF has a position and viewing direction as input. Like SDF and unlike NeRF, SDDF directly provides distance to the observed surface along the direction, rather than integrating along the view ray, allowing efficient view synthesis. To learn and predict scene-level SDDF efficiently, we develop a differentiable hybrid representation that combines explicit ellipsoid priors and implicit neural residuals. This approach allows the model to effectively handle large distance discontinuities around obstacle boundaries while preserving the ability for dense high-fidelity prediction. We show that SDDF is competitive with the state-of-the-art neural implicit scene models in terms of reconstruction accuracy and rendering efficiency, while allowing differentiable view prediction for robot trajectory optimization.  
  </ol>  
</details>  
  
  



