<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Correspondence-Guided-SfM-Free-3D-Gaussian-Splatting-for-NVS>Correspondence-Guided SfM-Free 3D Gaussian Splatting for NVS</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#VF-NeRF:-Learning-Neural-Vector-Fields-for-Indoor-Scene-Reconstruction>VF-NeRF: Learning Neural Vector Fields for Indoor Scene Reconstruction</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Correspondence-Guided SfM-Free 3D Gaussian Splatting for NVS](http://arxiv.org/abs/2408.08723)  
Wei Sun, Xiaosong Zhang, Fang Wan, Yanzhao Zhou, Yuan Li, Qixiang Ye, Jianbin Jiao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Novel View Synthesis (NVS) without Structure-from-Motion (SfM) pre-processed camera poses--referred to as SfM-free methods--is crucial for promoting rapid response capabilities and enhancing robustness against variable operating conditions. Recent SfM-free methods have integrated pose optimization, designing end-to-end frameworks for joint camera pose estimation and NVS. However, most existing works rely on per-pixel image loss functions, such as L2 loss. In SfM-free methods, inaccurate initial poses lead to misalignment issue, which, under the constraints of per-pixel image loss functions, results in excessive gradients, causing unstable optimization and poor convergence for NVS. In this study, we propose a correspondence-guided SfM-free 3D Gaussian splatting for NVS. We use correspondences between the target and the rendered result to achieve better pixel alignment, facilitating the optimization of relative poses between frames. We then apply the learned poses to optimize the entire scene. Each 2D screen-space pixel is associated with its corresponding 3D Gaussians through approximated surface rendering to facilitate gradient back propagation. Experimental results underline the superior performance and time efficiency of the proposed approach compared to the state-of-the-art baselines.  
  </ol>  
</details>  
**comments**: arXiv admin note: text overlap with arXiv:2312.07504 by other authors  
  
  



## NeRF  

### [VF-NeRF: Learning Neural Vector Fields for Indoor Scene Reconstruction](http://arxiv.org/abs/2408.08766)  
[[code](https://github.com/albertgassol1/vf-nerf)]  
Albert Gassol Puigjaner, Edoardo Mello Rella, Erik Sandström, Ajad Chhatkuli, Luc Van Gool  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Implicit surfaces via neural radiance fields (NeRF) have shown surprising accuracy in surface reconstruction. Despite their success in reconstructing richly textured surfaces, existing methods struggle with planar regions with weak textures, which account for the majority of indoor scenes. In this paper, we address indoor dense surface reconstruction by revisiting key aspects of NeRF in order to use the recently proposed Vector Field (VF) as the implicit representation. VF is defined by the unit vector directed to the nearest surface point. It therefore flips direction at the surface and equals to the explicit surface normals. Except for this flip, VF remains constant along planar surfaces and provides a strong inductive bias in representing planar surfaces. Concretely, we develop a novel density-VF relationship and a training scheme that allows us to learn VF via volume rendering By doing this, VF-NeRF can model large planar surfaces and sharp corners accurately. We show that, when depth cues are available, our method further improves and achieves state-of-the-art results in reconstructing indoor scenes and rendering novel views. We extensively evaluate VF-NeRF on indoor datasets and run ablations of its components.  
  </ol>  
</details>  
**comments**: 15 pages  
  
  



