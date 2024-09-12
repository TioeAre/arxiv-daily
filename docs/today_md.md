<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#DreamMesh:-Jointly-Manipulating-and-Texturing-Triangle-Meshes-for-Text-to-3D-Generation>DreamMesh: Jointly Manipulating and Texturing Triangle Meshes for Text-to-3D Generation</a></li>
        <li><a href=#ThermalGaussian:-Thermal-3D-Gaussian-Splatting>ThermalGaussian: Thermal 3D Gaussian Splatting</a></li>
      </ul>
    </li>
  </ol>
</details>

## NeRF  

### [DreamMesh: Jointly Manipulating and Texturing Triangle Meshes for Text-to-3D Generation](http://arxiv.org/abs/2409.07454)  
Haibo Yang, Yang Chen, Yingwei Pan, Ting Yao, Zhineng Chen, Zuxuan Wu, Yu-Gang Jiang, Tao Mei  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Learning radiance fields (NeRF) with powerful 2D diffusion models has garnered popularity for text-to-3D generation. Nevertheless, the implicit 3D representations of NeRF lack explicit modeling of meshes and textures over surfaces, and such surface-undefined way may suffer from the issues, e.g., noisy surfaces with ambiguous texture details or cross-view inconsistency. To alleviate this, we present DreamMesh, a novel text-to-3D architecture that pivots on well-defined surfaces (triangle meshes) to generate high-fidelity explicit 3D model. Technically, DreamMesh capitalizes on a distinctive coarse-to-fine scheme. In the coarse stage, the mesh is first deformed by text-guided Jacobians and then DreamMesh textures the mesh with an interlaced use of 2D diffusion models in a tuning free manner from multiple viewpoints. In the fine stage, DreamMesh jointly manipulates the mesh and refines the texture map, leading to high-quality triangle meshes with high-fidelity textured materials. Extensive experiments demonstrate that DreamMesh significantly outperforms state-of-the-art text-to-3D methods in faithfully generating 3D content with richer textual details and enhanced geometry. Our project page is available at https://dreammesh.github.io.  
  </ol>  
</details>  
**comments**: ECCV 2024. Project page is available at
  \url{https://dreammesh.github.io}  
  
### [ThermalGaussian: Thermal 3D Gaussian Splatting](http://arxiv.org/abs/2409.07200)  
Rongfeng Lu, Hangyu Chen, Zunjie Zhu, Yuhang Qin, Ming Lu, Le Zhang, Chenggang Yan, Anke Xue  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Thermography is especially valuable for the military and other users of surveillance cameras. Some recent methods based on Neural Radiance Fields (NeRF) are proposed to reconstruct the thermal scenes in 3D from a set of thermal and RGB images. However, unlike NeRF, 3D Gaussian splatting (3DGS) prevails due to its rapid training and real-time rendering. In this work, we propose ThermalGaussian, the first thermal 3DGS approach capable of rendering high-quality images in RGB and thermal modalities. We first calibrate the RGB camera and the thermal camera to ensure that both modalities are accurately aligned. Subsequently, we use the registered images to learn the multimodal 3D Gaussians. To prevent the overfitting of any single modality, we introduce several multimodal regularization constraints. We also develop smoothing constraints tailored to the physical characteristics of the thermal modality. Besides, we contribute a real-world dataset named RGBT-Scenes, captured by a hand-hold thermal-infrared camera, facilitating future research on thermal scene reconstruction. We conduct comprehensive experiments to show that ThermalGaussian achieves photorealistic rendering of thermal images and improves the rendering quality of RGB images. With the proposed multimodal regularization constraints, we also reduced the model's storage cost by 90\%. The code and dataset will be released.  
  </ol>  
</details>  
**comments**: 10 pages, 7 figures  
  
  



