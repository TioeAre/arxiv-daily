<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#DiffusionSfM:-Predicting-Structure-and-Motion-via-Ray-Origin-and-Endpoint-Diffusion>DiffusionSfM: Predicting Structure and Motion via Ray Origin and Endpoint Diffusion</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#3D-Scene-Generation:-A-Survey>3D Scene Generation: A Survey</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [DiffusionSfM: Predicting Structure and Motion via Ray Origin and Endpoint Diffusion](http://arxiv.org/abs/2505.05473)  
Qitao Zhao, Amy Lin, Jeff Tan, Jason Y. Zhang, Deva Ramanan, Shubham Tulsiani  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Current Structure-from-Motion (SfM) methods typically follow a two-stage pipeline, combining learned or geometric pairwise reasoning with a subsequent global optimization step. In contrast, we propose a data-driven multi-view reasoning approach that directly infers 3D scene geometry and camera poses from multi-view images. Our framework, DiffusionSfM, parameterizes scene geometry and cameras as pixel-wise ray origins and endpoints in a global frame and employs a transformer-based denoising diffusion model to predict them from multi-view inputs. To address practical challenges in training diffusion models with missing data and unbounded scene coordinates, we introduce specialized mechanisms that ensure robust learning. We empirically validate DiffusionSfM on both synthetic and real datasets, demonstrating that it outperforms classical and learning-based approaches while naturally modeling uncertainty.  
  </ol>  
</details>  
**comments**: CVPR 2025. Project website: https://qitaozhao.github.io/DiffusionSfM  
  
  



## NeRF  

### [3D Scene Generation: A Survey](http://arxiv.org/abs/2505.05474)  
[[code](https://github.com/hzxie/awesome-3d-scene-generation)]  
Beichen Wen, Haozhe Xie, Zhaoxi Chen, Fangzhou Hong, Ziwei Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D scene generation seeks to synthesize spatially structured, semantically meaningful, and photorealistic environments for applications such as immersive media, robotics, autonomous driving, and embodied AI. Early methods based on procedural rules offered scalability but limited diversity. Recent advances in deep generative models (e.g., GANs, diffusion models) and 3D representations (e.g., NeRF, 3D Gaussians) have enabled the learning of real-world scene distributions, improving fidelity, diversity, and view consistency. Recent advances like diffusion models bridge 3D scene synthesis and photorealism by reframing generation as image or video synthesis problems. This survey provides a systematic overview of state-of-the-art approaches, organizing them into four paradigms: procedural generation, neural 3D-based generation, image-based generation, and video-based generation. We analyze their technical foundations, trade-offs, and representative results, and review commonly used datasets, evaluation protocols, and downstream applications. We conclude by discussing key challenges in generation capacity, 3D representation, data and annotations, and evaluation, and outline promising directions including higher fidelity, physics-aware and interactive generation, and unified perception-generation models. This review organizes recent advances in 3D scene generation and highlights promising directions at the intersection of generative AI, 3D vision, and embodied intelligence. To track ongoing developments, we maintain an up-to-date project page: https://github.com/hzxie/Awesome-3D-Scene-Generation.  
  </ol>  
</details>  
**comments**: Project Page: https://github.com/hzxie/Awesome-3D-Scene-Generation  
  
  



