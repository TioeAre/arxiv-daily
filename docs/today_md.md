<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Affine-based-Deformable-Attention-and-Selective-Fusion-for-Semi-dense-Matching>Affine-based Deformable Attention and Selective Fusion for Semi-dense Matching</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#NeRF-Casting:-Improved-View-Dependent-Appearance-with-Consistent-Reflections>NeRF-Casting: Improved View-Dependent Appearance with Consistent Reflections</a></li>
        <li><a href=#Neural-Directional-Encoding-for-Efficient-and-Accurate-View-Dependent-Appearance-Modeling>Neural Directional Encoding for Efficient and Accurate View-Dependent Appearance Modeling</a></li>
        <li><a href=#Camera-Relocalization-in-Shadow-free-Neural-Radiance-Fields>Camera Relocalization in Shadow-free Neural Radiance Fields</a></li>
        <li><a href=#LDM:-Large-Tensorial-SDF-Model-for-Textured-Mesh-Generation>LDM: Large Tensorial SDF Model for Textured Mesh Generation</a></li>
        <li><a href=#JointRF:-End-to-End-Joint-Optimization-for-Dynamic-Neural-Radiance-Field-Representation-and-Compression>JointRF: End-to-End Joint Optimization for Dynamic Neural Radiance Field Representation and Compression</a></li>
        <li><a href=#DoGaussian:-Distributed-Oriented-Gaussian-Splatting-for-Large-Scale-3D-Reconstruction-Via-Gaussian-Consensus>DoGaussian: Distributed-Oriented Gaussian Splatting for Large-Scale 3D Reconstruction Via Gaussian Consensus</a></li>
        <li><a href=#Gaussian-Time-Machine:-A-Real-Time-Rendering-Methodology-for-Time-Variant-Appearances>Gaussian Time Machine: A Real-Time Rendering Methodology for Time-Variant Appearances</a></li>
      </ul>
    </li>
  </ol>
</details>

## Image Matching  

### [Affine-based Deformable Attention and Selective Fusion for Semi-dense Matching](http://arxiv.org/abs/2405.13874)  
Hongkai Chen, Zixin Luo, Yurun Tian, Xuyang Bai, Ziyu Wang, Lei Zhou, Mingmin Zhen, Tian Fang, David McKinnon, Yanghai Tsin, Long Quan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Identifying robust and accurate correspondences across images is a fundamental problem in computer vision that enables various downstream tasks. Recent semi-dense matching methods emphasize the effectiveness of fusing relevant cross-view information through Transformer. In this paper, we propose several improvements upon this paradigm. Firstly, we introduce affine-based local attention to model cross-view deformations. Secondly, we present selective fusion to merge local and global messages from cross attention. Apart from network structure, we also identify the importance of enforcing spatial smoothness in loss design, which has been omitted by previous works. Based on these augmentations, our network demonstrate strong matching capacity under different settings. The full version of our network achieves state-of-the-art performance among semi-dense matching methods at a similar cost to LoFTR, while the slim version reaches LoFTR baseline's performance with only 15% computation cost and 18% parameters.  
  </ol>  
</details>  
**comments**: Accepted to CVPR2024 Image Matching Workshop  
  
  



## NeRF  

### [NeRF-Casting: Improved View-Dependent Appearance with Consistent Reflections](http://arxiv.org/abs/2405.14871)  
Dor Verbin, Pratul P. Srinivasan, Peter Hedman, Ben Mildenhall, Benjamin Attal, Richard Szeliski, Jonathan T. Barron  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRFs) typically struggle to reconstruct and render highly specular objects, whose appearance varies quickly with changes in viewpoint. Recent works have improved NeRF's ability to render detailed specular appearance of distant environment illumination, but are unable to synthesize consistent reflections of closer content. Moreover, these techniques rely on large computationally-expensive neural networks to model outgoing radiance, which severely limits optimization and rendering speed. We address these issues with an approach based on ray tracing: instead of querying an expensive neural network for the outgoing view-dependent radiance at points along each camera ray, our model casts reflection rays from these points and traces them through the NeRF representation to render feature vectors which are decoded into color using a small inexpensive network. We demonstrate that our model outperforms prior methods for view synthesis of scenes containing shiny objects, and that it is the only existing NeRF method that can synthesize photorealistic specular appearance and reflections in real-world scenes, while requiring comparable optimization time to current state-of-the-art view synthesis models.  
  </ol>  
</details>  
**comments**: Project page: http://nerf-casting.github.io  
  
### [Neural Directional Encoding for Efficient and Accurate View-Dependent Appearance Modeling](http://arxiv.org/abs/2405.14847)  
Liwen Wu, Sai Bi, Zexiang Xu, Fujun Luan, Kai Zhang, Iliyan Georgiev, Kalyan Sunkavalli, Ravi Ramamoorthi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Novel-view synthesis of specular objects like shiny metals or glossy paints remains a significant challenge. Not only the glossy appearance but also global illumination effects, including reflections of other objects in the environment, are critical components to faithfully reproduce a scene. In this paper, we present Neural Directional Encoding (NDE), a view-dependent appearance encoding of neural radiance fields (NeRF) for rendering specular objects. NDE transfers the concept of feature-grid-based spatial encoding to the angular domain, significantly improving the ability to model high-frequency angular signals. In contrast to previous methods that use encoding functions with only angular input, we additionally cone-trace spatial features to obtain a spatially varying directional encoding, which addresses the challenging interreflection effects. Extensive experiments on both synthetic and real datasets show that a NeRF model with NDE (1) outperforms the state of the art on view synthesis of specular objects, and (2) works with small networks to allow fast (real-time) inference. The project webpage and source code are available at: \url{https://lwwu2.github.io/nde/}.  
  </ol>  
</details>  
**comments**: Accepted to CVPR 2024  
  
### [Camera Relocalization in Shadow-free Neural Radiance Fields](http://arxiv.org/abs/2405.14824)  
Shiyao Xu, Caiyun Liu, Yuantao Chen, Zhenxin Zhu, Zike Yan, Yongliang Shi, Hao Zhao, Guyue Zhou  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Camera relocalization is a crucial problem in computer vision and robotics. Recent advancements in neural radiance fields (NeRFs) have shown promise in synthesizing photo-realistic images. Several works have utilized NeRFs for refining camera poses, but they do not account for lighting changes that can affect scene appearance and shadow regions, causing a degraded pose optimization process. In this paper, we propose a two-staged pipeline that normalizes images with varying lighting and shadow conditions to improve camera relocalization. We implement our scene representation upon a hash-encoded NeRF which significantly boosts up the pose optimization process. To account for the noisy image gradient computing problem in grid-based NeRFs, we further propose a re-devised truncated dynamic low-pass filter (TDLF) and a numerical gradient averaging technique to smoothen the process. Experimental results on several datasets with varying lighting conditions demonstrate that our method achieves state-of-the-art results in camera relocalization under varying lighting conditions. Code and data will be made publicly available.  
  </ol>  
</details>  
**comments**: Accepted by ICRA 2024. 8 pages, 5 figures, 3 tables. Codes and
  dataset: https://github.com/hnrna/ShadowfreeNeRF-CameraReloc  
  
### [LDM: Large Tensorial SDF Model for Textured Mesh Generation](http://arxiv.org/abs/2405.14580)  
Rengan Xie, Wenting Zheng, Kai Huang, Yizheng Chen, Qi Wang, Qi Ye, Wei Chen, Yuchi Huo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Previous efforts have managed to generate production-ready 3D assets from text or images. However, these methods primarily employ NeRF or 3D Gaussian representations, which are not adept at producing smooth, high-quality geometries required by modern rendering pipelines. In this paper, we propose LDM, a novel feed-forward framework capable of generating high-fidelity, illumination-decoupled textured mesh from a single image or text prompts. We firstly utilize a multi-view diffusion model to generate sparse multi-view inputs from single images or text prompts, and then a transformer-based model is trained to predict a tensorial SDF field from these sparse multi-view image inputs. Finally, we employ a gradient-based mesh optimization layer to refine this model, enabling it to produce an SDF field from which high-quality textured meshes can be extracted. Extensive experiments demonstrate that our method can generate diverse, high-quality 3D mesh assets with corresponding decomposed RGB textures within seconds.  
  </ol>  
</details>  
  
### [JointRF: End-to-End Joint Optimization for Dynamic Neural Radiance Field Representation and Compression](http://arxiv.org/abs/2405.14452)  
Zihan Zheng, Houqiang Zhong, Qiang Hu, Xiaoyun Zhang, Li Song, Ya Zhang, Yanfeng Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Field (NeRF) excels in photo-realistically static scenes, inspiring numerous efforts to facilitate volumetric videos. However, rendering dynamic and long-sequence radiance fields remains challenging due to the significant data required to represent volumetric videos. In this paper, we propose a novel end-to-end joint optimization scheme of dynamic NeRF representation and compression, called JointRF, thus achieving significantly improved quality and compression efficiency against the previous methods. Specifically, JointRF employs a compact residual feature grid and a coefficient feature grid to represent the dynamic NeRF. This representation handles large motions without compromising quality while concurrently diminishing temporal redundancy. We also introduce a sequential feature compression subnetwork to further reduce spatial-temporal redundancy. Finally, the representation and compression subnetworks are end-to-end trained combined within the JointRF. Extensive experiments demonstrate that JointRF can achieve superior compression performance across various datasets.  
  </ol>  
</details>  
**comments**: 8 pages, 5 figures  
  
### [DoGaussian: Distributed-Oriented Gaussian Splatting for Large-Scale 3D Reconstruction Via Gaussian Consensus](http://arxiv.org/abs/2405.13943)  
Yu Chen, Gim Hee Lee  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The recent advances in 3D Gaussian Splatting (3DGS) show promising results on the novel view synthesis (NVS) task. With its superior rendering performance and high-fidelity rendering quality, 3DGS is excelling at its previous NeRF counterparts. The most recent 3DGS method focuses either on improving the instability of rendering efficiency or reducing the model size. On the other hand, the training efficiency of 3DGS on large-scale scenes has not gained much attention. In this work, we propose DoGaussian, a method that trains 3DGS distributedly. Our method first decomposes a scene into K blocks and then introduces the Alternating Direction Method of Multipliers (ADMM) into the training procedure of 3DGS. During training, our DoGaussian maintains one global 3DGS model on the master node and K local 3DGS models on the slave nodes. The K local 3DGS models are dropped after training and we only query the global 3DGS model during inference. The training time is reduced by scene decomposition, and the training convergence and stability are guaranteed through the consensus on the shared 3D Gaussians. Our method accelerates the training of 3DGS by 6+ times when evaluated on large-scale scenes while concurrently achieving state-of-the-art rendering quality. Our project page is available at https://aibluefisher.github.io/DoGaussian.  
  </ol>  
</details>  
  
### [Gaussian Time Machine: A Real-Time Rendering Methodology for Time-Variant Appearances](http://arxiv.org/abs/2405.13694)  
Licheng Shen, Ho Ngai Chow, Lingyun Wang, Tong Zhang, Mengqiu Wang, Yuxing Han  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in neural rendering techniques have significantly enhanced the fidelity of 3D reconstruction. Notably, the emergence of 3D Gaussian Splatting (3DGS) has marked a significant milestone by adopting a discrete scene representation, facilitating efficient training and real-time rendering. Several studies have successfully extended the real-time rendering capability of 3DGS to dynamic scenes. However, a challenge arises when training images are captured under vastly differing weather and lighting conditions. This scenario poses a challenge for 3DGS and its variants in achieving accurate reconstructions. Although NeRF-based methods (NeRF-W, CLNeRF) have shown promise in handling such challenging conditions, their computational demands hinder real-time rendering capabilities. In this paper, we present Gaussian Time Machine (GTM) which models the time-dependent attributes of Gaussian primitives with discrete time embedding vectors decoded by a lightweight Multi-Layer-Perceptron(MLP). By adjusting the opacity of Gaussian primitives, we can reconstruct visibility changes of objects. We further propose a decomposed color model for improved geometric consistency. GTM achieved state-of-the-art rendering fidelity on 3 datasets and is 100 times faster than NeRF-based counterparts in rendering. Moreover, GTM successfully disentangles the appearance changes and renders smooth appearance interpolation.  
  </ol>  
</details>  
**comments**: 14 pages, 6 figures  
  
  



