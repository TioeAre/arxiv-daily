<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#SplineGS:-Robust-Motion-Adaptive-Spline-for-Real-Time-Dynamic-3D-Gaussians-from-Monocular-Video>SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Agtech-Framework-for-Cranberry-Ripening-Analysis-Using-Vision-Foundation-Models>Agtech Framework for Cranberry-Ripening Analysis Using Vision Foundation Models</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#NeRF-Texture:-Synthesizing-Neural-Radiance-Field-Textures>NeRF-Texture: Synthesizing Neural Radiance Field Textures</a></li>
        <li><a href=#Sharpening-Your-Density-Fields:-Spiking-Neuron-Aided-Fast-Geometry-Learning>Sharpening Your Density Fields: Spiking Neuron Aided Fast Geometry Learning</a></li>
        <li><a href=#PBR-NeRF:-Inverse-Rendering-with-Physics-Based-Neural-Fields>PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video](http://arxiv.org/abs/2412.09982)  
Jongmin Park, Minh-Quan Viet Bui, Juan Luis Gonzalez Bello, Jaeho Moon, Jihyong Oh, Munchurl Kim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Synthesizing novel views from in-the-wild monocular videos is challenging due to scene dynamics and the lack of multi-view cues. To address this, we propose SplineGS, a COLMAP-free dynamic 3D Gaussian Splatting (3DGS) framework for high-quality reconstruction and fast rendering from monocular videos. At its core is a novel Motion-Adaptive Spline (MAS) method, which represents continuous dynamic 3D Gaussian trajectories using cubic Hermite splines with a small number of control points. For MAS, we introduce a Motion-Adaptive Control points Pruning (MACP) method to model the deformation of each dynamic 3D Gaussian across varying motions, progressively pruning control points while maintaining dynamic modeling integrity. Additionally, we present a joint optimization strategy for camera parameter estimation and 3D Gaussian attributes, leveraging photometric and geometric consistency. This eliminates the need for Structure-from-Motion preprocessing and enhances SplineGS's robustness in real-world conditions. Experiments show that SplineGS significantly outperforms state-of-the-art methods in novel view synthesis quality for dynamic scenes from monocular videos, achieving thousands times faster rendering speed.  
  </ol>  
</details>  
**comments**: The first two authors contributed equally to this work (equal
  contribution). The last two authors advised equally to this work. Please
  visit our project page at this https://kaist-viclab.github.io/splinegs-site/  
  
  



## Keypoint Detection  

### [Agtech Framework for Cranberry-Ripening Analysis Using Vision Foundation Models](http://arxiv.org/abs/2412.09739)  
Faith Johnson, Ryan Meegan, Jack Lowry, Peter Oudemans, Kristin Dana  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Agricultural domains are being transformed by recent advances in AI and computer vision that support quantitative visual evaluation. Using aerial and ground imaging over a time series, we develop a framework for characterizing the ripening process of cranberry crops, a crucial component for precision agriculture tasks such as comparing crop breeds (high-throughput phenotyping) and detecting disease. Using drone imaging, we capture images from 20 waypoints across multiple bogs, and using ground-based imaging (hand-held camera), we image same bog patch using fixed fiducial markers. Both imaging methods are repeated to gather a multi-week time series spanning the entire growing season. Aerial imaging provides multiple samples to compute a distribution of albedo values. Ground imaging enables tracking of individual berries for a detailed view of berry appearance changes. Using vision transformers (ViT) for feature detection after segmentation, we extract a high dimensional feature descriptor of berry appearance. Interpretability of appearance is critical for plant biologists and cranberry growers to support crop breeding decisions (e.g.\ comparison of berry varieties from breeding programs). For interpretability, we create a 2D manifold of cranberry appearance by using a UMAP dimensionality reduction on ViT features. This projection enables quantification of ripening paths and a useful metric of ripening rate. We demonstrate the comparison of four cranberry varieties based on our ripening assessments. This work is the first of its kind and has future impact for cranberries and for other crops including wine grapes, olives, blueberries, and maize. Aerial and ground datasets are made publicly available.  
  </ol>  
</details>  
**comments**: arXiv admin note: substantial text overlap with arXiv:2309.00028  
  
  



## NeRF  

### [NeRF-Texture: Synthesizing Neural Radiance Field Textures](http://arxiv.org/abs/2412.10004)  
Yi-Hua Huang, Yan-Pei Cao, Yu-Kun Lai, Ying Shan, Lin Gao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Texture synthesis is a fundamental problem in computer graphics that would benefit various applications. Existing methods are effective in handling 2D image textures. In contrast, many real-world textures contain meso-structure in the 3D geometry space, such as grass, leaves, and fabrics, which cannot be effectively modeled using only 2D image textures. We propose a novel texture synthesis method with Neural Radiance Fields (NeRF) to capture and synthesize textures from given multi-view images. In the proposed NeRF texture representation, a scene with fine geometric details is disentangled into the meso-structure textures and the underlying base shape. This allows textures with meso-structure to be effectively learned as latent features situated on the base shape, which are fed into a NeRF decoder trained simultaneously to represent the rich view-dependent appearance. Using this implicit representation, we can synthesize NeRF-based textures through patch matching of latent features. However, inconsistencies between the metrics of the reconstructed content space and the latent feature space may compromise the synthesis quality. To enhance matching performance, we further regularize the distribution of latent features by incorporating a clustering constraint. In addition to generating NeRF textures over a planar domain, our method can also synthesize NeRF textures over curved surfaces, which are practically useful. Experimental results and evaluations demonstrate the effectiveness of our approach.  
  </ol>  
</details>  
  
### [Sharpening Your Density Fields: Spiking Neuron Aided Fast Geometry Learning](http://arxiv.org/abs/2412.09881)  
Yi Gu, Zhaorui Wang, Dongjun Ye, Renjing Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) have achieved remarkable progress in neural rendering. Extracting geometry from NeRF typically relies on the Marching Cubes algorithm, which uses a hand-crafted threshold to define the level set. However, this threshold-based approach requires laborious and scenario-specific tuning, limiting its practicality for real-world applications. In this work, we seek to enhance the efficiency of this method during the training time. To this end, we introduce a spiking neuron mechanism that dynamically adjusts the threshold, eliminating the need for manual selection. Despite its promise, directly training with the spiking neuron often results in model collapse and noisy outputs. To overcome these challenges, we propose a round-robin strategy that stabilizes the training process and enables the geometry network to achieve a sharper and more precise density distribution with minimal computational overhead. We validate our approach through extensive experiments on both synthetic and real-world datasets. The results show that our method significantly improves the performance of threshold-based techniques, offering a more robust and efficient solution for NeRF geometry extraction.  
  </ol>  
</details>  
  
### [PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields](http://arxiv.org/abs/2412.09680)  
[[code](https://github.com/s3anwu/pbrnerf)]  
Sean Wu, Shamik Basu, Tim Broedermann, Luc Van Gool, Christos Sakaridis  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We tackle the ill-posed inverse rendering problem in 3D reconstruction with a Neural Radiance Field (NeRF) approach informed by Physics-Based Rendering (PBR) theory, named PBR-NeRF. Our method addresses a key limitation in most NeRF and 3D Gaussian Splatting approaches: they estimate view-dependent appearance without modeling scene materials and illumination. To address this limitation, we present an inverse rendering (IR) model capable of jointly estimating scene geometry, materials, and illumination. Our model builds upon recent NeRF-based IR approaches, but crucially introduces two novel physics-based priors that better constrain the IR estimation. Our priors are rigorously formulated as intuitive loss terms and achieve state-of-the-art material estimation without compromising novel view synthesis quality. Our method is easily adaptable to other inverse rendering and 3D reconstruction frameworks that require material estimation. We demonstrate the importance of extending current neural rendering approaches to fully model scene properties beyond geometry and view-dependent appearance. Code is publicly available at https://github.com/s3anwu/pbrnerf  
  </ol>  
</details>  
**comments**: 16 pages, 7 figures. Code is publicly available at
  https://github.com/s3anwu/pbrnerf  
  
  



