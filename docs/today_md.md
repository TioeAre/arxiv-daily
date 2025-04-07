<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#An-Algebraic-Geometry-Approach-to-Viewing-Graph-Solvability>An Algebraic Geometry Approach to Viewing Graph Solvability</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#REJEPA:-A-Novel-Joint-Embedding-Predictive-Architecture-for-Efficient-Remote-Sensing-Image-Retrieval>REJEPA: A Novel Joint-Embedding Predictive Architecture for Efficient Remote Sensing Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#NeRFlex:-Resource-aware-Real-time-High-quality-Rendering-of-Complex-Scenes-on-Mobile-Devices>NeRFlex: Resource-aware Real-time High-quality Rendering of Complex Scenes on Mobile Devices</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [An Algebraic Geometry Approach to Viewing Graph Solvability](http://arxiv.org/abs/2504.03637)  
Federica Arrigoni, Kathlén Kohn, Andrea Fusiello, Tomas Pajdla  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The concept of viewing graph solvability has gained significant interest in the context of structure-from-motion. A viewing graph is a mathematical structure where nodes are associated to cameras and edges represent the epipolar geometry connecting overlapping views. Solvability studies under which conditions the cameras are uniquely determined by the graph. In this paper we propose a novel framework for analyzing solvability problems based on Algebraic Geometry, demonstrating its potential in understanding structure-from-motion graphs and proving a conjecture that was previously proposed.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [REJEPA: A Novel Joint-Embedding Predictive Architecture for Efficient Remote Sensing Image Retrieval](http://arxiv.org/abs/2504.03169)  
Shabnam Choudhury, Yash Salunkhe, Sarthak Mehrotra, Biplab Banerjee  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The rapid expansion of remote sensing image archives demands the development of strong and efficient techniques for content-based image retrieval (RS-CBIR). This paper presents REJEPA (Retrieval with Joint-Embedding Predictive Architecture), an innovative self-supervised framework designed for unimodal RS-CBIR. REJEPA utilises spatially distributed context token encoding to forecast abstract representations of target tokens, effectively capturing high-level semantic features and eliminating unnecessary pixel-level details. In contrast to generative methods that focus on pixel reconstruction or contrastive techniques that depend on negative pairs, REJEPA functions within feature space, achieving a reduction in computational complexity of 40-60% when compared to pixel-reconstruction baselines like Masked Autoencoders (MAE). To guarantee strong and varied representations, REJEPA incorporates Variance-Invariance-Covariance Regularisation (VICReg), which prevents encoder collapse by promoting feature diversity and reducing redundancy. The method demonstrates an estimated enhancement in retrieval accuracy of 5.1% on BEN-14K (S1), 7.4% on BEN-14K (S2), 6.0% on FMoW-RGB, and 10.1% on FMoW-Sentinel compared to prominent SSL techniques, including CSMAE-SESD, Mask-VLM, SatMAE, ScaleMAE, and SatMAE++, on extensive RS benchmarks BEN-14K (multispectral and SAR data), FMoW-RGB and FMoW-Sentinel. Through effective generalisation across sensor modalities, REJEPA establishes itself as a sensor-agnostic benchmark for efficient, scalable, and precise RS-CBIR, addressing challenges like varying resolutions, high object density, and complex backgrounds with computational efficiency.  
  </ol>  
</details>  
**comments**: 14 pages  
  
  



## NeRF  

### [NeRFlex: Resource-aware Real-time High-quality Rendering of Complex Scenes on Mobile Devices](http://arxiv.org/abs/2504.03415)  
Zhe Wang, Yifei Zhu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) is a cutting-edge neural network-based technique for novel view synthesis in 3D reconstruction. However, its significant computational demands pose challenges for deployment on mobile devices. While mesh-based NeRF solutions have shown potential in achieving real-time rendering on mobile platforms, they often fail to deliver high-quality reconstructions when rendering practical complex scenes. Additionally, the non-negligible memory overhead caused by pre-computed intermediate results complicates their practical application. To overcome these challenges, we present NeRFlex, a resource-aware, high-resolution, real-time rendering framework for complex scenes on mobile devices. NeRFlex integrates mobile NeRF rendering with multi-NeRF representations that decompose a scene into multiple sub-scenes, each represented by an individual NeRF network. Crucially, NeRFlex considers both memory and computation constraints as first-class citizens and redesigns the reconstruction process accordingly. NeRFlex first designs a detail-oriented segmentation module to identify sub-scenes with high-frequency details. For each NeRF network, a lightweight profiler, built on domain knowledge, is used to accurately map configurations to visual quality and memory usage. Based on these insights and the resource constraints on mobile devices, NeRFlex presents a dynamic programming algorithm to efficiently determine configurations for all NeRF representations, despite the NP-hardness of the original decision problem. Extensive experiments on real-world datasets and mobile devices demonstrate that NeRFlex achieves real-time, high-quality rendering on commercial mobile devices.  
  </ol>  
</details>  
**comments**: This paper is accepted by 45th IEEE International Conference on
  Distributed Computing Systems (ICDCS 2025)  
  
  



