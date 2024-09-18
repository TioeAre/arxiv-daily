<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#GS-Net:-Generalizable-Plug-and-Play-3D-Gaussian-Splatting-Module>GS-Net: Generalizable Plug-and-Play 3D Gaussian Splatting Module</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Improving-the-Efficiency-of-Visually-Augmented-Language-Models>Improving the Efficiency of Visually Augmented Language Models</a></li>
        <li><a href=#HGSLoc:-3DGS-based-Heuristic-Camera-Pose-Refinement>HGSLoc: 3DGS-based Heuristic Camera Pose Refinement</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#RenderWorld:-World-Model-with-Self-Supervised-3D-Label>RenderWorld: World Model with Self-Supervised 3D Label</a></li>
        <li><a href=#HGSLoc:-3DGS-based-Heuristic-Camera-Pose-Refinement>HGSLoc: 3DGS-based Heuristic Camera Pose Refinement</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [GS-Net: Generalizable Plug-and-Play 3D Gaussian Splatting Module](http://arxiv.org/abs/2409.11307)  
Yichen Zhang, Zihan Wang, Jiali Han, Peilin Li, Jiaxun Zhang, Jianqiang Wang, Lei He, Keqiang Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D Gaussian Splatting (3DGS) integrates the strengths of primitive-based representations and volumetric rendering techniques, enabling real-time, high-quality rendering. However, 3DGS models typically overfit to single-scene training and are highly sensitive to the initialization of Gaussian ellipsoids, heuristically derived from Structure from Motion (SfM) point clouds, which limits both generalization and practicality. To address these limitations, we propose GS-Net, a generalizable, plug-and-play 3DGS module that densifies Gaussian ellipsoids from sparse SfM point clouds, enhancing geometric structure representation. To the best of our knowledge, GS-Net is the first plug-and-play 3DGS module with cross-scene generalization capabilities. Additionally, we introduce the CARLA-NVS dataset, which incorporates additional camera viewpoints to thoroughly evaluate reconstruction and rendering quality. Extensive experiments demonstrate that applying GS-Net to 3DGS yields a PSNR improvement of 2.08 dB for conventional viewpoints and 1.86 dB for novel viewpoints, confirming the method's effectiveness and robustness.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Improving the Efficiency of Visually Augmented Language Models](http://arxiv.org/abs/2409.11148)  
Paula Ontalvilla, Aitor Ormazabal, Gorka Azkune  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Despite the impressive performance of autoregressive Language Models (LM) it has been shown that due to reporting bias, LMs lack visual knowledge, i.e. they do not know much about the visual world and its properties. To augment LMs with visual knowledge, existing solutions often rely on explicit images, requiring time-consuming retrieval or image generation systems. This paper shows that explicit images are not necessary to visually augment an LM. Instead, we use visually-grounded text representations obtained from the well-known CLIP multimodal system. For a fair comparison, we modify VALM, a visually-augmented LM which uses image retrieval and representation, to work directly with visually-grounded text representations. We name this new model BLIND-VALM. We show that BLIND-VALM performs on par with VALM for Visual Language Understanding (VLU), Natural Language Understanding (NLU) and Language Modeling tasks, despite being significantly more efficient and simpler. We also show that scaling up our model within the compute budget of VALM, either increasing the model or pre-training corpus size, we outperform VALM for all the evaluation tasks.  
  </ol>  
</details>  
  
### [HGSLoc: 3DGS-based Heuristic Camera Pose Refinement](http://arxiv.org/abs/2409.10925)  
Zhongyan Niu, Zhen Tan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual localization refers to the process of determining camera poses and orientation within a known scene representation. This task is often complicated by factors such as illumination changes and variations in viewing angles. In this paper, we propose HGSLoc, a novel lightweight, plug and-play pose optimization framework, which integrates 3D reconstruction with a heuristic refinement strategy to achieve higher pose estimation accuracy. Specifically, we introduce an explicit geometric map for 3D representation and high-fidelity rendering, allowing the generation of high-quality synthesized views to support accurate visual localization. Our method demonstrates a faster rendering speed and higher localization accuracy compared to NeRF-based neural rendering localization approaches. We introduce a heuristic refinement strategy, its efficient optimization capability can quickly locate the target node, while we set the step-level optimization step to enhance the pose accuracy in the scenarios with small errors. With carefully designed heuristic functions, it offers efficient optimization capabilities, enabling rapid error reduction in rough localization estimations. Our method mitigates the dependence on complex neural network models while demonstrating improved robustness against noise and higher localization accuracy in challenging environments, as compared to neural network joint optimization strategies. The optimization framework proposed in this paper introduces novel approaches to visual localization by integrating the advantages of 3D reconstruction and heuristic refinement strategy, which demonstrates strong performance across multiple benchmark datasets, including 7Scenes and DB dataset.  
  </ol>  
</details>  
  
  



## NeRF  

### [RenderWorld: World Model with Self-Supervised 3D Label](http://arxiv.org/abs/2409.11356)  
Ziyang Yan, Wenzhen Dong, Yihua Shao, Yuhang Lu, Liu Haiyang, Jingwen Liu, Haozhe Wang, Zhe Wang, Yan Wang, Fabio Remondino, Yuexin Ma  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    End-to-end autonomous driving with vision-only is not only more cost-effective compared to LiDAR-vision fusion but also more reliable than traditional methods. To achieve a economical and robust purely visual autonomous driving system, we propose RenderWorld, a vision-only end-to-end autonomous driving framework, which generates 3D occupancy labels using a self-supervised gaussian-based Img2Occ Module, then encodes the labels by AM-VAE, and uses world model for forecasting and planning. RenderWorld employs Gaussian Splatting to represent 3D scenes and render 2D images greatly improves segmentation accuracy and reduces GPU memory consumption compared with NeRF-based methods. By applying AM-VAE to encode air and non-air separately, RenderWorld achieves more fine-grained scene element representation, leading to state-of-the-art performance in both 4D occupancy forecasting and motion planning from autoregressive world model.  
  </ol>  
</details>  
  
### [HGSLoc: 3DGS-based Heuristic Camera Pose Refinement](http://arxiv.org/abs/2409.10925)  
Zhongyan Niu, Zhen Tan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual localization refers to the process of determining camera poses and orientation within a known scene representation. This task is often complicated by factors such as illumination changes and variations in viewing angles. In this paper, we propose HGSLoc, a novel lightweight, plug and-play pose optimization framework, which integrates 3D reconstruction with a heuristic refinement strategy to achieve higher pose estimation accuracy. Specifically, we introduce an explicit geometric map for 3D representation and high-fidelity rendering, allowing the generation of high-quality synthesized views to support accurate visual localization. Our method demonstrates a faster rendering speed and higher localization accuracy compared to NeRF-based neural rendering localization approaches. We introduce a heuristic refinement strategy, its efficient optimization capability can quickly locate the target node, while we set the step-level optimization step to enhance the pose accuracy in the scenarios with small errors. With carefully designed heuristic functions, it offers efficient optimization capabilities, enabling rapid error reduction in rough localization estimations. Our method mitigates the dependence on complex neural network models while demonstrating improved robustness against noise and higher localization accuracy in challenging environments, as compared to neural network joint optimization strategies. The optimization framework proposed in this paper introduces novel approaches to visual localization by integrating the advantages of 3D reconstruction and heuristic refinement strategy, which demonstrates strong performance across multiple benchmark datasets, including 7Scenes and DB dataset.  
  </ol>  
</details>  
  
  



