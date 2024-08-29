<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#ES-PTAM:-Event-based-Stereo-Parallel-Tracking-and-Mapping>ES-PTAM: Event-based Stereo Parallel Tracking and Mapping</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Temporal-Attention-for-Cross-View-Sequential-Image-Localization>Temporal Attention for Cross-View Sequential Image Localization</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Towards-Realistic-Example-based-Modeling-via-3D-Gaussian-Stitching>Towards Realistic Example-based Modeling via 3D Gaussian Stitching</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [ES-PTAM: Event-based Stereo Parallel Tracking and Mapping](http://arxiv.org/abs/2408.15605)  
Suman Ghosh, Valentina Cavinato, Guillermo Gallego  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Odometry (VO) and SLAM are fundamental components for spatial perception in mobile robots. Despite enormous progress in the field, current VO/SLAM systems are limited by their sensors' capability. Event cameras are novel visual sensors that offer advantages to overcome the limitations of standard cameras, enabling robots to expand their operating range to challenging scenarios, such as high-speed motion and high dynamic range illumination. We propose a novel event-based stereo VO system by combining two ideas: a correspondence-free mapping module that estimates depth by maximizing ray density fusion and a tracking module that estimates camera poses by maximizing edge-map alignment. We evaluate the system comprehensively on five real-world datasets, spanning a variety of camera types (manufacturers and spatial resolutions) and scenarios (driving, flying drone, hand-held, egocentric, etc). The quantitative and qualitative results demonstrate that our method outperforms the state of the art in majority of the test sequences by a margin, e.g., trajectory error reduction of 45% on RPG dataset, 61% on DSEC dataset, and 21% on TUM-VIE dataset. To benefit the community and foster research on event-based perception systems, we release the source code and results: https://github.com/tub-rip/ES-PTAM  
  </ol>  
</details>  
**comments**: 17 pages, 7 figures, 4 tables, https://github.com/tub-rip/ES-PTAM  
  
  



## Visual Localization  

### [Temporal Attention for Cross-View Sequential Image Localization](http://arxiv.org/abs/2408.15569)  
[[code](https://github.com/UQ-DongYuan/CVSeqLocation)]  
Dong Yuan, Frederic Maire, Feras Dayoub  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper introduces a novel approach to enhancing cross-view localization, focusing on the fine-grained, sequential localization of street-view images within a single known satellite image patch, a significant departure from traditional one-to-one image retrieval methods. By expanding to sequential image fine-grained localization, our model, equipped with a novel Temporal Attention Module (TAM), leverages contextual information to significantly improve sequential image localization accuracy. Our method shows substantial reductions in both mean and median localization errors on the Cross-View Image Sequence (CVIS) dataset, outperforming current state-of-the-art single-image localization techniques. Additionally, by adapting the KITTI-CVL dataset into sequential image sets, we not only offer a more realistic dataset for future research but also demonstrate our model's robust generalization capabilities across varying times and areas, evidenced by a 75.3% reduction in mean distance error in cross-view sequential image localization.  
  </ol>  
</details>  
**comments**: Accepted to IROS 2024  
  
  



## NeRF  

### [Towards Realistic Example-based Modeling via 3D Gaussian Stitching](http://arxiv.org/abs/2408.15708)  
Xinyu Gao, Ziyi Yang, Bingchen Gong, Xiaoguang Han, Sipeng Yang, Xiaogang Jin  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Using parts of existing models to rebuild new models, commonly termed as example-based modeling, is a classical methodology in the realm of computer graphics. Previous works mostly focus on shape composition, making them very hard to use for realistic composition of 3D objects captured from real-world scenes. This leads to combining multiple NeRFs into a single 3D scene to achieve seamless appearance blending. However, the current SeamlessNeRF method struggles to achieve interactive editing and harmonious stitching for real-world scenes due to its gradient-based strategy and grid-based representation. To this end, we present an example-based modeling method that combines multiple Gaussian fields in a point-based representation using sample-guided synthesis. Specifically, as for composition, we create a GUI to segment and transform multiple fields in real time, easily obtaining a semantically meaningful composition of models represented by 3D Gaussian Splatting (3DGS). For texture blending, due to the discrete and irregular nature of 3DGS, straightforwardly applying gradient propagation as SeamlssNeRF is not supported. Thus, a novel sampling-based cloning method is proposed to harmonize the blending while preserving the original rich texture and content. Our workflow consists of three steps: 1) real-time segmentation and transformation of a Gaussian model using a well-tailored GUI, 2) KNN analysis to identify boundary points in the intersecting area between the source and target models, and 3) two-phase optimization of the target model using sampling-based cloning and gradient constraints. Extensive experimental results validate that our approach significantly outperforms previous works in terms of realistic synthesis, demonstrating its practicality. More demos are available at https://ingra14m.github.io/gs_stitching_website.  
  </ol>  
</details>  
  
  



