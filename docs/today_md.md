<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#A-Miniature-Vision-Based-Localization-System-for-Indoor-Blimps>A Miniature Vision-Based Localization System for Indoor Blimps</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#A-Miniature-Vision-Based-Localization-System-for-Indoor-Blimps>A Miniature Vision-Based Localization System for Indoor Blimps</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Potamoi:-Accelerating-Neural-Rendering-via-a-Unified-Streaming-Architecture>Potamoi: Accelerating Neural Rendering via a Unified Streaming Architecture</a></li>
        <li><a href=#ActiveNeRF:-Learning-Accurate-3D-Geometry-by-Active-Pattern-Projection>ActiveNeRF: Learning Accurate 3D Geometry by Active Pattern Projection</a></li>
        <li><a href=#HDRGS:-High-Dynamic-Range-Gaussian-Splatting>HDRGS: High Dynamic Range Gaussian Splatting</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [A Miniature Vision-Based Localization System for Indoor Blimps](http://arxiv.org/abs/2408.06648)  
Shicong Ma  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    With increasing attention paid to blimp research, I hope to build an indoor blimp to interact with humans. To begin with, I propose developing a visual localization system to enable blimps to localize themselves in an indoor environment autonomously. This system initially reconstructs an indoor environment by employing Structure from Motion with Superpoint visual features. Next, with the previously built sparse point cloud map, the system generates camera poses by continuously employing pose estimation on matched visual features observed from the map. In this project, the blimp only serves as a reference mobile platform that constrains the weight of the perception system. The perception system contains one monocular camera and a WiFi adaptor to capture and transmit visual data to a ground PC station where the algorithms will be executed. The success of this project will transform remote-controlled indoor blimps into autonomous indoor blimps, which can be utilized for applications such as surveillance, advertisement, and indoor mapping.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [A Miniature Vision-Based Localization System for Indoor Blimps](http://arxiv.org/abs/2408.06648)  
Shicong Ma  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    With increasing attention paid to blimp research, I hope to build an indoor blimp to interact with humans. To begin with, I propose developing a visual localization system to enable blimps to localize themselves in an indoor environment autonomously. This system initially reconstructs an indoor environment by employing Structure from Motion with Superpoint visual features. Next, with the previously built sparse point cloud map, the system generates camera poses by continuously employing pose estimation on matched visual features observed from the map. In this project, the blimp only serves as a reference mobile platform that constrains the weight of the perception system. The perception system contains one monocular camera and a WiFi adaptor to capture and transmit visual data to a ground PC station where the algorithms will be executed. The success of this project will transform remote-controlled indoor blimps into autonomous indoor blimps, which can be utilized for applications such as surveillance, advertisement, and indoor mapping.  
  </ol>  
</details>  
  
  



## NeRF  

### [Potamoi: Accelerating Neural Rendering via a Unified Streaming Architecture](http://arxiv.org/abs/2408.06608)  
Yu Feng, Weikai Lin, Zihan Liu, Jingwen Leng, Minyi Guo, Han Zhao, Xiaofeng Hou, Jieru Zhao, Yuhao Zhu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Field (NeRF) has emerged as a promising alternative for photorealistic rendering. Despite recent algorithmic advancements, achieving real-time performance on today's resource-constrained devices remains challenging. In this paper, we identify the primary bottlenecks in current NeRF algorithms and introduce a unified algorithm-architecture co-design, Potamoi, designed to accommodate various NeRF algorithms. Specifically, we introduce a runtime system featuring a plug-and-play algorithm, SpaRW, which significantly reduces the per-frame computational workload and alleviates compute inefficiencies. Furthermore, our unified streaming pipeline coupled with customized hardware support effectively tames both SRAM and DRAM inefficiencies by minimizing repetitive DRAM access and completely eliminating SRAM bank conflicts. When evaluated against a baseline utilizing a dedicated DNN accelerator, our framework demonstrates a speed-up and energy reduction of 53.1 $\times$ and 67.7$\times$ , respectively, all while maintaining high visual quality with less than a 1.0 dB reduction in peak signal-to-noise ratio.  
  </ol>  
</details>  
**comments**: arXiv admin note: substantial text overlap with arXiv:2404.11852  
  
### [ActiveNeRF: Learning Accurate 3D Geometry by Active Pattern Projection](http://arxiv.org/abs/2408.06592)  
[[code](https://github.com/hcp16/active_nerf)]  
Jianyu Tao, Changping Hu, Edward Yang, Jing Xu, Rui Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    NeRFs have achieved incredible success in novel view synthesis. However, the accuracy of the implicit geometry is unsatisfactory because the passive static environmental illumination has low spatial frequency and cannot provide enough information for accurate geometry reconstruction. In this work, we propose ActiveNeRF, a 3D geometry reconstruction framework, which improves the geometry quality of NeRF by actively projecting patterns of high spatial frequency onto the scene using a projector which has a constant relative pose to the camera. We design a learnable active pattern rendering pipeline which jointly learns the scene geometry and the active pattern. We find that, by adding the active pattern and imposing its consistency across different views, our proposed method outperforms state of the art geometry reconstruction methods qualitatively and quantitatively in both simulation and real experiments. Code is avaliable at https://github.com/hcp16/active_nerf  
  </ol>  
</details>  
**comments**: 18 pages, 10 figures  
  
### [HDRGS: High Dynamic Range Gaussian Splatting](http://arxiv.org/abs/2408.06543)  
[[code](https://github.com/wujh2001/hdrgs)]  
Jiahao Wu, Lu Xiao, Chao Wang, Rui Peng, Kaiqiang Xiong, Ronggang Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent years have witnessed substantial advancements in the field of 3D reconstruction from 2D images, particularly following the introduction of the neural radiance field (NeRF) technique. However, reconstructing a 3D high dynamic range (HDR) radiance field, which aligns more closely with real-world conditions, from 2D multi-exposure low dynamic range (LDR) images continues to pose significant challenges. Approaches to this issue fall into two categories: grid-based and implicit-based. Implicit methods, using multi-layer perceptrons (MLP), face inefficiencies, limited solvability, and overfitting risks. Conversely, grid-based methods require significant memory and struggle with image quality and long training times. In this paper, we introduce Gaussian Splatting-a recent, high-quality, real-time 3D reconstruction technique-into this domain. We further develop the High Dynamic Range Gaussian Splatting (HDR-GS) method, designed to address the aforementioned challenges. This method enhances color dimensionality by including luminance and uses an asymmetric grid for tone-mapping, swiftly and precisely converting pixel irradiance to color. Our approach improves HDR scene recovery accuracy and integrates a novel coarse-to-fine strategy to speed up model convergence, enhancing robustness against sparse viewpoints and exposure extremes, and preventing local optima. Extensive testing confirms that our method surpasses current state-of-the-art techniques in both synthetic and real-world scenarios. Code will be released at \url{https://github.com/WuJH2001/HDRGS}  
  </ol>  
</details>  
  
  



