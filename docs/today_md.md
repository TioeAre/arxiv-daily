<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#LoGS:-Visual-Localization-via-Gaussian-Splatting-with-Fewer-Training-Images>LoGS: Visual Localization via Gaussian Splatting with Fewer Training Images</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#LoGS:-Visual-Localization-via-Gaussian-Splatting-with-Fewer-Training-Images>LoGS: Visual Localization via Gaussian Splatting with Fewer Training Images</a></li>
        <li><a href=#Multiview-Scene-Graph>Multiview Scene Graph</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#LoGS:-Visual-Localization-via-Gaussian-Splatting-with-Fewer-Training-Images>LoGS: Visual Localization via Gaussian Splatting with Fewer Training Images</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [LoGS: Visual Localization via Gaussian Splatting with Fewer Training Images](http://arxiv.org/abs/2410.11505)  
Yuzhou Cheng, Jianhao Jiao, Yue Wang, Dimitrios Kanoulas  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual localization involves estimating a query image's 6-DoF (degrees of freedom) camera pose, which is a fundamental component in various computer vision and robotic tasks. This paper presents LoGS, a vision-based localization pipeline utilizing the 3D Gaussian Splatting (GS) technique as scene representation. This novel representation allows high-quality novel view synthesis. During the mapping phase, structure-from-motion (SfM) is applied first, followed by the generation of a GS map. During localization, the initial position is obtained through image retrieval, local feature matching coupled with a PnP solver, and then a high-precision pose is achieved through the analysis-by-synthesis manner on the GS map. Experimental results on four large-scale datasets demonstrate the proposed approach's SoTA accuracy in estimating camera poses and robustness under challenging few-shot conditions.  
  </ol>  
</details>  
**comments**: 8 pages  
  
  



## Visual Localization  

### [LoGS: Visual Localization via Gaussian Splatting with Fewer Training Images](http://arxiv.org/abs/2410.11505)  
Yuzhou Cheng, Jianhao Jiao, Yue Wang, Dimitrios Kanoulas  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual localization involves estimating a query image's 6-DoF (degrees of freedom) camera pose, which is a fundamental component in various computer vision and robotic tasks. This paper presents LoGS, a vision-based localization pipeline utilizing the 3D Gaussian Splatting (GS) technique as scene representation. This novel representation allows high-quality novel view synthesis. During the mapping phase, structure-from-motion (SfM) is applied first, followed by the generation of a GS map. During localization, the initial position is obtained through image retrieval, local feature matching coupled with a PnP solver, and then a high-precision pose is achieved through the analysis-by-synthesis manner on the GS map. Experimental results on four large-scale datasets demonstrate the proposed approach's SoTA accuracy in estimating camera poses and robustness under challenging few-shot conditions.  
  </ol>  
</details>  
**comments**: 8 pages  
  
### [Multiview Scene Graph](http://arxiv.org/abs/2410.11187)  
Juexiao Zhang, Gao Zhu, Sihang Li, Xinhao Liu, Haorui Song, Xinran Tang, Chen Feng  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    A proper scene representation is central to the pursuit of spatial intelligence where agents can robustly reconstruct and efficiently understand 3D scenes. A scene representation is either metric, such as landmark maps in 3D reconstruction, 3D bounding boxes in object detection, or voxel grids in occupancy prediction, or topological, such as pose graphs with loop closures in SLAM or visibility graphs in SfM. In this work, we propose to build Multiview Scene Graphs (MSG) from unposed images, representing a scene topologically with interconnected place and object nodes. The task of building MSG is challenging for existing representation learning methods since it needs to jointly address both visual place recognition, object detection, and object association from images with limited fields of view and potentially large viewpoint changes. To evaluate any method tackling this task, we developed an MSG dataset and annotation based on a public 3D dataset. We also propose an evaluation metric based on the intersection-over-union score of MSG edges. Moreover, we develop a novel baseline method built on mainstream pretrained vision models, combining visual place recognition and object association into one Transformer decoder architecture. Experiments demonstrate our method has superior performance compared to existing relevant baselines.  
  </ol>  
</details>  
**comments**: To be published in NeurIPS 2024. Website at
  https://ai4ce.github.io/MSG/  
  
  



## Image Matching  

### [LoGS: Visual Localization via Gaussian Splatting with Fewer Training Images](http://arxiv.org/abs/2410.11505)  
Yuzhou Cheng, Jianhao Jiao, Yue Wang, Dimitrios Kanoulas  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual localization involves estimating a query image's 6-DoF (degrees of freedom) camera pose, which is a fundamental component in various computer vision and robotic tasks. This paper presents LoGS, a vision-based localization pipeline utilizing the 3D Gaussian Splatting (GS) technique as scene representation. This novel representation allows high-quality novel view synthesis. During the mapping phase, structure-from-motion (SfM) is applied first, followed by the generation of a GS map. During localization, the initial position is obtained through image retrieval, local feature matching coupled with a PnP solver, and then a high-precision pose is achieved through the analysis-by-synthesis manner on the GS map. Experimental results on four large-scale datasets demonstrate the proposed approach's SoTA accuracy in estimating camera poses and robustness under challenging few-shot conditions.  
  </ol>  
</details>  
**comments**: 8 pages  
  
  



