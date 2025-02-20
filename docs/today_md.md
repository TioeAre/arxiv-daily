<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Active-Illumination-for-Visual-Ego-Motion-Estimation-in-the-Dark>Active Illumination for Visual Ego-Motion Estimation in the Dark</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#3D-Gaussian-Splatting-aided-Localization-for-Large-and-Complex-Indoor-Environments>3D Gaussian Splatting aided Localization for Large and Complex Indoor-Environments</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#2.5D-U-Net-with-Depth-Reduction-for-3D-CryoET-Object-Identification>2.5D U-Net with Depth Reduction for 3D CryoET Object Identification</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Geometry-Aware-Diffusion-Models-for-Multiview-Scene-Inpainting>Geometry-Aware Diffusion Models for Multiview Scene Inpainting</a></li>
        <li><a href=#GS-QA:-Comprehensive-Quality-Assessment-Benchmark-for-Gaussian-Splatting-View-Synthesis>GS-QA: Comprehensive Quality Assessment Benchmark for Gaussian Splatting View Synthesis</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Active Illumination for Visual Ego-Motion Estimation in the Dark](http://arxiv.org/abs/2502.13708)  
Francesco Crocetti, Alberto Dionigi, Raffaele Brilli, Gabriele Costante, Paolo Valigi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Odometry (VO) and Visual SLAM (V-SLAM) systems often struggle in low-light and dark environments due to the lack of robust visual features. In this paper, we propose a novel active illumination framework to enhance the performance of VO and V-SLAM algorithms in these challenging conditions. The developed approach dynamically controls a moving light source to illuminate highly textured areas, thereby improving feature extraction and tracking. Specifically, a detector block, which incorporates a deep learning-based enhancing network, identifies regions with relevant features. Then, a pan-tilt controller is responsible for guiding the light beam toward these areas, so that to provide information-rich images to the ego-motion estimation algorithm. Experimental results on a real robotic platform demonstrate the effectiveness of the proposed method, showing a reduction in the pose estimation error up to 75% with respect to a traditional fixed lighting technique.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [3D Gaussian Splatting aided Localization for Large and Complex Indoor-Environments](http://arxiv.org/abs/2502.13803)  
Vincent Ress, Jonas Meyer, Wei Zhang, David Skuddis, Uwe Soergel, Norbert Haala  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The field of visual localization has been researched for several decades and has meanwhile found many practical applications. Despite the strong progress in this field, there are still challenging situations in which established methods fail. We present an approach to significantly improve the accuracy and reliability of established visual localization methods by adding rendered images. In detail, we first use a modern visual SLAM approach that provides a 3D Gaussian Splatting (3DGS) based map to create reference data. We demonstrate that enriching reference data with images rendered from 3DGS at randomly sampled poses significantly improves the performance of both geometry-based visual localization and Scene Coordinate Regression (SCR) methods. Through comprehensive evaluation in a large industrial environment, we analyze the performance impact of incorporating these additional rendered views.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [2.5D U-Net with Depth Reduction for 3D CryoET Object Identification](http://arxiv.org/abs/2502.13484)  
Yusuke Uchida, Takaaki Fukui  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Cryo-electron tomography (cryoET) is a crucial technique for unveiling the structure of protein complexes. Automatically analyzing tomograms captured by cryoET is an essential step toward understanding cellular structures. In this paper, we introduce the 4th place solution from the CZII - CryoET Object Identification competition, which was organized to advance the development of automated tomogram analysis techniques. Our solution adopted a heatmap-based keypoint detection approach, utilizing an ensemble of two different types of 2.5D U-Net models with depth reduction. Despite its highly unified and simple architecture, our method achieved 4th place, demonstrating its effectiveness.  
  </ol>  
</details>  
  
  



## NeRF  

### [Geometry-Aware Diffusion Models for Multiview Scene Inpainting](http://arxiv.org/abs/2502.13335)  
Ahmad Salimi, Tristan Aumentado-Armstrong, Marcus A. Brubaker, Konstantinos G. Derpanis  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we focus on 3D scene inpainting, where parts of an input image set, captured from different viewpoints, are masked out. The main challenge lies in generating plausible image completions that are geometrically consistent across views. Most recent work addresses this challenge by combining generative models with a 3D radiance field to fuse information across viewpoints. However, a major drawback of these methods is that they often produce blurry images due to the fusion of inconsistent cross-view images. To avoid blurry inpaintings, we eschew the use of an explicit or implicit radiance field altogether and instead fuse cross-view information in a learned space. In particular, we introduce a geometry-aware conditional generative model, capable of inpainting multi-view consistent images based on both geometric and appearance cues from reference images. A key advantage of our approach over existing methods is its unique ability to inpaint masked scenes with a limited number of views (i.e., few-view inpainting), whereas previous methods require relatively large image sets for their 3D model fitting step. Empirically, we evaluate and compare our scene-centric inpainting method on two datasets, SPIn-NeRF and NeRFiller, which contain images captured at narrow and wide baselines, respectively, and achieve state-of-the-art 3D inpainting performance on both. Additionally, we demonstrate the efficacy of our approach in the few-view setting compared to prior methods.  
  </ol>  
</details>  
**comments**: Our project page is available at https://geomvi.github.io  
  
### [GS-QA: Comprehensive Quality Assessment Benchmark for Gaussian Splatting View Synthesis](http://arxiv.org/abs/2502.13196)  
Pedro Martin, António Rodrigues, João Ascenso, Maria Paula Queluz  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Gaussian Splatting (GS) offers a promising alternative to Neural Radiance Fields (NeRF) for real-time 3D scene rendering. Using a set of 3D Gaussians to represent complex geometry and appearance, GS achieves faster rendering times and reduced memory consumption compared to the neural network approach used in NeRF. However, quality assessment of GS-generated static content is not yet explored in-depth. This paper describes a subjective quality assessment study that aims to evaluate synthesized videos obtained with several static GS state-of-the-art methods. The methods were applied to diverse visual scenes, covering both 360-degree and forward-facing (FF) camera trajectories. Moreover, the performance of 18 objective quality metrics was analyzed using the scores resulting from the subjective study, providing insights into their strengths, limitations, and alignment with human perception. All videos and scores are made available providing a comprehensive database that can be used as benchmark on GS view synthesis and objective quality metrics.  
  </ol>  
</details>  
  
  



