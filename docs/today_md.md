<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#From-2D-to-3D:-AISG-SLA-Visual-Localization-Challenge>From 2D to 3D: AISG-SLA Visual Localization Challenge</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#SHIC:-Shape-Image-Correspondences-with-no-Keypoint-Supervision>SHIC: Shape-Image Correspondences with no Keypoint Supervision</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#PIV3CAMS:-a-multi-camera-dataset-for-multiple-computer-vision-problems-and-its-application-to-novel-view-point-synthesis>PIV3CAMS: a multi-camera dataset for multiple computer vision problems and its application to novel view-point synthesis</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#IOVS4NeRF:Incremental-Optimal-View-Selection-for-Large-Scale-NeRFs>IOVS4NeRF:Incremental Optimal View Selection for Large-Scale NeRFs</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [From 2D to 3D: AISG-SLA Visual Localization Challenge](http://arxiv.org/abs/2407.18590)  
Jialin Gao, Bill Ong, Darld Lwi, Zhen Hao Ng, Xun Wei Yee, Mun-Thye Mak, Wee Siong Ng, See-Kiong Ng, Hui Ying Teo, Victor Khoo, Georg Bökman, Johan Edstedt, Kirill Brodt, Clémentin Boittiaux, Maxime Ferrera, Stepan Konev  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Research in 3D mapping is crucial for smart city applications, yet the cost of acquiring 3D data often hinders progress. Visual localization, particularly monocular camera position estimation, offers a solution by determining the camera's pose solely through visual cues. However, this task is challenging due to limited data from a single camera. To tackle these challenges, we organized the AISG-SLA Visual Localization Challenge (VLC) at IJCAI 2023 to explore how AI can accurately extract camera pose data from 2D images in 3D space. The challenge attracted over 300 participants worldwide, forming 50+ teams. Winning teams achieved high accuracy in pose estimation using images from a car-mounted camera with low frame rates. The VLC dataset is available for research purposes upon request via vlc-dataset@aisingapore.org.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [SHIC: Shape-Image Correspondences with no Keypoint Supervision](http://arxiv.org/abs/2407.18907)  
Aleksandar Shtedritski, Christian Rupprecht, Andrea Vedaldi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Canonical surface mapping generalizes keypoint detection by assigning each pixel of an object to a corresponding point in a 3D template. Popularised by DensePose for the analysis of humans, authors have since attempted to apply the concept to more categories, but with limited success due to the high cost of manual supervision. In this work, we introduce SHIC, a method to learn canonical maps without manual supervision which achieves better results than supervised methods for most categories. Our idea is to leverage foundation computer vision models such as DINO and Stable Diffusion that are open-ended and thus possess excellent priors over natural categories. SHIC reduces the problem of estimating image-to-template correspondences to predicting image-to-image correspondences using features from the foundation models. The reduction works by matching images of the object to non-photorealistic renders of the template, which emulates the process of collecting manual annotations for this task. These correspondences are then used to supervise high-quality canonical maps for any object of interest. We also show that image generators can further improve the realism of the template views, which provide an additional source of supervision for the model.  
  </ol>  
</details>  
**comments**: ECCV 2024. Project website
  https://www.robots.ox.ac.uk/~vgg/research/shic/  
  
  



## Image Matching  

### [PIV3CAMS: a multi-camera dataset for multiple computer vision problems and its application to novel view-point synthesis](http://arxiv.org/abs/2407.18695)  
Sohyeong Kim, Martin Danelljan, Radu Timofte, Luc Van Gool, Jean-Philippe Thiran  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The modern approaches for computer vision tasks significantly rely on machine learning, which requires a large number of quality images. While there is a plethora of image datasets with a single type of images, there is a lack of datasets collected from multiple cameras. In this thesis, we introduce Paired Image and Video data from three CAMeraS, namely PIV3CAMS, aimed at multiple computer vision tasks. The PIV3CAMS dataset consists of 8385 pairs of images and 82 pairs of videos taken from three different cameras: Canon D5 Mark IV, Huawei P20, and ZED stereo camera. The dataset includes various indoor and outdoor scenes from different locations in Zurich (Switzerland) and Cheonan (South Korea). Some of the computer vision applications that can benefit from the PIV3CAMS dataset are image/video enhancement, view interpolation, image matching, and much more. We provide a careful explanation of the data collection process and detailed analysis of the data. The second part of this thesis studies the usage of depth information in the view synthesizing task. In addition to the regeneration of a current state-of-the-art algorithm, we investigate several proposed alternative models that integrate depth information geometrically. Through extensive experiments, we show that the effect of depth is crucial in small view changes. Finally, we apply our model to the introduced PIV3CAMS dataset to synthesize novel target views as an example application of PIV3CAMS.  
  </ol>  
</details>  
  
  



## NeRF  

### [IOVS4NeRF:Incremental Optimal View Selection for Large-Scale NeRFs](http://arxiv.org/abs/2407.18611)  
Jingpeng Xie, Shiyu Tan, Yuanlei Wang, Yizhen Lao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Urban-level three-dimensional reconstruction for modern applications demands high rendering fidelity while minimizing computational costs. The advent of Neural Radiance Fields (NeRF) has enhanced 3D reconstruction, yet it exhibits artifacts under multiple viewpoints. In this paper, we propose a new NeRF framework method to address these issues. Our method uses image content and pose data to iteratively plan the next best view. A crucial aspect of this method involves uncertainty estimation, guiding the selection of views with maximum information gain from a candidate set. This iterative process enhances rendering quality over time. Simultaneously, we introduce the Vonoroi diagram and threshold sampling together with flight classifier to boost the efficiency, while keep the original NeRF network intact. It can serve as a plug-in tool to assist in better rendering, outperforming baselines and similar prior works.  
  </ol>  
</details>  
  
  



