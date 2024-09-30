<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Exploiting-Motion-Prior-for-Accurate-Pose-Estimation-of-Dashboard-Cameras>Exploiting Motion Prior for Accurate Pose Estimation of Dashboard Cameras</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Search-and-Detect:-Training-Free-Long-Tail-Object-Detection-via-Web-Image-Retrieval>Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Exploiting-Motion-Prior-for-Accurate-Pose-Estimation-of-Dashboard-Cameras>Exploiting Motion Prior for Accurate Pose Estimation of Dashboard Cameras</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Exploiting Motion Prior for Accurate Pose Estimation of Dashboard Cameras](http://arxiv.org/abs/2409.18673)  
Yipeng Lu, Yifan Zhao, Haiping Wang, Zhiwei Ruan, Yuan Liu, Zhen Dong, Bisheng Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Dashboard cameras (dashcams) record millions of driving videos daily, offering a valuable potential data source for various applications, including driving map production and updates. A necessary step for utilizing these dashcam data involves the estimation of camera poses. However, the low-quality images captured by dashcams, characterized by motion blurs and dynamic objects, pose challenges for existing image-matching methods in accurately estimating camera poses. In this study, we propose a precise pose estimation method for dashcam images, leveraging the inherent camera motion prior. Typically, image sequences captured by dash cameras exhibit pronounced motion prior, such as forward movement or lateral turns, which serve as essential cues for correspondence estimation. Building upon this observation, we devise a pose regression module aimed at learning camera motion prior, subsequently integrating these prior into both correspondences and pose estimation processes. The experiment shows that, in real dashcams dataset, our method is 22% better than the baseline for pose estimation in AUC5\textdegree, and it can estimate poses for 19% more images with less reprojection error in Structure from Motion (SfM).  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval](http://arxiv.org/abs/2409.18733)  
Mankeerat Sidhu, Hetarth Chopra, Ansel Blume, Jeonghwan Kim, Revanth Gangi Reddy, Heng Ji  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we introduce SearchDet, a training-free long-tail object detection framework that significantly enhances open-vocabulary object detection performance. SearchDet retrieves a set of positive and negative images of an object to ground, embeds these images, and computes an input image-weighted query which is used to detect the desired concept in the image. Our proposed method is simple and training-free, yet achieves over 48.7% mAP improvement on ODinW and 59.1% mAP improvement on LVIS compared to state-of-the-art models such as GroundingDINO. We further show that our approach of basing object detection on a set of Web-retrieved exemplars is stable with respect to variations in the exemplars, suggesting a path towards eliminating costly data annotation and training procedures.  
  </ol>  
</details>  
  
  



## Image Matching  

### [Exploiting Motion Prior for Accurate Pose Estimation of Dashboard Cameras](http://arxiv.org/abs/2409.18673)  
Yipeng Lu, Yifan Zhao, Haiping Wang, Zhiwei Ruan, Yuan Liu, Zhen Dong, Bisheng Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Dashboard cameras (dashcams) record millions of driving videos daily, offering a valuable potential data source for various applications, including driving map production and updates. A necessary step for utilizing these dashcam data involves the estimation of camera poses. However, the low-quality images captured by dashcams, characterized by motion blurs and dynamic objects, pose challenges for existing image-matching methods in accurately estimating camera poses. In this study, we propose a precise pose estimation method for dashcam images, leveraging the inherent camera motion prior. Typically, image sequences captured by dash cameras exhibit pronounced motion prior, such as forward movement or lateral turns, which serve as essential cues for correspondence estimation. Building upon this observation, we devise a pose regression module aimed at learning camera motion prior, subsequently integrating these prior into both correspondences and pose estimation processes. The experiment shows that, in real dashcams dataset, our method is 22% better than the baseline for pose estimation in AUC5\textdegree, and it can estimate poses for 19% more images with less reprojection error in Structure from Motion (SfM).  
  </ol>  
</details>  
  
  



