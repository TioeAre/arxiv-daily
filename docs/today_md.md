<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#VDG:-Vision-Only-Dynamic-Gaussian-for-Driving-Simulation>VDG: Vision-Only Dynamic Gaussian for Driving Simulation</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#View-Invariant-Pixelwise-Anomaly-Detection-in-Multi-object-Scenes-with-Adaptive-View-Synthesis>View-Invariant Pixelwise Anomaly Detection in Multi-object Scenes with Adaptive View Synthesis</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [VDG: Vision-Only Dynamic Gaussian for Driving Simulation](http://arxiv.org/abs/2406.18198)  
Hao Li, Jingfeng Li, Dingwen Zhang, Chenming Wu, Jieqi Shi, Chen Zhao, Haocheng Feng, Errui Ding, Jingdong Wang, Junwei Han  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Dynamic Gaussian splatting has led to impressive scene reconstruction and image synthesis advances in novel views. Existing methods, however, heavily rely on pre-computed poses and Gaussian initialization by Structure from Motion (SfM) algorithms or expensive sensors. For the first time, this paper addresses this issue by integrating self-supervised VO into our pose-free dynamic Gaussian method (VDG) to boost pose and depth initialization and static-dynamic decomposition. Moreover, VDG can work with only RGB image input and construct dynamic scenes at a faster speed and larger scenes compared with the pose-free dynamic view-synthesis method. We demonstrate the robustness of our approach via extensive quantitative and qualitative experiments. Our results show favorable performance over the state-of-the-art dynamic view synthesis methods. Additional video and source code will be posted on our project page at https://3d-aigc.github.io/VDG.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [View-Invariant Pixelwise Anomaly Detection in Multi-object Scenes with Adaptive View Synthesis](http://arxiv.org/abs/2406.18012)  
Subin Varghese, Vedhus Hoskere  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The inspection and monitoring of infrastructure assets typically requires identifying visual anomalies in scenes periodically photographed over time. Images collected manually or with robots such as unmanned aerial vehicles from the same scene at different instances in time are typically not perfectly aligned. Supervised segmentation methods can be applied to identify known problems, but unsupervised anomaly detection approaches are required when unknown anomalies occur. Current unsupervised pixel-level anomaly detection methods have mainly been developed for industrial settings where the camera position is known and constant. However, we find that these methods fail to generalize to the case when images are not perfectly aligned. We term the problem of unsupervised anomaly detection between two such imperfectly aligned sets of images as Scene Anomaly Detection (Scene AD). We present a novel network termed OmniAD to address the Scene AD problem posed. Specifically, we refine the anomaly detection method reverse distillation to achieve a 40% increase in pixel-level anomaly detection performance. The network's performance is further demonstrated to improve with two new data augmentation strategies proposed that leverage novel view synthesis and camera localization to improve generalization. We validate our approach with qualitative and quantitative results on a new dataset, ToyCity, the first Scene AD dataset with multiple objects, as well as on the established single object-centric dataset, MAD. https://drags99.github.io/OmniAD/  
  </ol>  
</details>  
  
  



