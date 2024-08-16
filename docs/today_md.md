<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#CorrAdaptor:-Adaptive-Local-Context-Learning-for-Correspondence-Pruning>CorrAdaptor: Adaptive Local Context Learning for Correspondence Pruning</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#DM2RM:-Dual-Mode-Multimodal-Ranking-for-Target-Objects-and-Receptacles-Based-on-Open-Vocabulary-Instructions>DM2RM: Dual-Mode Multimodal Ranking for Target Objects and Receptacles Based on Open-Vocabulary Instructions</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Towards-Practical-Human-Motion-Prediction-with-LiDAR-Point-Clouds>Towards Practical Human Motion Prediction with LiDAR Point Clouds</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#WaterSplatting:-Fast-Underwater-3D-Scene-Reconstruction-Using-Gaussian-Splatting>WaterSplatting: Fast Underwater 3D Scene Reconstruction Using Gaussian Splatting</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [CorrAdaptor: Adaptive Local Context Learning for Correspondence Pruning](http://arxiv.org/abs/2408.08134)  
Wei Zhu, Yicheng Liu, Yuping He, Tangfei Liao, Kang Zheng, Xiaoqiu Xu, Tao Wang, Tong Lu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In the fields of computer vision and robotics, accurate pixel-level correspondences are essential for enabling advanced tasks such as structure-from-motion and simultaneous localization and mapping. Recent correspondence pruning methods usually focus on learning local consistency through k-nearest neighbors, which makes it difficult to capture robust context for each correspondence. We propose CorrAdaptor, a novel architecture that introduces a dual-branch structure capable of adaptively adjusting local contexts through both explicit and implicit local graph learning. Specifically, the explicit branch uses KNN-based graphs tailored for initial neighborhood identification, while the implicit branch leverages a learnable matrix to softly assign neighbors and adaptively expand the local context scope, significantly enhancing the model's robustness and adaptability to complex image variations. Moreover, we design a motion injection module to integrate motion consistency into the network to suppress the impact of outliers and refine local context learning, resulting in substantial performance improvements. The experimental results on extensive correspondence-based tasks indicate that our CorrAdaptor achieves state-of-the-art performance both qualitatively and quantitatively. The code and pre-trained models are available at https://github.com/TaoWangzj/CorrAdaptor.  
  </ol>  
</details>  
**comments**: 8 pages, 4 figures, accepted by ECAI  
  
  



## Visual Localization  

### [DM2RM: Dual-Mode Multimodal Ranking for Target Objects and Receptacles Based on Open-Vocabulary Instructions](http://arxiv.org/abs/2408.07910)  
Ryosuke Korekata, Kanta Kaneda, Shunya Nagashima, Yuto Imai, Komei Sugiura  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this study, we aim to develop a domestic service robot (DSR) that, guided by open-vocabulary instructions, can carry everyday objects to the specified pieces of furniture. Few existing methods handle mobile manipulation tasks with open-vocabulary instructions in the image retrieval setting, and most do not identify both the target objects and the receptacles. We propose the Dual-Mode Multimodal Ranking model (DM2RM), which enables images of both the target objects and receptacles to be retrieved using a single model based on multimodal foundation models. We introduce a switching mechanism that leverages a mode token and phrase identification via a large language model to switch the embedding space based on the prediction target. To evaluate the DM2RM, we construct a novel dataset including real-world images collected from hundreds of building-scale environments and crowd-sourced instructions with referring expressions. The evaluation results show that the proposed DM2RM outperforms previous approaches in terms of standard metrics in image retrieval settings. Furthermore, we demonstrate the application of the DM2RM on a standardized real-world DSR platform including fetch-and-carry actions, where it achieves a task success rate of 82% despite the zero-shot transfer setting. Demonstration videos, code, and more materials are available at https://kkrr10.github.io/dm2rm/.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Towards Practical Human Motion Prediction with LiDAR Point Clouds](http://arxiv.org/abs/2408.08202)  
Xiao Han, Yiming Ren, Yichen Yao, Yujing Sun, Yuexin Ma  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Human motion prediction is crucial for human-centric multimedia understanding and interacting. Current methods typically rely on ground truth human poses as observed input, which is not practical for real-world scenarios where only raw visual sensor data is available. To implement these methods in practice, a pre-phrase of pose estimation is essential. However, such two-stage approaches often lead to performance degradation due to the accumulation of errors. Moreover, reducing raw visual data to sparse keypoint representations significantly diminishes the density of information, resulting in the loss of fine-grained features. In this paper, we propose \textit{LiDAR-HMP}, the first single-LiDAR-based 3D human motion prediction approach, which receives the raw LiDAR point cloud as input and forecasts future 3D human poses directly. Building upon our novel structure-aware body feature descriptor, LiDAR-HMP adaptively maps the observed motion manifold to future poses and effectively models the spatial-temporal correlations of human motions for further refinement of prediction results. Extensive experiments show that our method achieves state-of-the-art performance on two public benchmarks and demonstrates remarkable robustness and efficacy in real-world deployments.  
  </ol>  
</details>  
  
  



## NeRF  

### [WaterSplatting: Fast Underwater 3D Scene Reconstruction Using Gaussian Splatting](http://arxiv.org/abs/2408.08206)  
Huapeng Li, Wenxuan Song, Tianao Xu, Alexandre Elsig, Jonas Kulhanek  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The underwater 3D scene reconstruction is a challenging, yet interesting problem with applications ranging from naval robots to VR experiences. The problem was successfully tackled by fully volumetric NeRF-based methods which can model both the geometry and the medium (water). Unfortunately, these methods are slow to train and do not offer real-time rendering. More recently, 3D Gaussian Splatting (3DGS) method offered a fast alternative to NeRFs. However, because it is an explicit method that renders only the geometry, it cannot render the medium and is therefore unsuited for underwater reconstruction. Therefore, we propose a novel approach that fuses volumetric rendering with 3DGS to handle underwater data effectively. Our method employs 3DGS for explicit geometry representation and a separate volumetric field (queried once per pixel) for capturing the scattering medium. This dual representation further allows the restoration of the scenes by removing the scattering medium. Our method outperforms state-of-the-art NeRF-based methods in rendering quality on the underwater SeaThru-NeRF dataset. Furthermore, it does so while offering real-time rendering performance, addressing the efficiency limitations of existing methods. Web: https://water-splatting.github.io  
  </ol>  
</details>  
**comments**: Web: https://water-splatting.github.io  
  
  



