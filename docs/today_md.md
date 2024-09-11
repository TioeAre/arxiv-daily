<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#GeoCalib:-Learning-Single-image-Calibration-with-Geometric-Optimization>GeoCalib: Learning Single-image Calibration with Geometric Optimization</a></li>
        <li><a href=#Weakly-supervised-Camera-Localization-by-Ground-to-satellite-Image-Registration>Weakly-supervised Camera Localization by Ground-to-satellite Image Registration</a></li>
        <li><a href=#A-Cross-Font-Image-Retrieval-Network-for-Recognizing-Undeciphered-Oracle-Bone-Inscriptions>A Cross-Font Image Retrieval Network for Recognizing Undeciphered Oracle Bone Inscriptions</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Weakly-supervised-Camera-Localization-by-Ground-to-satellite-Image-Registration>Weakly-supervised Camera Localization by Ground-to-satellite Image Registration</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#LEIA:-Latent-View-invariant-Embeddings-for-Implicit-3D-Articulation>LEIA: Latent View-invariant Embeddings for Implicit 3D Articulation</a></li>
        <li><a href=#Sources-of-Uncertainty-in-3D-Scene-Reconstruction>Sources of Uncertainty in 3D Scene Reconstruction</a></li>
        <li><a href=#LSE-NeRF:-Learning-Sensor-Modeling-Errors-for-Deblured-Neural-Radiance-Fields-with-RGB-Event-Stereo>LSE-NeRF: Learning Sensor Modeling Errors for Deblured Neural Radiance Fields with RGB-Event Stereo</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [GeoCalib: Learning Single-image Calibration with Geometric Optimization](http://arxiv.org/abs/2409.06704)  
[[code](https://github.com/cvg/geocalib)]  
Alexander Veicht, Paul-Edouard Sarlin, Philipp Lindenberger, Marc Pollefeys  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    From a single image, visual cues can help deduce intrinsic and extrinsic camera parameters like the focal length and the gravity direction. This single-image calibration can benefit various downstream applications like image editing and 3D mapping. Current approaches to this problem are based on either classical geometry with lines and vanishing points or on deep neural networks trained end-to-end. The learned approaches are more robust but struggle to generalize to new environments and are less accurate than their classical counterparts. We hypothesize that they lack the constraints that 3D geometry provides. In this work, we introduce GeoCalib, a deep neural network that leverages universal rules of 3D geometry through an optimization process. GeoCalib is trained end-to-end to estimate camera parameters and learns to find useful visual cues from the data. Experiments on various benchmarks show that GeoCalib is more robust and more accurate than existing classical and learned approaches. Its internal optimization estimates uncertainties, which help flag failure cases and benefit downstream applications like visual localization. The code and trained models are publicly available at https://github.com/cvg/GeoCalib.  
  </ol>  
</details>  
**comments**: Presented at ECCV 2024  
  
### [Weakly-supervised Camera Localization by Ground-to-satellite Image Registration](http://arxiv.org/abs/2409.06471)  
[[code](https://github.com/yujiaoshi/g2sweakly)]  
Yujiao Shi, Hongdong Li, Akhil Perincherry, Ankit Vora  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The ground-to-satellite image matching/retrieval was initially proposed for city-scale ground camera localization. This work addresses the problem of improving camera pose accuracy by ground-to-satellite image matching after a coarse location and orientation have been obtained, either from the city-scale retrieval or from consumer-level GPS and compass sensors. Existing learning-based methods for solving this task require accurate GPS labels of ground images for network training. However, obtaining such accurate GPS labels is difficult, often requiring an expensive {\color{black}Real Time Kinematics (RTK)} setup and suffering from signal occlusion, multi-path signal disruptions, \etc. To alleviate this issue, this paper proposes a weakly supervised learning strategy for ground-to-satellite image registration when only noisy pose labels for ground images are available for network training. It derives positive and negative satellite images for each ground image and leverages contrastive learning to learn feature representations for ground and satellite images useful for translation estimation. We also propose a self-supervision strategy for cross-view image relative rotation estimation, which trains the network by creating pseudo query and reference image pairs. Experimental results show that our weakly supervised learning strategy achieves the best performance on cross-area evaluation compared to recent state-of-the-art methods that are reliant on accurate pose labels for supervision.  
  </ol>  
</details>  
**comments**: Accepted by ECCV 2024  
  
### [A Cross-Font Image Retrieval Network for Recognizing Undeciphered Oracle Bone Inscriptions](http://arxiv.org/abs/2409.06381)  
Zhicong Wu, Qifeng Su, Ke Gu, Xiaodong Shi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Oracle Bone Inscription (OBI) is the earliest mature writing system known in China to date, which represents a crucial stage in the development of hieroglyphs. Nevertheless, the substantial quantity of undeciphered OBI characters continues to pose a persistent challenge for scholars, while conventional methods of ancient script research are both time-consuming and labor-intensive. In this paper, we propose a cross-font image retrieval network (CFIRN) to decipher OBI characters by establishing associations between OBI characters and other script forms, simulating the interpretive behavior of paleography scholars. Concretely, our network employs a siamese framework to extract deep features from character images of various fonts, fully exploring structure clues with different resolution by designed multiscale feature integration (MFI) module and multiscale refinement classifier (MRC). Extensive experiments on three challenging cross-font image retrieval datasets demonstrate that, given undeciphered OBI characters, our CFIRN can effectively achieve accurate matches with characters from other gallery fonts.  
  </ol>  
</details>  
  
  



## Image Matching  

### [Weakly-supervised Camera Localization by Ground-to-satellite Image Registration](http://arxiv.org/abs/2409.06471)  
[[code](https://github.com/yujiaoshi/g2sweakly)]  
Yujiao Shi, Hongdong Li, Akhil Perincherry, Ankit Vora  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The ground-to-satellite image matching/retrieval was initially proposed for city-scale ground camera localization. This work addresses the problem of improving camera pose accuracy by ground-to-satellite image matching after a coarse location and orientation have been obtained, either from the city-scale retrieval or from consumer-level GPS and compass sensors. Existing learning-based methods for solving this task require accurate GPS labels of ground images for network training. However, obtaining such accurate GPS labels is difficult, often requiring an expensive {\color{black}Real Time Kinematics (RTK)} setup and suffering from signal occlusion, multi-path signal disruptions, \etc. To alleviate this issue, this paper proposes a weakly supervised learning strategy for ground-to-satellite image registration when only noisy pose labels for ground images are available for network training. It derives positive and negative satellite images for each ground image and leverages contrastive learning to learn feature representations for ground and satellite images useful for translation estimation. We also propose a self-supervision strategy for cross-view image relative rotation estimation, which trains the network by creating pseudo query and reference image pairs. Experimental results show that our weakly supervised learning strategy achieves the best performance on cross-area evaluation compared to recent state-of-the-art methods that are reliant on accurate pose labels for supervision.  
  </ol>  
</details>  
**comments**: Accepted by ECCV 2024  
  
  



## NeRF  

### [LEIA: Latent View-invariant Embeddings for Implicit 3D Articulation](http://arxiv.org/abs/2409.06703)  
Archana Swaminathan, Anubhav Gupta, Kamal Gupta, Shishira R. Maiya, Vatsal Agarwal, Abhinav Shrivastava  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRFs) have revolutionized the reconstruction of static scenes and objects in 3D, offering unprecedented quality. However, extending NeRFs to model dynamic objects or object articulations remains a challenging problem. Previous works have tackled this issue by focusing on part-level reconstruction and motion estimation for objects, but they often rely on heuristics regarding the number of moving parts or object categories, which can limit their practical use. In this work, we introduce LEIA, a novel approach for representing dynamic 3D objects. Our method involves observing the object at distinct time steps or "states" and conditioning a hypernetwork on the current state, using this to parameterize our NeRF. This approach allows us to learn a view-invariant latent representation for each state. We further demonstrate that by interpolating between these states, we can generate novel articulation configurations in 3D space that were previously unseen. Our experimental results highlight the effectiveness of our method in articulating objects in a manner that is independent of the viewing angle and joint configuration. Notably, our approach outperforms previous methods that rely on motion information for articulation registration.  
  </ol>  
</details>  
**comments**: Accepted to ECCV 2024. Project Website at
  https://archana1998.github.io/leia/  
  
### [Sources of Uncertainty in 3D Scene Reconstruction](http://arxiv.org/abs/2409.06407)  
[[code](https://github.com/aaltoml/uncertainty-nerf-gs)]  
Marcus Klasson, Riccardo Mereu, Juho Kannala, Arno Solin  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The process of 3D scene reconstruction can be affected by numerous uncertainty sources in real-world scenes. While Neural Radiance Fields (NeRFs) and 3D Gaussian Splatting (GS) achieve high-fidelity rendering, they lack built-in mechanisms to directly address or quantify uncertainties arising from the presence of noise, occlusions, confounding outliers, and imprecise camera pose inputs. In this paper, we introduce a taxonomy that categorizes different sources of uncertainty inherent in these methods. Moreover, we extend NeRF- and GS-based methods with uncertainty estimation techniques, including learning uncertainty outputs and ensembles, and perform an empirical study to assess their ability to capture the sensitivity of the reconstruction. Our study highlights the need for addressing various uncertainty aspects when designing NeRF/GS-based methods for uncertainty-aware 3D reconstruction.  
  </ol>  
</details>  
**comments**: To appear in ECCV 2024 Workshop Proceedings. Project page at
  https://aaltoml.github.io/uncertainty-nerf-gs/  
  
### [LSE-NeRF: Learning Sensor Modeling Errors for Deblured Neural Radiance Fields with RGB-Event Stereo](http://arxiv.org/abs/2409.06104)  
Wei Zhi Tang, Daniel Rebain, Kostantinos G. Derpanis, Kwang Moo Yi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present a method for reconstructing a clear Neural Radiance Field (NeRF) even with fast camera motions. To address blur artifacts, we leverage both (blurry) RGB images and event camera data captured in a binocular configuration. Importantly, when reconstructing our clear NeRF, we consider the camera modeling imperfections that arise from the simple pinhole camera model as learned embeddings for each camera measurement, and further learn a mapper that connects event camera measurements with RGB data. As no previous dataset exists for our binocular setting, we introduce an event camera dataset with captures from a 3D-printed stereo configuration between RGB and event cameras. Empirically, we evaluate our introduced dataset and EVIMOv2 and show that our method leads to improved reconstructions. Our code and dataset are available at https://github.com/ubc-vision/LSENeRF.  
  </ol>  
</details>  
  
  



