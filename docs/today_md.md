<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#A-SCoRe:-Attention-based-Scene-Coordinate-Regression-for-wide-ranging-scenarios>A-SCoRe: Attention-based Scene Coordinate Regression for wide-ranging scenarios</a></li>
        <li><a href=#Improving-Geometric-Consistency-for-360-Degree-Neural-Radiance-Fields-in-Indoor-Scenarios>Improving Geometric Consistency for 360-Degree Neural Radiance Fields in Indoor Scenarios</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#3D-Densification-for-Multi-Map-Monocular-VSLAM-in-Endoscopy>3D Densification for Multi-Map Monocular VSLAM in Endoscopy</a></li>
        <li><a href=#A-SCoRe:-Attention-based-Scene-Coordinate-Regression-for-wide-ranging-scenarios>A-SCoRe: Attention-based Scene Coordinate Regression for wide-ranging scenarios</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Periodontal-Bone-Loss-Analysis-via-Keypoint-Detection-With-Heuristic-Post-Processing>Periodontal Bone Loss Analysis via Keypoint Detection With Heuristic Post-Processing</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Segmentation-Guided-Neural-Radiance-Fields-for-Novel-Street-View-Synthesis>Segmentation-Guided Neural Radiance Fields for Novel Street View Synthesis</a></li>
        <li><a href=#Improving-Geometric-Consistency-for-360-Degree-Neural-Radiance-Fields-in-Indoor-Scenarios>Improving Geometric Consistency for 360-Degree Neural Radiance Fields in Indoor Scenarios</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [A-SCoRe: Attention-based Scene Coordinate Regression for wide-ranging scenarios](http://arxiv.org/abs/2503.13982)  
Huy-Hoang Bui, Bach-Thuan Bui, Quang-Vinh Tran, Yasuyuki Fujii, Joo-Ho Lee  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual localization is considered to be one of the crucial parts in many robotic and vision systems. While state-of-the art methods that relies on feature matching have proven to be accurate for visual localization, its requirements for storage and compute are burdens. Scene coordinate regression (SCR) is an alternative approach that remove the barrier for storage by learning to map 2D pixels to 3D scene coordinates. Most popular SCR use Convolutional Neural Network (CNN) to extract 2D descriptor, which we would argue that it miss the spatial relationship between pixels. Inspired by the success of vision transformer architecture, we present a new SCR architecture, called A-ScoRe, an Attention-based model which leverage attention on descriptor map level to produce meaningful and high-semantic 2D descriptors. Since the operation is performed on descriptor map, our model can work with multiple data modality whether it is a dense or sparse from depth-map, SLAM to Structure-from-Motion (SfM). This versatility allows A-SCoRe to operate in different kind of environments, conditions and achieve the level of flexibility that is important for mobile robots. Results show our methods achieve comparable performance with State-of-the-art methods on multiple benchmark while being light-weighted and much more flexible. Code and pre-trained models are public in our repository: https://github.com/ais-lab/A-SCoRe.  
  </ol>  
</details>  
  
### [Improving Geometric Consistency for 360-Degree Neural Radiance Fields in Indoor Scenarios](http://arxiv.org/abs/2503.13710)  
Iryna Repinetska, Anna Hilsmann, Peter Eisert  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Photo-realistic rendering and novel view synthesis play a crucial role in human-computer interaction tasks, from gaming to path planning. Neural Radiance Fields (NeRFs) model scenes as continuous volumetric functions and achieve remarkable rendering quality. However, NeRFs often struggle in large, low-textured areas, producing cloudy artifacts known as ''floaters'' that reduce scene realism, especially in indoor environments with featureless architectural surfaces like walls, ceilings, and floors. To overcome this limitation, prior work has integrated geometric constraints into the NeRF pipeline, typically leveraging depth information derived from Structure from Motion or Multi-View Stereo. Yet, conventional RGB-feature correspondence methods face challenges in accurately estimating depth in textureless regions, leading to unreliable constraints. This challenge is further complicated in 360-degree ''inside-out'' views, where sparse visual overlap between adjacent images further hinders depth estimation. In order to address these issues, we propose an efficient and robust method for computing dense depth priors, specifically tailored for large low-textured architectural surfaces in indoor environments. We introduce a novel depth loss function to enhance rendering quality in these challenging, low-feature regions, while complementary depth-patch regularization further refines depth consistency across other areas. Experiments with Instant-NGP on two synthetic 360-degree indoor scenes demonstrate improved visual fidelity with our method compared to standard photometric loss and Mean Squared Error depth supervision.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [3D Densification for Multi-Map Monocular VSLAM in Endoscopy](http://arxiv.org/abs/2503.14346)  
X. Anadón, Javier Rodríguez-Puigvert, J. M. M. Montiel  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Multi-map Sparse Monocular visual Simultaneous Localization and Mapping applied to monocular endoscopic sequences has proven efficient to robustly recover tracking after the frequent losses in endoscopy due to motion blur, temporal occlusion, tools interaction or water jets. The sparse multi-maps are adequate for robust camera localization, however they are very poor for environment representation, they are noisy, with a high percentage of inaccurately reconstructed 3D points, including significant outliers, and more importantly with an unacceptable low density for clinical applications.   We propose a method to remove outliers and densify the maps of the state of the art for sparse endoscopy multi-map CudaSIFT-SLAM. The NN LightDepth for up-to-scale depth dense predictions are aligned with the sparse CudaSIFT submaps by means of the robust to spurious LMedS. Our system mitigates the inherent scale ambiguity in monocular depth estimation while filtering outliers, leading to reliable densified 3D maps.   We provide experimental evidence of accurate densified maps 4.15 mm RMS accuracy at affordable computing time in the C3VD phantom colon dataset. We report qualitative results on the real colonoscopy from the Endomapper dataset.  
  </ol>  
</details>  
  
### [A-SCoRe: Attention-based Scene Coordinate Regression for wide-ranging scenarios](http://arxiv.org/abs/2503.13982)  
Huy-Hoang Bui, Bach-Thuan Bui, Quang-Vinh Tran, Yasuyuki Fujii, Joo-Ho Lee  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual localization is considered to be one of the crucial parts in many robotic and vision systems. While state-of-the art methods that relies on feature matching have proven to be accurate for visual localization, its requirements for storage and compute are burdens. Scene coordinate regression (SCR) is an alternative approach that remove the barrier for storage by learning to map 2D pixels to 3D scene coordinates. Most popular SCR use Convolutional Neural Network (CNN) to extract 2D descriptor, which we would argue that it miss the spatial relationship between pixels. Inspired by the success of vision transformer architecture, we present a new SCR architecture, called A-ScoRe, an Attention-based model which leverage attention on descriptor map level to produce meaningful and high-semantic 2D descriptors. Since the operation is performed on descriptor map, our model can work with multiple data modality whether it is a dense or sparse from depth-map, SLAM to Structure-from-Motion (SfM). This versatility allows A-SCoRe to operate in different kind of environments, conditions and achieve the level of flexibility that is important for mobile robots. Results show our methods achieve comparable performance with State-of-the-art methods on multiple benchmark while being light-weighted and much more flexible. Code and pre-trained models are public in our repository: https://github.com/ais-lab/A-SCoRe.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Periodontal Bone Loss Analysis via Keypoint Detection With Heuristic Post-Processing](http://arxiv.org/abs/2503.13477)  
Ryan Banks, Vishal Thengane, María Eugenia Guerrero, Nelly Maria García-Madueño, Yunpeng Li, Hongying Tang, Akhilanand Chaurasia  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Calculating percentage bone loss is a critical test for periodontal disease staging but is sometimes imprecise and time consuming when manually calculated. This study evaluates the application of a deep learning keypoint and object detection model, YOLOv8-pose, for the automatic identification of localised periodontal bone loss landmarks, conditions and staging. YOLOv8-pose was fine-tuned on 193 annotated periapical radiographs. We propose a keypoint detection metric, Percentage of Relative Correct Keypoints (PRCK), which normalises the metric to the average tooth size of teeth in the image. We propose a heuristic post-processing module that adjusts certain keypoint predictions to align with the edge of the related tooth, using a supporting instance segmentation model trained on an open source auxiliary dataset. The model can sufficiently detect bone loss keypoints, tooth boxes, and alveolar ridge resorption, but has insufficient performance at detecting detached periodontal ligament and furcation involvement. The model with post-processing demonstrated a PRCK 0.25 of 0.726 and PRCK 0.05 of 0.401 for keypoint detection, mAP 0.5 of 0.715 for tooth object detection, mesial dice score of 0.593 for periodontal staging, and dice score of 0.280 for furcation involvement. Our annotation methodology provides a stage agnostic approach to periodontal disease detection, by ensuring most keypoints are present for each tooth in the image, allowing small imbalanced datasets. Our PRCK metric allows accurate evaluation of keypoints in dental domains. Our post-processing module adjusts predicted keypoints correctly but is dependent on a minimum quality of prediction by the pose detection and segmentation models. Code: https:// anonymous.4open.science/r/Bone-Loss-Keypoint-Detection-Code. Dataset: https://bit.ly/4hJ3aE7.  
  </ol>  
</details>  
**comments**: 31 pages, 7 tables, 5 figures, 3 equations, journal paper submitted
  to Computers in Biology and Medicine  
  
  



## NeRF  

### [Segmentation-Guided Neural Radiance Fields for Novel Street View Synthesis](http://arxiv.org/abs/2503.14219)  
Yizhou Li, Yusuke Monno, Masatoshi Okutomi, Yuuichi Tanaka, Seiichi Kataoka, Teruaki Kosiba  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advances in Neural Radiance Fields (NeRF) have shown great potential in 3D reconstruction and novel view synthesis, particularly for indoor and small-scale scenes. However, extending NeRF to large-scale outdoor environments presents challenges such as transient objects, sparse cameras and textures, and varying lighting conditions. In this paper, we propose a segmentation-guided enhancement to NeRF for outdoor street scenes, focusing on complex urban environments. Our approach extends ZipNeRF and utilizes Grounded SAM for segmentation mask generation, enabling effective handling of transient objects, modeling of the sky, and regularization of the ground. We also introduce appearance embeddings to adapt to inconsistent lighting across view sequences. Experimental results demonstrate that our method outperforms the baseline ZipNeRF, improving novel view synthesis quality with fewer artifacts and sharper details.  
  </ol>  
</details>  
**comments**: Presented at VISAPP2025. Project page:
  http://www.ok.sc.e.titech.ac.jp/res/NVS/index.html  
  
### [Improving Geometric Consistency for 360-Degree Neural Radiance Fields in Indoor Scenarios](http://arxiv.org/abs/2503.13710)  
Iryna Repinetska, Anna Hilsmann, Peter Eisert  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Photo-realistic rendering and novel view synthesis play a crucial role in human-computer interaction tasks, from gaming to path planning. Neural Radiance Fields (NeRFs) model scenes as continuous volumetric functions and achieve remarkable rendering quality. However, NeRFs often struggle in large, low-textured areas, producing cloudy artifacts known as ''floaters'' that reduce scene realism, especially in indoor environments with featureless architectural surfaces like walls, ceilings, and floors. To overcome this limitation, prior work has integrated geometric constraints into the NeRF pipeline, typically leveraging depth information derived from Structure from Motion or Multi-View Stereo. Yet, conventional RGB-feature correspondence methods face challenges in accurately estimating depth in textureless regions, leading to unreliable constraints. This challenge is further complicated in 360-degree ''inside-out'' views, where sparse visual overlap between adjacent images further hinders depth estimation. In order to address these issues, we propose an efficient and robust method for computing dense depth priors, specifically tailored for large low-textured architectural surfaces in indoor environments. We introduce a novel depth loss function to enhance rendering quality in these challenging, low-feature regions, while complementary depth-patch regularization further refines depth consistency across other areas. Experiments with Instant-NGP on two synthetic 360-degree indoor scenes demonstrate improved visual fidelity with our method compared to standard photometric loss and Mean Squared Error depth supervision.  
  </ol>  
</details>  
  
  



