<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Reprojection-Errors-as-Prompts-for-Efficient-Scene-Coordinate-Regression>Reprojection Errors as Prompts for Efficient Scene Coordinate Regression</a></li>
        <li><a href=#Matched-Filtering-based-LiDAR-Place-Recognition-for-Urban-and-Natural-Environments>Matched Filtering based LiDAR Place Recognition for Urban and Natural Environments</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#D4:-Text-guided-diffusion-model-based-domain-adaptive-data-augmentation-for-vineyard-shoot-detection>D4: Text-guided diffusion model-based domain adaptive data augmentation for vineyard shoot detection</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Reprojection Errors as Prompts for Efficient Scene Coordinate Regression](http://arxiv.org/abs/2409.04178)  
Ting-Ru Liu, Hsuan-Kung Yang, Jou-Min Liu, Chun-Wei Huang, Tsung-Chih Chiang, Quan Kong, Norimasa Kobori, Chun-Yi Lee  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Scene coordinate regression (SCR) methods have emerged as a promising area of research due to their potential for accurate visual localization. However, many existing SCR approaches train on samples from all image regions, including dynamic objects and texture-less areas. Utilizing these areas for optimization during training can potentially hamper the overall performance and efficiency of the model. In this study, we first perform an in-depth analysis to validate the adverse impacts of these areas. Drawing inspiration from our analysis, we then introduce an error-guided feature selection (EGFS) mechanism, in tandem with the use of the Segment Anything Model (SAM). This mechanism seeds low reprojection areas as prompts and expands them into error-guided masks, and then utilizes these masks to sample points and filter out problematic areas in an iterative manner. The experiments demonstrate that our method outperforms existing SCR approaches that do not rely on 3D information on the Cambridge Landmarks and Indoor6 datasets.  
  </ol>  
</details>  
**comments**: ECCV2024  
  
### [Matched Filtering based LiDAR Place Recognition for Urban and Natural Environments](http://arxiv.org/abs/2409.03998)  
Therese Joseph, Tobias Fischer, Michael Milford  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Place recognition is an important task within autonomous navigation, involving the re-identification of previously visited locations from an initial traverse. Unlike visual place recognition (VPR), LiDAR place recognition (LPR) is tolerant to changes in lighting, seasons, and textures, leading to high performance on benchmark datasets from structured urban environments. However, there is a growing need for methods that can operate in diverse environments with high performance and minimal training. In this paper, we propose a handcrafted matching strategy that performs roto-translation invariant place recognition and relative pose estimation for both urban and unstructured natural environments. Our approach constructs Birds Eye View (BEV) global descriptors and employs a two-stage search using matched filtering -- a signal processing technique for detecting known signals amidst noise. Extensive testing on the NCLT, Oxford Radar, and WildPlaces datasets consistently demonstrates state-of-the-art (SoTA) performance across place recognition and relative pose estimation metrics, with up to 15% higher recall than previous SoTA.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [D4: Text-guided diffusion model-based domain adaptive data augmentation for vineyard shoot detection](http://arxiv.org/abs/2409.04060)  
Kentaro Hirahara, Chikahito Nakane, Hajime Ebisawa, Tsuyoshi Kuroda, Yohei Iwaki, Tomoyoshi Utsumi, Yuichiro Nomura, Makoto Koike, Hiroshi Mineno  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In an agricultural field, plant phenotyping using object detection models is gaining attention. However, collecting the training data necessary to create generic and high-precision models is extremely challenging due to the difficulty of annotation and the diversity of domains. Furthermore, it is difficult to transfer training data across different crops, and although machine learning models effective for specific environments, conditions, or crops have been developed, they cannot be widely applied in actual fields. In this study, we propose a generative data augmentation method (D4) for vineyard shoot detection. D4 uses a pre-trained text-guided diffusion model based on a large number of original images culled from video data collected by unmanned ground vehicles or other means, and a small number of annotated datasets. The proposed method generates new annotated images with background information adapted to the target domain while retaining annotation information necessary for object detection. In addition, D4 overcomes the lack of training data in agriculture, including the difficulty of annotation and diversity of domains. We confirmed that this generative data augmentation method improved the mean average precision by up to 28.65% for the BBox detection task and the average precision by up to 13.73% for the keypoint detection task for vineyard shoot detection. Our generative data augmentation method D4 is expected to simultaneously solve the cost and domain diversity issues of training data generation in agriculture and improve the generalization performance of detection models.  
  </ol>  
</details>  
  
  



