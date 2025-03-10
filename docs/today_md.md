<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#THE-SEAN:-A-Heart-Rate-Variation-Inspired-Temporally-High-Order-Event-Based-Visual-Odometry-with-Self-Supervised-Spiking-Event-Accumulation-Networks>THE-SEAN: A Heart Rate Variation-Inspired Temporally High-Order Event-Based Visual Odometry with Self-Supervised Spiking Event Accumulation Networks</a></li>
        <li><a href=#MarsLGPR:-Mars-Rover-Localization-with-Ground-Penetrating-Radar>MarsLGPR: Mars Rover Localization with Ground Penetrating Radar</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Data-Efficient-Generalization-for-Zero-shot-Composed-Image-Retrieval>Data-Efficient Generalization for Zero-shot Composed Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Automatic-determination-of-quasicrystalline-patterns-from-microscopy-images>Automatic determination of quasicrystalline patterns from microscopy images</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [THE-SEAN: A Heart Rate Variation-Inspired Temporally High-Order Event-Based Visual Odometry with Self-Supervised Spiking Event Accumulation Networks](http://arxiv.org/abs/2503.05112)  
Chaoran Xiong, Litao Wei, Kehui Ma, Zhen Sun, Yan Xiang, Zihan Nan, Trieu-Kien Truong, Ling Pei  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Event-based visual odometry has recently gained attention for its high accuracy and real-time performance in fast-motion systems. Unlike traditional synchronous estimators that rely on constant-frequency (zero-order) triggers, event-based visual odometry can actively accumulate information to generate temporally high-order estimation triggers. However, existing methods primarily focus on adaptive event representation after estimation triggers, neglecting the decision-making process for efficient temporal triggering itself. This oversight leads to the computational redundancy and noise accumulation. In this paper, we introduce a temporally high-order event-based visual odometry with spiking event accumulation networks (THE-SEAN). To the best of our knowledge, it is the first event-based visual odometry capable of dynamically adjusting its estimation trigger decision in response to motion and environmental changes. Inspired by biological systems that regulate hormone secretion to modulate heart rate, a self-supervised spiking neural network is designed to generate estimation triggers. This spiking network extracts temporal features to produce triggers, with rewards based on block matching points and Fisher information matrix (FIM) trace acquired from the estimator itself. Finally, THE-SEAN is evaluated across several open datasets, thereby demonstrating average improvements of 13\% in estimation accuracy, 9\% in smoothness, and 38\% in triggering efficiency compared to the state-of-the-art methods.  
  </ol>  
</details>  
  
### [MarsLGPR: Mars Rover Localization with Ground Penetrating Radar](http://arxiv.org/abs/2503.04944)  
Anja Sheppard, Katherine A. Skinner  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this work, we propose the use of Ground Penetrating Radar (GPR) for rover localization on Mars. Precise pose estimation is an important task for mobile robots exploring planetary surfaces, as they operate in GPS-denied environments. Although visual odometry provides accurate localization, it is computationally expensive and can fail in dim or high-contrast lighting. Wheel encoders can also provide odometry estimation, but are prone to slipping on the sandy terrain encountered on Mars. Although traditionally a scientific surveying sensor, GPR has been used on Earth for terrain classification and localization through subsurface feature matching. The Perseverance rover and the upcoming ExoMars rover have GPR sensors already equipped to aid in the search of water and mineral resources. We propose to leverage GPR to aid in Mars rover localization. Specifically, we develop a novel GPR-based deep learning model that predicts 1D relative pose translation. We fuse our GPR pose prediction method with inertial and wheel encoder data in a filtering framework to output rover localization. We perform experiments in a Mars analog environment and demonstrate that our GPR-based displacement predictions both outperform wheel encoders and improve multi-modal filtering estimates in high-slip environments. Lastly, we present the first dataset aimed at GPR-based localization in Mars analog environments, which will be made publicly available upon publication.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Data-Efficient Generalization for Zero-shot Composed Image Retrieval](http://arxiv.org/abs/2503.05204)  
Zining Chen, Zhicheng Zhao, Fei Su, Xiaoqin Zhang, Shijian Lu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Zero-shot Composed Image Retrieval (ZS-CIR) aims to retrieve the target image based on a reference image and a text description without requiring in-distribution triplets for training. One prevalent approach follows the vision-language pretraining paradigm that employs a mapping network to transfer the image embedding to a pseudo-word token in the text embedding space. However, this approach tends to impede network generalization due to modality discrepancy and distribution shift between training and inference. To this end, we propose a Data-efficient Generalization (DeG) framework, including two novel designs, namely, Textual Supplement (TS) module and Semantic-Set (S-Set). The TS module exploits compositional textual semantics during training, enhancing the pseudo-word token with more linguistic semantics and thus mitigating the modality discrepancy effectively. The S-Set exploits the zero-shot capability of pretrained Vision-Language Models (VLMs), alleviating the distribution shift and mitigating the overfitting issue from the redundancy of the large-scale image-text data. Extensive experiments over four ZS-CIR benchmarks show that DeG outperforms the state-of-the-art (SOTA) methods with much less training data, and saves substantial training and inference time for practical usage.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Automatic determination of quasicrystalline patterns from microscopy images](http://arxiv.org/abs/2503.05472)  
Tano Kim Kender, Marco Corrias, Cesare Franchini  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Quasicrystals are aperiodically ordered solids that exhibit long-range order without translational periodicity, bridging the gap between crystalline and amorphous materials. Due to their lack of translational periodicity, information on atomic arrangements in quasicrystals cannot be extracted by current crystalline lattice recognition softwares. This work introduces a method to automatically detect quasicrystalline atomic arrangements and tiling using image feature recognition coupled with machine learning, tailored towards quasiperiodic tilings with 8-, 10- and 12-fold rotational symmetry. Atom positions are identified using clustering of feature descriptors. Subsequent nearest-neighbor analysis and border following on the interatomic connections deliver the tiling. Support vector machines further increase the quality of the results, reaching an accuracy consistent with those reported in the literature. A statistical analysis of the results is performed. The code is now part of the open-source package AiSurf.  
  </ol>  
</details>  
  
  



