<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#eNCApsulate:-NCA-for-Precision-Diagnosis-on-Capsule-Endoscopes>eNCApsulate: NCA for Precision Diagnosis on Capsule Endoscopes</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Emotion-Recognition-in-Contemporary-Dance-Performances-Using-Laban-Movement-Analysis>Emotion Recognition in Contemporary Dance Performances Using Laban Movement Analysis</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#GauSS-MI:-Gaussian-Splatting-Shannon-Mutual-Information-for-Active-3D-Reconstruction>GauSS-MI: Gaussian Splatting Shannon Mutual Information for Active 3D Reconstruction</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [eNCApsulate: NCA for Precision Diagnosis on Capsule Endoscopes](http://arxiv.org/abs/2504.21562)  
Henry John Krumb, Anirban Mukhopadhyay  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Wireless Capsule Endoscopy is a non-invasive imaging method for the entire gastrointestinal tract, and is a pain-free alternative to traditional endoscopy. It generates extensive video data that requires significant review time, and localizing the capsule after ingestion is a challenge. Techniques like bleeding detection and depth estimation can help with localization of pathologies, but deep learning models are typically too large to run directly on the capsule. Neural Cellular Automata (NCA) for bleeding segmentation and depth estimation are trained on capsule endoscopic images. For monocular depth estimation, we distill a large foundation model into the lean NCA architecture, by treating the outputs of the foundation model as pseudo ground truth. We then port the trained NCA to the ESP32 microcontroller, enabling efficient image processing on hardware as small as a camera capsule. NCA are more accurate (Dice) than other portable segmentation models, while requiring more than 100x fewer parameters stored in memory than other small-scale models. The visual results of NCA depth estimation look convincing, and in some cases beat the realism and detail of the pseudo ground truth. Runtime optimizations on the ESP32-S3 accelerate the average inference speed significantly, by more than factor 3. With several algorithmic adjustments and distillation, it is possible to eNCApsulate NCA models into microcontrollers that fit into wireless capsule endoscopes. This is the first work that enables reliable bleeding segmentation and depth estimation on a miniaturized device, paving the way for precise diagnosis combined with visual odometry as a means of precise localization of the capsule -- on the capsule.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Emotion Recognition in Contemporary Dance Performances Using Laban Movement Analysis](http://arxiv.org/abs/2504.21154)  
Muhammad Turab, Philippe Colantoni, Damien Muselet, Alain Tremeau  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper presents a novel framework for emotion recognition in contemporary dance by improving existing Laban Movement Analysis (LMA) feature descriptors and introducing robust, novel descriptors that capture both quantitative and qualitative aspects of the movement. Our approach extracts expressive characteristics from 3D keypoints data of professional dancers performing contemporary dance under various emotional states, and trains multiple classifiers, including Random Forests and Support Vector Machines. Additionally, we provide in-depth explanation of features and their impact on model predictions using explainable machine learning methods. Overall, our study improves emotion recognition in contemporary dance and offers promising applications in performance analysis, dance training, and human--computer interaction, with a highest accuracy of 96.85\%.  
  </ol>  
</details>  
  
  



## NeRF  

### [GauSS-MI: Gaussian Splatting Shannon Mutual Information for Active 3D Reconstruction](http://arxiv.org/abs/2504.21067)  
Yuhan Xie, Yixi Cai, Yinqiang Zhang, Lei Yang, Jia Pan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This research tackles the challenge of real-time active view selection and uncertainty quantification on visual quality for active 3D reconstruction. Visual quality is a critical aspect of 3D reconstruction. Recent advancements such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have notably enhanced the image rendering quality of reconstruction models. Nonetheless, the efficient and effective acquisition of input images for reconstruction-specifically, the selection of the most informative viewpoint-remains an open challenge, which is crucial for active reconstruction. Existing studies have primarily focused on evaluating geometric completeness and exploring unobserved or unknown regions, without direct evaluation of the visual uncertainty within the reconstruction model. To address this gap, this paper introduces a probabilistic model that quantifies visual uncertainty for each Gaussian. Leveraging Shannon Mutual Information, we formulate a criterion, Gaussian Splatting Shannon Mutual Information (GauSS-MI), for real-time assessment of visual mutual information from novel viewpoints, facilitating the selection of next best view. GauSS-MI is implemented within an active reconstruction system integrated with a view and motion planner. Extensive experiments across various simulated and real-world scenes showcase the superior visual quality and reconstruction efficiency performance of the proposed system.  
  </ol>  
</details>  
  
  



