<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Hardware-Aware-Feature-Extraction-Quantisation-for-Real-Time-Visual-Odometry-on-FPGA-Platforms>Hardware-Aware Feature Extraction Quantisation for Real-Time Visual Odometry on FPGA Platforms</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#SCREP:-Scene-Coordinate-Regression-and-Evidential-Learning-based-Perception-Aware-Trajectory-Generation>SCREP: Scene Coordinate Regression and Evidential Learning-based Perception-Aware Trajectory Generation</a></li>
        <li><a href=#VP-SelDoA:-Visual-prompted-Selective-DoA-Estimation-of-Target-Sound-via-Semantic-Spatial-Matching>VP-SelDoA: Visual-prompted Selective DoA Estimation of Target Sound via Semantic-Spatial Matching</a></li>
        <li><a href=#FACap:-A-Large-scale-Fashion-Dataset-for-Fine-grained-Composed-Image-Retrieval>FACap: A Large-scale Fashion Dataset for Fine-grained Composed Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Doodle-Your-Keypoints:-Sketch-Based-Few-Shot-Keypoint-Detection>Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#MUVOD:-A-Novel-Multi-view-Video-Object-Segmentation-Dataset-and-A-Benchmark-for-3D-Segmentation>MUVOD: A Novel Multi-view Video Object Segmentation Dataset and A Benchmark for 3D Segmentation</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Hardware-Aware Feature Extraction Quantisation for Real-Time Visual Odometry on FPGA Platforms](http://arxiv.org/abs/2507.07903)  
Mateusz Wasala, Mateusz Smolarczyk, Michal Danilowicz, Tomasz Kryjak  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Accurate position estimation is essential for modern navigation systems deployed in autonomous platforms, including ground vehicles, marine vessels, and aerial drones. In this context, Visual Simultaneous Localisation and Mapping (VSLAM) - which includes Visual Odometry - relies heavily on the reliable extraction of salient feature points from the visual input data. In this work, we propose an embedded implementation of an unsupervised architecture capable of detecting and describing feature points. It is based on a quantised SuperPoint convolutional neural network. Our objective is to minimise the computational demands of the model while preserving high detection quality, thus facilitating efficient deployment on platforms with limited resources, such as mobile or embedded systems. We implemented the solution on an FPGA System-on-Chip (SoC) platform, specifically the AMD/Xilinx Zynq UltraScale+, where we evaluated the performance of Deep Learning Processing Units (DPUs) and we also used the Brevitas library and the FINN framework to perform model quantisation and hardware-aware optimisation. This allowed us to process 640 x 480 pixel images at up to 54 fps on an FPGA platform, outperforming state-of-the-art solutions in the field. We conducted experiments on the TUM dataset to demonstrate and discuss the impact of different quantisation techniques on the accuracy and performance of the model in a visual odometry task.  
  </ol>  
</details>  
**comments**: Accepted for the DSD 2025 conference in Salerno, Italy  
  
  



## Visual Localization  

### [SCREP: Scene Coordinate Regression and Evidential Learning-based Perception-Aware Trajectory Generation](http://arxiv.org/abs/2507.07467)  
Juyeop Han, Lukas Lao Beyer, Guilherme V. Cavalheiro, Sertac Karaman  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Autonomous flight in GPS denied indoor spaces requires trajectories that keep visual localization error tightly bounded across varied missions. Whereas visual inertial odometry (VIO) accumulates drift over time, scene coordinate regression (SCR) yields drift-free, high accuracy absolute pose estimation. We present a perception-aware framework that couples an evidential learning-based SCR pose estimator with a receding horizon trajectory optimizer. The optimizer steers the onboard camera toward pixels whose uncertainty predicts reliable scene coordinates, while a fixed-lag smoother fuses the low rate SCR stream with high rate IMU data to close the perception control loop in real time. In simulation, our planner reduces translation (rotation) mean error by 54% / 15% (40% / 31%) relative to yaw fixed and forward-looking baselines, respectively. Moreover, hardware in the loop experiment validates the feasibility of our proposed framework.  
  </ol>  
</details>  
**comments**: 8 pages, 7 figures, 3 tables  
  
### [VP-SelDoA: Visual-prompted Selective DoA Estimation of Target Sound via Semantic-Spatial Matching](http://arxiv.org/abs/2507.07384)  
Yu Chen, Xinyuan Qian, Hongxu Zhu, Jiadong Wang, Kainan Chen, Haizhou Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Audio-visual sound source localization (AV-SSL) identifies the position of a sound source by exploiting the complementary strengths of auditory and visual signals. However, existing AV-SSL methods encounter three major challenges: 1) inability to selectively isolate the target sound source in multi-source scenarios, 2) misalignment between semantic visual features and spatial acoustic features, and 3) overreliance on paired audio-visual data. To overcome these limitations, we introduce Cross-Instance Audio-Visual Localization (CI-AVL), a novel task that leverages images from different instances of the same sound event category to localize target sound sources, thereby reducing dependence on paired data while enhancing generalization capabilities. Our proposed VP-SelDoA tackles this challenging task through a semantic-level modality fusion and employs a Frequency-Temporal ConMamba architecture to generate target-selective masks for sound isolation. We further develop a Semantic-Spatial Matching mechanism that aligns the heterogeneous semantic and spatial features via integrated cross- and self-attention mechanisms. To facilitate the CI-AVL research, we construct a large-scale dataset named VGG-SSL, comprising 13,981 spatial audio clips across 296 sound event categories. Extensive experiments show that our proposed method outperforms state-of-the-art audio-visual localization methods, achieving a mean absolute error (MAE) of 12.04 and an accuracy (ACC) of 78.23%.  
  </ol>  
</details>  
**comments**: Under Review  
  
### [FACap: A Large-scale Fashion Dataset for Fine-grained Composed Image Retrieval](http://arxiv.org/abs/2507.07135)  
François Gardères, Shizhe Chen, Camille-Sovanneary Gauthier, Jean Ponce  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The composed image retrieval (CIR) task is to retrieve target images given a reference image and a modification text. Recent methods for CIR leverage large pretrained vision-language models (VLMs) and achieve good performance on general-domain concepts like color and texture. However, they still struggle with application domains like fashion, because the rich and diverse vocabulary used in fashion requires specific fine-grained vision and language understanding. An additional difficulty is the lack of large-scale fashion datasets with detailed and relevant annotations, due to the expensive cost of manual annotation by specialists. To address these challenges, we introduce FACap, a large-scale, automatically constructed fashion-domain CIR dataset. It leverages web-sourced fashion images and a two-stage annotation pipeline powered by a VLM and a large language model (LLM) to generate accurate and detailed modification texts. Then, we propose a new CIR model FashionBLIP-2, which fine-tunes the general-domain BLIP-2 model on FACap with lightweight adapters and multi-head query-candidate matching to better account for fine-grained fashion-specific information. FashionBLIP-2 is evaluated with and without additional fine-tuning on the Fashion IQ benchmark and the enhanced evaluation dataset enhFashionIQ, leveraging our pipeline to obtain higher-quality annotations. Experimental results show that the combination of FashionBLIP-2 and pretraining with FACap significantly improves the model's performance in fashion CIR especially for retrieval with fine-grained modification texts, demonstrating the value of our dataset and approach in a highly demanding environment such as e-commerce websites. Code is available at https://fgxaos.github.io/facap-paper-website/.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection](http://arxiv.org/abs/2507.07994)  
Subhajit Maity, Ayan Kumar Bhunia, Subhadeep Koley, Pinaki Nath Chowdhury, Aneeshan Sain, Yi-Zhe Song  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Keypoint detection, integral to modern machine perception, faces challenges in few-shot learning, particularly when source data from the same distribution as the query is unavailable. This gap is addressed by leveraging sketches, a popular form of human expression, providing a source-free alternative. However, challenges arise in mastering cross-modal embeddings and handling user-specific sketch styles. Our proposed framework overcomes these hurdles with a prototypical setup, combined with a grid-based locator and prototypical domain adaptation. We also demonstrate success in few-shot convergence across novel keypoints and classes through extensive experiments.  
  </ol>  
</details>  
**comments**: Accepted at ICCV 2025. Project Page: https://subhajitmaity.me/DYKp  
  
  



## NeRF  

### [MUVOD: A Novel Multi-view Video Object Segmentation Dataset and A Benchmark for 3D Segmentation](http://arxiv.org/abs/2507.07519)  
Bangning Wei, Joshua Maraval, Meriem Outtas, Kidiyo Kpalma, Nicolas Ramin, Lu Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The application of methods based on Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3D GS) have steadily gained popularity in the field of 3D object segmentation in static scenes. These approaches demonstrate efficacy in a range of 3D scene understanding and editing tasks. Nevertheless, the 4D object segmentation of dynamic scenes remains an underexplored field due to the absence of a sufficiently extensive and accurately labelled multi-view video dataset. In this paper, we present MUVOD, a new multi-view video dataset for training and evaluating object segmentation in reconstructed real-world scenarios. The 17 selected scenes, describing various indoor or outdoor activities, are collected from different sources of datasets originating from various types of camera rigs. Each scene contains a minimum of 9 views and a maximum of 46 views. We provide 7830 RGB images (30 frames per video) with their corresponding segmentation mask in 4D motion, meaning that any object of interest in the scene could be tracked across temporal frames of a given view or across different views belonging to the same camera rig. This dataset, which contains 459 instances of 73 categories, is intended as a basic benchmark for the evaluation of multi-view video segmentation methods. We also present an evaluation metric and a baseline segmentation approach to encourage and evaluate progress in this evolving field. Additionally, we propose a new benchmark for 3D object segmentation task with a subset of annotated multi-view images selected from our MUVOD dataset. This subset contains 50 objects of different conditions in different scenarios, providing a more comprehensive analysis of state-of-the-art 3D object segmentation methods. Our proposed MUVOD dataset is available at https://volumetric-repository.labs.b-com.com/#/muvod.  
  </ol>  
</details>  
  
  



