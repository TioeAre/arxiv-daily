<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Semantically-correlated-memories-in-a-dense-associative-model>Semantically-correlated memories in a dense associative model</a></li>
        <li><a href=#Training-Free-Open-Vocabulary-Segmentation-with-Offline-Diffusion-Augmented-Prototype-Generation>Training-Free Open-Vocabulary Segmentation with Offline Diffusion-Augmented Prototype Generation</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#SplatPose-&-Detect:-Pose-Agnostic-3D-Anomaly-Detection>SplatPose & Detect: Pose-Agnostic 3D Anomaly Detection</a></li>
        <li><a href=#MonoSelfRecon:-Purely-Self-Supervised-Explicit-Generalizable-3D-Reconstruction-of-Indoor-Scenes-from-Monocular-RGB-Views>MonoSelfRecon: Purely Self-Supervised Explicit Generalizable 3D Reconstruction of Indoor Scenes from Monocular RGB Views</a></li>
        <li><a href=#Bayesian-NeRF:-Quantifying-Uncertainty-with-Volume-Density-in-Neural-Radiance-Fields>Bayesian NeRF: Quantifying Uncertainty with Volume Density in Neural Radiance Fields</a></li>
        <li><a href=#SpikeNVS:-Enhancing-Novel-View-Synthesis-from-Blurry-Images-via-Spike-Camera>SpikeNVS: Enhancing Novel View Synthesis from Blurry Images via Spike Camera</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Semantically-correlated memories in a dense associative model](http://arxiv.org/abs/2404.07123)  
Thomas F Burns  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    I introduce a novel associative memory model named Correlated Dense Associative Memory (CDAM), which integrates both auto- and hetero-association in a unified framework for continuous-valued memory patterns. Employing an arbitrary graph structure to semantically link memory patterns, CDAM is theoretically and numerically analysed, revealing four distinct dynamical modes: auto-association, narrow hetero-association, wide hetero-association, and neutral quiescence. Drawing inspiration from inhibitory modulation studies, I employ anti-Hebbian learning rules to control the range of hetero-association, extract multi-scale representations of community structures in graphs, and stabilise the recall of temporal sequences. Experimental demonstrations showcase CDAM's efficacy in handling real-world data, replicating a classical neuroscience experiment, performing image retrieval, and simulating arbitrary finite automata.  
  </ol>  
</details>  
**comments**: 35 pages, 32 figures  
  
### [Training-Free Open-Vocabulary Segmentation with Offline Diffusion-Augmented Prototype Generation](http://arxiv.org/abs/2404.06542)  
Luca Barsellotti, Roberto Amoroso, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Open-vocabulary semantic segmentation aims at segmenting arbitrary categories expressed in textual form. Previous works have trained over large amounts of image-caption pairs to enforce pixel-level multimodal alignments. However, captions provide global information about the semantics of a given image but lack direct localization of individual concepts. Further, training on large-scale datasets inevitably brings significant computational costs. In this paper, we propose FreeDA, a training-free diffusion-augmented method for open-vocabulary semantic segmentation, which leverages the ability of diffusion models to visually localize generated concepts and local-global similarities to match class-agnostic regions with semantic classes. Our approach involves an offline stage in which textual-visual reference embeddings are collected, starting from a large set of captions and leveraging visual and semantic contexts. At test time, these are queried to support the visual matching process, which is carried out by jointly considering class-agnostic regions and global semantic similarities. Extensive analyses demonstrate that FreeDA achieves state-of-the-art performance on five datasets, surpassing previous methods by more than 7.0 average points in terms of mIoU and without requiring any training.  
  </ol>  
</details>  
**comments**: CVPR 2024. Project page: https://aimagelab.github.io/freeda/  
  
  



## NeRF  

### [SplatPose & Detect: Pose-Agnostic 3D Anomaly Detection](http://arxiv.org/abs/2404.06832)  
[[code](https://github.com/m-kruse98/splatpose)]  
Mathis Kruse, Marco Rudolph, Dominik Woiwode, Bodo Rosenhahn  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Detecting anomalies in images has become a well-explored problem in both academia and industry. State-of-the-art algorithms are able to detect defects in increasingly difficult settings and data modalities. However, most current methods are not suited to address 3D objects captured from differing poses. While solutions using Neural Radiance Fields (NeRFs) have been proposed, they suffer from excessive computation requirements, which hinder real-world usability. For this reason, we propose the novel 3D Gaussian splatting-based framework SplatPose which, given multi-view images of a 3D object, accurately estimates the pose of unseen views in a differentiable manner, and detects anomalies in them. We achieve state-of-the-art results in both training and inference speed, and detection performance, even when using less training data than competing methods. We thoroughly evaluate our framework using the recently proposed Pose-agnostic Anomaly Detection benchmark and its multi-pose anomaly detection (MAD) data set.  
  </ol>  
</details>  
**comments**: Visual Anomaly and Novelty Detection 2.0 Workshop at CVPR 2024  
  
### [MonoSelfRecon: Purely Self-Supervised Explicit Generalizable 3D Reconstruction of Indoor Scenes from Monocular RGB Views](http://arxiv.org/abs/2404.06753)  
Runfa Li, Upal Mahbub, Vasudev Bhaskaran, Truong Nguyen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Current monocular 3D scene reconstruction (3DR) works are either fully-supervised, or not generalizable, or implicit in 3D representation. We propose a novel framework - MonoSelfRecon that for the first time achieves explicit 3D mesh reconstruction for generalizable indoor scenes with monocular RGB views by purely self-supervision on voxel-SDF (signed distance function). MonoSelfRecon follows an Autoencoder-based architecture, decodes voxel-SDF and a generalizable Neural Radiance Field (NeRF), which is used to guide voxel-SDF in self-supervision. We propose novel self-supervised losses, which not only support pure self-supervision, but can be used together with supervised signals to further boost supervised training. Our experiments show that "MonoSelfRecon" trained in pure self-supervision outperforms current best self-supervised indoor depth estimation models and is comparable to 3DR models trained in fully supervision with depth annotations. MonoSelfRecon is not restricted by specific model design, which can be used to any models with voxel-SDF for purely self-supervised manner.  
  </ol>  
</details>  
  
### [Bayesian NeRF: Quantifying Uncertainty with Volume Density in Neural Radiance Fields](http://arxiv.org/abs/2404.06727)  
[[code](https://github.com/lab-of-ai-and-robotics/bayesian_nerf)]  
Sibeak Lee, Kyeongsu Kang, Hyeonwoo Yu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present the Bayesian Neural Radiance Field (NeRF), which explicitly quantifies uncertainty in geometric volume structures without the need for additional networks, making it adept for challenging observations and uncontrolled images. NeRF diverges from traditional geometric methods by offering an enriched scene representation, rendering color and density in 3D space from various viewpoints. However, NeRF encounters limitations in relaxing uncertainties by using geometric structure information, leading to inaccuracies in interpretation under insufficient real-world observations. Recent research efforts aimed at addressing this issue have primarily relied on empirical methods or auxiliary networks. To fundamentally address this issue, we propose a series of formulational extensions to NeRF. By introducing generalized approximations and defining density-related uncertainty, our method seamlessly extends to manage uncertainty not only for RGB but also for depth, without the need for additional networks or empirical assumptions. In experiments we show that our method significantly enhances performance on RGB and depth images in the comprehensive dataset, demonstrating the reliability of the Bayesian NeRF approach to quantifying uncertainty based on the geometric structure.  
  </ol>  
</details>  
  
### [SpikeNVS: Enhancing Novel View Synthesis from Blurry Images via Spike Camera](http://arxiv.org/abs/2404.06710)  
Gaole Dai, Zhenyu Wang, Qinwen Xu, Wen Cheng, Ming Lu, Boxing Shi, Shanghang Zhang, Tiejun Huang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    One of the most critical factors in achieving sharp Novel View Synthesis (NVS) using neural field methods like Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) is the quality of the training images. However, Conventional RGB cameras are susceptible to motion blur. In contrast, neuromorphic cameras like event and spike cameras inherently capture more comprehensive temporal information, which can provide a sharp representation of the scene as additional training data. Recent methods have explored the integration of event cameras to improve the quality of NVS. The event-RGB approaches have some limitations, such as high training costs and the inability to work effectively in the background. Instead, our study introduces a new method that uses the spike camera to overcome these limitations. By considering texture reconstruction from spike streams as ground truth, we design the Texture from Spike (TfS) loss. Since the spike camera relies on temporal integration instead of temporal differentiation used by event cameras, our proposed TfS loss maintains manageable training costs. It handles foreground objects with backgrounds simultaneously. We also provide a real-world dataset captured with our spike-RGB camera system to facilitate future research endeavors. We conduct extensive experiments using synthetic and real-world datasets to demonstrate that our design can enhance novel view synthesis across NeRF and 3DGS. The code and dataset will be made available for public access.  
  </ol>  
</details>  
  
  



