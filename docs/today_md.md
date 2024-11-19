<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#The-Oxford-Spires-Dataset:-Benchmarking-Large-Scale-LiDAR-Visual-Localisation,-Reconstruction-and-Radiance-Field-Methods>The Oxford Spires Dataset: Benchmarking Large-Scale LiDAR-Visual Localisation, Reconstruction and Radiance Field Methods</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Exploring-Emerging-Trends-and-Research-Opportunities-in-Visual-Place-Recognition>Exploring Emerging Trends and Research Opportunities in Visual Place Recognition</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Towards-Degradation-Robust-Reconstruction-in-Generalizable-NeRF>Towards Degradation-Robust Reconstruction in Generalizable NeRF</a></li>
        <li><a href=#LeC$^2$O-NeRF:-Learning-Continuous-and-Compact-Large-Scale-Occupancy-for-Urban-Scenes>LeC$^2$O-NeRF: Learning Continuous and Compact Large-Scale Occupancy for Urban Scenes</a></li>
        <li><a href=#The-Oxford-Spires-Dataset:-Benchmarking-Large-Scale-LiDAR-Visual-Localisation,-Reconstruction-and-Radiance-Field-Methods>The Oxford Spires Dataset: Benchmarking Large-Scale LiDAR-Visual Localisation, Reconstruction and Radiance Field Methods</a></li>
        <li><a href=#USP-Gaussian:-Unifying-Spike-based-Image-Reconstruction,-Pose-Correction-and-Gaussian-Splatting>USP-Gaussian: Unifying Spike-based Image Reconstruction, Pose Correction and Gaussian Splatting</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [The Oxford Spires Dataset: Benchmarking Large-Scale LiDAR-Visual Localisation, Reconstruction and Radiance Field Methods](http://arxiv.org/abs/2411.10546)  
Yifu Tao, Miguel Ángel Muñoz-Bañón, Lintong Zhang, Jiahao Wang, Lanke Frank Tarimo Fu, Maurice Fallon  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper introduces a large-scale multi-modal dataset captured in and around well-known landmarks in Oxford using a custom-built multi-sensor perception unit as well as a millimetre-accurate map from a Terrestrial LiDAR Scanner (TLS). The perception unit includes three synchronised global shutter colour cameras, an automotive 3D LiDAR scanner, and an inertial sensor - all precisely calibrated. We also establish benchmarks for tasks involving localisation, reconstruction, and novel-view synthesis, which enable the evaluation of Simultaneous Localisation and Mapping (SLAM) methods, Structure-from-Motion (SfM) and Multi-view Stereo (MVS) methods as well as radiance field methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting. To evaluate 3D reconstruction the TLS 3D models are used as ground truth. Localisation ground truth is computed by registering the mobile LiDAR scans to the TLS 3D models. Radiance field methods are evaluated not only with poses sampled from the input trajectory, but also from viewpoints that are from trajectories which are distant from the training poses. Our evaluation demonstrates a key limitation of state-of-the-art radiance field methods: we show that they tend to overfit to the training poses/images and do not generalise well to out-of-sequence poses. They also underperform in 3D reconstruction compared to MVS systems using the same visual inputs. Our dataset and benchmarks are intended to facilitate better integration of radiance field methods and SLAM systems. The raw and processed data, along with software for parsing and evaluation, can be accessed at https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/.  
  </ol>  
</details>  
**comments**: Website: https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/  
  
  



## Visual Localization  

### [Exploring Emerging Trends and Research Opportunities in Visual Place Recognition](http://arxiv.org/abs/2411.11481)  
Antonios Gasteratos, Konstantinos A. Tsintotas, Tobias Fischer, Yiannis Aloimonos, Michael Milford  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual-based recognition, e.g., image classification, object detection, etc., is a long-standing challenge in computer vision and robotics communities. Concerning the roboticists, since the knowledge of the environment is a prerequisite for complex navigation tasks, visual place recognition is vital for most localization implementations or re-localization and loop closure detection pipelines within simultaneous localization and mapping (SLAM). More specifically, it corresponds to the system's ability to identify and match a previously visited location using computer vision tools. Towards developing novel techniques with enhanced accuracy and robustness, while motivated by the success presented in natural language processing methods, researchers have recently turned their attention to vision-language models, which integrate visual and textual data.  
  </ol>  
</details>  
**comments**: 2 pages, 1 figure. 40th Anniversary of the IEEE Conference on
  Robotics and Automation (ICRA@40), Rotterdam, Netherlands, September 23-26,
  2024  
  
  



## NeRF  

### [Towards Degradation-Robust Reconstruction in Generalizable NeRF](http://arxiv.org/abs/2411.11691)  
Chan Ho Park, Ka Leong Cheng, Zhicheng Wang, Qifeng Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Generalizable Neural Radiance Field (GNeRF) across scenes has been proven to be an effective way to avoid per-scene optimization by representing a scene with deep image features of source images. However, despite its potential for real-world applications, there has been limited research on the robustness of GNeRFs to different types of degradation present in the source images. The lack of such research is primarily attributed to the absence of a large-scale dataset fit for training a degradation-robust generalizable NeRF model. To address this gap and facilitate investigations into the degradation robustness of 3D reconstruction tasks, we construct the Objaverse Blur Dataset, comprising 50,000 images from over 1000 settings featuring multiple levels of blur degradation. In addition, we design a simple and model-agnostic module for enhancing the degradation robustness of GNeRFs. Specifically, by extracting 3D-aware features through a lightweight depth estimator and denoiser, the proposed module shows improvement on different popular methods in GNeRFs in terms of both quantitative and visual quality over varying degradation types and levels. Our dataset and code will be made publicly available.  
  </ol>  
</details>  
  
### [LeC $^2$ O-NeRF: Learning Continuous and Compact Large-Scale Occupancy for Urban Scenes](http://arxiv.org/abs/2411.11374)  
Zhenxing Mi, Dan Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In NeRF, a critical problem is to effectively estimate the occupancy to guide empty-space skipping and point sampling. Grid-based methods work well for small-scale scenes. However, on large-scale scenes, they are limited by predefined bounding boxes, grid resolutions, and high memory usage for grid updates, and thus struggle to speed up training for large-scale, irregularly bounded and complex urban scenes without sacrificing accuracy. In this paper, we propose to learn a continuous and compact large-scale occupancy network, which can classify 3D points as occupied or unoccupied points. We train this occupancy network end-to-end together with the radiance field in a self-supervised manner by three designs. First, we propose a novel imbalanced occupancy loss to regularize the occupancy network. It makes the occupancy network effectively control the ratio of unoccupied and occupied points, motivated by the prior that most of 3D scene points are unoccupied. Second, we design an imbalanced architecture containing a large scene network and a small empty space network to separately encode occupied and unoccupied points classified by the occupancy network. This imbalanced structure can effectively model the imbalanced nature of occupied and unoccupied regions. Third, we design an explicit density loss to guide the occupancy network, making the density of unoccupied points smaller. As far as we know, we are the first to learn a continuous and compact occupancy of large-scale NeRF by a network. In our experiments, our occupancy network can quickly learn more compact, accurate and smooth occupancy compared to the occupancy grid. With our learned occupancy as guidance for empty space skipping on challenging large-scale benchmarks, our method consistently obtains higher accuracy compared to the occupancy grid, and our method can speed up state-of-the-art NeRF methods without sacrificing accuracy.  
  </ol>  
</details>  
**comments**: 13 pages  
  
### [The Oxford Spires Dataset: Benchmarking Large-Scale LiDAR-Visual Localisation, Reconstruction and Radiance Field Methods](http://arxiv.org/abs/2411.10546)  
Yifu Tao, Miguel Ángel Muñoz-Bañón, Lintong Zhang, Jiahao Wang, Lanke Frank Tarimo Fu, Maurice Fallon  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper introduces a large-scale multi-modal dataset captured in and around well-known landmarks in Oxford using a custom-built multi-sensor perception unit as well as a millimetre-accurate map from a Terrestrial LiDAR Scanner (TLS). The perception unit includes three synchronised global shutter colour cameras, an automotive 3D LiDAR scanner, and an inertial sensor - all precisely calibrated. We also establish benchmarks for tasks involving localisation, reconstruction, and novel-view synthesis, which enable the evaluation of Simultaneous Localisation and Mapping (SLAM) methods, Structure-from-Motion (SfM) and Multi-view Stereo (MVS) methods as well as radiance field methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting. To evaluate 3D reconstruction the TLS 3D models are used as ground truth. Localisation ground truth is computed by registering the mobile LiDAR scans to the TLS 3D models. Radiance field methods are evaluated not only with poses sampled from the input trajectory, but also from viewpoints that are from trajectories which are distant from the training poses. Our evaluation demonstrates a key limitation of state-of-the-art radiance field methods: we show that they tend to overfit to the training poses/images and do not generalise well to out-of-sequence poses. They also underperform in 3D reconstruction compared to MVS systems using the same visual inputs. Our dataset and benchmarks are intended to facilitate better integration of radiance field methods and SLAM systems. The raw and processed data, along with software for parsing and evaluation, can be accessed at https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/.  
  </ol>  
</details>  
**comments**: Website: https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/  
  
### [USP-Gaussian: Unifying Spike-based Image Reconstruction, Pose Correction and Gaussian Splatting](http://arxiv.org/abs/2411.10504)  
[[code](https://github.com/chenkang455/usp-gaussian)]  
Kang Chen, Jiyuan Zhang, Zecheng Hao, Yajing Zheng, Tiejun Huang, Zhaofei Yu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Spike cameras, as an innovative neuromorphic camera that captures scenes with the 0-1 bit stream at 40 kHz, are increasingly employed for the 3D reconstruction task via Neural Radiance Fields (NeRF) or 3D Gaussian Splatting (3DGS). Previous spike-based 3D reconstruction approaches often employ a casecased pipeline: starting with high-quality image reconstruction from spike streams based on established spike-to-image reconstruction algorithms, then progressing to camera pose estimation and 3D reconstruction. However, this cascaded approach suffers from substantial cumulative errors, where quality limitations of initial image reconstructions negatively impact pose estimation, ultimately degrading the fidelity of the 3D reconstruction. To address these issues, we propose a synergistic optimization framework, \textbf{USP-Gaussian}, that unifies spike-based image reconstruction, pose correction, and Gaussian splatting into an end-to-end framework. Leveraging the multi-view consistency afforded by 3DGS and the motion capture capability of the spike camera, our framework enables a joint iterative optimization that seamlessly integrates information between the spike-to-image network and 3DGS. Experiments on synthetic datasets with accurate poses demonstrate that our method surpasses previous approaches by effectively eliminating cascading errors. Moreover, we integrate pose optimization to achieve robust 3D reconstruction in real-world scenarios with inaccurate initial poses, outperforming alternative methods by effectively reducing noise and preserving fine texture details. Our code, data and trained models will be available at \url{https://github.com/chenkang455/USP-Gaussian}.  
  </ol>  
</details>  
  
  



