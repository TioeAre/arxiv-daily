<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#ColabSfM:-Collaborative-Structure-from-Motion-by-Point-Cloud-Registration>ColabSfM: Collaborative Structure-from-Motion by Point Cloud Registration</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Missing-Target-Relevant-Information-Prediction-with-World-Model-for-Accurate-Zero-Shot-Composed-Image-Retrieval>Missing Target-Relevant Information Prediction with World Model for Accurate Zero-Shot Composed Image Retrieval</a></li>
        <li><a href=#Autonomous-Exploration-Based-Precise-Mapping-for-Mobile-Robots-through-Stepwise-and-Consistent-Motions>Autonomous Exploration-Based Precise Mapping for Mobile Robots through Stepwise and Consistent Motions</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#FFaceNeRF:-Few-shot-Face-Editing-in-Neural-Radiance-Fields>FFaceNeRF: Few-shot Face Editing in Neural Radiance Fields</a></li>
        <li><a href=#DroneSplat:-3D-Gaussian-Splatting-for-Robust-3D-Reconstruction-from-In-the-Wild-Drone-Imagery>DroneSplat: 3D Gaussian Splatting for Robust 3D Reconstruction from In-the-Wild Drone Imagery</a></li>
        <li><a href=#Digitally-Prototype-Your-Eye-Tracker:-Simulating-Hardware-Performance-using-3D-Synthetic-Data>Digitally Prototype Your Eye Tracker: Simulating Hardware Performance using 3D Synthetic Data</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [ColabSfM: Collaborative Structure-from-Motion by Point Cloud Registration](http://arxiv.org/abs/2503.17093)  
Johan Edstedt, André Mateus, Alberto Jaenal  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Structure-from-Motion (SfM) is the task of estimating 3D structure and camera poses from images. We define Collaborative SfM (ColabSfM) as sharing distributed SfM reconstructions. Sharing maps requires estimating a joint reference frame, which is typically referred to as registration. However, there is a lack of scalable methods and training datasets for registering SfM reconstructions. In this paper, we tackle this challenge by proposing the scalable task of point cloud registration for SfM reconstructions. We find that current registration methods cannot register SfM point clouds when trained on existing datasets. To this end, we propose a SfM registration dataset generation pipeline, leveraging partial reconstructions from synthetically generated camera trajectories for each scene. Finally, we propose a simple but impactful neural refiner on top of the SotA registration method RoITr that yields significant improvements, which we call RefineRoITr. Our extensive experimental evaluation shows that our proposed pipeline and model enables ColabSfM. Code is available at https://github.com/EricssonResearch/ColabSfM  
  </ol>  
</details>  
**comments**: CVPR 2025  
  
  



## Visual Localization  

### [Missing Target-Relevant Information Prediction with World Model for Accurate Zero-Shot Composed Image Retrieval](http://arxiv.org/abs/2503.17109)  
Yuanmin Tang, Jing Yu, Keke Gai, Jiamin Zhuang, Gang Xiong, Gaopeng Gou, Qi Wu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Zero-Shot Composed Image Retrieval (ZS-CIR) involves diverse tasks with a broad range of visual content manipulation intent across domain, scene, object, and attribute. The key challenge for ZS-CIR tasks is to modify a reference image according to manipulation text to accurately retrieve a target image, especially when the reference image is missing essential target content. In this paper, we propose a novel prediction-based mapping network, named PrediCIR, to adaptively predict the missing target visual content in reference images in the latent space before mapping for accurate ZS-CIR. Specifically, a world view generation module first constructs a source view by omitting certain visual content of a target view, coupled with an action that includes the manipulation intent derived from existing image-caption pairs. Then, a target content prediction module trains a world model as a predictor to adaptively predict the missing visual information guided by user intention in manipulating text at the latent space. The two modules map an image with the predicted relevant information to a pseudo-word token without extra supervision. Our model shows strong generalization ability on six ZS-CIR tasks. It obtains consistent and significant performance boosts ranging from 1.73% to 4.45% over the best methods and achieves new state-of-the-art results on ZS-CIR. Our code is available at https://github.com/Pter61/predicir.  
  </ol>  
</details>  
**comments**: This work has been accepted to CVPR 2025  
  
### [Autonomous Exploration-Based Precise Mapping for Mobile Robots through Stepwise and Consistent Motions](http://arxiv.org/abs/2503.17005)  
Muhua Zhang, Lei Ma, Ying Wu, Kai Shen, Yongkui Sun, Henry Leung  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper presents an autonomous exploration framework. It is designed for indoor ground mobile robots that utilize laser Simultaneous Localization and Mapping (SLAM), ensuring process completeness and precise mapping results. For frontier search, the local-global sampling architecture based on multiple Rapidly Exploring Random Trees (RRTs) is employed. Traversability checks during RRT expansion and global RRT pruning upon map updates eliminate unreachable frontiers, reducing potential collisions and deadlocks. Adaptive sampling density adjustments, informed by obstacle distribution, enhance exploration coverage potential. For frontier point navigation, a stepwise consistent motion strategy is adopted, wherein the robot strictly drives straight on approximately equidistant line segments in the polyline path and rotates in place at segment junctions. This simplified, decoupled motion pattern improves scan-matching stability and mitigates map drift. For process control, the framework serializes frontier point selection and navigation, avoiding oscillation caused by frequent goal changes in conventional parallelized processes. The waypoint retracing mechanism is introduced to generate repeated observations, triggering loop closure detection and backend optimization in graph-based SLAM, thereby improving map consistency and precision. Experiments in both simulation and real-world scenarios validate the effectiveness of the framework. It achieves improved mapping coverage and precision in more challenging environments compared to baseline 2D exploration algorithms. It also shows robustness in supporting resource-constrained robot platforms and maintaining mapping consistency across various LiDAR field-of-view (FoV) configurations.  
  </ol>  
</details>  
**comments**: 8 pages, 11 figures. This work has been submitted to the IEEE for
  possible publication  
  
  



## NeRF  

### [FFaceNeRF: Few-shot Face Editing in Neural Radiance Fields](http://arxiv.org/abs/2503.17095)  
[[code](https://github.com/kwanyun/FFaceNeRF)]  
Kwan Yun, Chaelin Kim, Hangyeul Shin, Junyong Noh  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent 3D face editing methods using masks have produced high-quality edited images by leveraging Neural Radiance Fields (NeRF). Despite their impressive performance, existing methods often provide limited user control due to the use of pre-trained segmentation masks. To utilize masks with a desired layout, an extensive training dataset is required, which is challenging to gather. We present FFaceNeRF, a NeRF-based face editing technique that can overcome the challenge of limited user control due to the use of fixed mask layouts. Our method employs a geometry adapter with feature injection, allowing for effective manipulation of geometry attributes. Additionally, we adopt latent mixing for tri-plane augmentation, which enables training with a few samples. This facilitates rapid model adaptation to desired mask layouts, crucial for applications in fields like personalized medical imaging or creative face editing. Our comparative evaluations demonstrate that FFaceNeRF surpasses existing mask based face editing methods in terms of flexibility, control, and generated image quality, paving the way for future advancements in customized and high-fidelity 3D face editing. The code is available on the {\href{https://kwanyun.github.io/FFaceNeRF_page/}{project-page}}.  
  </ol>  
</details>  
**comments**: CVPR2025, 11 pages, 14 figures  
  
### [DroneSplat: 3D Gaussian Splatting for Robust 3D Reconstruction from In-the-Wild Drone Imagery](http://arxiv.org/abs/2503.16964)  
Jiadong Tang, Yu Gao, Dianyi Yang, Liqi Yan, Yufeng Yue, Yi Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Drones have become essential tools for reconstructing wild scenes due to their outstanding maneuverability. Recent advances in radiance field methods have achieved remarkable rendering quality, providing a new avenue for 3D reconstruction from drone imagery. However, dynamic distractors in wild environments challenge the static scene assumption in radiance fields, while limited view constraints hinder the accurate capture of underlying scene geometry. To address these challenges, we introduce DroneSplat, a novel framework designed for robust 3D reconstruction from in-the-wild drone imagery. Our method adaptively adjusts masking thresholds by integrating local-global segmentation heuristics with statistical approaches, enabling precise identification and elimination of dynamic distractors in static scenes. We enhance 3D Gaussian Splatting with multi-view stereo predictions and a voxel-guided optimization strategy, supporting high-quality rendering under limited view constraints. For comprehensive evaluation, we provide a drone-captured 3D reconstruction dataset encompassing both dynamic and static scenes. Extensive experiments demonstrate that DroneSplat outperforms both 3DGS and NeRF baselines in handling in-the-wild drone imagery.  
  </ol>  
</details>  
  
### [Digitally Prototype Your Eye Tracker: Simulating Hardware Performance using 3D Synthetic Data](http://arxiv.org/abs/2503.16742)  
Esther Y. H. Lin, Yimin Ding, Jogendra Kundu, Yatong An, Mohamed T. El-Haddad, Alexander Fix  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Eye tracking (ET) is a key enabler for Augmented and Virtual Reality (AR/VR). Prototyping new ET hardware requires assessing the impact of hardware choices on eye tracking performance. This task is compounded by the high cost of obtaining data from sufficiently many variations of real hardware, especially for machine learning, which requires large training datasets. We propose a method for end-to-end evaluation of how hardware changes impact machine learning-based ET performance using only synthetic data. We utilize a dataset of real 3D eyes, reconstructed from light dome data using neural radiance fields (NeRF), to synthesize captured eyes from novel viewpoints and camera parameters. Using this framework, we demonstrate that we can predict the relative performance across various hardware configurations, accounting for variations in sensor noise, illumination brightness, and optical blur. We also compare our simulator with the publicly available eye tracking dataset from the Project Aria glasses, demonstrating a strong correlation with real-world performance. Finally, we present a first-of-its-kind analysis in which we vary ET camera positions, evaluating ET performance ranging from on-axis direct views of the eye to peripheral views on the frame. Such an analysis would have previously required manufacturing physical devices to capture evaluation data. In short, our method enables faster prototyping of ET hardware.  
  </ol>  
</details>  
**comments**: 14 pages, 12 figures  
  
  



