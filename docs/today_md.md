<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Leveraging-Consistent-Spatio-Temporal-Correspondence-for-Robust-Visual-Odometry>Leveraging Consistent Spatio-Temporal Correspondence for Robust Visual Odometry</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Reconstructing-People,-Places,-and-Cameras>Reconstructing People, Places, and Cameras</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Where-am-I?-Cross-View-Geo-localization-with-Natural-Language-Descriptions>Where am I? Cross-View Geo-localization with Natural Language Descriptions</a></li>
        <li><a href=#Large-Scale-UWB-Anchor-Calibration-and-One-Shot-Localization-Using-Gaussian-Process>Large-Scale UWB Anchor Calibration and One-Shot Localization Using Gaussian Process</a></li>
        <li><a href=#Open-Vocabulary-Mobile-Manipulation-Based-on-Double-Relaxed-Contrastive-Learning-with-Dense-Labeling>Open-Vocabulary Mobile Manipulation Based on Double Relaxed Contrastive Learning with Dense Labeling</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#A-Novel-Approach-to-Tomato-Harvesting-Using-a-Hybrid-Gripper-with-Semantic-Segmentation-and-Keypoint-Detection>A Novel Approach to Tomato Harvesting Using a Hybrid Gripper with Semantic Segmentation and Keypoint Detection</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Editing-Implicit-and-Explicit-Representations-of-Radiance-Fields:-A-Survey>Editing Implicit and Explicit Representations of Radiance Fields: A Survey</a></li>
        <li><a href=#Exploring-Dynamic-Novel-View-Synthesis-Technologies-for-Cinematography>Exploring Dynamic Novel View Synthesis Technologies for Cinematography</a></li>
        <li><a href=#LUCES-MV:-A-Multi-View-Dataset-for-Near-Field-Point-Light-Source-Photometric-Stereo>LUCES-MV: A Multi-View Dataset for Near-Field Point Light Source Photometric Stereo</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Leveraging Consistent Spatio-Temporal Correspondence for Robust Visual Odometry](http://arxiv.org/abs/2412.16923)  
Zhaoxing Zhang, Junda Cheng, Gangwei Xu, Xiaoxiang Wang, Can Zhang, Xin Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent approaches to VO have significantly improved performance by using deep networks to predict optical flow between video frames. However, existing methods still suffer from noisy and inconsistent flow matching, making it difficult to handle challenging scenarios and long-sequence estimation. To overcome these challenges, we introduce Spatio-Temporal Visual Odometry (STVO), a novel deep network architecture that effectively leverages inherent spatio-temporal cues to enhance the accuracy and consistency of multi-frame flow matching. With more accurate and consistent flow matching, STVO can achieve better pose estimation through the bundle adjustment (BA). Specifically, STVO introduces two innovative components: 1) the Temporal Propagation Module that utilizes multi-frame information to extract and propagate temporal cues across adjacent frames, maintaining temporal consistency; 2) the Spatial Activation Module that utilizes geometric priors from the depth maps to enhance spatial consistency while filtering out excessive noise and incorrect matches. Our STVO achieves state-of-the-art performance on TUM-RGBD, EuRoc MAV, ETH3D and KITTI Odometry benchmarks. Notably, it improves accuracy by 77.8% on ETH3D benchmark and 38.9% on KITTI Odometry benchmark over the previous best methods.  
  </ol>  
</details>  
  
  



## SFM  

### [Reconstructing People, Places, and Cameras](http://arxiv.org/abs/2412.17806)  
Lea Müller, Hongsuk Choi, Anthony Zhang, Brent Yi, Jitendra Malik, Angjoo Kanazawa  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present "Humans and Structure from Motion" (HSfM), a method for jointly reconstructing multiple human meshes, scene point clouds, and camera parameters in a metric world coordinate system from a sparse set of uncalibrated multi-view images featuring people. Our approach combines data-driven scene reconstruction with the traditional Structure-from-Motion (SfM) framework to achieve more accurate scene reconstruction and camera estimation, while simultaneously recovering human meshes. In contrast to existing scene reconstruction and SfM methods that lack metric scale information, our method estimates approximate metric scale by leveraging a human statistical model. Furthermore, it reconstructs multiple human meshes within the same world coordinate system alongside the scene point cloud, effectively capturing spatial relationships among individuals and their positions in the environment. We initialize the reconstruction of humans, scenes, and cameras using robust foundational models and jointly optimize these elements. This joint optimization synergistically improves the accuracy of each component. We compare our method to existing approaches on two challenging benchmarks, EgoHumans and EgoExo4D, demonstrating significant improvements in human localization accuracy within the world coordinate frame (reducing error from 3.51m to 1.04m in EgoHumans and from 2.9m to 0.56m in EgoExo4D). Notably, our results show that incorporating human data into the SfM pipeline improves camera pose estimation (e.g., increasing RRA@15 by 20.3% on EgoHumans). Additionally, qualitative results show that our approach improves overall scene reconstruction quality. Our code is available at: muelea.github.io/hsfm.  
  </ol>  
</details>  
**comments**: Project website: muelea.github.io/hsfm  
  
  



## Visual Localization  

### [Where am I? Cross-View Geo-localization with Natural Language Descriptions](http://arxiv.org/abs/2412.17007)  
Junyan Ye, Honglin Lin, Leyan Ou, Dairong Chen, Zihao Wang, Conghui He, Weijia Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Cross-view geo-localization identifies the locations of street-view images by matching them with geo-tagged satellite images or OSM. However, most studies focus on image-to-image retrieval, with fewer addressing text-guided retrieval, a task vital for applications like pedestrian navigation and emergency response. In this work, we introduce a novel task for cross-view geo-localization with natural language descriptions, which aims to retrieve corresponding satellite images or OSM database based on scene text. To support this task, we construct the CVG-Text dataset by collecting cross-view data from multiple cities and employing a scene text generation approach that leverages the annotation capabilities of Large Multimodal Models to produce high-quality scene text descriptions with localization details.Additionally, we propose a novel text-based retrieval localization method, CrossText2Loc, which improves recall by 10% and demonstrates excellent long-text retrieval capabilities. In terms of explainability, it not only provides similarity scores but also offers retrieval reasons. More information can be found at https://yejy53.github.io/CVG-Text/.  
  </ol>  
</details>  
**comments**: 11 pages, 6 figures  
  
### [Large-Scale UWB Anchor Calibration and One-Shot Localization Using Gaussian Process](http://arxiv.org/abs/2412.16880)  
Shenghai Yuan, Boyang Lou, Thien-Minh Nguyen, Pengyu Yin, Muqing Cao, Xinghang Xu, Jianping Li, Jie Xu, Siyu Chen, Lihua Xie  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Ultra-wideband (UWB) is gaining popularity with devices like AirTags for precise home item localization but faces significant challenges when scaled to large environments like seaports. The main challenges are calibration and localization in obstructed conditions, which are common in logistics environments. Traditional calibration methods, dependent on line-of-sight (LoS), are slow, costly, and unreliable in seaports and warehouses, making large-scale localization a significant pain point in the industry. To overcome these challenges, we propose a UWB-LiDAR fusion-based calibration and one-shot localization framework. Our method uses Gaussian Processes to estimate anchor position from continuous-time LiDAR Inertial Odometry with sampled UWB ranges. This approach ensures accurate and reliable calibration with just one round of sampling in large-scale areas, I.e., 600x450 square meter. With the LoS issues, UWB-only localization can be problematic, even when anchor positions are known. We demonstrate that by applying a UWB-range filter, the search range for LiDAR loop closure descriptors is significantly reduced, improving both accuracy and speed. This concept can be applied to other loop closure detection methods, enabling cost-effective localization in large-scale warehouses and seaports. It significantly improves precision in challenging environments where UWB-only and LiDAR-Inertial methods fall short, as shown in the video \url{https://youtu.be/oY8jQKdM7lU }. We will open-source our datasets and calibration codes for community use.  
  </ol>  
</details>  
**comments**: Submitted to ICRA 2025  
  
### [Open-Vocabulary Mobile Manipulation Based on Double Relaxed Contrastive Learning with Dense Labeling](http://arxiv.org/abs/2412.16576)  
[[code](https://github.com/keio-smilab24/relax-former)]  
Daichi Yashima, Ryosuke Korekata, Komei Sugiura  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Growing labor shortages are increasing the demand for domestic service robots (DSRs) to assist in various settings. In this study, we develop a DSR that transports everyday objects to specified pieces of furniture based on open-vocabulary instructions. Our approach focuses on retrieving images of target objects and receptacles from pre-collected images of indoor environments. For example, given an instruction "Please get the right red towel hanging on the metal towel rack and put it in the white washing machine on the left," the DSR is expected to carry the red towel to the washing machine based on the retrieved images. This is challenging because the correct images should be retrieved from thousands of collected images, which may include many images of similar towels and appliances. To address this, we propose RelaX-Former, which learns diverse and robust representations from among positive, unlabeled positive, and negative samples. We evaluated RelaX-Former on a dataset containing real-world indoor images and human annotated instructions including complex referring expressions. The experimental results demonstrate that RelaX-Former outperformed existing baseline models across standard image retrieval metrics. Moreover, we performed physical experiments using a DSR to evaluate the performance of our approach in a zero-shot transfer setting. The experiments involved the DSR to carry objects to specific receptacles based on open-vocabulary instructions, achieving an overall success rate of 75%.  
  </ol>  
</details>  
**comments**: Accepted for IEEE RA-L 2025  
  
  



## Keypoint Detection  

### [A Novel Approach to Tomato Harvesting Using a Hybrid Gripper with Semantic Segmentation and Keypoint Detection](http://arxiv.org/abs/2412.16755)  
Shahid Ansari, Mahendra Kumar Gohil, Bishakh Bhattacharya  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Current agriculture and farming industries are able to reap advancements in robotics and automation technology to harvest fruits and vegetables using robots with adaptive grasping forces based on the compliance or softness of the fruit or vegetable. A successful operation depends on using a gripper that can adapt to the mechanical properties of the crops. This paper proposes a new robotic harvesting approach for tomato fruit using a novel hybrid gripper with a soft caging effect. It uses its six flexible passive auxetic structures based on fingers with rigid outer exoskeletons for good gripping strength and shape conformability. The gripper is actuated through a scotch-yoke mechanism using a servo motor. To perform tomato picking operations through a gripper, a vision system based on a depth camera and RGB camera implements the fruit identification process. It incorporates deep learning-based keypoint detection of the tomato's pedicel and body for localization in an occluded and variable ambient light environment and semantic segmentation of ripe and unripe tomatoes. In addition, robust trajectory planning of the robotic arm based on input from the vision system and control of robotic gripper movements are carried out for secure tomato handling. The tunable grasping force of the gripper would allow the robotic handling of fruits with a broad range of compliance.  
  </ol>  
</details>  
  
  



## NeRF  

### [Editing Implicit and Explicit Representations of Radiance Fields: A Survey](http://arxiv.org/abs/2412.17628)  
Arthur Hubert, Gamal Elghazaly, Raphael Frank  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) revolutionized novel view synthesis in recent years by offering a new volumetric representation, which is compact and provides high-quality image rendering. However, the methods to edit those radiance fields developed slower than the many improvements to other aspects of NeRF. With the recent development of alternative radiance field-based representations inspired by NeRF as well as the worldwide rise in popularity of text-to-image models, many new opportunities and strategies have emerged to provide radiance field editing. In this paper, we deliver a comprehensive survey of the different editing methods present in the literature for NeRF and other similar radiance field representations. We propose a new taxonomy for classifying existing works based on their editing methodologies, review pioneering models, reflect on current and potential new applications of radiance field editing, and compare state-of-the-art approaches in terms of editing options and performance.  
  </ol>  
</details>  
  
### [Exploring Dynamic Novel View Synthesis Technologies for Cinematography](http://arxiv.org/abs/2412.17532)  
Adrian Azzarelli, Nantheera Anantrasirichai, David R Bull  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Novel view synthesis (NVS) has shown significant promise for applications in cinematographic production, particularly through the exploitation of Neural Radiance Fields (NeRF) and Gaussian Splatting (GS). These methods model real 3D scenes, enabling the creation of new shots that are challenging to capture in the real world due to set topology or expensive equipment requirement. This innovation also offers cinematographic advantages such as smooth camera movements, virtual re-shoots, slow-motion effects, etc. This paper explores dynamic NVS with the aim of facilitating the model selection process. We showcase its potential through a short montage filmed using various NVS models.  
  </ol>  
</details>  
  
### [LUCES-MV: A Multi-View Dataset for Near-Field Point Light Source Photometric Stereo](http://arxiv.org/abs/2412.16737)  
Fotios Logothetis, Ignas Budvytis, Stephan Liwicki, Roberto Cipolla  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The biggest improvements in Photometric Stereo (PS) field has recently come from adoption of differentiable volumetric rendering techniques such as NeRF or Neural SDF achieving impressive reconstruction error of 0.2mm on DiLiGenT-MV benchmark. However, while there are sizeable datasets for environment lit objects such as Digital Twin Catalogue (DTS), there are only several small Photometric Stereo datasets which often lack challenging objects (simple, smooth, untextured) and practical, small form factor (near-field) light setup.   To address this, we propose LUCES-MV, the first real-world, multi-view dataset designed for near-field point light source photometric stereo. Our dataset includes 15 objects with diverse materials, each imaged under varying light conditions from an array of 15 LEDs positioned 30 to 40 centimeters from the camera center. To facilitate transparent end-to-end evaluation, our dataset provides not only ground truth normals and ground truth object meshes and poses but also light and camera calibration images.   We evaluate state-of-the-art near-field photometric stereo algorithms, highlighting their strengths and limitations across different material and shape complexities. LUCES-MV dataset offers an important benchmark for developing more robust, accurate and scalable real-world Photometric Stereo based 3D reconstruction methods.  
  </ol>  
</details>  
  
  



