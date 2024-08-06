<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Deep-Patch-Visual-SLAM>Deep Patch Visual SLAM</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Birational-geometry-of-critical-loci-in-Algebraic-Vision>Birational geometry of critical loci in Algebraic Vision</a></li>
        <li><a href=#PanicleNeRF:-low-cost,-high-precision-in-field-phenotypingof-rice-panicles-with-smartphone>PanicleNeRF: low-cost, high-precision in-field phenotypingof rice panicles with smartphone</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#CMR-Agent:-Learning-a-Cross-Modal-Agent-for-Iterative-Image-to-Point-Cloud-Registration>CMR-Agent: Learning a Cross-Modal Agent for Iterative Image-to-Point Cloud Registration</a></li>
        <li><a href=#BEVPlace++:-Fast,-Robust,-and-Lightweight-LiDAR-Global-Localization-for-Unmanned-Ground-Vehicles>BEVPlace++: Fast, Robust, and Lightweight LiDAR Global Localization for Unmanned Ground Vehicles</a></li>
        <li><a href=#On-Validation-of-Search-&-Retrieval-of-Tissue-Images-in-Digital-Pathology>On Validation of Search & Retrieval of Tissue Images in Digital Pathology</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Improving-Neural-Surface-Reconstruction-with-Feature-Priors-from-Multi-View-Image>Improving Neural Surface Reconstruction with Feature Priors from Multi-View Image</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#PanicleNeRF:-low-cost,-high-precision-in-field-phenotypingof-rice-panicles-with-smartphone>PanicleNeRF: low-cost, high-precision in-field phenotypingof rice panicles with smartphone</a></li>
        <li><a href=#FBINeRF:-Feature-Based-Integrated-Recurrent-Network-for-Pinhole-and-Fisheye-Neural-Radiance-Fields>FBINeRF: Feature-Based Integrated Recurrent Network for Pinhole and Fisheye Neural Radiance Fields</a></li>
        <li><a href=#E$^3$NeRF:-Efficient-Event-Enhanced-Neural-Radiance-Fields-from-Blurry-Images>E$^3$NeRF: Efficient Event-Enhanced Neural Radiance Fields from Blurry Images</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Deep Patch Visual SLAM](http://arxiv.org/abs/2408.01654)  
Lahav Lipson, Zachary Teed, Jia Deng  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent work in visual SLAM has shown the effectiveness of using deep network backbones. Despite excellent accuracy, however, such approaches are often expensive to run or do not generalize well zero-shot. Their runtime can also fluctuate wildly while their frontend and backend fight for access to GPU resources. To address these problems, we introduce Deep Patch Visual (DPV) SLAM, a method for monocular visual SLAM on a single GPU. DPV-SLAM maintains a high minimum framerate and small memory overhead (5-7G) compared to existing deep SLAM systems. On real-world datasets, DPV-SLAM runs at 1x-4x real-time framerates. We achieve comparable accuracy to DROID-SLAM on EuRoC and TartanAir while running 2.5x faster using a fraction of the memory. DPV-SLAM is an extension to the DPVO visual odometry system; its code can be found in the same repository: https://github.com/princeton-vl/DPVO  
  </ol>  
</details>  
  
  



## SFM  

### [Birational geometry of critical loci in Algebraic Vision](http://arxiv.org/abs/2408.02067)  
Marina Bertolini, Roberto Notari, Cristina Turrini  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In Algebraic Vision, the projective reconstruction of the position of each camera and scene point from the knowledge of many enough corresponding points in the views is called the structure from motion problem. It is known that the reconstruction is ambiguous if the scene points are contained in particular algebraic varieties, called critical loci. To be more precise, from the definition of criticality, for the same reconstruction problem, two critical loci arise in a natural way. In the present paper, we investigate the relations between these two critical loci, and we prove that, under some mild smoothness hypotheses, (some of) their irreducible components are birational. To this end, we introduce a unified critical locus that restores the symmetry between the two critical loci, and a natural commutative diagram relating the unified critical locus and the two single critical loci. For technical reasons, but of interest in its own, we also consider how a critical locus change when one increases the number of views.  
  </ol>  
</details>  
  
### [PanicleNeRF: low-cost, high-precision in-field phenotypingof rice panicles with smartphone](http://arxiv.org/abs/2408.02053)  
Xin Yang, Xuqi Lu, Pengyao Xie, Ziyue Guo, Hui Fang, Haowei Fu, Xiaochun Hu, Zhenbiao Sun, Haiyan Cen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The rice panicle traits significantly influence grain yield, making them a primary target for rice phenotyping studies. However, most existing techniques are limited to controlled indoor environments and difficult to capture the rice panicle traits under natural growth conditions. Here, we developed PanicleNeRF, a novel method that enables high-precision and low-cost reconstruction of rice panicle three-dimensional (3D) models in the field using smartphone. The proposed method combined the large model Segment Anything Model (SAM) and the small model You Only Look Once version 8 (YOLOv8) to achieve high-precision segmentation of rice panicle images. The NeRF technique was then employed for 3D reconstruction using the images with 2D segmentation. Finally, the resulting point clouds are processed to successfully extract panicle traits. The results show that PanicleNeRF effectively addressed the 2D image segmentation task, achieving a mean F1 Score of 86.9% and a mean Intersection over Union (IoU) of 79.8%, with nearly double the boundary overlap (BO) performance compared to YOLOv8. As for point cloud quality, PanicleNeRF significantly outperformed traditional SfM-MVS (structure-from-motion and multi-view stereo) methods, such as COLMAP and Metashape. The panicle length was then accurately extracted with the rRMSE of 2.94% for indica and 1.75% for japonica rice. The panicle volume estimated from 3D point clouds strongly correlated with the grain number (R2 = 0.85 for indica and 0.82 for japonica) and grain mass (0.80 for indica and 0.76 for japonica). This method provides a low-cost solution for high-throughput in-field phenotyping of rice panicles, accelerating the efficiency of rice breeding.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [CMR-Agent: Learning a Cross-Modal Agent for Iterative Image-to-Point Cloud Registration](http://arxiv.org/abs/2408.02394)  
Gongxin Yao, Yixin Xuan, Xinyang Li, Yu Pan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image-to-point cloud registration aims to determine the relative camera pose of an RGB image with respect to a point cloud. It plays an important role in camera localization within pre-built LiDAR maps. Despite the modality gaps, most learning-based methods establish 2D-3D point correspondences in feature space without any feedback mechanism for iterative optimization, resulting in poor accuracy and interpretability. In this paper, we propose to reformulate the registration procedure as an iterative Markov decision process, allowing for incremental adjustments to the camera pose based on each intermediate state. To achieve this, we employ reinforcement learning to develop a cross-modal registration agent (CMR-Agent), and use imitation learning to initialize its registration policy for stability and quick-start of the training. According to the cross-modal observations, we propose a 2D-3D hybrid state representation that fully exploits the fine-grained features of RGB images while reducing the useless neutral states caused by the spatial truncation of camera frustum. Additionally, the overall framework is well-designed to efficiently reuse one-shot cross-modal embeddings, avoiding repetitive and time-consuming feature extraction. Extensive experiments on the KITTI-Odometry and NuScenes datasets demonstrate that CMR-Agent achieves competitive accuracy and efficiency in registration. Once the one-shot embeddings are completed, each iteration only takes a few milliseconds.  
  </ol>  
</details>  
**comments**: Accepted to IEEE/RSJ International Conference on Intelligent Robots
  and Systems (IROS) 2024  
  
### [BEVPlace++: Fast, Robust, and Lightweight LiDAR Global Localization for Unmanned Ground Vehicles](http://arxiv.org/abs/2408.01841)  
Lun Luo, Siyuan Cao, Xiaorui Li, Jintao Xu, Rui Ai, Zhu Yu, Xieyuanli Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This article introduces BEVPlace++, a novel, fast, and robust LiDAR global localization method for unmanned ground vehicles. It uses lightweight convolutional neural networks (CNNs) on Bird's Eye View (BEV) image-like representations of LiDAR data to achieve accurate global localization through place recognition followed by 3-DoF pose estimation. Our detailed analyses reveal an interesting fact that CNNs are inherently effective at extracting distinctive features from LiDAR BEV images. Remarkably, keypoints of two BEV images with large translations can be effectively matched using CNN-extracted features. Building on this insight, we design a rotation equivariant module (REM) to obtain distinctive features while enhancing robustness to rotational changes. A Rotation Equivariant and Invariant Network (REIN) is then developed by cascading REM and a descriptor generator, NetVLAD, to sequentially generate rotation equivariant local features and rotation invariant global descriptors. The global descriptors are used first to achieve robust place recognition, and the local features are used for accurate pose estimation. Experimental results on multiple public datasets demonstrate that BEVPlace++, even when trained on a small dataset (3000 frames of KITTI) only with place labels, generalizes well to unseen environments, performs consistently across different days and years, and adapts to various types of LiDAR scanners. BEVPlace++ achieves state-of-the-art performance in subtasks of global localization including place recognition, loop closure detection, and global localization. Additionally, BEVPlace++ is lightweight, runs in real-time, and does not require accurate pose supervision, making it highly convenient for deployment. The source codes are publicly available at \href{https://github.com/zjuluolun/BEVPlace}{https://github.com/zjuluolun/BEVPlace}.  
  </ol>  
</details>  
**comments**: Under review  
  
### [On Validation of Search & Retrieval of Tissue Images in Digital Pathology](http://arxiv.org/abs/2408.01570)  
H. R. Tizhoosh  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Medical images play a crucial role in modern healthcare by providing vital information for diagnosis, treatment planning, and disease monitoring. Fields such as radiology and pathology rely heavily on accurate image interpretation, with radiologists examining X-rays, CT scans, and MRIs to diagnose conditions from fractures to cancer, while pathologists use microscopy and digital images to detect cellular abnormalities for diagnosing cancers and infections. The technological advancements have exponentially increased the volume and complexity of medical images, necessitating efficient tools for management and retrieval. Content-Based Image Retrieval (CBIR) systems address this need by searching and retrieving images based on visual content, enhancing diagnostic accuracy by allowing clinicians to find similar cases and compare pathological patterns. Comprehensive validation of image search engines in medical applications involves evaluating performance metrics like accuracy, indexing, and search times, and storage overhead, ensuring reliable and efficient retrieval of accurate results, as demonstrated by recent validations in histopathology.  
  </ol>  
</details>  
  
  



## Image Matching  

### [Improving Neural Surface Reconstruction with Feature Priors from Multi-View Image](http://arxiv.org/abs/2408.02079)  
Xinlin Ren, Chenjie Cao, Yanwei Fu, Xiangyang Xue  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in Neural Surface Reconstruction (NSR) have significantly improved multi-view reconstruction when coupled with volume rendering. However, relying solely on photometric consistency in image space falls short of addressing complexities posed by real-world data, including occlusions and non-Lambertian surfaces. To tackle these challenges, we propose an investigation into feature-level consistent loss, aiming to harness valuable feature priors from diverse pretext visual tasks and overcome current limitations. It is crucial to note the existing gap in determining the most effective pretext visual task for enhancing NSR. In this study, we comprehensively explore multi-view feature priors from seven pretext visual tasks, comprising thirteen methods. Our main goal is to strengthen NSR training by considering a wide range of possibilities. Additionally, we examine the impact of varying feature resolutions and evaluate both pixel-wise and patch-wise consistent losses, providing insights into effective strategies for improving NSR performance. By incorporating pre-trained representations from MVSFormer and QuadTree, our approach can generate variations of MVS-NeuS and Match-NeuS, respectively. Our results, analyzed on DTU and EPFL datasets, reveal that feature priors from image matching and multi-view stereo outperform other pretext tasks. Moreover, we discover that extending patch-wise photometric consistency to the feature level surpasses the performance of pixel-wise approaches. These findings underscore the effectiveness of these techniques in enhancing NSR outcomes.  
  </ol>  
</details>  
**comments**: ECCV2024  
  
  



## NeRF  

### [PanicleNeRF: low-cost, high-precision in-field phenotypingof rice panicles with smartphone](http://arxiv.org/abs/2408.02053)  
Xin Yang, Xuqi Lu, Pengyao Xie, Ziyue Guo, Hui Fang, Haowei Fu, Xiaochun Hu, Zhenbiao Sun, Haiyan Cen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The rice panicle traits significantly influence grain yield, making them a primary target for rice phenotyping studies. However, most existing techniques are limited to controlled indoor environments and difficult to capture the rice panicle traits under natural growth conditions. Here, we developed PanicleNeRF, a novel method that enables high-precision and low-cost reconstruction of rice panicle three-dimensional (3D) models in the field using smartphone. The proposed method combined the large model Segment Anything Model (SAM) and the small model You Only Look Once version 8 (YOLOv8) to achieve high-precision segmentation of rice panicle images. The NeRF technique was then employed for 3D reconstruction using the images with 2D segmentation. Finally, the resulting point clouds are processed to successfully extract panicle traits. The results show that PanicleNeRF effectively addressed the 2D image segmentation task, achieving a mean F1 Score of 86.9% and a mean Intersection over Union (IoU) of 79.8%, with nearly double the boundary overlap (BO) performance compared to YOLOv8. As for point cloud quality, PanicleNeRF significantly outperformed traditional SfM-MVS (structure-from-motion and multi-view stereo) methods, such as COLMAP and Metashape. The panicle length was then accurately extracted with the rRMSE of 2.94% for indica and 1.75% for japonica rice. The panicle volume estimated from 3D point clouds strongly correlated with the grain number (R2 = 0.85 for indica and 0.82 for japonica) and grain mass (0.80 for indica and 0.76 for japonica). This method provides a low-cost solution for high-throughput in-field phenotyping of rice panicles, accelerating the efficiency of rice breeding.  
  </ol>  
</details>  
  
### [FBINeRF: Feature-Based Integrated Recurrent Network for Pinhole and Fisheye Neural Radiance Fields](http://arxiv.org/abs/2408.01878)  
Yifan Wu, Tianyi Cheng, Peixu Xin, Janusz Konrad  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Previous studies aiming to optimize and bundle-adjust camera poses using Neural Radiance Fields (NeRFs), such as BARF and DBARF, have demonstrated impressive capabilities in 3D scene reconstruction. However, these approaches have been designed for pinhole-camera pose optimization and do not perform well under radial image distortions such as those in fisheye cameras. Furthermore, inaccurate depth initialization in DBARF results in erroneous geometric information affecting the overall convergence and quality of results. In this paper, we propose adaptive GRUs with a flexible bundle-adjustment method adapted to radial distortions and incorporate feature-based recurrent neural networks to generate continuous novel views from fisheye datasets. Other NeRF methods for fisheye images, such as SCNeRF and OMNI-NeRF, use projected ray distance loss for distorted pose refinement, causing severe artifacts, long rendering time, and are difficult to use in downstream tasks, where the dense voxel representation generated by a NeRF method needs to be converted into a mesh representation. We also address depth initialization issues by adding MiDaS-based depth priors for pinhole images. Through extensive experiments, we demonstrate the generalization capacity of FBINeRF and show high-fidelity results for both pinhole-camera and fisheye-camera NeRFs.  
  </ol>  
</details>  
**comments**: 18 pages  
  
### [E $^3$ NeRF: Efficient Event-Enhanced Neural Radiance Fields from Blurry Images](http://arxiv.org/abs/2408.01840)  
Yunshan Qi, Jia Li, Yifan Zhao, Yu Zhang, Lin Zhu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) achieve impressive rendering performance by learning volumetric 3D representation from several images of different views. However, it is difficult to reconstruct a sharp NeRF from blurry input as it often occurs in the wild. To solve this problem, we propose a novel Efficient Event-Enhanced NeRF (E$^3$NeRF) by utilizing the combination of RGB images and event streams. To effectively introduce event streams into the neural volumetric representation learning process, we propose an event-enhanced blur rendering loss and an event rendering loss, which guide the network via modeling the real blur process and event generation process, respectively. Specifically, we leverage spatial-temporal information from the event stream to evenly distribute learning attention over temporal blur while simultaneously focusing on blurry texture through the spatial attention. Moreover, a camera pose estimation framework for real-world data is built with the guidance of the events to generalize the method to practical applications. Compared to previous image-based or event-based NeRF, our framework makes more profound use of the internal relationship between events and images. Extensive experiments on both synthetic data and real-world data demonstrate that E$^3$NeRF can effectively learn a sharp NeRF from blurry images, especially in non-uniform motion and low-light scenes.  
  </ol>  
</details>  
  
  



