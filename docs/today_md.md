<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Geometric-Prior-Guided-Neural-Implicit-Surface-Reconstruction-in-the-Wild>Geometric Prior-Guided Neural Implicit Surface Reconstruction in the Wild</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Enabling-Privacy-Aware-AI-Based-Ergonomic-Analysis>Enabling Privacy-Aware AI-Based Ergonomic Analysis</a></li>
        <li><a href=#My-Emotion-on-your-face:-The-use-of-Facial-Keypoint-Detection-to-preserve-Emotions-in-Latent-Space-Editing>My Emotion on your face: The use of Facial Keypoint Detection to preserve Emotions in Latent Space Editing</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Boosting-Global-Local-Feature-Matching-via-Anomaly-Synthesis-for-Multi-Class-Point-Cloud-Anomaly-Detection>Boosting Global-Local Feature Matching via Anomaly Synthesis for Multi-Class Point Cloud Anomaly Detection</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#TUM2TWIN:-Introducing-the-Large-Scale-Multimodal-Urban-Digital-Twin-Benchmark-Dataset>TUM2TWIN: Introducing the Large-Scale Multimodal Urban Digital Twin Benchmark Dataset</a></li>
        <li><a href=#Geometric-Prior-Guided-Neural-Implicit-Surface-Reconstruction-in-the-Wild>Geometric Prior-Guided Neural Implicit Surface Reconstruction in the Wild</a></li>
        <li><a href=#NeuGen:-Amplifying-the-'Neural'-in-Neural-Radiance-Fields-for-Domain-Generalization>NeuGen: Amplifying the 'Neural' in Neural Radiance Fields for Domain Generalization</a></li>
        <li><a href=#3D-Characterization-of-Smoke-Plume-Dispersion-Using-Multi-View-Drone-Swarm>3D Characterization of Smoke Plume Dispersion Using Multi-View Drone Swarm</a></li>
        <li><a href=#FlexNeRFer:-A-Multi-Dataflow,-Adaptive-Sparsity-Aware-Accelerator-for-On-Device-NeRF-Rendering>FlexNeRFer: A Multi-Dataflow, Adaptive Sparsity-Aware Accelerator for On-Device NeRF Rendering</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Geometric Prior-Guided Neural Implicit Surface Reconstruction in the Wild](http://arxiv.org/abs/2505.07373)  
Lintao Xiang, Hongpei Zheng, Bailin Deng, Hujun Yin  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural implicit surface reconstruction using volume rendering techniques has recently achieved significant advancements in creating high-fidelity surfaces from multiple 2D images. However, current methods primarily target scenes with consistent illumination and struggle to accurately reconstruct 3D geometry in uncontrolled environments with transient occlusions or varying appearances. While some neural radiance field (NeRF)-based variants can better manage photometric variations and transient objects in complex scenes, they are designed for novel view synthesis rather than precise surface reconstruction due to limited surface constraints. To overcome this limitation, we introduce a novel approach that applies multiple geometric constraints to the implicit surface optimization process, enabling more accurate reconstructions from unconstrained image collections. First, we utilize sparse 3D points from structure-from-motion (SfM) to refine the signed distance function estimation for the reconstructed surface, with a displacement compensation to accommodate noise in the sparse points. Additionally, we employ robust normal priors derived from a normal predictor, enhanced by edge prior filtering and multi-view consistency constraints, to improve alignment with the actual surface geometry. Extensive testing on the Heritage-Recon benchmark and other datasets has shown that the proposed method can accurately reconstruct surfaces from in-the-wild images, yielding geometries with superior accuracy and granularity compared to existing techniques. Our approach enables high-quality 3D reconstruction of various landmarks, making it applicable to diverse scenarios such as digital preservation of cultural heritage sites.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Enabling Privacy-Aware AI-Based Ergonomic Analysis](http://arxiv.org/abs/2505.07306)  
Sander De Coninck, Emilio Gamba, Bart Van Doninck, Abdellatif Bey-Temsamani, Sam Leroux, Pieter Simoens  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Musculoskeletal disorders (MSDs) are a leading cause of injury and productivity loss in the manufacturing industry, incurring substantial economic costs. Ergonomic assessments can mitigate these risks by identifying workplace adjustments that improve posture and reduce strain. Camera-based systems offer a non-intrusive, cost-effective method for continuous ergonomic tracking, but they also raise significant privacy concerns. To address this, we propose a privacy-aware ergonomic assessment framework utilizing machine learning techniques. Our approach employs adversarial training to develop a lightweight neural network that obfuscates video data, preserving only the essential information needed for human pose estimation. This obfuscation ensures compatibility with standard pose estimation algorithms, maintaining high accuracy while protecting privacy. The obfuscated video data is transmitted to a central server, where state-of-the-art keypoint detection algorithms extract body landmarks. Using multi-view integration, 3D keypoints are reconstructed and evaluated with the Rapid Entire Body Assessment (REBA) method. Our system provides a secure, effective solution for ergonomic monitoring in industrial environments, addressing both privacy and workplace safety concerns.  
  </ol>  
</details>  
**comments**: Accepted and presented at the 35th CIRP Design conference  
  
### [My Emotion on your face: The use of Facial Keypoint Detection to preserve Emotions in Latent Space Editing](http://arxiv.org/abs/2505.06436)  
Jingrui He, Andrew Stephen McGough  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Generative Adversarial Network approaches such as StyleGAN/2 provide two key benefits: the ability to generate photo-realistic face images and possessing a semantically structured latent space from which these images are created. Many approaches have emerged for editing images derived from vectors in the latent space of a pre-trained StyleGAN/2 models by identifying semantically meaningful directions (e.g., gender or age) in the latent space. By moving the vector in a specific direction, the ideal result would only change the target feature while preserving all the other features. Providing an ideal data augmentation approach for gesture research as it could be used to generate numerous image variations whilst keeping the facial expressions intact. However, entanglement issues, where changing one feature inevitably affects other features, impacts the ability to preserve facial expressions. To address this, we propose the use of an addition to the loss function of a Facial Keypoint Detection model to restrict changes to the facial expressions. Building on top of an existing model, adding the proposed Human Face Landmark Detection (HFLD) loss, provided by a pre-trained Facial Keypoint Detection model, to the original loss function. We quantitatively and qualitatively evaluate the existing and our extended model, showing the effectiveness of our approach in addressing the entanglement issue and maintaining the facial expression. Our approach achieves up to 49% reduction in the change of emotion in our experiments. Moreover, we show the benefit of our approach by comparing with state-of-the-art models. By increasing the ability to preserve the facial gesture and expression during facial transformation, we present a way to create human face images with fixed expression but different appearances, making it a reliable data augmentation approach for Facial Gesture and Expression research.  
  </ol>  
</details>  
**comments**: Submitted to 2nd International Workshop on Synthetic Data for Face
  and Gesture Analysis at IEEE FG 2025  
  
  



## Image Matching  

### [Boosting Global-Local Feature Matching via Anomaly Synthesis for Multi-Class Point Cloud Anomaly Detection](http://arxiv.org/abs/2505.07375)  
[[code](https://github.com/hustCYQ/GLFM-Multi-class-3DAD)]  
Yuqi Cheng, Yunkang Cao, Dongfang Wang, Weiming Shen, Wenlong Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Point cloud anomaly detection is essential for various industrial applications. The huge computation and storage costs caused by the increasing product classes limit the application of single-class unsupervised methods, necessitating the development of multi-class unsupervised methods. However, the feature similarity between normal and anomalous points from different class data leads to the feature confusion problem, which greatly hinders the performance of multi-class methods. Therefore, we introduce a multi-class point cloud anomaly detection method, named GLFM, leveraging global-local feature matching to progressively separate data that are prone to confusion across multiple classes. Specifically, GLFM is structured into three stages: Stage-I proposes an anomaly synthesis pipeline that stretches point clouds to create abundant anomaly data that are utilized to adapt the point cloud feature extractor for better feature representation. Stage-II establishes the global and local memory banks according to the global and local feature distributions of all the training data, weakening the impact of feature confusion on the establishment of the memory bank. Stage-III implements anomaly detection of test data leveraging its feature distance from global and local memory banks. Extensive experiments on the MVTec 3D-AD, Real3D-AD and actual industry parts dataset showcase our proposed GLFM's superior point cloud anomaly detection performance. The code is available at https://github.com/hustCYQ/GLFM-Multi-class-3DAD.  
  </ol>  
</details>  
**comments**: 12 pages, 12 figures  
  
  



## NeRF  

### [TUM2TWIN: Introducing the Large-Scale Multimodal Urban Digital Twin Benchmark Dataset](http://arxiv.org/abs/2505.07396)  
Olaf Wysocki, Benedikt Schwab, Manoj Kumar Biswanath, Qilin Zhang, Jingwei Zhu, Thomas Froech, Medhini Heeramaglore, Ihab Hijazi, Khaoula Kanna, Mathias Pechinger, Zhaiyu Chen, Yao Sun, Alejandro Rueda Segura, Ziyang Xu, Omar AbdelGafar, Mansour Mehranfar, Chandan Yeshwanth, Yueh-Cheng Liu, Hadi Yazdi, Jiapan Wang, Stefan Auer, Katharina Anders, Klaus Bogenberger, Andre Borrmann, Angela Dai, Ludwig Hoegner, Christoph Holst, Thomas H. Kolbe, Ferdinand Ludwig, Matthias Nießner, Frank Petzold, Xiao Xiang Zhu, Boris Jutzi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Urban Digital Twins (UDTs) have become essential for managing cities and integrating complex, heterogeneous data from diverse sources. Creating UDTs involves challenges at multiple process stages, including acquiring accurate 3D source data, reconstructing high-fidelity 3D models, maintaining models' updates, and ensuring seamless interoperability to downstream tasks. Current datasets are usually limited to one part of the processing chain, hampering comprehensive UDTs validation. To address these challenges, we introduce the first comprehensive multimodal Urban Digital Twin benchmark dataset: TUM2TWIN. This dataset includes georeferenced, semantically aligned 3D models and networks along with various terrestrial, mobile, aerial, and satellite observations boasting 32 data subsets over roughly 100,000 $m^2$ and currently 767 GB of data. By ensuring georeferenced indoor-outdoor acquisition, high accuracy, and multimodal data integration, the benchmark supports robust analysis of sensors and the development of advanced reconstruction methods. Additionally, we explore downstream tasks demonstrating the potential of TUM2TWIN, including novel view synthesis of NeRF and Gaussian Splatting, solar potential analysis, point cloud semantic segmentation, and LoD3 building reconstruction. We are convinced this contribution lays a foundation for overcoming current limitations in UDT creation, fostering new research directions and practical solutions for smarter, data-driven urban environments. The project is available under: https://tum2t.win  
  </ol>  
</details>  
**comments**: Submitted to the ISPRS Journal of Photogrammetry and Remote Sensing  
  
### [Geometric Prior-Guided Neural Implicit Surface Reconstruction in the Wild](http://arxiv.org/abs/2505.07373)  
Lintao Xiang, Hongpei Zheng, Bailin Deng, Hujun Yin  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural implicit surface reconstruction using volume rendering techniques has recently achieved significant advancements in creating high-fidelity surfaces from multiple 2D images. However, current methods primarily target scenes with consistent illumination and struggle to accurately reconstruct 3D geometry in uncontrolled environments with transient occlusions or varying appearances. While some neural radiance field (NeRF)-based variants can better manage photometric variations and transient objects in complex scenes, they are designed for novel view synthesis rather than precise surface reconstruction due to limited surface constraints. To overcome this limitation, we introduce a novel approach that applies multiple geometric constraints to the implicit surface optimization process, enabling more accurate reconstructions from unconstrained image collections. First, we utilize sparse 3D points from structure-from-motion (SfM) to refine the signed distance function estimation for the reconstructed surface, with a displacement compensation to accommodate noise in the sparse points. Additionally, we employ robust normal priors derived from a normal predictor, enhanced by edge prior filtering and multi-view consistency constraints, to improve alignment with the actual surface geometry. Extensive testing on the Heritage-Recon benchmark and other datasets has shown that the proposed method can accurately reconstruct surfaces from in-the-wild images, yielding geometries with superior accuracy and granularity compared to existing techniques. Our approach enables high-quality 3D reconstruction of various landmarks, making it applicable to diverse scenarios such as digital preservation of cultural heritage sites.  
  </ol>  
</details>  
  
### [NeuGen: Amplifying the 'Neural' in Neural Radiance Fields for Domain Generalization](http://arxiv.org/abs/2505.06894)  
Ahmed Qazi, Abdul Basit, Asim Iqbal  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) have significantly advanced the field of novel view synthesis, yet their generalization across diverse scenes and conditions remains challenging. Addressing this, we propose the integration of a novel brain-inspired normalization technique Neural Generalization (NeuGen) into leading NeRF architectures which include MVSNeRF and GeoNeRF. NeuGen extracts the domain-invariant features, thereby enhancing the models' generalization capabilities. It can be seamlessly integrated into NeRF architectures and cultivates a comprehensive feature set that significantly improves accuracy and robustness in image rendering. Through this integration, NeuGen shows improved performance on benchmarks on diverse datasets across state-of-the-art NeRF architectures, enabling them to generalize better across varied scenes. Our comprehensive evaluations, both quantitative and qualitative, confirm that our approach not only surpasses existing models in generalizability but also markedly improves rendering quality. Our work exemplifies the potential of merging neuroscientific principles with deep learning frameworks, setting a new precedent for enhanced generalizability and efficiency in novel view synthesis. A demo of our study is available at https://neugennerf.github.io.  
  </ol>  
</details>  
**comments**: 18 pages, 6 figures  
  
### [3D Characterization of Smoke Plume Dispersion Using Multi-View Drone Swarm](http://arxiv.org/abs/2505.06638)  
Nikil Krishnakumar, Shashank Sharma, Srijan Kumar Pal, Jiarong Hong  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This study presents an advanced multi-view drone swarm imaging system for the three-dimensional characterization of smoke plume dispersion dynamics. The system comprises a manager drone and four worker drones, each equipped with high-resolution cameras and precise GPS modules. The manager drone uses image feedback to autonomously detect and position itself above the plume, then commands the worker drones to orbit the area in a synchronized circular flight pattern, capturing multi-angle images. The camera poses of these images are first estimated, then the images are grouped in batches and processed using Neural Radiance Fields (NeRF) to generate high-resolution 3D reconstructions of plume dynamics over time. Field tests demonstrated the ability of the system to capture critical plume characteristics including volume dynamics, wind-driven directional shifts, and lofting behavior at a temporal resolution of about 1 s. The 3D reconstructions generated by this system provide unique field data for enhancing the predictive models of smoke plume dispersion and fire spread. Broadly, the drone swarm system offers a versatile platform for high resolution measurements of pollutant emissions and transport in wildfires, volcanic eruptions, prescribed burns, and industrial processes, ultimately supporting more effective fire control decisions and mitigating wildfire risks.  
  </ol>  
</details>  
**comments**: 10 pages, 8 figures  
  
### [FlexNeRFer: A Multi-Dataflow, Adaptive Sparsity-Aware Accelerator for On-Device NeRF Rendering](http://arxiv.org/abs/2505.06504)  
Seock-Hwan Noh, Banseok Shin, Jeik Choi, Seungpyo Lee, Jaeha Kung, Yeseong Kim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF), an AI-driven approach for 3D view reconstruction, has demonstrated impressive performance, sparking active research across fields. As a result, a range of advanced NeRF models has emerged, leading on-device applications to increasingly adopt NeRF for highly realistic scene reconstructions. With the advent of diverse NeRF models, NeRF-based applications leverage a variety of NeRF frameworks, creating the need for hardware capable of efficiently supporting these models. However, GPUs fail to meet the performance, power, and area (PPA) cost demanded by these on-device applications, or are specialized for specific NeRF algorithms, resulting in lower efficiency when applied to other NeRF models. To address this limitation, in this work, we introduce FlexNeRFer, an energy-efficient versatile NeRF accelerator. The key components enabling the enhancement of FlexNeRFer include: i) a flexible network-on-chip (NoC) supporting multi-dataflow and sparsity on precision-scalable MAC array, and ii) efficient data storage using an optimal sparsity format based on the sparsity ratio and precision modes. To evaluate the effectiveness of FlexNeRFer, we performed a layout implementation using 28nm CMOS technology. Our evaluation shows that FlexNeRFer achieves 8.2~243.3x speedup and 24.1~520.3x improvement in energy efficiency over a GPU (i.e., NVIDIA RTX 2080 Ti), while demonstrating 4.2~86.9x speedup and 2.3~47.5x improvement in energy efficiency compared to a state-of-the-art NeRF accelerator (i.e., NeuRex).  
  </ol>  
</details>  
**comments**: Accepted for publication at the 52nd IEEE/ACM International Symposium
  on Computer Architecture (ISCA-52), 2025  
  
  



