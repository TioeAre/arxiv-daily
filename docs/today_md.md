<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#GV-Bench:-Benchmarking-Local-Feature-Matching-for-Geometric-Verification-of-Long-term-Loop-Closure-Detection>GV-Bench: Benchmarking Local Feature Matching for Geometric Verification of Long-term Loop Closure Detection</a></li>
        <li><a href=#EndoFinder:-Online-Image-Retrieval-for-Explainable-Colorectal-Polyp-Diagnosis>EndoFinder: Online Image Retrieval for Explainable Colorectal Polyp Diagnosis</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#GV-Bench:-Benchmarking-Local-Feature-Matching-for-Geometric-Verification-of-Long-term-Loop-Closure-Detection>GV-Bench: Benchmarking Local Feature Matching for Geometric Verification of Long-term Loop Closure Detection</a></li>
        <li><a href=#REMM:Rotation-Equivariant-Framework-for-End-to-End-Multimodal-Image-Matching>REMM:Rotation-Equivariant Framework for End-to-End Multimodal Image Matching</a></li>
        <li><a href=#A-Self-Correcting-Strategy-of-the-Digital-Volume-Correlation-Displacement-Field-Based-on-Image-Matching:-Application-to-Poor-Speckles-Quality-and-Complex-Large-Deformation>A Self-Correcting Strategy of the Digital Volume Correlation Displacement Field Based on Image Matching: Application to Poor Speckles Quality and Complex-Large Deformation</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Motion-Oriented-Compositional-Neural-Radiance-Fields-for-Monocular-Dynamic-Human-Modeling>Motion-Oriented Compositional Neural Radiance Fields for Monocular Dynamic Human Modeling</a></li>
        <li><a href=#IPA-NeRF:-Illusory-Poisoning-Attack-Against-Neural-Radiance-Fields>IPA-NeRF: Illusory Poisoning Attack Against Neural Radiance Fields</a></li>
        <li><a href=#DreamCatalyst:-Fast-and-High-Quality-3D-Editing-via-Controlling-Editability-and-Identity-Preservation>DreamCatalyst: Fast and High-Quality 3D Editing via Controlling Editability and Identity Preservation</a></li>
        <li><a href=#Evaluating-geometric-accuracy-of-NeRF-reconstructions-compared-to-SLAM-method>Evaluating geometric accuracy of NeRF reconstructions compared to SLAM method</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [GV-Bench: Benchmarking Local Feature Matching for Geometric Verification of Long-term Loop Closure Detection](http://arxiv.org/abs/2407.11736)  
[[code](https://github.com/jarvisyjw/gv-bench)]  
Jingwen Yu, Hanjing Ye, Jianhao Jiao, Ping Tan, Hong Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual loop closure detection is an important module in visual simultaneous localization and mapping (SLAM), which associates current camera observation with previously visited places. Loop closures correct drifts in trajectory estimation to build a globally consistent map. However, a false loop closure can be fatal, so verification is required as an additional step to ensure robustness by rejecting the false positive loops. Geometric verification has been a well-acknowledged solution that leverages spatial clues provided by local feature matching to find true positives. Existing feature matching methods focus on homography and pose estimation in long-term visual localization, lacking references for geometric verification. To fill the gap, this paper proposes a unified benchmark targeting geometric verification of loop closure detection under long-term conditional variations. Furthermore, we evaluate six representative local feature matching methods (handcrafted and learning-based) under the benchmark, with in-depth analysis for limitations and future directions.  
  </ol>  
</details>  
**comments**: 9 pages, 11 figures, Accepted by IROS(2024)  
  
### [EndoFinder: Online Image Retrieval for Explainable Colorectal Polyp Diagnosis](http://arxiv.org/abs/2407.11401)  
Ruijie Yang, Yan Zhu, Peiyao Fu, Yizhe Zhang, Zhihua Wang, Quanlin Li, Pinghong Zhou, Xian Yang, Shuo Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Determining the necessity of resecting malignant polyps during colonoscopy screen is crucial for patient outcomes, yet challenging due to the time-consuming and costly nature of histopathology examination. While deep learning-based classification models have shown promise in achieving optical biopsy with endoscopic images, they often suffer from a lack of explainability. To overcome this limitation, we introduce EndoFinder, a content-based image retrieval framework to find the 'digital twin' polyp in the reference database given a newly detected polyp. The clinical semantics of the new polyp can be inferred referring to the matched ones. EndoFinder pioneers a polyp-aware image encoder that is pre-trained on a large polyp dataset in a self-supervised way, merging masked image modeling with contrastive learning. This results in a generic embedding space ready for different downstream clinical tasks based on image retrieval. We validate the framework on polyp re-identification and optical biopsy tasks, with extensive experiments demonstrating that EndoFinder not only achieves explainable diagnostics but also matches the performance of supervised classification models. EndoFinder's reliance on image retrieval has the potential to support diverse downstream decision-making tasks during real-time colonoscopy procedures.  
  </ol>  
</details>  
**comments**: MICCAI 2024  
  
  



## Image Matching  

### [GV-Bench: Benchmarking Local Feature Matching for Geometric Verification of Long-term Loop Closure Detection](http://arxiv.org/abs/2407.11736)  
[[code](https://github.com/jarvisyjw/gv-bench)]  
Jingwen Yu, Hanjing Ye, Jianhao Jiao, Ping Tan, Hong Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual loop closure detection is an important module in visual simultaneous localization and mapping (SLAM), which associates current camera observation with previously visited places. Loop closures correct drifts in trajectory estimation to build a globally consistent map. However, a false loop closure can be fatal, so verification is required as an additional step to ensure robustness by rejecting the false positive loops. Geometric verification has been a well-acknowledged solution that leverages spatial clues provided by local feature matching to find true positives. Existing feature matching methods focus on homography and pose estimation in long-term visual localization, lacking references for geometric verification. To fill the gap, this paper proposes a unified benchmark targeting geometric verification of loop closure detection under long-term conditional variations. Furthermore, we evaluate six representative local feature matching methods (handcrafted and learning-based) under the benchmark, with in-depth analysis for limitations and future directions.  
  </ol>  
</details>  
**comments**: 9 pages, 11 figures, Accepted by IROS(2024)  
  
### [REMM:Rotation-Equivariant Framework for End-to-End Multimodal Image Matching](http://arxiv.org/abs/2407.11637)  
Han Nie, Bin Luo, Jun Liu, Zhitao Fu, Weixing Liu, Xin Su  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present REMM, a rotation-equivariant framework for end-to-end multimodal image matching, which fully encodes rotational differences of descriptors in the whole matching pipeline. Previous learning-based methods mainly focus on extracting modal-invariant descriptors, while consistently ignoring the rotational invariance. In this paper, we demonstrate that our REMM is very useful for multimodal image matching, including multimodal feature learning module and cyclic shift module. We first learn modal-invariant features through the multimodal feature learning module. Then, we design the cyclic shift module to rotationally encode the descriptors, greatly improving the performance of rotation-equivariant matching, which makes them robust to any angle. To validate our method, we establish a comprehensive rotation and scale-matching benchmark for evaluating the anti-rotation performance of multimodal images, which contains a combination of multi-angle and multi-scale transformations from four publicly available datasets. Extensive experiments show that our method outperforms existing methods in benchmarking and generalizes well to independent datasets. Additionally, we conducted an in-depth analysis of the key components of the REMM to validate the improvements brought about by the cyclic shift module. Code and dataset at https://github.com/HanNieWHU/REMM.  
  </ol>  
</details>  
**comments**: 13 pages, 13 figures  
  
### [A Self-Correcting Strategy of the Digital Volume Correlation Displacement Field Based on Image Matching: Application to Poor Speckles Quality and Complex-Large Deformation](http://arxiv.org/abs/2407.11287)  
Chengsheng Li, Zhijun Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Digital Volume Correlation (DVC) is widely used for the analysis of three-dimensional displacement and strain fields based on CT scans. However, the applicability of DVC methods is limited when it comes to geomaterials: CT speckles are directly correlated with the material's microstructure, and the speckle structure cannot be artificially altered, with generally poor speckle quality. Additionally, most geomaterials exhibit elastoplastic properties and will undergo complex-large deformations under external loading, sometimes leading to strain localization phenomena. These factors contribute to inaccuracies in the displacement field obtained through DVC, and at present, there is a shortage of correction methods and accuracy assessment techniques for the displacement field. If the accuracy of the DVC displacement field is sufficiently high, the gray residue of the two volume images before and after deformation should be minimal, utilizing this characteristic to develop a correction method for the displacement field is feasible. The proposed self-correcting strategy of the DVC displacement field based on image matching, which from the experimental measurement error. We demonstrated the effectiveness of the proposed method by CT triaxial tests of granite residual soil. Without adding other parameters or adjusting the original parameters of DVC, the gray residue showed that the proposed method can effectively improve the accuracy of the displacement field. Additionally, the accuracy evaluation method can reasonably estimate the accuracy of the displacement field. The proposed method can effectively improve the accuracy of DVC three-dimensional displacement field for the state of speckles with poor quality and complex-large deformation.  
  </ol>  
</details>  
  
  



## NeRF  

### [Motion-Oriented Compositional Neural Radiance Fields for Monocular Dynamic Human Modeling](http://arxiv.org/abs/2407.11962)  
Jaehyeok Kim, Dongyoon Wee, Dan Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper introduces Motion-oriented Compositional Neural Radiance Fields (MoCo-NeRF), a framework designed to perform free-viewpoint rendering of monocular human videos via novel non-rigid motion modeling approach. In the context of dynamic clothed humans, complex cloth dynamics generate non-rigid motions that are intrinsically distinct from skeletal articulations and critically important for the rendering quality. The conventional approach models non-rigid motions as spatial (3D) deviations in addition to skeletal transformations. However, it is either time-consuming or challenging to achieve optimal quality due to its high learning complexity without a direct supervision. To target this problem, we propose a novel approach of modeling non-rigid motions as radiance residual fields to benefit from more direct color supervision in the rendering and utilize the rigid radiance fields as a prior to reduce the complexity of the learning process. Our approach utilizes a single multiresolution hash encoding (MHE) to concurrently learn the canonical T-pose representation from rigid skeletal motions and the radiance residual field for non-rigid motions. Additionally, to further improve both training efficiency and usability, we extend MoCo-NeRF to support simultaneous training of multiple subjects within a single framework, thanks to our effective design for modeling non-rigid motions. This scalability is achieved through the integration of a global MHE and learnable identity codes in addition to multiple local MHEs. We present extensive results on ZJU-MoCap and MonoCap, clearly demonstrating state-of-the-art performance in both single- and multi-subject settings. The code and model will be made publicly available at the project page: https://stevejaehyeok.github.io/publications/moco-nerf.  
  </ol>  
</details>  
**comments**: Accepted by ECCV2024  
  
### [IPA-NeRF: Illusory Poisoning Attack Against Neural Radiance Fields](http://arxiv.org/abs/2407.11921)  
Wenxiang Jiang, Hanwei Zhang, Shuo Zhao, Zhongwen Guo, Hao Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Field (NeRF) represents a significant advancement in computer vision, offering implicit neural network-based scene representation and novel view synthesis capabilities. Its applications span diverse fields including robotics, urban mapping, autonomous navigation, virtual reality/augmented reality, etc., some of which are considered high-risk AI applications. However, despite its widespread adoption, the robustness and security of NeRF remain largely unexplored. In this study, we contribute to this area by introducing the Illusory Poisoning Attack against Neural Radiance Fields (IPA-NeRF). This attack involves embedding a hidden backdoor view into NeRF, allowing it to produce predetermined outputs, i.e. illusory, when presented with the specified backdoor view while maintaining normal performance with standard inputs. Our attack is specifically designed to deceive users or downstream models at a particular position while ensuring that any abnormalities in NeRF remain undetectable from other viewpoints. Experimental results demonstrate the effectiveness of our Illusory Poisoning Attack, successfully presenting the desired illusory on the specified viewpoint without impacting other views. Notably, we achieve this attack by introducing small perturbations solely to the training set. The code can be found at https://github.com/jiang-wenxiang/IPA-NeRF.  
  </ol>  
</details>  
  
### [DreamCatalyst: Fast and High-Quality 3D Editing via Controlling Editability and Identity Preservation](http://arxiv.org/abs/2407.11394)  
[[code](https://github.com/kaist-cvml-lab/DreamCatalyst)]  
Jiwook Kim, Seonho Lee, Jaeyo Shin, Jiho Choi, Hyunjung Shim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Score distillation sampling (SDS) has emerged as an effective framework in text-driven 3D editing tasks due to its inherent 3D consistency. However, existing SDS-based 3D editing methods suffer from extensive training time and lead to low-quality results, primarily because these methods deviate from the sampling dynamics of diffusion models. In this paper, we propose DreamCatalyst, a novel framework that interprets SDS-based editing as a diffusion reverse process. Our objective function considers the sampling dynamics, thereby making the optimization process of DreamCatalyst an approximation of the diffusion reverse process in editing tasks. DreamCatalyst aims to reduce training time and improve editing quality. DreamCatalyst presents two modes: (1) a faster mode, which edits the NeRF scene in only about 25 minutes, and (2) a high-quality mode, which produces superior results in less than 70 minutes. Specifically, our high-quality mode outperforms current state-of-the-art NeRF editing methods both in terms of speed and quality. See more extensive results on our project page: https://dream-catalyst.github.io.  
  </ol>  
</details>  
  
### [Evaluating geometric accuracy of NeRF reconstructions compared to SLAM method](http://arxiv.org/abs/2407.11238)  
Adam Korycki, Colleen Josephson, Steve McGuire  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    As Neural Radiance Field (NeRF) implementations become faster, more efficient and accurate, their applicability to real world mapping tasks becomes more accessible. Traditionally, 3D mapping, or scene reconstruction, has relied on expensive LiDAR sensing. Photogrammetry can perform image-based 3D reconstruction but is computationally expensive and requires extremely dense image representation to recover complex geometry and photorealism. NeRFs perform 3D scene reconstruction by training a neural network on sparse image and pose data, achieving superior results to photogrammetry with less input data. This paper presents an evaluation of two NeRF scene reconstructions for the purpose of estimating the diameter of a vertical PVC cylinder. One of these are trained on commodity iPhone data and the other is trained on robot-sourced imagery and poses. This neural-geometry is compared to state-of-the-art lidar-inertial SLAM in terms of scene noise and metric-accuracy.  
  </ol>  
</details>  
  
  



