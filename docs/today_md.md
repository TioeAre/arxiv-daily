<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#A-Robust-and-Efficient-Visual-Inertial-Initialization-with-Probabilistic-Normal-Epipolar-Constraint>A Robust and Efficient Visual-Inertial Initialization with Probabilistic Normal Epipolar Constraint</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Context-Based-Visual-Language-Place-Recognition>Context-Based Visual-Language Place Recognition</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Content-Aware-Radiance-Fields:-Aligning-Model-Complexity-with-Scene-Intricacy-Through-Learned-Bitwidth-Quantization>Content-Aware Radiance Fields: Aligning Model Complexity with Scene Intricacy Through Learned Bitwidth Quantization</a></li>
        <li><a href=#Evaluation-of-strategies-for-efficient-rate-distortion-NeRF-streaming>Evaluation of strategies for efficient rate-distortion NeRF streaming</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [A Robust and Efficient Visual-Inertial Initialization with Probabilistic Normal Epipolar Constraint](http://arxiv.org/abs/2410.19473)  
Changshi Mu, Daquan Feng, Qi Zheng, Yuan Zhuang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Accurate and robust initialization is essential for Visual-Inertial Odometry (VIO), as poor initialization can severely degrade pose accuracy. During initialization, it is crucial to estimate parameters such as accelerometer bias, gyroscope bias, initial velocity, and gravity, etc. The IMU sensor requires precise estimation of gyroscope bias because gyroscope bias affects rotation, velocity and position. Most existing VIO initialization methods adopt Structure from Motion (SfM) to solve for gyroscope bias. However, SfM is not stable and efficient enough in fast motion or degenerate scenes. To overcome these limitations, we extended the rotation-translation-decoupling framework by adding new uncertainty parameters and optimization modules. First, we adopt a gyroscope bias optimizer that incorporates probabilistic normal epipolar constraints. Second, we fuse IMU and visual measurements to solve for velocity, gravity, and scale efficiently. Finally, we design an additional refinement module that effectively diminishes gravity and scale errors. Extensive initialization tests on the EuRoC dataset show that our method reduces the gyroscope bias and rotation estimation error by an average of 16% and 4% respectively. It also significantly reduces the gravity error, with an average reduction of 29%.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Context-Based Visual-Language Place Recognition](http://arxiv.org/abs/2410.19341)  
[[code](https://github.com/woo-soojin/context-based-vlpr)]  
Soojin Woo, Seong-Woo Kim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In vision-based robot localization and SLAM, Visual Place Recognition (VPR) is essential. This paper addresses the problem of VPR, which involves accurately recognizing the location corresponding to a given query image. A popular approach to vision-based place recognition relies on low-level visual features. Despite significant progress in recent years, place recognition based on low-level visual features is challenging when there are changes in scene appearance. To address this, end-to-end training approaches have been proposed to overcome the limitations of hand-crafted features. However, these approaches still fail under drastic changes and require large amounts of labeled data to train models, presenting a significant limitation. Methods that leverage high-level semantic information, such as objects or categories, have been proposed to handle variations in appearance. In this paper, we introduce a novel VPR approach that remains robust to scene changes and does not require additional training. Our method constructs semantic image descriptors by extracting pixel-level embeddings using a zero-shot, language-driven semantic segmentation model. We validate our approach in challenging place recognition scenarios using real-world public dataset. The experiments demonstrate that our method outperforms non-learned image representation techniques and off-the-shelf convolutional neural network (CNN) descriptors. Our code is available at https: //github.com/woo-soojin/context-based-vlpr.  
  </ol>  
</details>  
  
  



## NeRF  

### [Content-Aware Radiance Fields: Aligning Model Complexity with Scene Intricacy Through Learned Bitwidth Quantization](http://arxiv.org/abs/2410.19483)  
[[code](https://github.com/weihangliu2024/content_aware_nerf)]  
Weihang Liu, Xue Xian Zheng, Jingyi Yu, Xin Lou  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The recent popular radiance field models, exemplified by Neural Radiance Fields (NeRF), Instant-NGP and 3D Gaussian Splat?ting, are designed to represent 3D content by that training models for each individual scene. This unique characteristic of scene representation and per-scene training distinguishes radiance field models from other neural models, because complex scenes necessitate models with higher representational capacity and vice versa. In this paper, we propose content?aware radiance fields, aligning the model complexity with the scene intricacies through Adversarial Content-Aware Quantization (A-CAQ). Specifically, we make the bitwidth of parameters differentiable and train?able, tailored to the unique characteristics of specific scenes and requirements. The proposed framework has been assessed on Instant-NGP, a well-known NeRF variant and evaluated using various datasets. Experimental results demonstrate a notable reduction in computational complexity, while preserving the requisite reconstruction and rendering quality, making it beneficial for practical deployment of radiance fields models. Codes are available at https://github.com/WeihangLiu2024/Content_Aware_NeRF.  
  </ol>  
</details>  
**comments**: accepted by ECCV2024  
  
### [Evaluation of strategies for efficient rate-distortion NeRF streaming](http://arxiv.org/abs/2410.19459)  
Pedro Martin, António Rodrigues, João Ascenso, Maria Paula Queluz  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) have revolutionized the field of 3D visual representation by enabling highly realistic and detailed scene reconstructions from a sparse set of images. NeRF uses a volumetric functional representation that maps 3D points to their corresponding colors and opacities, allowing for photorealistic view synthesis from arbitrary viewpoints. Despite its advancements, the efficient streaming of NeRF content remains a significant challenge due to the large amount of data involved. This paper investigates the rate-distortion performance of two NeRF streaming strategies: pixel-based and neural network (NN) parameter-based streaming. While in the former, images are coded and then transmitted throughout the network, in the latter, the respective NeRF model parameters are coded and transmitted instead. This work also highlights the trade-offs in complexity and performance, demonstrating that the NN parameter-based strategy generally offers superior efficiency, making it suitable for one-to-many streaming scenarios.  
  </ol>  
</details>  
  
  



