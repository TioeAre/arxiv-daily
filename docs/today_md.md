<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#FOR:-Finetuning-for-Object-Level-Open-Vocabulary-Image-Retrieval>FOR: Finetuning for Object Level Open Vocabulary Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#MINIMA:-Modality-Invariant-Image-Matching>MINIMA: Modality Invariant Image Matching</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Learning-Radiance-Fields-from-a-Single-Snapshot-Compressive-Image>Learning Radiance Fields from a Single Snapshot Compressive Image</a></li>
        <li><a href=#BeSplat----Gaussian-Splatting-from-a-Single-Blurry-Image-and-Event-Stream>BeSplat -- Gaussian Splatting from a Single Blurry Image and Event Stream</a></li>
        <li><a href=#Generating-Editable-Head-Avatars-with-3D-Gaussian-GANs>Generating Editable Head Avatars with 3D Gaussian GANs</a></li>
        <li><a href=#MVS-GS:-High-Quality-3D-Gaussian-Splatting-Mapping-via-Online-Multi-View-Stereo>MVS-GS: High-Quality 3D Gaussian Splatting Mapping via Online Multi-View Stereo</a></li>
        <li><a href=#Humans-as-a-Calibration-Pattern:-Dynamic-3D-Scene-Reconstruction-from-Unsynchronized-and-Uncalibrated-Videos>Humans as a Calibration Pattern: Dynamic 3D Scene Reconstruction from Unsynchronized and Uncalibrated Videos</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [FOR: Finetuning for Object Level Open Vocabulary Image Retrieval](http://arxiv.org/abs/2412.18806)  
Hila Levi, Guy Heller, Dan Levi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    As working with large datasets becomes standard, the task of accurately retrieving images containing objects of interest by an open set textual query gains practical importance. The current leading approach utilizes a pre-trained CLIP model without any adaptation to the target domain, balancing accuracy and efficiency through additional post-processing. In this work, we propose FOR: Finetuning for Object-centric Open-vocabulary Image Retrieval, which allows finetuning on a target dataset using closed-set labels while keeping the visual-language association crucial for open vocabulary retrieval. FOR is based on two design elements: a specialized decoder variant of the CLIP head customized for the intended task, and its coupling within a multi-objective training framework. Together, these design choices result in a significant increase in accuracy, showcasing improvements of up to 8 mAP@50 points over SoTA across three datasets. Additionally, we demonstrate that FOR is also effective in a semi-supervised setting, achieving impressive results even when only a small portion of the dataset is labeled.  
  </ol>  
</details>  
**comments**: WACV 2025  
  
  



## Image Matching  

### [MINIMA: Modality Invariant Image Matching](http://arxiv.org/abs/2412.19412)  
[[code](https://github.com/LSXI7/MINIMA)]  
Xingyu Jiang, Jiangwei Ren, Zizhuo Li, Xin Zhou, Dingkang Liang, Xiang Bai  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image matching for both cross-view and cross-modality plays a critical role in multimodal perception. In practice, the modality gap caused by different imaging systems/styles poses great challenges to the matching task. Existing works try to extract invariant features for specific modalities and train on limited datasets, showing poor generalization. In this paper, we present MINIMA, a unified image matching framework for multiple cross-modal cases. Without pursuing fancy modules, our MINIMA aims to enhance universal performance from the perspective of data scaling up. For such purpose, we propose a simple yet effective data engine that can freely produce a large dataset containing multiple modalities, rich scenarios, and accurate matching labels. Specifically, we scale up the modalities from cheap but rich RGB-only matching data, by means of generative models. Under this setting, the matching labels and rich diversity of the RGB dataset are well inherited by the generated multimodal data. Benefiting from this, we construct MD-syn, a new comprehensive dataset that fills the data gap for general multimodal image matching. With MD-syn, we can directly train any advanced matching pipeline on randomly selected modality pairs to obtain cross-modal ability. Extensive experiments on in-domain and zero-shot matching tasks, including $19$ cross-modal cases, demonstrate that our MINIMA can significantly outperform the baselines and even surpass modality-specific methods. The dataset and code are available at https://github.com/LSXI7/MINIMA .  
  </ol>  
</details>  
**comments**: The dataset and code are available at https://github.com/LSXI7/MINIMA  
  
  



## NeRF  

### [Learning Radiance Fields from a Single Snapshot Compressive Image](http://arxiv.org/abs/2412.19483)  
Yunhao Li, Xiang Liu, Xiaodong Wang, Xin Yuan, Peidong Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we explore the potential of Snapshot Compressive Imaging (SCI) technique for recovering the underlying 3D scene structure from a single temporal compressed image. SCI is a cost-effective method that enables the recording of high-dimensional data, such as hyperspectral or temporal information, into a single image using low-cost 2D imaging sensors. To achieve this, a series of specially designed 2D masks are usually employed, reducing storage and transmission requirements and offering potential privacy protection. Inspired by this, we take one step further to recover the encoded 3D scene information leveraging powerful 3D scene representation capabilities of neural radiance fields (NeRF). Specifically, we propose SCINeRF, in which we formulate the physical imaging process of SCI as part of the training of NeRF, allowing us to exploit its impressive performance in capturing complex scene structures. In addition, we further integrate the popular 3D Gaussian Splatting (3DGS) framework and propose SCISplat to improve 3D scene reconstruction quality and training/rendering speed by explicitly optimizing point clouds into 3D Gaussian representations. To assess the effectiveness of our method, we conduct extensive evaluations using both synthetic data and real data captured by our SCI system. Experimental results demonstrate that our proposed approach surpasses the state-of-the-art methods in terms of image reconstruction and novel view synthesis. Moreover, our method also exhibits the ability to render high frame-rate multi-view consistent images in real time by leveraging SCI and the rendering capabilities of 3DGS. Codes will be available at: https://github.com/WU- CVGL/SCISplat.  
  </ol>  
</details>  
  
### [BeSplat -- Gaussian Splatting from a Single Blurry Image and Event Stream](http://arxiv.org/abs/2412.19370)  
Gopi Raju Matta, Reddypalli Trisha, Kaushik Mitra  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Novel view synthesis has been greatly enhanced by the development of radiance field methods. The introduction of 3D Gaussian Splatting (3DGS) has effectively addressed key challenges, such as long training times and slow rendering speeds, typically associated with Neural Radiance Fields (NeRF), while maintaining high-quality reconstructions. In this work (BeSplat), we demonstrate the recovery of sharp radiance field (Gaussian splats) from a single motion-blurred image and its corresponding event stream. Our method jointly learns the scene representation via Gaussian Splatting and recovers the camera motion through Bezier SE(3) formulation effectively, minimizing discrepancies between synthesized and real-world measurements of both blurry image and corresponding event stream. We evaluate our approach on both synthetic and real datasets, showcasing its ability to render view-consistent, sharp images from the learned radiance field and the estimated camera trajectory. To the best of our knowledge, ours is the first work to address this highly challenging ill-posed problem in a Gaussian Splatting framework with the effective incorporation of temporal information captured using the event stream.  
  </ol>  
</details>  
**comments**: Accepted for publication at EVGEN2025, WACV-25 Workshop  
  
### [Generating Editable Head Avatars with 3D Gaussian GANs](http://arxiv.org/abs/2412.19149)  
[[code](https://github.com/liguohao96/egg3d)]  
Guohao Li, Hongyu Yang, Yifang Men, Di Huang, Weixin Li, Ruijie Yang, Yunhong Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Generating animatable and editable 3D head avatars is essential for various applications in computer vision and graphics. Traditional 3D-aware generative adversarial networks (GANs), often using implicit fields like Neural Radiance Fields (NeRF), achieve photorealistic and view-consistent 3D head synthesis. However, these methods face limitations in deformation flexibility and editability, hindering the creation of lifelike and easily modifiable 3D heads. We propose a novel approach that enhances the editability and animation control of 3D head avatars by incorporating 3D Gaussian Splatting (3DGS) as an explicit 3D representation. This method enables easier illumination control and improved editability. Central to our approach is the Editable Gaussian Head (EG-Head) model, which combines a 3D Morphable Model (3DMM) with texture maps, allowing precise expression control and flexible texture editing for accurate animation while preserving identity. To capture complex non-facial geometries like hair, we use an auxiliary set of 3DGS and tri-plane features. Extensive experiments demonstrate that our approach delivers high-quality 3D-aware synthesis with state-of-the-art controllability. Our code and models are available at https://github.com/liguohao96/EGG3D.  
  </ol>  
</details>  
  
### [MVS-GS: High-Quality 3D Gaussian Splatting Mapping via Online Multi-View Stereo](http://arxiv.org/abs/2412.19130)  
Byeonggwon Lee, Junkyu Park, Khang Truong Giang, Sungho Jo, Soohwan Song  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This study addresses the challenge of online 3D model generation for neural rendering using an RGB image stream. Previous research has tackled this issue by incorporating Neural Radiance Fields (NeRF) or 3D Gaussian Splatting (3DGS) as scene representations within dense SLAM methods. However, most studies focus primarily on estimating coarse 3D scenes rather than achieving detailed reconstructions. Moreover, depth estimation based solely on images is often ambiguous, resulting in low-quality 3D models that lead to inaccurate renderings. To overcome these limitations, we propose a novel framework for high-quality 3DGS modeling that leverages an online multi-view stereo (MVS) approach. Our method estimates MVS depth using sequential frames from a local time window and applies comprehensive depth refinement techniques to filter out outliers, enabling accurate initialization of Gaussians in 3DGS. Furthermore, we introduce a parallelized backend module that optimizes the 3DGS model efficiently, ensuring timely updates with each new keyframe. Experimental results demonstrate that our method outperforms state-of-the-art dense SLAM methods, particularly excelling in challenging outdoor environments.  
  </ol>  
</details>  
**comments**: 7 pages, 6 figures, submitted to IEEE ICRA 2025  
  
### [Humans as a Calibration Pattern: Dynamic 3D Scene Reconstruction from Unsynchronized and Uncalibrated Videos](http://arxiv.org/abs/2412.19089)  
Changwoon Choi, Jeongjun Kim, Geonho Cha, Minkwan Kim, Dongyoon Wee, Young Min Kim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent works on dynamic neural field reconstruction assume input from synchronized multi-view videos with known poses. These input constraints are often unmet in real-world setups, making the approach impractical. We demonstrate that unsynchronized videos with unknown poses can generate dynamic neural fields if the videos capture human motion. Humans are one of the most common dynamic subjects whose poses can be estimated using state-of-the-art methods. While noisy, the estimated human shape and pose parameters provide a decent initialization for the highly non-convex and under-constrained problem of training a consistent dynamic neural representation. Given the sequences of pose and shape of humans, we estimate the time offsets between videos, followed by camera pose estimations by analyzing 3D joint locations. Then, we train dynamic NeRF employing multiresolution rids while simultaneously refining both time offsets and camera poses. The setup still involves optimizing many parameters, therefore, we introduce a robust progressive learning strategy to stabilize the process. Experiments show that our approach achieves accurate spatiotemporal calibration and high-quality scene reconstruction in challenging conditions.  
  </ol>  
</details>  
  
  



