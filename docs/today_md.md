<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#A-Constrained-Optimization-Approach-for-Gaussian-Splatting-from-Coarsely-posed-Images-and-Noisy-Lidar-Point-Clouds>A Constrained Optimization Approach for Gaussian Splatting from Coarsely-posed Images and Noisy Lidar Point Clouds</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Focus-on-Local:-Finding-Reliable-Discriminative-Regions-for-Visual-Place-Recognition>Focus on Local: Finding Reliable Discriminative Regions for Visual Place Recognition</a></li>
        <li><a href=#Evolved-Hierarchical-Masking-for-Self-Supervised-Learning>Evolved Hierarchical Masking for Self-Supervised Learning</a></li>
        <li><a href=#HAL-NeRF:-High-Accuracy-Localization-Leveraging-Neural-Radiance-Fields>HAL-NeRF: High Accuracy Localization Leveraging Neural Radiance Fields</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#LL-Gaussian:-Low-Light-Scene-Reconstruction-and-Enhancement-via-Gaussian-Splatting-for-Novel-View-Synthesis>LL-Gaussian: Low-Light Scene Reconstruction and Enhancement via Gaussian Splatting for Novel View Synthesis</a></li>
        <li><a href=#MCBlock:-Boosting-Neural-Radiance-Field-Training-Speed-by-MCTS-based-Dynamic-Resolution-Ray-Sampling>MCBlock: Boosting Neural Radiance Field Training Speed by MCTS-based Dynamic-Resolution Ray Sampling</a></li>
        <li><a href=#NeRF-Based-Transparent-Object-Grasping-Enhanced-by-Shape-Priors>NeRF-Based Transparent Object Grasping Enhanced by Shape Priors</a></li>
        <li><a href=#HAL-NeRF:-High-Accuracy-Localization-Leveraging-Neural-Radiance-Fields>HAL-NeRF: High Accuracy Localization Leveraging Neural Radiance Fields</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [A Constrained Optimization Approach for Gaussian Splatting from Coarsely-posed Images and Noisy Lidar Point Clouds](http://arxiv.org/abs/2504.09129)  
Jizong Peng, Tze Ho Elden Tse, Kai Xu, Wenchao Gao, Angela Yao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D Gaussian Splatting (3DGS) is a powerful reconstruction technique, but it needs to be initialized from accurate camera poses and high-fidelity point clouds. Typically, the initialization is taken from Structure-from-Motion (SfM) algorithms; however, SfM is time-consuming and restricts the application of 3DGS in real-world scenarios and large-scale scene reconstruction. We introduce a constrained optimization method for simultaneous camera pose estimation and 3D reconstruction that does not require SfM support. Core to our approach is decomposing a camera pose into a sequence of camera-to-(device-)center and (device-)center-to-world optimizations. To facilitate, we propose two optimization constraints conditioned to the sensitivity of each parameter group and restricts each parameter's search space. In addition, as we learn the scene geometry directly from the noisy point clouds, we propose geometric constraints to improve the reconstruction quality. Experiments demonstrate that the proposed method significantly outperforms the existing (multi-modal) 3DGS baseline and methods supplemented by COLMAP on both our collected dataset and two public benchmarks.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Focus on Local: Finding Reliable Discriminative Regions for Visual Place Recognition](http://arxiv.org/abs/2504.09881)  
Changwei Wang, Shunpeng Chen, Yukun Song, Rongtao Xu, Zherui Zhang, Jiguang Zhang, Haoran Yang, Yu Zhang, Kexue Fu, Shide Du, Zhiwei Xu, Longxiang Gao, Li Guo, Shibiao Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Place Recognition (VPR) is aimed at predicting the location of a query image by referencing a database of geotagged images. For VPR task, often fewer discriminative local regions in an image produce important effects while mundane background regions do not contribute or even cause perceptual aliasing because of easy overlap. However, existing methods lack precisely modeling and full exploitation of these discriminative regions. In this paper, we propose the Focus on Local (FoL) approach to stimulate the performance of image retrieval and re-ranking in VPR simultaneously by mining and exploiting reliable discriminative local regions in images and introducing pseudo-correlation supervision. First, we design two losses, Extraction-Aggregation Spatial Alignment Loss (SAL) and Foreground-Background Contrast Enhancement Loss (CEL), to explicitly model reliable discriminative local regions and use them to guide the generation of global representations and efficient re-ranking. Second, we introduce a weakly-supervised local feature training strategy based on pseudo-correspondences obtained from aggregating global features to alleviate the lack of local correspondences ground truth for the VPR task. Third, we suggest an efficient re-ranking pipeline that is efficiently and precisely based on discriminative region guidance. Finally, experimental results show that our FoL achieves the state-of-the-art on multiple VPR benchmarks in both image retrieval and re-ranking stages and also significantly outperforms existing two-stage VPR methods in terms of computational efficiency. Code and models are available at https://github.com/chenshunpeng/FoL  
  </ol>  
</details>  
**comments**: Accepted by AAAI 2025  
  
### [Evolved Hierarchical Masking for Self-Supervised Learning](http://arxiv.org/abs/2504.09155)  
Zhanzhou Feng, Shiliang Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Existing Masked Image Modeling methods apply fixed mask patterns to guide the self-supervised training. As those mask patterns resort to different criteria to depict image contents, sticking to a fixed pattern leads to a limited vision cues modeling capability.This paper introduces an evolved hierarchical masking method to pursue general visual cues modeling in self-supervised learning. The proposed method leverages the vision model being trained to parse the input visual cues into a hierarchy structure, which is hence adopted to generate masks accordingly. The accuracy of hierarchy is on par with the capability of the model being trained, leading to evolved mask patterns at different training stages. Initially, generated masks focus on low-level visual cues to grasp basic textures, then gradually evolve to depict higher-level cues to reinforce the learning of more complicated object semantics and contexts. Our method does not require extra pre-trained models or annotations and ensures training efficiency by evolving the training difficulty. We conduct extensive experiments on seven downstream tasks including partial-duplicate image retrieval relying on low-level details, as well as image classification and semantic segmentation that require semantic parsing capability. Experimental results demonstrate that it substantially boosts performance across these tasks. For instance, it surpasses the recent MAE by 1.1\% in imageNet-1K classification and 1.4\% in ADE20K segmentation with the same training epochs. We also align the proposed method with the current research focus on LLMs. The proposed approach bridges the gap with large-scale pre-training on semantic demanding tasks and enhances intricate detail perception in tasks requiring low-level feature recognition.  
  </ol>  
</details>  
  
### [HAL-NeRF: High Accuracy Localization Leveraging Neural Radiance Fields](http://arxiv.org/abs/2504.08901)  
Asterios Reppas, Grigorios-Aris Cheimariotis, Panos K. Papadopoulos, Panagiotis Frasiolas, Dimitrios Zarpalas  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Precise camera localization is a critical task in XR applications and robotics. Using only the camera captures as input to a system is an inexpensive option that enables localization in large indoor and outdoor environments, but it presents challenges in achieving high accuracy. Specifically, camera relocalization methods, such as Absolute Pose Regression (APR), can localize cameras with a median translation error of more than $0.5m$ in outdoor scenes. This paper presents HAL-NeRF, a high-accuracy localization method that combines a CNN pose regressor with a refinement module based on a Monte Carlo particle filter. The Nerfacto model, an implementation of Neural Radiance Fields (NeRFs), is used to augment the data for training the pose regressor and to measure photometric loss in the particle filter refinement module. HAL-NeRF leverages Nerfacto's ability to synthesize high-quality novel views, significantly improving the performance of the localization pipeline. HAL-NeRF achieves state-of-the-art results that are conventionally measured as the average of the median per scene errors. The translation error was $0.025m$ and the rotation error was $0.59$ degrees and 0.04m and 0.58 degrees on the 7-Scenes dataset and Cambridge Landmarks datasets respectively, with the trade-off of increased computational time. This work highlights the potential of combining APR with NeRF-based refinement techniques to advance monocular camera relocalization accuracy.  
  </ol>  
</details>  
**comments**: 8 pages, 4 figures  
  
  



## NeRF  

### [LL-Gaussian: Low-Light Scene Reconstruction and Enhancement via Gaussian Splatting for Novel View Synthesis](http://arxiv.org/abs/2504.10331)  
Hao Sun, Fenggen Yu, Huiyao Xu, Tao Zhang, Changqing Zou  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Novel view synthesis (NVS) in low-light scenes remains a significant challenge due to degraded inputs characterized by severe noise, low dynamic range (LDR) and unreliable initialization. While recent NeRF-based approaches have shown promising results, most suffer from high computational costs, and some rely on carefully captured or pre-processed data--such as RAW sensor inputs or multi-exposure sequences--which severely limits their practicality. In contrast, 3D Gaussian Splatting (3DGS) enables real-time rendering with competitive visual fidelity; however, existing 3DGS-based methods struggle with low-light sRGB inputs, resulting in unstable Gaussian initialization and ineffective noise suppression. To address these challenges, we propose LL-Gaussian, a novel framework for 3D reconstruction and enhancement from low-light sRGB images, enabling pseudo normal-light novel view synthesis. Our method introduces three key innovations: 1) an end-to-end Low-Light Gaussian Initialization Module (LLGIM) that leverages dense priors from learning-based MVS approach to generate high-quality initial point clouds; 2) a dual-branch Gaussian decomposition model that disentangles intrinsic scene properties (reflectance and illumination) from transient interference, enabling stable and interpretable optimization; 3) an unsupervised optimization strategy guided by both physical constrains and diffusion prior to jointly steer decomposition and enhancement. Additionally, we contribute a challenging dataset collected in extreme low-light environments and demonstrate the effectiveness of LL-Gaussian. Compared to state-of-the-art NeRF-based methods, LL-Gaussian achieves up to 2,000 times faster inference and reduces training time to just 2%, while delivering superior reconstruction and rendering quality.  
  </ol>  
</details>  
  
### [MCBlock: Boosting Neural Radiance Field Training Speed by MCTS-based Dynamic-Resolution Ray Sampling](http://arxiv.org/abs/2504.09878)  
Yunpeng Tan, Junlin Hao, Jiangkai Wu, Liming Liu, Qingyang Li, Xinggong Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Field (NeRF) is widely known for high-fidelity novel view synthesis. However, even the state-of-the-art NeRF model, Gaussian Splatting, requires minutes for training, far from the real-time performance required by multimedia scenarios like telemedicine. One of the obstacles is its inefficient sampling, which is only partially addressed by existing works. Existing point-sampling algorithms uniformly sample simple-texture regions (easy to fit) and complex-texture regions (hard to fit), while existing ray-sampling algorithms sample these regions all in the finest granularity (i.e. the pixel level), both wasting GPU training resources. Actually, regions with different texture intensities require different sampling granularities. To this end, we propose a novel dynamic-resolution ray-sampling algorithm, MCBlock, which employs Monte Carlo Tree Search (MCTS) to partition each training image into pixel blocks with different sizes for active block-wise training. Specifically, the trees are initialized according to the texture of training images to boost the initialization speed, and an expansion/pruning module dynamically optimizes the block partition. MCBlock is implemented in Nerfstudio, an open-source toolset, and achieves a training acceleration of up to 2.33x, surpassing other ray-sampling algorithms. We believe MCBlock can apply to any cone-tracing NeRF model and contribute to the multimedia community.  
  </ol>  
</details>  
  
### [NeRF-Based Transparent Object Grasping Enhanced by Shape Priors](http://arxiv.org/abs/2504.09868)  
Yi Han, Zixin Lin, Dongjie Li, Lvping Chen, Yongliang Shi, Gan Ma  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Transparent object grasping remains a persistent challenge in robotics, largely due to the difficulty of acquiring precise 3D information. Conventional optical 3D sensors struggle to capture transparent objects, and machine learning methods are often hindered by their reliance on high-quality datasets. Leveraging NeRF's capability for continuous spatial opacity modeling, our proposed architecture integrates a NeRF-based approach for reconstructing the 3D information of transparent objects. Despite this, certain portions of the reconstructed 3D information may remain incomplete. To address these deficiencies, we introduce a shape-prior-driven completion mechanism, further refined by a geometric pose estimation method we have developed. This allows us to obtain a complete and reliable 3D information of transparent objects. Utilizing this refined data, we perform scene-level grasp prediction and deploy the results in real-world robotic systems. Experimental validation demonstrates the efficacy of our architecture, showcasing its capability to reliably capture 3D information of various transparent objects in cluttered scenes, and correspondingly, achieve high-quality, stables, and executable grasp predictions.  
  </ol>  
</details>  
  
### [HAL-NeRF: High Accuracy Localization Leveraging Neural Radiance Fields](http://arxiv.org/abs/2504.08901)  
Asterios Reppas, Grigorios-Aris Cheimariotis, Panos K. Papadopoulos, Panagiotis Frasiolas, Dimitrios Zarpalas  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Precise camera localization is a critical task in XR applications and robotics. Using only the camera captures as input to a system is an inexpensive option that enables localization in large indoor and outdoor environments, but it presents challenges in achieving high accuracy. Specifically, camera relocalization methods, such as Absolute Pose Regression (APR), can localize cameras with a median translation error of more than $0.5m$ in outdoor scenes. This paper presents HAL-NeRF, a high-accuracy localization method that combines a CNN pose regressor with a refinement module based on a Monte Carlo particle filter. The Nerfacto model, an implementation of Neural Radiance Fields (NeRFs), is used to augment the data for training the pose regressor and to measure photometric loss in the particle filter refinement module. HAL-NeRF leverages Nerfacto's ability to synthesize high-quality novel views, significantly improving the performance of the localization pipeline. HAL-NeRF achieves state-of-the-art results that are conventionally measured as the average of the median per scene errors. The translation error was $0.025m$ and the rotation error was $0.59$ degrees and 0.04m and 0.58 degrees on the 7-Scenes dataset and Cambridge Landmarks datasets respectively, with the trade-off of increased computational time. This work highlights the potential of combining APR with NeRF-based refinement techniques to advance monocular camera relocalization accuracy.  
  </ol>  
</details>  
**comments**: 8 pages, 4 figures  
  
  



