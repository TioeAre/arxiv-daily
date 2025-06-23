<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Class-Agnostic-Instance-level-Descriptor-for-Visual-Instance-Search>Class Agnostic Instance-level Descriptor for Visual Instance Search</a></li>
        <li><a href=#MambaHash:-Visual-State-Space-Deep-Hashing-Model-for-Large-Scale-Image-Retrieval>MambaHash: Visual State Space Deep Hashing Model for Large-Scale Image Retrieval</a></li>
        <li><a href=#Fine-grained-Image-Retrieval-via-Dual-Vision-Adaptation>Fine-grained Image Retrieval via Dual-Vision Adaptation</a></li>
        <li><a href=#Adversarial-Attacks-and-Detection-in-Visual-Place-Recognition-for-Safer-Robot-Navigation>Adversarial Attacks and Detection in Visual Place Recognition for Safer Robot Navigation</a></li>
        <li><a href=#Semantic-and-Feature-Guided-Uncertainty-Quantification-of-Visual-Localization-for-Autonomous-Vehicles>Semantic and Feature Guided Uncertainty Quantification of Visual Localization for Autonomous Vehicles</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#R3eVision:-A-Survey-on-Robust-Rendering,-Restoration,-and-Enhancement-for-3D-Low-Level-Vision>R3eVision: A Survey on Robust Rendering, Restoration, and Enhancement for 3D Low-Level Vision</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Class Agnostic Instance-level Descriptor for Visual Instance Search](http://arxiv.org/abs/2506.16745)  
Qi-Ying Sun, Wan-Lei Zhao, Yi-Bo Miao, Chong-Wah Ngo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Despite the great success of the deep features in content-based image retrieval, the visual instance search remains challenging due to the lack of effective instance level feature representation. Supervised or weakly supervised object detection methods are not among the options due to their poor performance on the unknown object categories. In this paper, based on the feature set output from self-supervised ViT, the instance level region discovery is modeled as detecting the compact feature subsets in a hierarchical fashion. The hierarchical decomposition results in a hierarchy of feature subsets. The non-leaf nodes and leaf nodes on the hierarchy correspond to the various instance regions in an image of different semantic scales. The hierarchical decomposition well addresses the problem of object embedding and occlusions, which are widely observed in the real scenarios. The features derived from the nodes on the hierarchy make up a comprehensive representation for the latent instances in the image. Our instance-level descriptor remains effective on both the known and unknown object categories. Empirical studies on three instance search benchmarks show that it outperforms state-of-the-art methods considerably.  
  </ol>  
</details>  
  
### [MambaHash: Visual State Space Deep Hashing Model for Large-Scale Image Retrieval](http://arxiv.org/abs/2506.16353)  
Chao He, Hongxi Wei  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Deep image hashing aims to enable effective large-scale image retrieval by mapping the input images into simple binary hash codes through deep neural networks. More recently, Vision Mamba with linear time complexity has attracted extensive attention from researchers by achieving outstanding performance on various computer tasks. Nevertheless, the suitability of Mamba for large-scale image retrieval tasks still needs to be explored. Towards this end, we propose a visual state space hashing model, called MambaHash. Concretely, we propose a backbone network with stage-wise architecture, in which grouped Mamba operation is introduced to model local and global information by utilizing Mamba to perform multi-directional scanning along different groups of the channel. Subsequently, the proposed channel interaction attention module is used to enhance information communication across channels. Finally, we meticulously design an adaptive feature enhancement module to increase feature diversity and enhance the visual representation capability of the model. We have conducted comprehensive experiments on three widely used datasets: CIFAR-10, NUS-WIDE and IMAGENET. The experimental results demonstrate that compared with the state-of-the-art deep hashing methods, our proposed MambaHash has well efficiency and superior performance to effectively accomplish large-scale image retrieval tasks. Source code is available https://github.com/shuaichaochao/MambaHash.git  
  </ol>  
</details>  
**comments**: Accepted by ICMR2025. arXiv admin note: text overlap with
  arXiv:2405.07524  
  
### [Fine-grained Image Retrieval via Dual-Vision Adaptation](http://arxiv.org/abs/2506.16273)  
Xin Jiang, Meiqi Cao, Hao Tang, Fei Shen, Zechao Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Fine-Grained Image Retrieval~(FGIR) faces challenges in learning discriminative visual representations to retrieve images with similar fine-grained features. Current leading FGIR solutions typically follow two regimes: enforce pairwise similarity constraints in the semantic embedding space, or incorporate a localization sub-network to fine-tune the entire model. However, such two regimes tend to overfit the training data while forgetting the knowledge gained from large-scale pre-training, thus reducing their generalization ability. In this paper, we propose a Dual-Vision Adaptation (DVA) approach for FGIR, which guides the frozen pre-trained model to perform FGIR through collaborative sample and feature adaptation. Specifically, we design Object-Perceptual Adaptation, which modifies input samples to help the pre-trained model perceive critical objects and elements within objects that are helpful for category prediction. Meanwhile, we propose In-Context Adaptation, which introduces a small set of parameters for feature adaptation without modifying the pre-trained parameters. This makes the FGIR task using these adjusted features closer to the task solved during the pre-training. Additionally, to balance retrieval efficiency and performance, we propose Discrimination Perception Transfer to transfer the discriminative knowledge in the object-perceptual adaptation to the image encoder using the knowledge distillation mechanism. Extensive experiments show that DVA has fewer learnable parameters and performs well on three in-distribution and three out-of-distribution fine-grained datasets.  
  </ol>  
</details>  
  
### [Adversarial Attacks and Detection in Visual Place Recognition for Safer Robot Navigation](http://arxiv.org/abs/2506.15988)  
Connor Malone, Owen Claxton, Iman Shames, Michael Milford  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Stand-alone Visual Place Recognition (VPR) systems have little defence against a well-designed adversarial attack, which can lead to disastrous consequences when deployed for robot navigation. This paper extensively analyzes the effect of four adversarial attacks common in other perception tasks and four novel VPR-specific attacks on VPR localization performance. We then propose how to close the loop between VPR, an Adversarial Attack Detector (AAD), and active navigation decisions by demonstrating the performance benefit of simulated AADs in a novel experiment paradigm -- which we detail for the robotics community to use as a system framework. In the proposed experiment paradigm, we see the addition of AADs across a range of detection accuracies can improve performance over baseline; demonstrating a significant improvement -- such as a ~50% reduction in the mean along-track localization error -- can be achieved with True Positive and False Positive detection rates of only 75% and up to 25% respectively. We examine a variety of metrics including: Along-Track Error, Percentage of Time Attacked, Percentage of Time in an `Unsafe' State, and Longest Continuous Time Under Attack. Expanding further on these results, we provide the first investigation into the efficacy of the Fast Gradient Sign Method (FGSM) adversarial attack for VPR. The analysis in this work highlights the need for AADs in real-world systems for trustworthy navigation, and informs quantitative requirements for system design.  
  </ol>  
</details>  
  
### [Semantic and Feature Guided Uncertainty Quantification of Visual Localization for Autonomous Vehicles](http://arxiv.org/abs/2506.15851)  
Qiyuan Wu, Mark Campbell  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The uncertainty quantification of sensor measurements coupled with deep learning networks is crucial for many robotics systems, especially for safety-critical applications such as self-driving cars. This paper develops an uncertainty quantification approach in the context of visual localization for autonomous driving, where locations are selected based on images. Key to our approach is to learn the measurement uncertainty using light-weight sensor error model, which maps both image feature and semantic information to 2-dimensional error distribution. Our approach enables uncertainty estimation conditioned on the specific context of the matched image pair, implicitly capturing other critical, unannotated factors (e.g., city vs highway, dynamic vs static scenes, winter vs summer) in a latent manner. We demonstrate the accuracy of our uncertainty prediction framework using the Ithaca365 dataset, which includes variations in lighting and weather (sunny, night, snowy). Both the uncertainty quantification of the sensor+network is evaluated, along with Bayesian localization filters using unique sensor gating method. Results show that the measurement error does not follow a Gaussian distribution with poor weather and lighting conditions, and is better predicted by our Gaussian Mixture model.  
  </ol>  
</details>  
**comments**: Accepted by ICRA 2025  
  
  



## NeRF  

### [R3eVision: A Survey on Robust Rendering, Restoration, and Enhancement for 3D Low-Level Vision](http://arxiv.org/abs/2506.16262)  
Weeyoung Kwon, Jeahun Sung, Minkyu Jeon, Chanho Eom, Jihyong Oh  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural rendering methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have achieved significant progress in photorealistic 3D scene reconstruction and novel view synthesis. However, most existing models assume clean and high-resolution (HR) multi-view inputs, which limits their robustness under real-world degradations such as noise, blur, low-resolution (LR), and weather-induced artifacts. To address these limitations, the emerging field of 3D Low-Level Vision (3D LLV) extends classical 2D Low-Level Vision tasks including super-resolution (SR), deblurring, weather degradation removal, restoration, and enhancement into the 3D spatial domain. This survey, referred to as R\textsuperscript{3}eVision, provides a comprehensive overview of robust rendering, restoration, and enhancement for 3D LLV by formalizing the degradation-aware rendering problem and identifying key challenges related to spatio-temporal consistency and ill-posed optimization. Recent methods that integrate LLV into neural rendering frameworks are categorized to illustrate how they enable high-fidelity 3D reconstruction under adverse conditions. Application domains such as autonomous driving, AR/VR, and robotics are also discussed, where reliable 3D perception from degraded inputs is critical. By reviewing representative methods, datasets, and evaluation protocols, this work positions 3D LLV as a fundamental direction for robust 3D content generation and scene-level reconstruction in real-world environments.  
  </ol>  
</details>  
**comments**: Please visit our project page at
  https://github.com/CMLab-Korea/Awesome-3D-Low-Level-Vision  
  
  



