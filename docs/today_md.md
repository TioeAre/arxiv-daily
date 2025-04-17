<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#An-Online-Adaptation-Method-for-Robust-Depth-Estimation-and-Visual-Odometry-in-the-Open-World>An Online Adaptation Method for Robust Depth Estimation and Visual Odometry in the Open World</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Generalized-Visual-Relation-Detection-with-Diffusion-Models>Generalized Visual Relation Detection with Diffusion Models</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#R-Meshfusion:-Reinforcement-Learning-Powered-Sparse-View-Mesh-Reconstruction-with-Diffusion-Priors>R-Meshfusion: Reinforcement Learning Powered Sparse-View Mesh Reconstruction with Diffusion Priors</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [An Online Adaptation Method for Robust Depth Estimation and Visual Odometry in the Open World](http://arxiv.org/abs/2504.11698)  
Xingwu Ji, Haochen Niu, Dexin Duan, Rendong Ying, Fei Wen, Peilin Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recently, learning-based robotic navigation systems have gained extensive research attention and made significant progress. However, the diversity of open-world scenarios poses a major challenge for the generalization of such systems to practical scenarios. Specifically, learned systems for scene measurement and state estimation tend to degrade when the application scenarios deviate from the training data, resulting to unreliable depth and pose estimation. Toward addressing this problem, this work aims to develop a visual odometry system that can fast adapt to diverse novel environments in an online manner. To this end, we construct a self-supervised online adaptation framework for monocular visual odometry aided by an online-updated depth estimation module. Firstly, we design a monocular depth estimation network with lightweight refiner modules, which enables efficient online adaptation. Then, we construct an objective for self-supervised learning of the depth estimation module based on the output of the visual odometry system and the contextual semantic information of the scene. Specifically, a sparse depth densification module and a dynamic consistency enhancement module are proposed to leverage camera poses and contextual semantics to generate pseudo-depths and valid masks for the online adaptation. Finally, we demonstrate the robustness and generalization capability of the proposed method in comparison with state-of-the-art learning-based approaches on urban, in-house datasets and a robot platform. Code is publicly available at: https://github.com/jixingwu/SOL-SLAM.  
  </ol>  
</details>  
**comments**: 11 pages, 14 figures  
  
  



## Visual Localization  

### [Generalized Visual Relation Detection with Diffusion Models](http://arxiv.org/abs/2504.12100)  
Kaifeng Gao, Siqi Chen, Hanwang Zhang, Jun Xiao, Yueting Zhuang, Qianru Sun  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual relation detection (VRD) aims to identify relationships (or interactions) between object pairs in an image. Although recent VRD models have achieved impressive performance, they are all restricted to pre-defined relation categories, while failing to consider the semantic ambiguity characteristic of visual relations. Unlike objects, the appearance of visual relations is always subtle and can be described by multiple predicate words from different perspectives, e.g., ``ride'' can be depicted as ``race'' and ``sit on'', from the sports and spatial position views, respectively. To this end, we propose to model visual relations as continuous embeddings, and design diffusion models to achieve generalized VRD in a conditional generative manner, termed Diff-VRD. We model the diffusion process in a latent space and generate all possible relations in the image as an embedding sequence. During the generation, the visual and text embeddings of subject-object pairs serve as conditional signals and are injected via cross-attention. After the generation, we design a subsequent matching stage to assign the relation words to subject-object pairs by considering their semantic similarities. Benefiting from the diffusion-based generative process, our Diff-VRD is able to generate visual relations beyond the pre-defined category labels of datasets. To properly evaluate this generalized VRD task, we introduce two evaluation metrics, i.e., text-to-image retrieval and SPICE PR Curve inspired by image captioning. Extensive experiments in both human-object interaction (HOI) detection and scene graph generation (SGG) benchmarks attest to the superiority and effectiveness of Diff-VRD.  
  </ol>  
</details>  
**comments**: Under review at IEEE TCSVT. The Appendix is provided additionally  
  
  



## NeRF  

### [R-Meshfusion: Reinforcement Learning Powered Sparse-View Mesh Reconstruction with Diffusion Priors](http://arxiv.org/abs/2504.11946)  
Haoyang Wang, Liming Liu, Peiheng Wang, Junlin Hao, Jiangkai Wu, Xinggong Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Mesh reconstruction from multi-view images is a fundamental problem in computer vision, but its performance degrades significantly under sparse-view conditions, especially in unseen regions where no ground-truth observations are available. While recent advances in diffusion models have demonstrated strong capabilities in synthesizing novel views from limited inputs, their outputs often suffer from visual artifacts and lack 3D consistency, posing challenges for reliable mesh optimization. In this paper, we propose a novel framework that leverages diffusion models to enhance sparse-view mesh reconstruction in a principled and reliable manner. To address the instability of diffusion outputs, we propose a Consensus Diffusion Module that filters unreliable generations via interquartile range (IQR) analysis and performs variance-aware image fusion to produce robust pseudo-supervision. Building on this, we design an online reinforcement learning strategy based on the Upper Confidence Bound (UCB) to adaptively select the most informative viewpoints for enhancement, guided by diffusion loss. Finally, the fused images are used to jointly supervise a NeRF-based model alongside sparse-view ground truth, ensuring consistency across both geometry and appearance. Extensive experiments demonstrate that our method achieves significant improvements in both geometric quality and rendering quality.  
  </ol>  
</details>  
  
  



