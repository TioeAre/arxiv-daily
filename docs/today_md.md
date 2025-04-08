<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#3R-GS:-Best-Practice-in-Optimizing-Camera-Poses-Along-with-3DGS>3R-GS: Best Practice in Optimizing Camera Poses Along with 3DGS</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#NCL-CIR:-Noise-aware-Contrastive-Learning-for-Composed-Image-Retrieval>NCL-CIR: Noise-aware Contrastive Learning for Composed Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Learning-Affine-Correspondences-by-Integrating-Geometric-Constraints>Learning Affine Correspondences by Integrating Geometric Constraints</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#DeclutterNeRF:-Generative-Free-3D-Scene-Recovery-for-Occlusion-Removal>DeclutterNeRF: Generative-Free 3D Scene Recovery for Occlusion Removal</a></li>
        <li><a href=#Thermoxels:-a-voxel-based-method-to-generate-simulation-ready-3D-thermal-models>Thermoxels: a voxel-based method to generate simulation-ready 3D thermal models</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [3R-GS: Best Practice in Optimizing Camera Poses Along with 3DGS](http://arxiv.org/abs/2504.04294)  
Zhisheng Huang, Peng Wang, Jingdong Zhang, Yuan Liu, Xin Li, Wenping Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D Gaussian Splatting (3DGS) has revolutionized neural rendering with its efficiency and quality, but like many novel view synthesis methods, it heavily depends on accurate camera poses from Structure-from-Motion (SfM) systems. Although recent SfM pipelines have made impressive progress, questions remain about how to further improve both their robust performance in challenging conditions (e.g., textureless scenes) and the precision of camera parameter estimation simultaneously. We present 3R-GS, a 3D Gaussian Splatting framework that bridges this gap by jointly optimizing 3D Gaussians and camera parameters from large reconstruction priors MASt3R-SfM. We note that naively performing joint 3D Gaussian and camera optimization faces two challenges: the sensitivity to the quality of SfM initialization, and its limited capacity for global optimization, leading to suboptimal reconstruction results. Our 3R-GS, overcomes these issues by incorporating optimized practices, enabling robust scene reconstruction even with imperfect camera registration. Extensive experiments demonstrate that 3R-GS delivers high-quality novel view synthesis and precise camera pose estimation while remaining computationally efficient. Project page: https://zsh523.github.io/3R-GS/  
  </ol>  
</details>  
  
  



## Visual Localization  

### [NCL-CIR: Noise-aware Contrastive Learning for Composed Image Retrieval](http://arxiv.org/abs/2504.04339)  
Peng Gao, Yujian Lee, Zailong Chen, Hui zhang, Xubo Liu, Yiyang Hu, Guquang Jing  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed Image Retrieval (CIR) seeks to find a target image using a multi-modal query, which combines an image with modification text to pinpoint the target. While recent CIR methods have shown promise, they mainly focus on exploring relationships between the query pairs (image and text) through data augmentation or model design. These methods often assume perfect alignment between queries and target images, an idealized scenario rarely encountered in practice. In reality, pairs are often partially or completely mismatched due to issues like inaccurate modification texts, low-quality target images, and annotation errors. Ignoring these mismatches leads to numerous False Positive Pair (FFPs) denoted as noise pairs in the dataset, causing the model to overfit and ultimately reducing its performance. To address this problem, we propose the Noise-aware Contrastive Learning for CIR (NCL-CIR), comprising two key components: the Weight Compensation Block (WCB) and the Noise-pair Filter Block (NFB). The WCB coupled with diverse weight maps can ensure more stable token representations of multi-modal queries and target images. Meanwhile, the NFB, in conjunction with the Gaussian Mixture Model (GMM) predicts noise pairs by evaluating loss distributions, and generates soft labels correspondingly, allowing for the design of the soft-label based Noise Contrastive Estimation (NCE) loss function. Consequently, the overall architecture helps to mitigate the influence of mismatched and partially matched samples, with experimental results demonstrating that NCL-CIR achieves exceptional performance on the benchmark datasets.  
  </ol>  
</details>  
**comments**: Has been accepted by ICASSP2025  
  
  



## Image Matching  

### [Learning Affine Correspondences by Integrating Geometric Constraints](http://arxiv.org/abs/2504.04834)  
Pengju Sun, Banglei Guan, Zhenbao Yu, Yang Shang, Qifeng Yu, Daniel Barath  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Affine correspondences have received significant attention due to their benefits in tasks like image matching and pose estimation. Existing methods for extracting affine correspondences still have many limitations in terms of performance; thus, exploring a new paradigm is crucial. In this paper, we present a new pipeline designed for extracting accurate affine correspondences by integrating dense matching and geometric constraints. Specifically, a novel extraction framework is introduced, with the aid of dense matching and a novel keypoint scale and orientation estimator. For this purpose, we propose loss functions based on geometric constraints, which can effectively improve accuracy by supervising neural networks to learn feature geometry. The experimental show that the accuracy and robustness of our method outperform the existing ones in image matching tasks. To further demonstrate the effectiveness of the proposed method, we applied it to relative pose estimation. Affine correspondences extracted by our method lead to more accurate poses than the baselines on a range of real-world datasets. The code is available at https://github.com/stilcrad/DenseAffine.  
  </ol>  
</details>  
  
  



## NeRF  

### [DeclutterNeRF: Generative-Free 3D Scene Recovery for Occlusion Removal](http://arxiv.org/abs/2504.04679)  
Wanzhou Liu, Zhexiao Xiong, Xinyu Li, Nathan Jacobs  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent novel view synthesis (NVS) techniques, including Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have greatly advanced 3D scene reconstruction with high-quality rendering and realistic detail recovery. Effectively removing occlusions while preserving scene details can further enhance the robustness and applicability of these techniques. However, existing approaches for object and occlusion removal predominantly rely on generative priors, which, despite filling the resulting holes, introduce new artifacts and blurriness. Moreover, existing benchmark datasets for evaluating occlusion removal methods lack realistic complexity and viewpoint variations. To address these issues, we introduce DeclutterSet, a novel dataset featuring diverse scenes with pronounced occlusions distributed across foreground, midground, and background, exhibiting substantial relative motion across viewpoints. We further introduce DeclutterNeRF, an occlusion removal method free from generative priors. DeclutterNeRF introduces joint multi-view optimization of learnable camera parameters, occlusion annealing regularization, and employs an explainable stochastic structural similarity loss, ensuring high-quality, artifact-free reconstructions from incomplete images. Experiments demonstrate that DeclutterNeRF significantly outperforms state-of-the-art methods on our proposed DeclutterSet, establishing a strong baseline for future research.  
  </ol>  
</details>  
**comments**: Accepted by CVPR 2025 4th CV4Metaverse Workshop. 15 pages, 10
  figures. Code and data at: https://github.com/wanzhouliu/declutter-nerf  
  
### [Thermoxels: a voxel-based method to generate simulation-ready 3D thermal models](http://arxiv.org/abs/2504.04448)  
Etienne Chassaing, Florent Forest, Olga Fink, Malcolm Mielle  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In the European Union, buildings account for 42% of energy use and 35% of greenhouse gas emissions. Since most existing buildings will still be in use by 2050, retrofitting is crucial for emissions reduction. However, current building assessment methods rely mainly on qualitative thermal imaging, which limits data-driven decisions for energy savings. On the other hand, quantitative assessments using finite element analysis (FEA) offer precise insights but require manual CAD design, which is tedious and error-prone. Recent advances in 3D reconstruction, such as Neural Radiance Fields (NeRF) and Gaussian Splatting, enable precise 3D modeling from sparse images but lack clearly defined volumes and the interfaces between them needed for FEA. We propose Thermoxels, a novel voxel-based method able to generate FEA-compatible models, including both geometry and temperature, from a sparse set of RGB and thermal images. Using pairs of RGB and thermal images as input, Thermoxels represents a scene's geometry as a set of voxels comprising color and temperature information. After optimization, a simple process is used to transform Thermoxels' models into tetrahedral meshes compatible with FEA. We demonstrate Thermoxels' capability to generate RGB+Thermal meshes of 3D scenes, surpassing other state-of-the-art methods. To showcase the practical applications of Thermoxels' models, we conduct a simple heat conduction simulation using FEA, achieving convergence from an initial state defined by Thermoxels' thermal reconstruction. Additionally, we compare Thermoxels' image synthesis abilities with current state-of-the-art methods, showing competitive results, and discuss the limitations of existing metrics in assessing mesh quality.  
  </ol>  
</details>  
**comments**: 7 pages, 2 figures  
  
  



