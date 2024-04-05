<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#GaSpCT:-Gaussian-Splatting-for-Novel-CT-Projection-View-Synthesis>GaSpCT: Gaussian Splatting for Novel CT Projection View Synthesis</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#RaFE:-Generative-Radiance-Fields-Restoration>RaFE: Generative Radiance Fields Restoration</a></li>
        <li><a href=#OpenNeRF:-Open-Set-3D-Neural-Scene-Segmentation-with-Pixel-Wise-Features-and-Rendered-Novel-Views>OpenNeRF: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views</a></li>
        <li><a href=#VF-NeRF:-Viewshed-Fields-for-Rigid-NeRF-Registration>VF-NeRF: Viewshed Fields for Rigid NeRF Registration</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [GaSpCT: Gaussian Splatting for Novel CT Projection View Synthesis](http://arxiv.org/abs/2404.03126)  
Emmanouil Nikolakakis, Utkarsh Gupta, Jonathan Vengosh, Justin Bui, Razvan Marinescu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present GaSpCT, a novel view synthesis and 3D scene representation method used to generate novel projection views for Computer Tomography (CT) scans. We adapt the Gaussian Splatting framework to enable novel view synthesis in CT based on limited sets of 2D image projections and without the need for Structure from Motion (SfM) methodologies. Therefore, we reduce the total scanning duration and the amount of radiation dose the patient receives during the scan. We adapted the loss function to our use-case by encouraging a stronger background and foreground distinction using two sparsity promoting regularizers: a beta loss and a total variation (TV) loss. Finally, we initialize the Gaussian locations across the 3D space using a uniform prior distribution of where the brain's positioning would be expected to be within the field of view. We evaluate the performance of our model using brain CT scans from the Parkinson's Progression Markers Initiative (PPMI) dataset and demonstrate that the rendered novel views closely match the original projection views of the simulated scan, and have better performance than other implicit 3D scene representations methodologies. Furthermore, we empirically observe reduced training time compared to neural network based image synthesis for sparse-view CT image reconstruction. Finally, the memory requirements of the Gaussian Splatting representations are reduced by 17% compared to the equivalent voxel grid image representations.  
  </ol>  
</details>  
**comments**: Under Review Process for MICCAI 2024  
  
  



## NeRF  

### [RaFE: Generative Radiance Fields Restoration](http://arxiv.org/abs/2404.03654)  
Zhongkai Wu, Ziyu Wan, Jing Zhang, Jing Liao, Dong Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    NeRF (Neural Radiance Fields) has demonstrated tremendous potential in novel view synthesis and 3D reconstruction, but its performance is sensitive to input image quality, which struggles to achieve high-fidelity rendering when provided with low-quality sparse input viewpoints. Previous methods for NeRF restoration are tailored for specific degradation type, ignoring the generality of restoration. To overcome this limitation, we propose a generic radiance fields restoration pipeline, named RaFE, which applies to various types of degradations, such as low resolution, blurriness, noise, compression artifacts, or their combinations. Our approach leverages the success of off-the-shelf 2D restoration methods to recover the multi-view images individually. Instead of reconstructing a blurred NeRF by averaging inconsistencies, we introduce a novel approach using Generative Adversarial Networks (GANs) for NeRF generation to better accommodate the geometric and appearance inconsistencies present in the multi-view images. Specifically, we adopt a two-level tri-plane architecture, where the coarse level remains fixed to represent the low-quality NeRF, and a fine-level residual tri-plane to be added to the coarse level is modeled as a distribution with GAN to capture potential variations in restoration. We validate RaFE on both synthetic and real cases for various restoration tasks, demonstrating superior performance in both quantitative and qualitative evaluations, surpassing other 3D restoration methods specific to single task. Please see our project website https://zkaiwu.github.io/RaFE-Project/.  
  </ol>  
</details>  
**comments**: Project Page: https://zkaiwu.github.io/RaFE-Project/  
  
### [OpenNeRF: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views](http://arxiv.org/abs/2404.03650)  
Francis Engelmann, Fabian Manhardt, Michael Niemeyer, Keisuke Tateno, Marc Pollefeys, Federico Tombari  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Large visual-language models (VLMs), like CLIP, enable open-set image segmentation to segment arbitrary concepts from an image in a zero-shot manner. This goes beyond the traditional closed-set assumption, i.e., where models can only segment classes from a pre-defined training set. More recently, first works on open-set segmentation in 3D scenes have appeared in the literature. These methods are heavily influenced by closed-set 3D convolutional approaches that process point clouds or polygon meshes. However, these 3D scene representations do not align well with the image-based nature of the visual-language models. Indeed, point cloud and 3D meshes typically have a lower resolution than images and the reconstructed 3D scene geometry might not project well to the underlying 2D image sequences used to compute pixel-aligned CLIP features. To address these challenges, we propose OpenNeRF which naturally operates on posed images and directly encodes the VLM features within the NeRF. This is similar in spirit to LERF, however our work shows that using pixel-wise VLM features (instead of global CLIP features) results in an overall less complex architecture without the need for additional DINO regularization. Our OpenNeRF further leverages NeRF's ability to render novel views and extract open-set VLM features from areas that are not well observed in the initial posed images. For 3D point cloud segmentation on the Replica dataset, OpenNeRF outperforms recent open-vocabulary methods such as LERF and OpenScene by at least +4.9 mIoU.  
  </ol>  
</details>  
**comments**: ICLR 2024, Project page: https://opennerf.github.io  
  
### [VF-NeRF: Viewshed Fields for Rigid NeRF Registration](http://arxiv.org/abs/2404.03349)  
Leo Segre, Shai Avidan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D scene registration is a fundamental problem in computer vision that seeks the best 6-DoF alignment between two scenes. This problem was extensively investigated in the case of point clouds and meshes, but there has been relatively limited work regarding Neural Radiance Fields (NeRF). In this paper, we consider the problem of rigid registration between two NeRFs when the position of the original cameras is not given. Our key novelty is the introduction of Viewshed Fields (VF), an implicit function that determines, for each 3D point, how likely it is to be viewed by the original cameras. We demonstrate how VF can help in the various stages of NeRF registration, with an extensive evaluation showing that VF-NeRF achieves SOTA results on various datasets with different capturing approaches such as LLFF and Objaverese.  
  </ol>  
</details>  
  
  



