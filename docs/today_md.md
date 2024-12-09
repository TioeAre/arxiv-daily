<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#DAug:-Diffusion-based-Channel-Augmentation-for-Radiology-Image-Retrieval-and-Classification>DAug: Diffusion-based Channel Augmentation for Radiology Image Retrieval and Classification</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Perturb-and-Revise:-Flexible-3D-Editing-with-Generative-Trajectories>Perturb-and-Revise: Flexible 3D Editing with Generative Trajectories</a></li>
        <li><a href=#MixedGaussianAvatar:-Realistically-and-Geometrically-Accurate-Head-Avatar-via-Mixed-2D-3D-Gaussian-Splatting>MixedGaussianAvatar: Realistically and Geometrically Accurate Head Avatar via Mixed 2D-3D Gaussian Splatting</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [DAug: Diffusion-based Channel Augmentation for Radiology Image Retrieval and Classification](http://arxiv.org/abs/2412.04828)  
Ying Jin, Zhuoran Zhou, Haoquan Fang, Jenq-Neng Hwang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Medical image understanding requires meticulous examination of fine visual details, with particular regions requiring additional attention. While radiologists build such expertise over years of experience, it is challenging for AI models to learn where to look with limited amounts of training data. This limitation results in unsatisfying robustness in medical image understanding. To address this issue, we propose Diffusion-based Feature Augmentation (DAug), a portable method that improves a perception model's performance with a generative model's output. Specifically, we extend a radiology image to multiple channels, with the additional channels being the heatmaps of regions where diseases tend to develop. A diffusion-based image-to-image translation model was used to generate such heatmaps conditioned on selected disease classes. Our method is motivated by the fact that generative models learn the distribution of normal and abnormal images, and such knowledge is complementary to image understanding tasks. In addition, we propose the Image-Text-Class Hybrid Contrastive learning to utilize both text and class labels. With two novel approaches combined, our method surpasses baseline models without changing the model architecture, and achieves state-of-the-art performance on both medical image retrieval and classification tasks.  
  </ol>  
</details>  
  
  



## NeRF  

### [Perturb-and-Revise: Flexible 3D Editing with Generative Trajectories](http://arxiv.org/abs/2412.05279)  
Susung Hong, Johanna Karras, Ricardo Martin-Brualla, Ira Kemelmacher-Shlizerman  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The fields of 3D reconstruction and text-based 3D editing have advanced significantly with the evolution of text-based diffusion models. While existing 3D editing methods excel at modifying color, texture, and style, they struggle with extensive geometric or appearance changes, thus limiting their applications. We propose Perturb-and-Revise, which makes possible a variety of NeRF editing. First, we perturb the NeRF parameters with random initializations to create a versatile initialization. We automatically determine the perturbation magnitude through analysis of the local loss landscape. Then, we revise the edited NeRF via generative trajectories. Combined with the generative process, we impose identity-preserving gradients to refine the edited NeRF. Extensive experiments demonstrate that Perturb-and-Revise facilitates flexible, effective, and consistent editing of color, appearance, and geometry in 3D. For 360{\deg} results, please visit our project page: https://susunghong.github.io/Perturb-and-Revise.  
  </ol>  
</details>  
**comments**: Project page: https://susunghong.github.io/Perturb-and-Revise  
  
### [MixedGaussianAvatar: Realistically and Geometrically Accurate Head Avatar via Mixed 2D-3D Gaussian Splatting](http://arxiv.org/abs/2412.04955)  
[[code](https://github.com/chenvoid/mga)]  
Peng Chen, Xiaobao Wei, Qingpo Wuwu, Xinyi Wang, Xingyu Xiao, Ming Lu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Reconstructing high-fidelity 3D head avatars is crucial in various applications such as virtual reality. The pioneering methods reconstruct realistic head avatars with Neural Radiance Fields (NeRF), which have been limited by training and rendering speed. Recent methods based on 3D Gaussian Splatting (3DGS) significantly improve the efficiency of training and rendering. However, the surface inconsistency of 3DGS results in subpar geometric accuracy; later, 2DGS uses 2D surfels to enhance geometric accuracy at the expense of rendering fidelity. To leverage the benefits of both 2DGS and 3DGS, we propose a novel method named MixedGaussianAvatar for realistically and geometrically accurate head avatar reconstruction. Our main idea is to utilize 2D Gaussians to reconstruct the surface of the 3D head, ensuring geometric accuracy. We attach the 2D Gaussians to the triangular mesh of the FLAME model and connect additional 3D Gaussians to those 2D Gaussians where the rendering quality of 2DGS is inadequate, creating a mixed 2D-3D Gaussian representation. These 2D-3D Gaussians can then be animated using FLAME parameters. We further introduce a progressive training strategy that first trains the 2D Gaussians and then fine-tunes the mixed 2D-3D Gaussians. We demonstrate the superiority of MixedGaussianAvatar through comprehensive experiments. The code will be released at: https://github.com/ChenVoid/MGA/.  
  </ol>  
</details>  
**comments**: Project: https://chenvoid.github.io/MGA/  
  
  



