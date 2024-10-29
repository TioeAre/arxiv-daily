<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Generative-Adversarial-Patches-for-Physical-Attacks-on-Cross-Modal-Pedestrian-Re-Identification>Generative Adversarial Patches for Physical Attacks on Cross-Modal Pedestrian Re-Identification</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#EEG-Driven-3D-Object-Reconstruction-with-Color-Consistency-and-Diffusion-Prior>EEG-Driven 3D Object Reconstruction with Color Consistency and Diffusion Prior</a></li>
        <li><a href=#ODGS:-3D-Scene-Reconstruction-from-Omnidirectional-Images-with-3D-Gaussian-Splattings>ODGS: 3D Scene Reconstruction from Omnidirectional Images with 3D Gaussian Splattings</a></li>
        <li><a href=#GUMBEL-NERF:-Representing-Unseen-Objects-as-Part-Compositional-Neural-Radiance-Fields>GUMBEL-NERF: Representing Unseen Objects as Part-Compositional Neural Radiance Fields</a></li>
      </ul>
    </li>
  </ol>
</details>

## Image Matching  

### [Generative Adversarial Patches for Physical Attacks on Cross-Modal Pedestrian Re-Identification](http://arxiv.org/abs/2410.20097)  
Yue Su, Hao Li, Maoguo Gong  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visible-infrared pedestrian Re-identification (VI-ReID) aims to match pedestrian images captured by infrared cameras and visible cameras. However, VI-ReID, like other traditional cross-modal image matching tasks, poses significant challenges due to its human-centered nature. This is evidenced by the shortcomings of existing methods, which struggle to extract common features across modalities, while losing valuable information when bridging the gap between them in the implicit feature space, potentially compromising security. To address this vulnerability, this paper introduces the first physical adversarial attack against VI-ReID models. Our method, termed Edge-Attack, specifically tests the models' ability to leverage deep-level implicit features by focusing on edge information, the most salient explicit feature differentiating individuals across modalities. Edge-Attack utilizes a novel two-step approach. First, a multi-level edge feature extractor is trained in a self-supervised manner to capture discriminative edge representations for each individual. Second, a generative model based on Vision Transformer Generative Adversarial Networks (ViTGAN) is employed to generate adversarial patches conditioned on the extracted edge features. By applying these patches to pedestrian clothing, we create realistic, physically-realizable adversarial samples. This black-box, self-supervised approach ensures the generalizability of our attack against various VI-ReID models. Extensive experiments on SYSU-MM01 and RegDB datasets, including real-world deployments, demonstrate the effectiveness of Edge- Attack in significantly degrading the performance of state-of-the-art VI-ReID methods.  
  </ol>  
</details>  
  
  



## NeRF  

### [EEG-Driven 3D Object Reconstruction with Color Consistency and Diffusion Prior](http://arxiv.org/abs/2410.20981)  
Xin Xiang, Wenhui Zhou, Guojun Dai  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    EEG-based visual perception reconstruction has become a current research hotspot. Neuroscientific studies have shown that humans can perceive various types of visual information, such as color, shape, and texture, when observing objects. However, existing technical methods often face issues such as inconsistencies in texture, shape, and color between the visual stimulus images and the reconstructed images. In this paper, we propose a method for reconstructing 3D objects with color consistency based on EEG signals. The method adopts a two-stage strategy: in the first stage, we train an implicit neural EEG encoder with the capability of perceiving 3D objects, enabling it to capture regional semantic features; in the second stage, based on the latent EEG codes obtained in the first stage, we integrate a diffusion model, neural style loss, and NeRF to implicitly decode the 3D objects. Finally, through experimental validation, we demonstrate that our method can reconstruct 3D objects with color consistency using EEG.  
  </ol>  
</details>  
  
### [ODGS: 3D Scene Reconstruction from Omnidirectional Images with 3D Gaussian Splattings](http://arxiv.org/abs/2410.20686)  
Suyoung Lee, Jaeyoung Chung, Jaeyoo Huh, Kyoung Mu Lee  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Omnidirectional (or 360-degree) images are increasingly being used for 3D applications since they allow the rendering of an entire scene with a single image. Existing works based on neural radiance fields demonstrate successful 3D reconstruction quality on egocentric videos, yet they suffer from long training and rendering times. Recently, 3D Gaussian splatting has gained attention for its fast optimization and real-time rendering. However, directly using a perspective rasterizer to omnidirectional images results in severe distortion due to the different optical properties between two image domains. In this work, we present ODGS, a novel rasterization pipeline for omnidirectional images, with geometric interpretation. For each Gaussian, we define a tangent plane that touches the unit sphere and is perpendicular to the ray headed toward the Gaussian center. We then leverage a perspective camera rasterizer to project the Gaussian onto the corresponding tangent plane. The projected Gaussians are transformed and combined into the omnidirectional image, finalizing the omnidirectional rasterization process. This interpretation reveals the implicit assumptions within the proposed pipeline, which we verify through mathematical proofs. The entire rasterization process is parallelized using CUDA, achieving optimization and rendering speeds 100 times faster than NeRF-based methods. Our comprehensive experiments highlight the superiority of ODGS by delivering the best reconstruction and perceptual quality across various datasets. Additionally, results on roaming datasets demonstrate that ODGS restores fine details effectively, even when reconstructing large 3D scenes. The source code is available on our project page (https://github.com/esw0116/ODGS).  
  </ol>  
</details>  
  
### [GUMBEL-NERF: Representing Unseen Objects as Part-Compositional Neural Radiance Fields](http://arxiv.org/abs/2410.20306)  
Yusuke Sekikawa, Chingwei Hsu, Satoshi Ikehata, Rei Kawakami, Ikuro Sato  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose Gumbel-NeRF, a mixture-of-expert (MoE) neural radiance fields (NeRF) model with a hindsight expert selection mechanism for synthesizing novel views of unseen objects. Previous studies have shown that the MoE structure provides high-quality representations of a given large-scale scene consisting of many objects. However, we observe that such a MoE NeRF model often produces low-quality representations in the vicinity of experts' boundaries when applied to the task of novel view synthesis of an unseen object from one/few-shot input. We find that this deterioration is primarily caused by the foresight expert selection mechanism, which may leave an unnatural discontinuity in the object shape near the experts' boundaries. Gumbel-NeRF adopts a hindsight expert selection mechanism, which guarantees continuity in the density field even near the experts' boundaries. Experiments using the SRN cars dataset demonstrate the superiority of Gumbel-NeRF over the baselines in terms of various image quality metrics.  
  </ol>  
</details>  
**comments**: 7 pages. Presented at ICIP2024  
  
  



