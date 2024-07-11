<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Raising-the-Ceiling:-Conflict-Free-Local-Feature-Matching-with-Dynamic-View-Switching>Raising the Ceiling: Conflict-Free Local Feature Matching with Dynamic View Switching</a></li>
        <li><a href=#Mutual-Information-calculation-on-different-appearances>Mutual Information calculation on different appearances</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Protecting-NeRFs'-Copyright-via-Plug-And-Play-Watermarking-Base-Model>Protecting NeRFs' Copyright via Plug-And-Play Watermarking Base Model</a></li>
        <li><a href=#Drantal-NeRF:-Diffusion-Based-Restoration-for-Anti-aliasing-Neural-Radiance-Field>Drantal-NeRF: Diffusion-Based Restoration for Anti-aliasing Neural Radiance Field</a></li>
        <li><a href=#Reference-based-Controllable-Scene-Stylization-with-Gaussian-Splatting>Reference-based Controllable Scene Stylization with Gaussian Splatting</a></li>
      </ul>
    </li>
  </ol>
</details>

## Image Matching  

### [Raising the Ceiling: Conflict-Free Local Feature Matching with Dynamic View Switching](http://arxiv.org/abs/2407.07789)  
Xiaoyong Lu, Songlin Du  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Current feature matching methods prioritize improving modeling capabilities to better align outputs with ground-truth matches, which are the theoretical upper bound on matching results, metaphorically depicted as the "ceiling". However, these enhancements fail to address the underlying issues that directly hinder ground-truth matches, including the scarcity of matchable points in small scale images, matching conflicts in dense methods, and the keypoint-repeatability reliance in sparse methods. We propose a novel feature matching method named RCM, which Raises the Ceiling of Matching from three aspects. 1) RCM introduces a dynamic view switching mechanism to address the scarcity of matchable points in source images by strategically switching image pairs. 2) RCM proposes a conflict-free coarse matching module, addressing matching conflicts in the target image through a many-to-one matching strategy. 3) By integrating the semi-sparse paradigm and the coarse-to-fine architecture, RCM preserves the benefits of both high efficiency and global search, mitigating the reliance on keypoint repeatability. As a result, RCM enables more matchable points in the source image to be matched in an exhaustive and conflict-free manner in the target image, leading to a substantial 260% increase in ground-truth matches. Comprehensive experiments show that RCM exhibits remarkable performance and efficiency in comparison to state-of-the-art methods.  
  </ol>  
</details>  
**comments**: Accepted at ECCV 2024  
  
### [Mutual Information calculation on different appearances](http://arxiv.org/abs/2407.07410)  
Jiecheng Liao, Junhao Lu, Jeff Ji, Jiacheng He  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Mutual information has many applications in image alignment and matching, mainly due to its ability to measure the statistical dependence between two images, even if the two images are from different modalities (e.g., CT and MRI). It considers not only the pixel intensities of the images but also the spatial relationships between the pixels. In this project, we apply the mutual information formula to image matching, where image A is the moving object and image B is the target object and calculate the mutual information between them to evaluate the similarity between the images. For comparison, we also used entropy and information-gain methods to test the dependency of the images. We also investigated the effect of different environments on the mutual information of the same image and used experiments and plots to demonstrate.  
  </ol>  
</details>  
**comments**: demo for the work: elucidator.cn/demo-mi/  
  
  



## NeRF  

### [Protecting NeRFs' Copyright via Plug-And-Play Watermarking Base Model](http://arxiv.org/abs/2407.07735)  
Qi Song, Ziyuan Luo, Ka Chun Cheung, Simon See, Renjie Wan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRFs) have become a key method for 3D scene representation. With the rising prominence and influence of NeRF, safeguarding its intellectual property has become increasingly important. In this paper, we propose \textbf{NeRFProtector}, which adopts a plug-and-play strategy to protect NeRF's copyright during its creation. NeRFProtector utilizes a pre-trained watermarking base model, enabling NeRF creators to embed binary messages directly while creating their NeRF. Our plug-and-play property ensures NeRF creators can flexibly choose NeRF variants without excessive modifications. Leveraging our newly designed progressive distillation, we demonstrate performance on par with several leading-edge neural rendering methods. Our project is available at: \url{https://qsong2001.github.io/NeRFProtector}.  
  </ol>  
</details>  
**comments**: Accepted by ECCV2024  
  
### [Drantal-NeRF: Diffusion-Based Restoration for Anti-aliasing Neural Radiance Field](http://arxiv.org/abs/2407.07461)  
Ganlin Yang, Kaidong Zhang, Jingjing Fu, Dong Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Aliasing artifacts in renderings produced by Neural Radiance Field (NeRF) is a long-standing but complex issue in the field of 3D implicit representation, which arises from a multitude of intricate causes and was mitigated by designing more advanced but complex scene parameterization methods before. In this paper, we present a Diffusion-based restoration method for anti-aliasing Neural Radiance Field (Drantal-NeRF). We consider the anti-aliasing issue from a low-level restoration perspective by viewing aliasing artifacts as a kind of degradation model added to clean ground truths. By leveraging the powerful prior knowledge encapsulated in diffusion model, we could restore the high-realism anti-aliasing renderings conditioned on aliased low-quality counterparts. We further employ a feature-wrapping operation to ensure multi-view restoration consistency and finetune the VAE decoder to better adapt to the scene-specific data distribution. Our proposed method is easy to implement and agnostic to various NeRF backbones. We conduct extensive experiments on challenging large-scale urban scenes as well as unbounded 360-degree scenes and achieve substantial qualitative and quantitative improvements.  
  </ol>  
</details>  
  
### [Reference-based Controllable Scene Stylization with Gaussian Splatting](http://arxiv.org/abs/2407.07220)  
Yiqun Mei, Jiacong Xu, Vishal M. Patel  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Referenced-based scene stylization that edits the appearance based on a content-aligned reference image is an emerging research area. Starting with a pretrained neural radiance field (NeRF), existing methods typically learn a novel appearance that matches the given style. Despite their effectiveness, they inherently suffer from time-consuming volume rendering, and thus are impractical for many real-time applications. In this work, we propose ReGS, which adapts 3D Gaussian Splatting (3DGS) for reference-based stylization to enable real-time stylized view synthesis. Editing the appearance of a pretrained 3DGS is challenging as it uses discrete Gaussians as 3D representation, which tightly bind appearance with geometry. Simply optimizing the appearance as prior methods do is often insufficient for modeling continuous textures in the given reference image. To address this challenge, we propose a novel texture-guided control mechanism that adaptively adjusts local responsible Gaussians to a new geometric arrangement, serving for desired texture details. The proposed process is guided by texture clues for effective appearance editing, and regularized by scene depth for preserving original geometric structure. With these novel designs, we show ReGs can produce state-of-the-art stylization results that respect the reference texture while embracing real-time rendering speed for free-view navigation.  
  </ol>  
</details>  
  
  



