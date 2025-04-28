<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#From-Mapping-to-Composing:-A-Two-Stage-Framework-for-Zero-shot-Composed-Image-Retrieval>From Mapping to Composing: A Two-Stage Framework for Zero-shot Composed Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#RGS-DR:-Reflective-Gaussian-Surfels-with-Deferred-Rendering-for-Shiny-Objects>RGS-DR: Reflective Gaussian Surfels with Deferred Rendering for Shiny Objects</a></li>
        <li><a href=#Visibility-Uncertainty-guided-3D-Gaussian-Inpainting-via-Scene-Conceptional-Learning>Visibility-Uncertainty-guided 3D Gaussian Inpainting via Scene Conceptional Learning</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [From Mapping to Composing: A Two-Stage Framework for Zero-shot Composed Image Retrieval](http://arxiv.org/abs/2504.17990)  
Yabing Wang, Zhuotao Tian, Qingpei Guo, Zheng Qin, Sanping Zhou, Ming Yang, Le Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed Image Retrieval (CIR) is a challenging multimodal task that retrieves a target image based on a reference image and accompanying modification text. Due to the high cost of annotating CIR triplet datasets, zero-shot (ZS) CIR has gained traction as a promising alternative. Existing studies mainly focus on projection-based methods, which map an image to a single pseudo-word token. However, these methods face three critical challenges: (1) insufficient pseudo-word token representation capacity, (2) discrepancies between training and inference phases, and (3) reliance on large-scale synthetic data. To address these issues, we propose a two-stage framework where the training is accomplished from mapping to composing. In the first stage, we enhance image-to-pseudo-word token learning by introducing a visual semantic injection module and a soft text alignment objective, enabling the token to capture richer and fine-grained image information. In the second stage, we optimize the text encoder using a small amount of synthetic triplet data, enabling it to effectively extract compositional semantics by combining pseudo-word tokens with modification text for accurate target image retrieval. The strong visual-to-pseudo mapping established in the first stage provides a solid foundation for the second stage, making our approach compatible with both high- and low-quality synthetic data, and capable of achieving significant performance gains with only a small amount of synthetic data. Extensive experiments were conducted on three public datasets, achieving superior performance compared to existing approaches.  
  </ol>  
</details>  
  
  



## NeRF  

### [RGS-DR: Reflective Gaussian Surfels with Deferred Rendering for Shiny Objects](http://arxiv.org/abs/2504.18468)  
Georgios Kouros, Minye Wu, Tinne Tuytelaars  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce RGS-DR, a novel inverse rendering method for reconstructing and rendering glossy and reflective objects with support for flexible relighting and scene editing. Unlike existing methods (e.g., NeRF and 3D Gaussian Splatting), which struggle with view-dependent effects, RGS-DR utilizes a 2D Gaussian surfel representation to accurately estimate geometry and surface normals, an essential property for high-quality inverse rendering. Our approach explicitly models geometric and material properties through learnable primitives rasterized into a deferred shading pipeline, effectively reducing rendering artifacts and preserving sharp reflections. By employing a multi-level cube mipmap, RGS-DR accurately approximates environment lighting integrals, facilitating high-quality reconstruction and relighting. A residual pass with spherical-mipmap-based directional encoding further refines the appearance modeling. Experiments demonstrate that RGS-DR achieves high-quality reconstruction and rendering quality for shiny objects, often outperforming reconstruction-exclusive state-of-the-art methods incapable of relighting.  
  </ol>  
</details>  
  
### [Visibility-Uncertainty-guided 3D Gaussian Inpainting via Scene Conceptional Learning](http://arxiv.org/abs/2504.17815)  
Mingxuan Cui, Qing Guo, Yuyi Wang, Hongkai Yu, Di Lin, Qin Zou, Ming-Ming Cheng, Xi Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D Gaussian Splatting (3DGS) has emerged as a powerful and efficient 3D representation for novel view synthesis. This paper extends 3DGS capabilities to inpainting, where masked objects in a scene are replaced with new contents that blend seamlessly with the surroundings. Unlike 2D image inpainting, 3D Gaussian inpainting (3DGI) is challenging in effectively leveraging complementary visual and semantic cues from multiple input views, as occluded areas in one view may be visible in others. To address this, we propose a method that measures the visibility uncertainties of 3D points across different input views and uses them to guide 3DGI in utilizing complementary visual cues. We also employ uncertainties to learn a semantic concept of scene without the masked object and use a diffusion model to fill masked objects in input images based on the learned concept. Finally, we build a novel 3DGI framework, VISTA, by integrating VISibility-uncerTainty-guided 3DGI with scene conceptuAl learning. VISTA generates high-quality 3DGS models capable of synthesizing artifact-free and naturally inpainted novel views. Furthermore, our approach extends to handling dynamic distractors arising from temporal object changes, enhancing its versatility in diverse scene reconstruction scenarios. We demonstrate the superior performance of our method over state-of-the-art techniques using two challenging datasets: the SPIn-NeRF dataset, featuring 10 diverse static 3D inpainting scenes, and an underwater 3D inpainting dataset derived from UTB180, including fast-moving fish as inpainting targets.  
  </ol>  
</details>  
**comments**: 14 pages, 12 figures, ICCV  
  
  



