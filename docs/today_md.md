<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#XPoint:-A-Self-Supervised-Visual-State-Space-based-Architecture-for-Multispectral-Image-Registration>XPoint: A Self-Supervised Visual-State-Space based Architecture for Multispectral Image Registration</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Material-Transforms-from-Disentangled-NeRF-Representations>Material Transforms from Disentangled NeRF Representations</a></li>
      </ul>
    </li>
  </ol>
</details>

## Image Matching  

### [XPoint: A Self-Supervised Visual-State-Space based Architecture for Multispectral Image Registration](http://arxiv.org/abs/2411.07430)  
[[code](https://github.com/canyagmur/xpoint)]  
Ismail Can Yagmur, Hasan F. Ates, Bahadir K. Gunturk  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Accurate multispectral image matching presents significant challenges due to non-linear intensity variations across spectral modalities, extreme viewpoint changes, and the scarcity of labeled datasets. Current state-of-the-art methods are typically specialized for a single spectral difference, such as visibleinfrared, and struggle to adapt to other modalities due to their reliance on expensive supervision, such as depth maps or camera poses. To address the need for rapid adaptation across modalities, we introduce XPoint, a self-supervised, modular image-matching framework designed for adaptive training and fine-tuning on aligned multispectral datasets, allowing users to customize key components based on their specific tasks. XPoint employs modularity and self-supervision to allow for the adjustment of elements such as the base detector, which generates pseudoground truth keypoints invariant to viewpoint and spectrum variations. The framework integrates a VMamba encoder, pretrained on segmentation tasks, for robust feature extraction, and includes three joint decoder heads: two are dedicated to interest point and descriptor extraction; and a task-specific homography regression head imposes geometric constraints for superior performance in tasks like image registration. This flexible architecture enables quick adaptation to a wide range of modalities, demonstrated by training on Optical-Thermal data and fine-tuning on settings such as visual-near infrared, visual-infrared, visual-longwave infrared, and visual-synthetic aperture radar. Experimental results show that XPoint consistently outperforms or matches state-ofthe-art methods in feature matching and image registration tasks across five distinct multispectral datasets. Our source code is available at https://github.com/canyagmur/XPoint.  
  </ol>  
</details>  
**comments**: 13 pages, 11 figures, 1 table, Journal  
  
  



## NeRF  

### [Material Transforms from Disentangled NeRF Representations](http://arxiv.org/abs/2411.08037)  
[[code](https://github.com/astra-vision/brdftransform)]  
Ivan Lopes, Jean-François Lalonde, Raoul de Charette  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we first propose a novel method for transferring material transformations across different scenes. Building on disentangled Neural Radiance Field (NeRF) representations, our approach learns to map Bidirectional Reflectance Distribution Functions (BRDF) from pairs of scenes observed in varying conditions, such as dry and wet. The learned transformations can then be applied to unseen scenes with similar materials, therefore effectively rendering the transformation learned with an arbitrary level of intensity. Extensive experiments on synthetic scenes and real-world objects validate the effectiveness of our approach, showing that it can learn various transformations such as wetness, painting, coating, etc. Our results highlight not only the versatility of our method but also its potential for practical applications in computer graphics. We publish our method implementation, along with our synthetic/real datasets on https://github.com/astra-vision/BRDFTransform  
  </ol>  
</details>  
  
  



