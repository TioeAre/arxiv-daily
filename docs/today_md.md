<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Structured-Pruning-for-Efficient-Visual-Place-Recognition>Structured Pruning for Efficient Visual Place Recognition</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#DreamHOI:-Subject-Driven-Generation-of-3D-Human-Object-Interactions-with-Diffusion-Priors>DreamHOI: Subject-Driven Generation of 3D Human-Object Interactions with Diffusion Priors</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Structured Pruning for Efficient Visual Place Recognition](http://arxiv.org/abs/2409.07834)  
Oliver Grainge, Michael Milford, Indu Bodala, Sarvapali D. Ramchurn, Shoaib Ehsan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Place Recognition (VPR) is fundamental for the global re-localization of robots and devices, enabling them to recognize previously visited locations based on visual inputs. This capability is crucial for maintaining accurate mapping and localization over large areas. Given that VPR methods need to operate in real-time on embedded systems, it is critical to optimize these systems for minimal resource consumption. While the most efficient VPR approaches employ standard convolutional backbones with fixed descriptor dimensions, these often lead to redundancy in the embedding space as well as in the network architecture. Our work introduces a novel structured pruning method, to not only streamline common VPR architectures but also to strategically remove redundancies within the feature embedding space. This dual focus significantly enhances the efficiency of the system, reducing both map and model memory requirements and decreasing feature extraction and retrieval latencies. Our approach has reduced memory usage and latency by 21% and 16%, respectively, across models, while minimally impacting recall@1 accuracy by less than 1%. This significant improvement enhances real-time applications on edge devices with negligible accuracy loss.  
  </ol>  
</details>  
  
  



## NeRF  

### [DreamHOI: Subject-Driven Generation of 3D Human-Object Interactions with Diffusion Priors](http://arxiv.org/abs/2409.08278)  
Thomas Hanwen Zhu, Ruining Li, Tomas Jakab  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present DreamHOI, a novel method for zero-shot synthesis of human-object interactions (HOIs), enabling a 3D human model to realistically interact with any given object based on a textual description. This task is complicated by the varying categories and geometries of real-world objects and the scarcity of datasets encompassing diverse HOIs. To circumvent the need for extensive data, we leverage text-to-image diffusion models trained on billions of image-caption pairs. We optimize the articulation of a skinned human mesh using Score Distillation Sampling (SDS) gradients obtained from these models, which predict image-space edits. However, directly backpropagating image-space gradients into complex articulation parameters is ineffective due to the local nature of such gradients. To overcome this, we introduce a dual implicit-explicit representation of a skinned mesh, combining (implicit) neural radiance fields (NeRFs) with (explicit) skeleton-driven mesh articulation. During optimization, we transition between implicit and explicit forms, grounding the NeRF generation while refining the mesh articulation. We validate our approach through extensive experiments, demonstrating its effectiveness in generating realistic HOIs.  
  </ol>  
</details>  
**comments**: Project page: https://DreamHOI.github.io/  
  
  



