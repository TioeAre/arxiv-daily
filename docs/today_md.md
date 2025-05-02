<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Cues3D:-Unleashing-the-Power-of-Sole-NeRF-for-Consistent-and-Unique-Instances-in-Open-Vocabulary-3D-Panoptic-Segmentation>Cues3D: Unleashing the Power of Sole NeRF for Consistent and Unique Instances in Open-Vocabulary 3D Panoptic Segmentation</a></li>
      </ul>
    </li>
  </ol>
</details>

## NeRF  

### [Cues3D: Unleashing the Power of Sole NeRF for Consistent and Unique Instances in Open-Vocabulary 3D Panoptic Segmentation](http://arxiv.org/abs/2505.00378)  
Feng Xue, Wenzhuang Xu, Guofeng Zhong, Anlong Minga, Nicu Sebe  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Open-vocabulary 3D panoptic segmentation has recently emerged as a significant trend. Top-performing methods currently integrate 2D segmentation with geometry-aware 3D primitives. However, the advantage would be lost without high-fidelity 3D point clouds, such as methods based on Neural Radiance Field (NeRF). These methods are limited by the insufficient capacity to maintain consistency across partial observations. To address this, recent works have utilized contrastive loss or cross-view association pre-processing for view consensus. In contrast to them, we present Cues3D, a compact approach that relies solely on NeRF instead of pre-associations. The core idea is that NeRF's implicit 3D field inherently establishes a globally consistent geometry, enabling effective object distinction without explicit cross-view supervision. We propose a three-phase training framework for NeRF, initialization-disambiguation-refinement, whereby the instance IDs are corrected using the initially-learned knowledge. Additionally, an instance disambiguation method is proposed to match NeRF-rendered 3D masks and ensure globally unique 3D instance identities. With the aid of Cues3D, we obtain highly consistent and unique 3D instance ID for each object across views with a balanced version of NeRF. Our experiments are conducted on ScanNet v2, ScanNet200, ScanNet++, and Replica datasets for 3D instance, panoptic, and semantic segmentation tasks. Cues3D outperforms other 2D image-based methods and competes with the latest 2D-3D merging based methods, while even surpassing them when using additional 3D point clouds. The code link could be found in the appendix and will be released on \href{https://github.com/mRobotit/Cues3D}{github}  
  </ol>  
</details>  
**comments**: Accepted by Information Fusion  
  
  



