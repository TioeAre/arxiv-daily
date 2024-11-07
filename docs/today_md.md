<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Structure-Consistent-Gaussian-Splatting-with-Matching-Prior-for-Few-shot-Novel-View-Synthesis>Structure Consistent Gaussian Splatting with Matching Prior for Few-shot Novel View Synthesis</a></li>
        <li><a href=#Enhancing-Exploratory-Capability-of-Visual-Navigation-Using-Uncertainty-of-Implicit-Scene-Representation>Enhancing Exploratory Capability of Visual Navigation Using Uncertainty of Implicit Scene Representation</a></li>
      </ul>
    </li>
  </ol>
</details>

## NeRF  

### [Structure Consistent Gaussian Splatting with Matching Prior for Few-shot Novel View Synthesis](http://arxiv.org/abs/2411.03637)  
[[code](https://github.com/prstrive/scgaussian)]  
Rui Peng, Wangze Xu, Luyang Tang, Liwei Liao, Jianbo Jiao, Ronggang Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Despite the substantial progress of novel view synthesis, existing methods, either based on the Neural Radiance Fields (NeRF) or more recently 3D Gaussian Splatting (3DGS), suffer significant degradation when the input becomes sparse. Numerous efforts have been introduced to alleviate this problem, but they still struggle to synthesize satisfactory results efficiently, especially in the large scene. In this paper, we propose SCGaussian, a Structure Consistent Gaussian Splatting method using matching priors to learn 3D consistent scene structure. Considering the high interdependence of Gaussian attributes, we optimize the scene structure in two folds: rendering geometry and, more importantly, the position of Gaussian primitives, which is hard to be directly constrained in the vanilla 3DGS due to the non-structure property. To achieve this, we present a hybrid Gaussian representation. Besides the ordinary non-structure Gaussian primitives, our model also consists of ray-based Gaussian primitives that are bound to matching rays and whose optimization of their positions is restricted along the ray. Thus, we can utilize the matching correspondence to directly enforce the position of these Gaussian primitives to converge to the surface points where rays intersect. Extensive experiments on forward-facing, surrounding, and complex large scenes show the effectiveness of our approach with state-of-the-art performance and high efficiency. Code is available at https://github.com/prstrive/SCGaussian.  
  </ol>  
</details>  
**comments**: NeurIPS 2024 Accepted  
  
### [Enhancing Exploratory Capability of Visual Navigation Using Uncertainty of Implicit Scene Representation](http://arxiv.org/abs/2411.03487)  
Yichen Wang, Qiming Liu, Zhe Liu, Hesheng Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In the context of visual navigation in unknown scenes, both "exploration" and "exploitation" are equally crucial. Robots must first establish environmental cognition through exploration and then utilize the cognitive information to accomplish target searches. However, most existing methods for image-goal navigation prioritize target search over the generation of exploratory behavior. To address this, we propose the Navigation with Uncertainty-driven Exploration (NUE) pipeline, which uses an implicit and compact scene representation, NeRF, as a cognitive structure. We estimate the uncertainty of NeRF and augment the exploratory ability by the uncertainty to in turn facilitate the construction of implicit representation. Simultaneously, we extract memory information from NeRF to enhance the robot's reasoning ability for determining the location of the target. Ultimately, we seamlessly combine the two generated abilities to produce navigational actions. Our pipeline is end-to-end, with the environmental cognitive structure being constructed online. Extensive experimental results on image-goal navigation demonstrate the capability of our pipeline to enhance exploratory behaviors, while also enabling a natural transition from the exploration to exploitation phase. This enables our model to outperform existing memory-based cognitive navigation structures in terms of navigation performance.  
  </ol>  
</details>  
  
  



