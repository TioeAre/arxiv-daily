<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Fast-Global-Localization-on-Neural-Radiance-Field>Fast Global Localization on Neural Radiance Field</a></li>
        <li><a href=#TutteNet:-Injective-3D-Deformations-by-Composition-of-2D-Mesh-Deformations>TutteNet: Injective 3D Deformations by Composition of 2D Mesh Deformations</a></li>
        <li><a href=#DistillNeRF:-Perceiving-3D-Scenes-from-Single-Glance-Images-by-Distilling-Neural-Fields-and-Foundation-Model-Features>DistillNeRF: Perceiving 3D Scenes from Single-Glance Images by Distilling Neural Fields and Foundation Model Features</a></li>
        <li><a href=#Uncertainty-modeling-for-fine-tuned-implicit-functions>Uncertainty modeling for fine-tuned implicit functions</a></li>
      </ul>
    </li>
  </ol>
</details>

## NeRF  

### [Fast Global Localization on Neural Radiance Field](http://arxiv.org/abs/2406.12202)  
Mangyu Kong, Seongwon Lee, Jaewon Lee, Euntai Kim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) presented a novel way to represent scenes, allowing for high-quality 3D reconstruction from 2D images. Following its remarkable achievements, global localization within NeRF maps is an essential task for enabling a wide range of applications. Recently, Loc-NeRF demonstrated a localization approach that combines traditional Monte Carlo Localization with NeRF, showing promising results for using NeRF as an environment map. However, despite its advancements, Loc-NeRF encounters the challenge of a time-intensive ray rendering process, which can be a significant limitation in practical applications. To address this issue, we introduce Fast Loc-NeRF, which leverages a coarse-to-fine approach to enable more efficient and accurate NeRF map-based global localization. Specifically, Fast Loc-NeRF matches rendered pixels and observed images on a multi-resolution from low to high resolution. As a result, it speeds up the costly particle update process while maintaining precise localization results. Additionally, to reject the abnormal particles, we propose particle rejection weighting, which estimates the uncertainty of particles by exploiting NeRF's characteristics and considers them in the particle weighting process. Our Fast Loc-NeRF sets new state-of-the-art localization performances on several benchmarks, convincing its accuracy and efficiency.  
  </ol>  
</details>  
**comments**: Preprint, Under review  
  
### [TutteNet: Injective 3D Deformations by Composition of 2D Mesh Deformations](http://arxiv.org/abs/2406.12121)  
Bo Sun, Thibault Groueix, Chen Song, Qixing Huang, Noam Aigerman  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This work proposes a novel representation of injective deformations of 3D space, which overcomes existing limitations of injective methods: inaccuracy, lack of robustness, and incompatibility with general learning and optimization frameworks. The core idea is to reduce the problem to a deep composition of multiple 2D mesh-based piecewise-linear maps. Namely, we build differentiable layers that produce mesh deformations through Tutte's embedding (guaranteed to be injective in 2D), and compose these layers over different planes to create complex 3D injective deformations of the 3D volume. We show our method provides the ability to efficiently and accurately optimize and learn complex deformations, outperforming other injective approaches. As a main application, we produce complex and artifact-free NeRF and SDF deformations.  
  </ol>  
</details>  
  
### [DistillNeRF: Perceiving 3D Scenes from Single-Glance Images by Distilling Neural Fields and Foundation Model Features](http://arxiv.org/abs/2406.12095)  
Letian Wang, Seung Wook Kim, Jiawei Yang, Cunjun Yu, Boris Ivanovic, Steven L. Waslander, Yue Wang, Sanja Fidler, Marco Pavone, Peter Karkus  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose DistillNeRF, a self-supervised learning framework addressing the challenge of understanding 3D environments from limited 2D observations in autonomous driving. Our method is a generalizable feedforward model that predicts a rich neural scene representation from sparse, single-frame multi-view camera inputs, and is trained self-supervised with differentiable rendering to reconstruct RGB, depth, or feature images. Our first insight is to exploit per-scene optimized Neural Radiance Fields (NeRFs) by generating dense depth and virtual camera targets for training, thereby helping our model to learn 3D geometry from sparse non-overlapping image inputs. Second, to learn a semantically rich 3D representation, we propose distilling features from pre-trained 2D foundation models, such as CLIP or DINOv2, thereby enabling various downstream tasks without the need for costly 3D human annotations. To leverage these two insights, we introduce a novel model architecture with a two-stage lift-splat-shoot encoder and a parameterized sparse hierarchical voxel representation. Experimental results on the NuScenes dataset demonstrate that DistillNeRF significantly outperforms existing comparable self-supervised methods for scene reconstruction, novel view synthesis, and depth estimation; and it allows for competitive zero-shot 3D semantic occupancy prediction, as well as open-world scene understanding through distilled foundation model features. Demos and code will be available at https://distillnerf.github.io/.  
  </ol>  
</details>  
  
### [Uncertainty modeling for fine-tuned implicit functions](http://arxiv.org/abs/2406.12082)  
Anna Susmelj, Mael Macuglia, Nataša Tagasovska, Reto Sutter, Sebastiano Caprara, Jean-Philippe Thiran, Ender Konukoglu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Implicit functions such as Neural Radiance Fields (NeRFs), occupancy networks, and signed distance functions (SDFs) have become pivotal in computer vision for reconstructing detailed object shapes from sparse views. Achieving optimal performance with these models can be challenging due to the extreme sparsity of inputs and distribution shifts induced by data corruptions. To this end, large, noise-free synthetic datasets can serve as shape priors to help models fill in gaps, but the resulting reconstructions must be approached with caution. Uncertainty estimation is crucial for assessing the quality of these reconstructions, particularly in identifying areas where the model is uncertain about the parts it has inferred from the prior. In this paper, we introduce Dropsembles, a novel method for uncertainty estimation in tuned implicit functions. We demonstrate the efficacy of our approach through a series of experiments, starting with toy examples and progressing to a real-world scenario. Specifically, we train a Convolutional Occupancy Network on synthetic anatomical data and test it on low-resolution MRI segmentations of the lumbar spine. Our results show that Dropsembles achieve the accuracy and calibration levels of deep ensembles but with significantly less computational cost.  
  </ol>  
</details>  
  
  



