<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Uncertainty-Quantification-Framework-for-Aerial-and-UAV-Photogrammetry-through-Error-Propagation>Uncertainty Quantification Framework for Aerial and UAV Photogrammetry through Error Propagation</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#TimeNeRF:-Building-Generalizable-Neural-Radiance-Fields-across-Time-from-Few-Shot-Input-Views>TimeNeRF: Building Generalizable Neural Radiance Fields across Time from Few-Shot Input Views</a></li>
        <li><a href=#EPSilon:-Efficient-Point-Sampling-for-Lightening-of-Hybrid-based-3D-Avatar-Generation>EPSilon: Efficient Point Sampling for Lightening of Hybrid-based 3D Avatar Generation</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Uncertainty Quantification Framework for Aerial and UAV Photogrammetry through Error Propagation](http://arxiv.org/abs/2507.13486)  
Debao Huang, Rongjun Qin  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Uncertainty quantification of the photogrammetry process is essential for providing per-point accuracy credentials of the point clouds. Unlike airborne LiDAR, which typically delivers consistent accuracy across various scenes, the accuracy of photogrammetric point clouds is highly scene-dependent, since it relies on algorithm-generated measurements (i.e., stereo or multi-view stereo). Generally, errors of the photogrammetric point clouds propagate through a two-step process: Structure-from-Motion (SfM) with Bundle adjustment (BA), followed by Multi-view Stereo (MVS). While uncertainty estimation in the SfM stage has been well studied using the first-order statistics of the reprojection error function, that in the MVS stage remains largely unsolved and non-standardized, primarily due to its non-differentiable and multi-modal nature (i.e., from pixel values to geometry). In this paper, we present an uncertainty quantification framework closing this gap by associating an error covariance matrix per point accounting for this two-step photogrammetry process. Specifically, to estimate the uncertainty in the MVS stage, we propose a novel, self-calibrating method by taking reliable n-view points (n>=6) per-view to regress the disparity uncertainty using highly relevant cues (such as matching cost values) from the MVS stage. Compared to existing approaches, our method uses self-contained, reliable 3D points extracted directly from the MVS process, with the benefit of being self-supervised and naturally adhering to error propagation path of the photogrammetry process, thereby providing a robust and certifiable uncertainty quantification across diverse scenes. We evaluate the framework using a variety of publicly available airborne and UAV imagery datasets. Results demonstrate that our method outperforms existing approaches by achieving high bounding rates without overestimating uncertainty.  
  </ol>  
</details>  
**comments**: 16 pages, 9 figures, this manuscript has been submitted to ISPRS
  Journal of Photogrammetry and Remote Sensing for consideration  
  
  



## NeRF  

### [TimeNeRF: Building Generalizable Neural Radiance Fields across Time from Few-Shot Input Views](http://arxiv.org/abs/2507.13929)  
Hsiang-Hui Hung, Huu-Phu Do, Yung-Hui Li, Ching-Chun Huang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present TimeNeRF, a generalizable neural rendering approach for rendering novel views at arbitrary viewpoints and at arbitrary times, even with few input views. For real-world applications, it is expensive to collect multiple views and inefficient to re-optimize for unseen scenes. Moreover, as the digital realm, particularly the metaverse, strives for increasingly immersive experiences, the ability to model 3D environments that naturally transition between day and night becomes paramount. While current techniques based on Neural Radiance Fields (NeRF) have shown remarkable proficiency in synthesizing novel views, the exploration of NeRF's potential for temporal 3D scene modeling remains limited, with no dedicated datasets available for this purpose. To this end, our approach harnesses the strengths of multi-view stereo, neural radiance fields, and disentanglement strategies across diverse datasets. This equips our model with the capability for generalizability in a few-shot setting, allows us to construct an implicit content radiance field for scene representation, and further enables the building of neural radiance fields at any arbitrary time. Finally, we synthesize novel views of that time via volume rendering. Experiments show that TimeNeRF can render novel views in a few-shot setting without per-scene optimization. Most notably, it excels in creating realistic novel views that transition smoothly across different times, adeptly capturing intricate natural scene changes from dawn to dusk.  
  </ol>  
</details>  
**comments**: Accepted by MM 2024  
  
### [EPSilon: Efficient Point Sampling for Lightening of Hybrid-based 3D Avatar Generation](http://arxiv.org/abs/2507.13648)  
Seungjun Moon, Sangjoon Yu, Gyeong-Moon Park  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The rapid advancement of neural radiance fields (NeRF) has paved the way to generate animatable human avatars from a monocular video. However, the sole usage of NeRF suffers from a lack of details, which results in the emergence of hybrid representation that utilizes SMPL-based mesh together with NeRF representation. While hybrid-based models show photo-realistic human avatar generation qualities, they suffer from extremely slow inference due to their deformation scheme: to be aligned with the mesh, hybrid-based models use the deformation based on SMPL skinning weights, which needs high computational costs on each sampled point. We observe that since most of the sampled points are located in empty space, they do not affect the generation quality but result in inference latency with deformation. In light of this observation, we propose EPSilon, a hybrid-based 3D avatar generation scheme with novel efficient point sampling strategies that boost both training and inference. In EPSilon, we propose two methods to omit empty points at rendering; empty ray omission (ERO) and empty interval omission (EIO). In ERO, we wipe out rays that progress through the empty space. Then, EIO narrows down the sampling interval on the ray, which wipes out the region not occupied by either clothes or mesh. The delicate sampling scheme of EPSilon enables not only great computational cost reduction during deformation but also the designation of the important regions to be sampled, which enables a single-stage NeRF structure without hierarchical sampling. Compared to existing methods, EPSilon maintains the generation quality while using only 3.9% of sampled points and achieves around 20 times faster inference, together with 4 times faster training convergence. We provide video results on https://github.com/seungjun-moon/epsilon.  
  </ol>  
</details>  
  
  



