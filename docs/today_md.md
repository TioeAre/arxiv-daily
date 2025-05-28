<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Intern-GS:-Vision-Model-Guided-Sparse-View-3D-Gaussian-Splatting>Intern-GS: Vision Model Guided Sparse-View 3D Gaussian Splatting</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#ConText-CIR:-Learning-from-Concepts-in-Text-for-Composed-Image-Retrieval>ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval</a></li>
        <li><a href=#Visualized-Text-to-Image-Retrieval>Visualized Text-to-Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Structure-from-Collision>Structure from Collision</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Intern-GS: Vision Model Guided Sparse-View 3D Gaussian Splatting](http://arxiv.org/abs/2505.20729)  
Xiangyu Sun, Runnan Chen, Mingming Gong, Dong Xu, Tongliang Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Sparse-view scene reconstruction often faces significant challenges due to the constraints imposed by limited observational data. These limitations result in incomplete information, leading to suboptimal reconstructions using existing methodologies. To address this, we present Intern-GS, a novel approach that effectively leverages rich prior knowledge from vision foundation models to enhance the process of sparse-view Gaussian Splatting, thereby enabling high-quality scene reconstruction. Specifically, Intern-GS utilizes vision foundation models to guide both the initialization and the optimization process of 3D Gaussian splatting, effectively addressing the limitations of sparse inputs. In the initialization process, our method employs DUSt3R to generate a dense and non-redundant gaussian point cloud. This approach significantly alleviates the limitations encountered by traditional structure-from-motion (SfM) methods, which often struggle under sparse-view constraints. During the optimization process, vision foundation models predict depth and appearance for unobserved views, refining the 3D Gaussians to compensate for missing information in unseen regions. Extensive experiments demonstrate that Intern-GS achieves state-of-the-art rendering quality across diverse datasets, including both forward-facing and large-scale scenes, such as LLFF, DTU, and Tanks and Temples.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval](http://arxiv.org/abs/2505.20764)  
Eric Xing, Pranavi Kolouju, Robert Pless, Abby Stylianou, Nathan Jacobs  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed image retrieval (CIR) is the task of retrieving a target image specified by a query image and a relative text that describes a semantic modification to the query image. Existing methods in CIR struggle to accurately represent the image and the text modification, resulting in subpar performance. To address this limitation, we introduce a CIR framework, ConText-CIR, trained with a Text Concept-Consistency loss that encourages the representations of noun phrases in the text modification to better attend to the relevant parts of the query image. To support training with this loss function, we also propose a synthetic data generation pipeline that creates training data from existing CIR datasets or unlabeled images. We show that these components together enable stronger performance on CIR tasks, setting a new state-of-the-art in composed image retrieval in both the supervised and zero-shot settings on multiple benchmark datasets, including CIRR and CIRCO. Source code, model checkpoints, and our new datasets are available at https://github.com/mvrl/ConText-CIR.  
  </ol>  
</details>  
**comments**: 15 pages, 8 figures, 6 tables. CVPR 2025  
  
### [Visualized Text-to-Image Retrieval](http://arxiv.org/abs/2505.20291)  
Di Wu, Yixin Wan, Kai-Wei Chang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose Visualize-then-Retrieve (VisRet), a new paradigm for Text-to-Image (T2I) retrieval that mitigates the limitations of cross-modal similarity alignment of existing multi-modal embeddings. VisRet first projects textual queries into the image modality via T2I generation. Then, it performs retrieval within the image modality to bypass the weaknesses of cross-modal retrievers in recognizing subtle visual-spatial features. Experiments on three knowledge-intensive T2I retrieval benchmarks, including a newly introduced multi-entity benchmark, demonstrate that VisRet consistently improves T2I retrieval by 24.5% to 32.7% NDCG@10 across different embedding models. VisRet also significantly benefits downstream visual question answering accuracy when used in retrieval-augmented generation pipelines. The method is plug-and-play and compatible with off-the-shelf retrievers, making it an effective module for knowledge-intensive multi-modal systems. Our code and the new benchmark are publicly available at https://github.com/xiaowu0162/Visualize-then-Retrieve.  
  </ol>  
</details>  
**comments**: Work in Progress  
  
  



## NeRF  

### [Structure from Collision](http://arxiv.org/abs/2505.21335)  
Takuhiro Kaneko  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in neural 3D representations, such as neural radiance fields (NeRF) and 3D Gaussian splatting (3DGS), have enabled the accurate estimation of 3D structures from multiview images. However, this capability is limited to estimating the visible external structure, and identifying the invisible internal structure hidden behind the surface is difficult. To overcome this limitation, we address a new task called Structure from Collision (SfC), which aims to estimate the structure (including the invisible internal structure) of an object from appearance changes during collision. To solve this problem, we propose a novel model called SfC-NeRF that optimizes the invisible internal structure of an object through a video sequence under physical, appearance (i.e., visible external structure)-preserving, and keyframe constraints. In particular, to avoid falling into undesirable local optima owing to its ill-posed nature, we propose volume annealing; that is, searching for global optima by repeatedly reducing and expanding the volume. Extensive experiments on 115 objects involving diverse structures (i.e., various cavity shapes, locations, and sizes) and material properties revealed the properties of SfC and demonstrated the effectiveness of the proposed SfC-NeRF.  
  </ol>  
</details>  
**comments**: Accepted to CVPR 2025 (Highlight). Project page:
  https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/sfc/  
  
  



