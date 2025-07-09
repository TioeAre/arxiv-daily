<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Automatic-Synthesis-of-High-Quality-Triplet-Data-for-Composed-Image-Retrieval>Automatic Synthesis of High-Quality Triplet Data for Composed Image Retrieval</a></li>
        <li><a href=#OFFSET:-Segmentation-based-Focus-Shift-Revision-for-Composed-Image-Retrieval>OFFSET: Segmentation-based Focus Shift Revision for Composed Image Retrieval</a></li>
        <li><a href=#Llama-Nemoretriever-Colembed:-Top-Performing-Text-Image-Retrieval-Model>Llama Nemoretriever Colembed: Top-Performing Text-Image Retrieval Model</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Reflections-Unlock:-Geometry-Aware-Reflection-Disentanglement-in-3D-Gaussian-Splatting-for-Photorealistic-Scenes-Rendering>Reflections Unlock: Geometry-Aware Reflection Disentanglement in 3D Gaussian Splatting for Photorealistic Scenes Rendering</a></li>
        <li><a href=#DreamArt:-Generating-Interactable-Articulated-Objects-from-a-Single-Image>DreamArt: Generating Interactable Articulated Objects from a Single Image</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Automatic Synthesis of High-Quality Triplet Data for Composed Image Retrieval](http://arxiv.org/abs/2507.05970)  
Haiwen Li, Delong Liu, Zhaohui Hou, Zhicheng Zhao, Fei Su  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    As a challenging vision-language (VL) task, Composed Image Retrieval (CIR) aims to retrieve target images using multimodal (image+text) queries. Although many existing CIR methods have attained promising performance, their reliance on costly, manually labeled triplets hinders scalability and zero-shot capability. To address this issue, we propose a scalable pipeline for automatic triplet generation, along with a fully synthetic dataset named Composed Image Retrieval on High-quality Synthetic Triplets (CIRHS). Our pipeline leverages a large language model (LLM) to generate diverse prompts, controlling a text-to-image generative model to produce image pairs with identical elements in each pair, which are then filtered and reorganized to form the CIRHS dataset. In addition, we introduce Hybrid Contextual Alignment (CoAlign), a novel CIR framework, which can accomplish global alignment and local reasoning within a broader context, enabling the model to learn more robust and informative representations. By utilizing the synthetic CIRHS dataset, CoAlign achieves outstanding zero-shot performance on three commonly used benchmarks, demonstrating for the first time the feasibility of training CIR models on a fully synthetic dataset. Furthermore, under supervised training, our method outperforms all the state-of-the-art supervised CIR approaches, validating the effectiveness of our proposed retrieval framework. The code and the CIRHS dataset will be released soon.  
  </ol>  
</details>  
**comments**: This paper was originally submitted to ACM MM 2025 on April 12, 2025  
  
### [OFFSET: Segmentation-based Focus Shift Revision for Composed Image Retrieval](http://arxiv.org/abs/2507.05631)  
Zhiwei Chen, Yupeng Hu, Zixu Li, Zhiheng Fu, Xuemeng Song, Liqiang Nie  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed Image Retrieval (CIR) represents a novel retrieval paradigm that is capable of expressing users' intricate retrieval requirements flexibly. It enables the user to give a multimodal query, comprising a reference image and a modification text, and subsequently retrieve the target image. Notwithstanding the considerable advances made by prevailing methodologies, CIR remains in its nascent stages due to two limitations: 1) inhomogeneity between dominant and noisy portions in visual data is ignored, leading to query feature degradation, and 2) the priority of textual data in the image modification process is overlooked, which leads to a visual focus bias. To address these two limitations, this work presents a focus mapping-based feature extractor, which consists of two modules: dominant portion segmentation and dual focus mapping. It is designed to identify significant dominant portions in images and guide the extraction of visual and textual data features, thereby reducing the impact of noise interference. Subsequently, we propose a textually guided focus revision module, which can utilize the modification requirements implied in the text to perform adaptive focus revision on the reference image, thereby enhancing the perception of the modification focus on the composed features. The aforementioned modules collectively constitute the segmentatiOn-based Focus shiFt reviSion nETwork (\mbox{OFFSET}), and comprehensive experiments on four benchmark datasets substantiate the superiority of our proposed method. The codes and data are available on https://zivchen-ty.github.io/OFFSET.github.io/  
  </ol>  
</details>  
  
### [Llama Nemoretriever Colembed: Top-Performing Text-Image Retrieval Model](http://arxiv.org/abs/2507.05513)  
Mengyao Xu, Gabriel Moreira, Ronay Ak, Radek Osmulski, Yauhen Babakhin, Zhiding Yu, Benedikt Schifferer, Even Oldridge  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Motivated by the growing demand for retrieval systems that operate across modalities, we introduce llama-nemoretriever-colembed, a unified text-image retrieval model that delivers state-of-the-art performance across multiple benchmarks. We release two model variants, 1B and 3B. The 3B model achieves state of the art performance, scoring NDCG@5 91.0 on ViDoRe V1 and 63.5 on ViDoRe V2, placing first on both leaderboards as of June 27, 2025.   Our approach leverages the NVIDIA Eagle2 Vision-Language model (VLM), modifies its architecture by replacing causal attention with bidirectional attention, and integrates a ColBERT-style late interaction mechanism to enable fine-grained multimodal retrieval in a shared embedding space. While this mechanism delivers superior retrieval accuracy, it introduces trade-offs in storage and efficiency. We provide a comprehensive analysis of these trade-offs. Additionally, we adopt a two-stage training strategy to enhance the model's retrieval capabilities.  
  </ol>  
</details>  
  
  



## NeRF  

### [Reflections Unlock: Geometry-Aware Reflection Disentanglement in 3D Gaussian Splatting for Photorealistic Scenes Rendering](http://arxiv.org/abs/2507.06103)  
Jiayi Song, Zihan Ye, Qingyuan Zhou, Weidong Yang, Ben Fei, Jingyi Xu, Ying He, Wanli Ouyang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Accurately rendering scenes with reflective surfaces remains a significant challenge in novel view synthesis, as existing methods like Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) often misinterpret reflections as physical geometry, resulting in degraded reconstructions. Previous methods rely on incomplete and non-generalizable geometric constraints, leading to misalignment between the positions of Gaussian splats and the actual scene geometry. When dealing with real-world scenes containing complex geometry, the accumulation of Gaussians further exacerbates surface artifacts and results in blurred reconstructions. To address these limitations, in this work, we propose Ref-Unlock, a novel geometry-aware reflection modeling framework based on 3D Gaussian Splatting, which explicitly disentangles transmitted and reflected components to better capture complex reflections and enhance geometric consistency in real-world scenes. Our approach employs a dual-branch representation with high-order spherical harmonics to capture high-frequency reflective details, alongside a reflection removal module providing pseudo reflection-free supervision to guide clean decomposition. Additionally, we incorporate pseudo-depth maps and a geometry-aware bilateral smoothness constraint to enhance 3D geometric consistency and stability in decomposition. Extensive experiments demonstrate that Ref-Unlock significantly outperforms classical GS-based reflection methods and achieves competitive results with NeRF-based models, while enabling flexible vision foundation models (VFMs) driven reflection editing. Our method thus offers an efficient and generalizable solution for realistic rendering of reflective scenes. Our code is available at https://ref-unlock.github.io/.  
  </ol>  
</details>  
  
### [DreamArt: Generating Interactable Articulated Objects from a Single Image](http://arxiv.org/abs/2507.05763)  
Ruijie Lu, Yu Liu, Jiaxiang Tang, Junfeng Ni, Yuxiang Wang, Diwen Wan, Gang Zeng, Yixin Chen, Siyuan Huang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Generating articulated objects, such as laptops and microwaves, is a crucial yet challenging task with extensive applications in Embodied AI and AR/VR. Current image-to-3D methods primarily focus on surface geometry and texture, neglecting part decomposition and articulation modeling. Meanwhile, neural reconstruction approaches (e.g., NeRF or Gaussian Splatting) rely on dense multi-view or interaction data, limiting their scalability. In this paper, we introduce DreamArt, a novel framework for generating high-fidelity, interactable articulated assets from single-view images. DreamArt employs a three-stage pipeline: firstly, it reconstructs part-segmented and complete 3D object meshes through a combination of image-to-3D generation, mask-prompted 3D segmentation, and part amodal completion. Second, we fine-tune a video diffusion model to capture part-level articulation priors, leveraging movable part masks as prompt and amodal images to mitigate ambiguities caused by occlusion. Finally, DreamArt optimizes the articulation motion, represented by a dual quaternion, and conducts global texture refinement and repainting to ensure coherent, high-quality textures across all parts. Experimental results demonstrate that DreamArt effectively generates high-quality articulated objects, possessing accurate part shape, high appearance fidelity, and plausible articulation, thereby providing a scalable solution for articulated asset generation. Our project page is available at https://dream-art-0.github.io/DreamArt/.  
  </ol>  
</details>  
**comments**: Technical Report  
  
  



