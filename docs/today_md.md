<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#DeCo:-Decoupling-Token-Compression-from-Semantic-Abstraction-in-Multimodal-Large-Language-Models>DeCo: Decoupling Token Compression from Semantic Abstraction in Multimodal Large Language Models</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#R$^2$-Gaussian:-Rectifying-Radiative-Gaussian-Splatting-for-Tomographic-Reconstruction>R$^2$-Gaussian: Rectifying Radiative Gaussian Splatting for Tomographic Reconstruction</a></li>
        <li><a href=#4Diffusion:-Multi-view-Video-Diffusion-Model-for-4D-Generation>4Diffusion: Multi-view Video Diffusion Model for 4D Generation</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [DeCo: Decoupling Token Compression from Semantic Abstraction in Multimodal Large Language Models](http://arxiv.org/abs/2405.20985)  
Linli Yao, Lei Li, Shuhuai Ren, Lean Wang, Yuanxin Liu, Xu Sun, Lu Hou  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The visual projector, which bridges the vision and language modalities and facilitates cross-modal alignment, serves as a crucial component in MLLMs. However, measuring the effectiveness of projectors in vision-language alignment remains under-explored, which currently can only be inferred from the performance of MLLMs on downstream tasks. Motivated by the problem, this study examines the projector module by interpreting the vision-language semantic flow within MLLMs. Specifically, we trace back the semantic relevance flow from generated language tokens to raw visual encoder patches and the intermediate outputs produced by projectors. Our findings reveal that compressive projectors (e.g., QFormer), abstract visual patches into a limited set of semantic concepts, such as objects or attributes, resulting in a 'double abstraction' phenomenon. This involves a first visual semantic abstraction by the projector referring to pre-defined query tokens, and a second extraction by the LLM based on text instructions. The double abstraction is inefficient in training and will result in cumulative vision semantics deficiency. To mitigate this issue, we propose the key insight of 'Decouple Compression from Abstraction (DeCo), that is compressing the visual token number at the patch level by projectors and allowing the LLM to handle visual semantic abstraction entirely. Consequently, we adopt a simple compressor, i.e., 2D Adaptive Pooling, to downsample visual patches in a parameter-free manner. Empirical evaluation demonstrates that DeCo surpasses traditional compressive projectors regarding both performance and efficiency. It achieves performance gains of 0.9%, 7.1%, and 2.9% across the MLLM Benchmarks, Visual Localization, and Open-ended VQA tasks with fewer trainable parameters and faster convergence speed.  
  </ol>  
</details>  
  
  



## NeRF  

### [R $^2$ -Gaussian: Rectifying Radiative Gaussian Splatting for Tomographic Reconstruction](http://arxiv.org/abs/2405.20693)  
Ruyi Zha, Tao Jun Lin, Yuanhao Cai, Jiwen Cao, Yanhao Zhang, Hongdong Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D Gaussian splatting (3DGS) has shown promising results in image rendering and surface reconstruction. However, its potential in volumetric reconstruction tasks, such as X-ray computed tomography, remains under-explored. This paper introduces R2-Gaussian, the first 3DGS-based framework for sparse-view tomographic reconstruction. By carefully deriving X-ray rasterization functions, we discover a previously unknown integration bias in the standard 3DGS formulation, which hampers accurate volume retrieval. To address this issue, we propose a novel rectification technique via refactoring the projection from 3D to 2D Gaussians. Our new method presents three key innovations: (1) introducing tailored Gaussian kernels, (2) extending rasterization to X-ray imaging, and (3) developing a CUDA-based differentiable voxelizer. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches by 0.93 dB in PSNR and 0.014 in SSIM. Crucially, it delivers high-quality results in 3 minutes, which is 12x faster than NeRF-based methods and on par with traditional algorithms. The superior performance and rapid convergence of our method highlight its practical value.  
  </ol>  
</details>  
  
### [4Diffusion: Multi-view Video Diffusion Model for 4D Generation](http://arxiv.org/abs/2405.20674)  
Haiyu Zhang, Xinyuan Chen, Yaohui Wang, Xihui Liu, Yunhong Wang, Yu Qiao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Current 4D generation methods have achieved noteworthy efficacy with the aid of advanced diffusion generative models. However, these methods lack multi-view spatial-temporal modeling and encounter challenges in integrating diverse prior knowledge from multiple diffusion models, resulting in inconsistent temporal appearance and flickers. In this paper, we propose a novel 4D generation pipeline, namely 4Diffusion aimed at generating spatial-temporally consistent 4D content from a monocular video. We first design a unified diffusion model tailored for multi-view video generation by incorporating a learnable motion module into a frozen 3D-aware diffusion model to capture multi-view spatial-temporal correlations. After training on a curated dataset, our diffusion model acquires reasonable temporal consistency and inherently preserves the generalizability and spatial consistency of the 3D-aware diffusion model. Subsequently, we propose 4D-aware Score Distillation Sampling loss, which is based on our multi-view video diffusion model, to optimize 4D representation parameterized by dynamic NeRF. This aims to eliminate discrepancies arising from multiple diffusion models, allowing for generating spatial-temporally consistent 4D content. Moreover, we devise an anchor loss to enhance the appearance details and facilitate the learning of dynamic NeRF. Extensive qualitative and quantitative experiments demonstrate that our method achieves superior performance compared to previous methods.  
  </ol>  
</details>  
**comments**: Project Page: https://aejion.github.io/4diffusion/  
  
  



