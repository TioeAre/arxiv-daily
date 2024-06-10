<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#PQPP:-A-Joint-Benchmark-for-Text-to-Image-Prompt-and-Query-Performance-Prediction>PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Multiplane-Prior-Guided-Few-Shot-Aerial-Scene-Rendering>Multiplane Prior Guided Few-Shot Aerial Scene Rendering</a></li>
        <li><a href=#Multi-style-Neural-Radiance-Field-with-AdaIN>Multi-style Neural Radiance Field with AdaIN</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction](http://arxiv.org/abs/2406.04746)  
[[code](https://github.com/eduard6421/pqpp)]  
Eduard Poesina, Adriana Valentina Costache, Adrian-Gabriel Chifu, Josiane Mothe, Radu Tudor Ionescu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Text-to-image generation has recently emerged as a viable alternative to text-to-image retrieval, due to the visually impressive results of generative diffusion models. Although query performance prediction is an active research topic in information retrieval, to the best of our knowledge, there is no prior study that analyzes the difficulty of queries (prompts) in text-to-image generation, based on human judgments. To this end, we introduce the first dataset of prompts which are manually annotated in terms of image generation performance. In order to determine the difficulty of the same prompts in image retrieval, we also collect manual annotations that represent retrieval performance. We thus propose the first benchmark for joint text-to-image prompt and query performance prediction, comprising 10K queries. Our benchmark enables: (i) the comparative assessment of the difficulty of prompts/queries in image generation and image retrieval, and (ii) the evaluation of prompt/query performance predictors addressing both generation and retrieval. We present results with several pre-generation/retrieval and post-generation/retrieval performance predictors, thus providing competitive baselines for future research. Our benchmark and code is publicly available under the CC BY 4.0 license at https://github.com/Eduard6421/PQPP.  
  </ol>  
</details>  
  
  



## NeRF  

### [Multiplane Prior Guided Few-Shot Aerial Scene Rendering](http://arxiv.org/abs/2406.04961)  
Zihan Gao, Licheng Jiao, Lingling Li, Xu Liu, Fang Liu, Puhua Chen, Yuwei Guo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) have been successfully applied in various aerial scenes, yet they face challenges with sparse views due to limited supervision. The acquisition of dense aerial views is often prohibitive, as unmanned aerial vehicles (UAVs) may encounter constraints in perspective range and energy constraints. In this work, we introduce Multiplane Prior guided NeRF (MPNeRF), a novel approach tailored for few-shot aerial scene rendering-marking a pioneering effort in this domain. Our key insight is that the intrinsic geometric regularities specific to aerial imagery could be leveraged to enhance NeRF in sparse aerial scenes. By investigating NeRF's and Multiplane Image (MPI)'s behavior, we propose to guide the training process of NeRF with a Multiplane Prior. The proposed Multiplane Prior draws upon MPI's benefits and incorporates advanced image comprehension through a SwinV2 Transformer, pre-trained via SimMIM. Our extensive experiments demonstrate that MPNeRF outperforms existing state-of-the-art methods applied in non-aerial contexts, by tripling the performance in SSIM and LPIPS even with three views available. We hope our work offers insights into the development of NeRF-based applications in aerial scenes with limited data.  
  </ol>  
</details>  
**comments**: 17 pages, 8 figures, accepted at CVPR 2024  
  
### [Multi-style Neural Radiance Field with AdaIN](http://arxiv.org/abs/2406.04960)  
Yu-Wen Pao, An-Jie Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this work, we propose a novel pipeline that combines AdaIN and NeRF for the task of stylized Novel View Synthesis. Compared to previous works, we make the following contributions: 1) We simplify the pipeline. 2) We extend the capabilities of model to handle the multi-style task. 3) We modify the model architecture to perform well on styles with strong brush strokes. 4) We implement style interpolation on the multi-style model, allowing us to control the style between any two styles and the style intensity between the stylized output and the original scene, providing better control over the stylization strength.  
  </ol>  
</details>  
  
  



