<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Interactive-Text-to-Image-Retrieval-with-Large-Language-Models:-A-Plug-and-Play-Approach>Interactive Text-to-Image Retrieval with Large Language Models: A Plug-and-Play Approach</a></li>
        <li><a href=#MeshVPR:-Citywide-Visual-Place-Recognition-Using-3D-Meshes>MeshVPR: Citywide Visual Place Recognition Using 3D Meshes</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#A-Self-Supervised-Denoising-Strategy-for-Underwater-Acoustic-Camera-Imageries>A Self-Supervised Denoising Strategy for Underwater Acoustic Camera Imageries</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#3D-HGS:-3D-Half-Gaussian-Splatting>3D-HGS: 3D Half-Gaussian Splatting</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Interactive Text-to-Image Retrieval with Large Language Models: A Plug-and-Play Approach](http://arxiv.org/abs/2406.03411)  
[[code](https://github.com/saehyung-lee/plugir)]  
Saehyung Lee, Sangwon Yu, Junsung Park, Jihun Yi, Sungroh Yoon  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we primarily address the issue of dialogue-form context query within the interactive text-to-image retrieval task. Our methodology, PlugIR, actively utilizes the general instruction-following capability of LLMs in two ways. First, by reformulating the dialogue-form context, we eliminate the necessity of fine-tuning a retrieval model on existing visual dialogue data, thereby enabling the use of any arbitrary black-box model. Second, we construct the LLM questioner to generate non-redundant questions about the attributes of the target image, based on the information of retrieval candidate images in the current context. This approach mitigates the issues of noisiness and redundancy in the generated questions. Beyond our methodology, we propose a novel evaluation metric, Best log Rank Integral (BRI), for a comprehensive assessment of the interactive retrieval system. PlugIR demonstrates superior performance compared to both zero-shot and fine-tuned baselines in various benchmarks. Additionally, the two methodologies comprising PlugIR can be flexibly applied together or separately in various situations. Our codes are available at https://github.com/Saehyung-Lee/PlugIR.  
  </ol>  
</details>  
**comments**: To appear in ACL 2024 Main  
  
### [MeshVPR: Citywide Visual Place Recognition Using 3D Meshes](http://arxiv.org/abs/2406.02776)  
Gabriele Berton, Lorenz Junglas, Riccardo Zaccone, Thomas Pollok, Barbara Caputo, Carlo Masone  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Mesh-based scene representation offers a promising direction for simplifying large-scale hierarchical visual localization pipelines, combining a visual place recognition step based on global features (retrieval) and a visual localization step based on local features. While existing work demonstrates the viability of meshes for visual localization, the impact of using synthetic databases rendered from them in visual place recognition remains largely unexplored. In this work we investigate using dense 3D textured meshes for large-scale Visual Place Recognition (VPR) and identify a significant performance drop when using synthetic mesh-based databases compared to real-world images for retrieval. To address this, we propose MeshVPR, a novel VPR pipeline that utilizes a lightweight features alignment framework to bridge the gap between real-world and synthetic domains. MeshVPR leverages pre-trained VPR models and it is efficient and scalable for city-wide deployments. We introduce novel datasets with freely available 3D meshes and manually collected queries from Berlin, Paris, and Melbourne. Extensive evaluations demonstrate that MeshVPR achieves competitive performance with standard VPR pipelines, paving the way for mesh-based localization systems. Our contributions include the new task of citywide mesh-based VPR, the new benchmark datasets, MeshVPR, and a thorough analysis of open challenges. Data, code, and interactive visualizations are available at https://mesh-vpr.github.io  
  </ol>  
</details>  
**comments**: Website: https://mesh-vpr.github.io/  
  
  



## Image Matching  

### [A Self-Supervised Denoising Strategy for Underwater Acoustic Camera Imageries](http://arxiv.org/abs/2406.02914)  
Xiaoteng Zhou, Katsunori Mizuno, Yilong Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In low-visibility marine environments characterized by turbidity and darkness, acoustic cameras serve as visual sensors capable of generating high-resolution 2D sonar images. However, acoustic camera images are interfered with by complex noise and are difficult to be directly ingested by downstream visual algorithms. This paper introduces a novel strategy for denoising acoustic camera images using deep learning techniques, which comprises two principal components: a self-supervised denoising framework and a fine feature-guided block. Additionally, the study explores the relationship between the level of image denoising and the improvement in feature-matching performance. Experimental results show that the proposed denoising strategy can effectively filter acoustic camera images without prior knowledge of the noise model. The denoising process is nearly end-to-end without complex parameter tuning and post-processing. It successfully removes noise while preserving fine feature details, thereby enhancing the performance of local feature matching.  
  </ol>  
</details>  
**comments**: 8 pages  
  
  



## NeRF  

### [3D-HGS: 3D Half-Gaussian Splatting](http://arxiv.org/abs/2406.02720)  
Haolin Li, Jinyang Liu, Mario Sznaier, Octavia Camps  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Photo-realistic 3D Reconstruction is a fundamental problem in 3D computer vision. This domain has seen considerable advancements owing to the advent of recent neural rendering techniques. These techniques predominantly aim to focus on learning volumetric representations of 3D scenes and refining these representations via loss functions derived from rendering. Among these, 3D Gaussian Splatting (3D-GS) has emerged as a significant method, surpassing Neural Radiance Fields (NeRFs). 3D-GS uses parameterized 3D Gaussians for modeling both spatial locations and color information, combined with a tile-based fast rendering technique. Despite its superior rendering performance and speed, the use of 3D Gaussian kernels has inherent limitations in accurately representing discontinuous functions, notably at edges and corners for shape discontinuities, and across varying textures for color discontinuities. To address this problem, we propose to employ 3D Half-Gaussian (3D-HGS) kernels, which can be used as a plug-and-play kernel. Our experiments demonstrate their capability to improve the performance of current 3D-GS related methods and achieve state-of-the-art rendering performance on various datasets without compromising rendering speed.  
  </ol>  
</details>  
**comments**: 9 pages, 6 figures  
  
  



