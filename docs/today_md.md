<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Unposed-Sparse-Views-Room-Layout-Reconstruction-in-the-Age-of-Pretrain-Model>Unposed Sparse Views Room Layout Reconstruction in the Age of Pretrain Model</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#MegaLoc:-One-Retrieval-to-Place-Them-All>MegaLoc: One Retrieval to Place Them All</a></li>
        <li><a href=#Visual-RAG:-Benchmarking-Text-to-Image-Retrieval-Augmented-Generation-for-Visual-Knowledge-Intensive-Queries>Visual-RAG: Benchmarking Text-to-Image Retrieval Augmented Generation for Visual Knowledge Intensive Queries</a></li>
        <li><a href=#SelaVPR++:-Towards-Seamless-Adaptation-of-Foundation-Models-for-Efficient-Place-Recognition>SelaVPR++: Towards Seamless Adaptation of Foundation Models for Efficient Place Recognition</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Unposed-Sparse-Views-Room-Layout-Reconstruction-in-the-Age-of-Pretrain-Model>Unposed Sparse Views Room Layout Reconstruction in the Age of Pretrain Model</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Semantic-Neural-Radiance-Fields-for-Multi-Date-Satellite-Data>Semantic Neural Radiance Fields for Multi-Date Satellite Data</a></li>
        <li><a href=#AquaNeRF:-Neural-Radiance-Fields-in-Underwater-Media-with-Distractor-Removal>AquaNeRF: Neural Radiance Fields in Underwater Media with Distractor Removal</a></li>
        <li><a href=#DualNeRF:-Text-Driven-3D-Scene-Editing-via-Dual-Field-Representation>DualNeRF: Text-Driven 3D Scene Editing via Dual-Field Representation</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Unposed Sparse Views Room Layout Reconstruction in the Age of Pretrain Model](http://arxiv.org/abs/2502.16779)  
Yaxuan Huang, Xili Dai, Jianan Wang, Xianbiao Qi, Yixing Yuan, Xiangyu Yue  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Room layout estimation from multiple-perspective images is poorly investigated due to the complexities that emerge from multi-view geometry, which requires muti-step solutions such as camera intrinsic and extrinsic estimation, image matching, and triangulation. However, in 3D reconstruction, the advancement of recent 3D foundation models such as DUSt3R has shifted the paradigm from the traditional multi-step structure-from-motion process to an end-to-end single-step approach. To this end, we introduce Plane-DUSt3R}, a novel method for multi-view room layout estimation leveraging the 3D foundation model DUSt3R. Plane-DUSt3R incorporates the DUSt3R framework and fine-tunes on a room layout dataset (Structure3D) with a modified objective to estimate structural planes. By generating uniform and parsimonious results, Plane-DUSt3R enables room layout estimation with only a single post-processing step and 2D detection results. Unlike previous methods that rely on single-perspective or panorama image, Plane-DUSt3R extends the setting to handle multiple-perspective images. Moreover, it offers a streamlined, end-to-end solution that simplifies the process and reduces error accumulation. Experimental results demonstrate that Plane-DUSt3R not only outperforms state-of-the-art methods on the synthetic dataset but also proves robust and effective on in the wild data with different image styles such as cartoon.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [MegaLoc: One Retrieval to Place Them All](http://arxiv.org/abs/2502.17237)  
Gabriele Berton, Carlo Masone  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Retrieving images from the same location as a given query is an important component of multiple computer vision tasks, like Visual Place Recognition, Landmark Retrieval, Visual Localization, 3D reconstruction, and SLAM. However, existing solutions are built to specifically work for one of these tasks, and are known to fail when the requirements slightly change or when they meet out-of-distribution data. In this paper we combine a variety of existing methods, training techniques, and datasets to train a retrieval model, called MegaLoc, that is performant on multiple tasks. We find that MegaLoc (1) achieves state of the art on a large number of Visual Place Recognition datasets, (2) impressive results on common Landmark Retrieval datasets, and (3) sets a new state of the art for Visual Localization on the LaMAR datasets, where we only changed the retrieval method to the existing localization pipeline. The code for MegaLoc is available at https://github.com/gmberton/MegaLoc  
  </ol>  
</details>  
**comments**: Tech Report  
  
### [Visual-RAG: Benchmarking Text-to-Image Retrieval Augmented Generation for Visual Knowledge Intensive Queries](http://arxiv.org/abs/2502.16636)  
Yin Wu, Quanyu Long, Jing Li, Jianfei Yu, Wenya Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Retrieval-Augmented Generation (RAG) is a popular approach for enhancing Large Language Models (LLMs) by addressing their limitations in verifying facts and answering knowledge-intensive questions. As the research in LLM extends their capability to handle input modality other than text, e.g. image, several multimodal RAG benchmarks are proposed. Nonetheless, they mainly use textual knowledge bases as the primary source of evidences for augmentation. There still lack benchmarks designed to evaluate images as augmentation in RAG systems and how they leverage visual knowledge. We propose Visual-RAG, a novel Question Answering benchmark that emphasizes visual knowledge intensive questions. Unlike prior works relying on text-based evidence, Visual-RAG necessitates text-to-image retrieval and integration of relevant clue images to extract visual knowledge as evidence. With Visual-RAG, we evaluate 5 open-sourced and 3 proprietary Multimodal LLMs (MLLMs), revealing that images can serve as good evidence in RAG; however, even the SoTA models struggle with effectively extracting and utilizing visual knowledge  
  </ol>  
</details>  
**comments**: 23 pages, 6 figures  
  
### [SelaVPR++: Towards Seamless Adaptation of Foundation Models for Efficient Place Recognition](http://arxiv.org/abs/2502.16601)  
Feng Lu, Tong Jin, Xiangyuan Lan, Lijun Zhang, Yunpeng Liu, Yaowei Wang, Chun Yuan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent studies show that the visual place recognition (VPR) method using pre-trained visual foundation models can achieve promising performance. In our previous work, we propose a novel method to realize seamless adaptation of foundation models to VPR (SelaVPR). This method can produce both global and local features that focus on discriminative landmarks to recognize places for two-stage VPR by a parameter-efficient adaptation approach. Although SelaVPR has achieved competitive results, we argue that the previous adaptation is inefficient in training time and GPU memory usage, and the re-ranking paradigm is also costly in retrieval latency and storage usage. In pursuit of higher efficiency and better performance, we propose an extension of the SelaVPR, called SelaVPR++. Concretely, we first design a parameter-, time-, and memory-efficient adaptation method that uses lightweight multi-scale convolution (MultiConv) adapters to refine intermediate features from the frozen foundation backbone. This adaptation method does not back-propagate gradients through the backbone during training, and the MultiConv adapter facilitates feature interactions along the spatial axes and introduces proper local priors, thus achieving higher efficiency and better performance. Moreover, we propose an innovative re-ranking paradigm for more efficient VPR. Instead of relying on local features for re-ranking, which incurs huge overhead in latency and storage, we employ compact binary features for initial retrieval and robust floating-point (global) features for re-ranking. To obtain such binary features, we propose a similarity-constrained deep hashing method, which can be easily integrated into the VPR pipeline. Finally, we improve our training strategy and unify the training protocol of several common training datasets to merge them for better training of VPR models. Extensive experiments show that ......  
  </ol>  
</details>  
  
  



## Image Matching  

### [Unposed Sparse Views Room Layout Reconstruction in the Age of Pretrain Model](http://arxiv.org/abs/2502.16779)  
Yaxuan Huang, Xili Dai, Jianan Wang, Xianbiao Qi, Yixing Yuan, Xiangyu Yue  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Room layout estimation from multiple-perspective images is poorly investigated due to the complexities that emerge from multi-view geometry, which requires muti-step solutions such as camera intrinsic and extrinsic estimation, image matching, and triangulation. However, in 3D reconstruction, the advancement of recent 3D foundation models such as DUSt3R has shifted the paradigm from the traditional multi-step structure-from-motion process to an end-to-end single-step approach. To this end, we introduce Plane-DUSt3R}, a novel method for multi-view room layout estimation leveraging the 3D foundation model DUSt3R. Plane-DUSt3R incorporates the DUSt3R framework and fine-tunes on a room layout dataset (Structure3D) with a modified objective to estimate structural planes. By generating uniform and parsimonious results, Plane-DUSt3R enables room layout estimation with only a single post-processing step and 2D detection results. Unlike previous methods that rely on single-perspective or panorama image, Plane-DUSt3R extends the setting to handle multiple-perspective images. Moreover, it offers a streamlined, end-to-end solution that simplifies the process and reduces error accumulation. Experimental results demonstrate that Plane-DUSt3R not only outperforms state-of-the-art methods on the synthetic dataset but also proves robust and effective on in the wild data with different image styles such as cartoon.  
  </ol>  
</details>  
  
  



## NeRF  

### [Semantic Neural Radiance Fields for Multi-Date Satellite Data](http://arxiv.org/abs/2502.16992)  
Valentin Wagner, Sebastian Bullinger, Christoph Bodensteiner, Michael Arens  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this work we propose a satellite specific Neural Radiance Fields (NeRF) model capable to obtain a three-dimensional semantic representation (neural semantic field) of the scene. The model derives the output from a set of multi-date satellite images with corresponding pixel-wise semantic labels. We demonstrate the robustness of our approach and its capability to improve noisy input labels. We enhance the color prediction by utilizing the semantic information to address temporal image inconsistencies caused by non-stationary categories such as vehicles. To facilitate further research in this domain, we present a dataset comprising manually generated labels for popular multi-view satellite images. Our code and dataset are available at https://github.com/wagnva/semantic-nerf-for-satellite-data.  
  </ol>  
</details>  
**comments**: Accepted at the CV4EO Workshop at WACV 2025  
  
### [AquaNeRF: Neural Radiance Fields in Underwater Media with Distractor Removal](http://arxiv.org/abs/2502.16351)  
Luca Gough, Adrian Azzarelli, Fan Zhang, Nantheera Anantrasirichai  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural radiance field (NeRF) research has made significant progress in modeling static video content captured in the wild. However, current models and rendering processes rarely consider scenes captured underwater, which are useful for studying and filming ocean life. They fail to address visual artifacts unique to underwater scenes, such as moving fish and suspended particles. This paper introduces a novel NeRF renderer and optimization scheme for an implicit MLP-based NeRF model. Our renderer reduces the influence of floaters and moving objects that interfere with static objects of interest by estimating a single surface per ray. We use a Gaussian weight function with a small offset to ensure that the transmittance of the surrounding media remains constant. Additionally, we enhance our model with a depth-based scaling function to upscale gradients for near-camera volumes. Overall, our method outperforms the baseline Nerfacto by approximately 7.5\% and SeaThru-NeRF by 6.2% in terms of PSNR. Subjective evaluation also shows a significant reduction of artifacts while preserving details of static targets and background compared to the state of the arts.  
  </ol>  
</details>  
**comments**: Accepted by 2025 IEEE International Symposium on Circuits and Systems  
  
### [DualNeRF: Text-Driven 3D Scene Editing via Dual-Field Representation](http://arxiv.org/abs/2502.16302)  
Yuxuan Xiong, Yue Shi, Yishun Dou, Bingbing Ni  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recently, denoising diffusion models have achieved promising results in 2D image generation and editing. Instruct-NeRF2NeRF (IN2N) introduces the success of diffusion into 3D scene editing through an "Iterative dataset update" (IDU) strategy. Though achieving fascinating results, IN2N suffers from problems of blurry backgrounds and trapping in local optima. The first problem is caused by IN2N's lack of efficient guidance for background maintenance, while the second stems from the interaction between image editing and NeRF training during IDU. In this work, we introduce DualNeRF to deal with these problems. We propose a dual-field representation to preserve features of the original scene and utilize them as additional guidance to the model for background maintenance during IDU. Moreover, a simulated annealing strategy is embedded into IDU to endow our model with the power of addressing local optima issues. A CLIP-based consistency indicator is used to further improve the editing quality by filtering out low-quality edits. Extensive experiments demonstrate that our method outperforms previous methods both qualitatively and quantitatively.  
  </ol>  
</details>  
  
  



