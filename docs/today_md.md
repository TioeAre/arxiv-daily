<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Are-They-the-Same-Picture?-Adapting-Concept-Bottleneck-Models-for-Human-AI-Collaboration-in-Image-Retrieval>Are They the Same Picture? Adapting Concept Bottleneck Models for Human-AI Collaboration in Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Radiance-Fields-from-Photons>Radiance Fields from Photons</a></li>
        <li><a href=#HPC:-Hierarchical-Progressive-Coding-Framework-for-Volumetric-Video>HPC: Hierarchical Progressive Coding Framework for Volumetric Video</a></li>
        <li><a href=#Feasibility-of-Neural-Radiance-Fields-for-Crime-Scene-Video-Reconstruction>Feasibility of Neural Radiance Fields for Crime Scene Video Reconstruction</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Are They the Same Picture? Adapting Concept Bottleneck Models for Human-AI Collaboration in Image Retrieval](http://arxiv.org/abs/2407.08908)  
[[code](https://github.com/realize-lab/chair)]  
Vaibhav Balloli, Sara Beery, Elizabeth Bondi-Kelly  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image retrieval plays a pivotal role in applications from wildlife conservation to healthcare, for finding individual animals or relevant images to aid diagnosis. Although deep learning techniques for image retrieval have advanced significantly, their imperfect real-world performance often necessitates including human expertise. Human-in-the-loop approaches typically rely on humans completing the task independently and then combining their opinions with an AI model in various ways, as these models offer very little interpretability or \textit{correctability}. To allow humans to intervene in the AI model instead, thereby saving human time and effort, we adapt the Concept Bottleneck Model (CBM) and propose \texttt{CHAIR}. \texttt{CHAIR} (a) enables humans to correct intermediate concepts, which helps \textit{improve} embeddings generated, and (b) allows for flexible levels of intervention that accommodate varying levels of human expertise for better retrieval. To show the efficacy of \texttt{CHAIR}, we demonstrate that our method performs better than similar models on image retrieval metrics without any external intervention. Furthermore, we also showcase how human intervention helps further improve retrieval performance, thereby achieving human-AI complementarity.  
  </ol>  
</details>  
**comments**: Accepted at Human-Centred AI Track at IJCAI 2024  
  
  



## NeRF  

### [Radiance Fields from Photons](http://arxiv.org/abs/2407.09386)  
Sacha Jungerman, Mohit Gupta  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural radiance fields, or NeRFs, have become the de facto approach for high-quality view synthesis from a collection of images captured from multiple viewpoints. However, many issues remain when capturing images in-the-wild under challenging conditions, such as low light, high dynamic range, or rapid motion leading to smeared reconstructions with noticeable artifacts. In this work, we introduce quanta radiance fields, a novel class of neural radiance fields that are trained at the granularity of individual photons using single-photon cameras (SPCs). We develop theory and practical computational techniques for building radiance fields and estimating dense camera poses from unconventional, stochastic, and high-speed binary frame sequences captured by SPCs. We demonstrate, both via simulations and a SPC hardware prototype, high-fidelity reconstructions under high-speed motion, in low light, and for extreme dynamic range settings.  
  </ol>  
</details>  
  
### [HPC: Hierarchical Progressive Coding Framework for Volumetric Video](http://arxiv.org/abs/2407.09026)  
Zihan Zheng, Houqiang Zhong, Qiang Hu, Xiaoyun Zhang, Li Song, Ya Zhang, Yanfeng Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Volumetric video based on Neural Radiance Field (NeRF) holds vast potential for various 3D applications, but its substantial data volume poses significant challenges for compression and transmission. Current NeRF compression lacks the flexibility to adjust video quality and bitrate within a single model for various network and device capacities. To address these issues, we propose HPC, a novel hierarchical progressive volumetric video coding framework achieving variable bitrate using a single model. Specifically, HPC introduces a hierarchical representation with a multi-resolution residual radiance field to reduce temporal redundancy in long-duration sequences while simultaneously generating various levels of detail. Then, we propose an end-to-end progressive learning approach with a multi-rate-distortion loss function to jointly optimize both hierarchical representation and compression. Our HPC trained only once can realize multiple compression levels, while the current methods need to train multiple fixed-bitrate models for different rate-distortion (RD) tradeoffs. Extensive experiments demonstrate that HPC achieves flexible quality levels with variable bitrate by a single model and exhibits competitive RD performance, even outperforming fixed-bitrate models across various datasets.  
  </ol>  
</details>  
**comments**: 11 pages, 7 figures  
  
### [Feasibility of Neural Radiance Fields for Crime Scene Video Reconstruction](http://arxiv.org/abs/2407.08795)  
Shariq Nadeem Malik, Min Hao Chee, Dayan Mario Anthony Perera, Chern Hong Lim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper aims to review and determine the feasibility of using variations of NeRF models in order to reconstruct crime scenes given input videos of the scene. We focus on three main innovations of NeRF when it comes to reconstructing crime scenes: Multi-object Synthesis, Deformable Synthesis, and Lighting. From there, we analyse its innovation progress against the requirements to be met in order to be able to reconstruct crime scenes with given videos of such scenes.  
  </ol>  
</details>  
**comments**: 4 pages, 1 table  
  
  



