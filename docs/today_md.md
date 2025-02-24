<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#ELIP:-Enhanced-Visual-Language-Foundation-Models-for-Image-Retrieval>ELIP: Enhanced Visual-Language Foundation Models for Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Para-Lane:-Multi-Lane-Dataset-Registering-Parallel-Scans-for-Benchmarking-Novel-View-Synthesis>Para-Lane: Multi-Lane Dataset Registering Parallel Scans for Benchmarking Novel View Synthesis</a></li>
        <li><a href=#Hier-SLAM++:-Neuro-Symbolic-Semantic-SLAM-with-a-Hierarchically-Categorical-Gaussian-Splatting>Hier-SLAM++: Neuro-Symbolic Semantic SLAM with a Hierarchically Categorical Gaussian Splatting</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [ELIP: Enhanced Visual-Language Foundation Models for Image Retrieval](http://arxiv.org/abs/2502.15682)  
Guanqi Zhan, Yuanpei Liu, Kai Han, Weidi Xie, Andrew Zisserman  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The objective in this paper is to improve the performance of text-to-image retrieval. To this end, we introduce a new framework that can boost the performance of large-scale pre-trained vision-language models, so that they can be used for text-to-image re-ranking. The approach, Enhanced Language-Image Pre-training (ELIP), uses the text query to predict a set of visual prompts to condition the ViT image encoding. ELIP can easily be applied to the commonly used CLIP/SigLIP and the state-of-the-art BLIP-2 architectures. To train the architecture with limited computing resources, we develop a 'student friendly' best practice involving global hard sample mining, and selection and curation of a large-scale dataset. On the evaluation side, we set up two new out-of-distribution benchmarks, Occluded COCO and ImageNet-R, to assess the zero-shot generalisation of the models to different domains. Benefiting from the novel architecture and data curation, experiments show our enhanced network significantly boosts CLIP/SigLIP performance and outperforms the state-of-the-art BLIP-2 model on text-to-image retrieval.  
  </ol>  
</details>  
  
  



## NeRF  

### [Para-Lane: Multi-Lane Dataset Registering Parallel Scans for Benchmarking Novel View Synthesis](http://arxiv.org/abs/2502.15635)  
Ziqian Ni, Sicong Du, Zhenghua Hou, Chenming Wu, Sheng Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    To evaluate end-to-end autonomous driving systems, a simulation environment based on Novel View Synthesis (NVS) techniques is essential, which synthesizes photo-realistic images and point clouds from previously recorded sequences under new vehicle poses, particularly in cross-lane scenarios. Therefore, the development of a multi-lane dataset and benchmark is necessary. While recent synthetic scene-based NVS datasets have been prepared for cross-lane benchmarking, they still lack the realism of captured images and point clouds. To further assess the performance of existing methods based on NeRF and 3DGS, we present the first multi-lane dataset registering parallel scans specifically for novel driving view synthesis dataset derived from real-world scans, comprising 25 groups of associated sequences, including 16,000 front-view images, 64,000 surround-view images, and 16,000 LiDAR frames. All frames are labeled to differentiate moving objects from static elements. Using this dataset, we evaluate the performance of existing approaches in various testing scenarios at different lanes and distances. Additionally, our method provides the solution for solving and assessing the quality of multi-sensor poses for multi-modal data alignment for curating such a dataset in real-world. We plan to continually add new sequences to test the generalization of existing methods across different scenarios. The dataset is released publicly at the project page: https://nizqleo.github.io/paralane-dataset/.  
  </ol>  
</details>  
  
### [Hier-SLAM++: Neuro-Symbolic Semantic SLAM with a Hierarchically Categorical Gaussian Splatting](http://arxiv.org/abs/2502.14931)  
Boying Li, Vuong Chi Hao, Peter J. Stuckey, Ian Reid, Hamid Rezatofighi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose Hier-SLAM++, a comprehensive Neuro-Symbolic semantic 3D Gaussian Splatting SLAM method with both RGB-D and monocular input featuring an advanced hierarchical categorical representation, which enables accurate pose estimation as well as global 3D semantic mapping. The parameter usage in semantic SLAM systems increases significantly with the growing complexity of the environment, making scene understanding particularly challenging and costly. To address this problem, we introduce a novel and general hierarchical representation that encodes both semantic and geometric information in a compact form into 3D Gaussian Splatting, leveraging the capabilities of large language models (LLMs) as well as the 3D generative model. By utilizing the proposed hierarchical tree structure, semantic information is symbolically represented and learned in an end-to-end manner. We further introduce a novel semantic loss designed to optimize hierarchical semantic information through both inter-level and cross-level optimization. Additionally, we propose an improved SLAM system to support both RGB-D and monocular inputs using a feed-forward model. To the best of our knowledge, this is the first semantic monocular Gaussian Splatting SLAM system, significantly reducing sensor requirements for 3D semantic understanding and broadening the applicability of semantic Gaussian SLAM system. We conduct experiments on both synthetic and real-world datasets, demonstrating superior or on-par performance with state-of-the-art NeRF-based and Gaussian-based SLAM systems, while significantly reducing storage and training time requirements.  
  </ol>  
</details>  
**comments**: 15 pages. Under review  
  
  



