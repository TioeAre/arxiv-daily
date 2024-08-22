<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#UniFashion:-A-Unified-Vision-Language-Model-for-Multimodal-Fashion-Retrieval-and-Generation>UniFashion: A Unified Vision-Language Model for Multimodal Fashion Retrieval and Generation</a></li>
        <li><a href=#GSLoc:-Efficient-Camera-Pose-Refinement-via-3D-Gaussian-Splatting>GSLoc: Efficient Camera Pose Refinement via 3D Gaussian Splatting</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Irregularity-Inspection-using-Neural-Radiance-Field>Irregularity Inspection using Neural Radiance Field</a></li>
        <li><a href=#GSLoc:-Efficient-Camera-Pose-Refinement-via-3D-Gaussian-Splatting>GSLoc: Efficient Camera Pose Refinement via 3D Gaussian Splatting</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [UniFashion: A Unified Vision-Language Model for Multimodal Fashion Retrieval and Generation](http://arxiv.org/abs/2408.11305)  
[[code](https://github.com/xiangyu-mm/unifashion)]  
Xiangyu Zhao, Yuehan Zhang, Wenlong Zhang, Xiao-Ming Wu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The fashion domain encompasses a variety of real-world multimodal tasks, including multimodal retrieval and multimodal generation. The rapid advancements in artificial intelligence generated content, particularly in technologies like large language models for text generation and diffusion models for visual generation, have sparked widespread research interest in applying these multimodal models in the fashion domain. However, tasks involving embeddings, such as image-to-text or text-to-image retrieval, have been largely overlooked from this perspective due to the diverse nature of the multimodal fashion domain. And current research on multi-task single models lack focus on image generation. In this work, we present UniFashion, a unified framework that simultaneously tackles the challenges of multimodal generation and retrieval tasks within the fashion domain, integrating image generation with retrieval tasks and text generation tasks. UniFashion unifies embedding and generative tasks by integrating a diffusion model and LLM, enabling controllable and high-fidelity generation. Our model significantly outperforms previous single-task state-of-the-art models across diverse fashion tasks, and can be readily adapted to manage complex vision-language tasks. This work demonstrates the potential learning synergy between multimodal generation and retrieval, offering a promising direction for future research in the fashion domain. The source code is available at https://github.com/xiangyu-mm/UniFashion.  
  </ol>  
</details>  
  
### [GSLoc: Efficient Camera Pose Refinement via 3D Gaussian Splatting](http://arxiv.org/abs/2408.11085)  
Changkun Liu, Shuai Chen, Yash Bhalgat, Siyan Hu, Zirui Wang, Ming Cheng, Victor Adrian Prisacariu, Tristan Braud  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We leverage 3D Gaussian Splatting (3DGS) as a scene representation and propose a novel test-time camera pose refinement framework, GSLoc. This framework enhances the localization accuracy of state-of-the-art absolute pose regression and scene coordinate regression methods. The 3DGS model renders high-quality synthetic images and depth maps to facilitate the establishment of 2D-3D correspondences. GSLoc obviates the need for training feature extractors or descriptors by operating directly on RGB images, utilizing the 3D vision foundation model, MASt3R, for precise 2D matching. To improve the robustness of our model in challenging outdoor environments, we incorporate an exposure-adaptive module within the 3DGS framework. Consequently, GSLoc enables efficient pose refinement given a single RGB query and a coarse initial pose estimation. Our proposed approach surpasses leading NeRF-based optimization methods in both accuracy and runtime across indoor and outdoor visual localization benchmarks, achieving state-of-the-art accuracy on two indoor datasets.  
  </ol>  
</details>  
**comments**: The project page is available at https://gsloc.active.vision  
  
  



## NeRF  

### [Irregularity Inspection using Neural Radiance Field](http://arxiv.org/abs/2408.11251)  
Tianqi Ding, Dawei Xiang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    With the increasing growth of industrialization, more and more industries are relying on machine automation for production. However, defect detection in large-scale production machinery is becoming increasingly important. Due to their large size and height, it is often challenging for professionals to conduct defect inspections on such large machinery. For example, the inspection of aging and misalignment of components on tall machinery like towers requires companies to assign dedicated personnel. Employees need to climb the towers and either visually inspect or take photos to detect safety hazards in these large machines. Direct visual inspection is limited by its low level of automation, lack of precision, and safety concerns associated with personnel climbing the towers. Therefore, in this paper, we propose a system based on neural network modeling (NeRF) of 3D twin models. By comparing two digital models, this system enables defect detection at the 3D interface of an object.  
  </ol>  
</details>  
  
### [GSLoc: Efficient Camera Pose Refinement via 3D Gaussian Splatting](http://arxiv.org/abs/2408.11085)  
Changkun Liu, Shuai Chen, Yash Bhalgat, Siyan Hu, Zirui Wang, Ming Cheng, Victor Adrian Prisacariu, Tristan Braud  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We leverage 3D Gaussian Splatting (3DGS) as a scene representation and propose a novel test-time camera pose refinement framework, GSLoc. This framework enhances the localization accuracy of state-of-the-art absolute pose regression and scene coordinate regression methods. The 3DGS model renders high-quality synthetic images and depth maps to facilitate the establishment of 2D-3D correspondences. GSLoc obviates the need for training feature extractors or descriptors by operating directly on RGB images, utilizing the 3D vision foundation model, MASt3R, for precise 2D matching. To improve the robustness of our model in challenging outdoor environments, we incorporate an exposure-adaptive module within the 3DGS framework. Consequently, GSLoc enables efficient pose refinement given a single RGB query and a coarse initial pose estimation. Our proposed approach surpasses leading NeRF-based optimization methods in both accuracy and runtime across indoor and outdoor visual localization benchmarks, achieving state-of-the-art accuracy on two indoor datasets.  
  </ol>  
</details>  
**comments**: The project page is available at https://gsloc.active.vision  
  
  



