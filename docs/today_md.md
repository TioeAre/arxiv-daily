<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Efficient-Camera-Exposure-Control-for-Visual-Odometry-via-Deep-Reinforcement-Learning>Efficient Camera Exposure Control for Visual Odometry via Deep Reinforcement Learning</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Augmented-Reality-without-Borders:-Achieving-Precise-Localization-Without-Maps>Augmented Reality without Borders: Achieving Precise Localization Without Maps</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Augmented-Reality-without-Borders:-Achieving-Precise-Localization-Without-Maps>Augmented Reality without Borders: Achieving Precise Localization Without Maps</a></li>
        <li><a href=#RISSOLE:-Parameter-efficient-Diffusion-Models-via-Block-wise-Generation-and-Retrieval-Guidance>RISSOLE: Parameter-efficient Diffusion Models via Block-wise Generation and Retrieval-Guidance</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#ConDense:-Consistent-2D/3D-Pre-training-for-Dense-and-Sparse-Features-from-Multi-View-Images>ConDense: Consistent 2D/3D Pre-training for Dense and Sparse Features from Multi-View Images</a></li>
        <li><a href=#GameIR:-A-Large-Scale-Synthesized-Ground-Truth-Dataset-for-Image-Restoration-over-Gaming-Content>GameIR: A Large-Scale Synthesized Ground-Truth Dataset for Image Restoration over Gaming Content</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Efficient Camera Exposure Control for Visual Odometry via Deep Reinforcement Learning](http://arxiv.org/abs/2408.17005)  
[[code](https://github.com/shuyanguni/drl_exposure_ctrl)]  
Shuyang Zhang, Jinhao He, Yilong Zhu, Jin Wu, Jie Yuan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The stability of visual odometry (VO) systems is undermined by degraded image quality, especially in environments with significant illumination changes. This study employs a deep reinforcement learning (DRL) framework to train agents for exposure control, aiming to enhance imaging performance in challenging conditions. A lightweight image simulator is developed to facilitate the training process, enabling the diversification of image exposure and sequence trajectory. This setup enables completely offline training, eliminating the need for direct interaction with camera hardware and the real environments. Different levels of reward functions are crafted to enhance the VO systems, equipping the DRL agents with varying intelligence. Extensive experiments have shown that our exposure control agents achieve superior efficiency-with an average inference duration of 1.58 ms per frame on a CPU-and respond more quickly than traditional feedback control schemes. By choosing an appropriate reward function, agents acquire an intelligent understanding of motion trends and anticipate future illumination changes. This predictive capability allows VO systems to deliver more stable and precise odometry results. The codes and datasets are available at https://github.com/ShuyangUni/drl_exposure_ctrl.  
  </ol>  
</details>  
**comments**: 8 pages, 7 figures  
  
  



## SFM  

### [Augmented Reality without Borders: Achieving Precise Localization Without Maps](http://arxiv.org/abs/2408.17373)  
Albert Gassol Puigjaner, Irvin Aloise, Patrik Schmuck  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual localization is crucial for Computer Vision and Augmented Reality (AR) applications, where determining the camera or device's position and orientation is essential to accurately interact with the physical environment. Traditional methods rely on detailed 3D maps constructed using Structure from Motion (SfM) or Simultaneous Localization and Mapping (SLAM), which is computationally expensive and impractical for dynamic or large-scale environments. We introduce MARLOC, a novel localization framework for AR applications that uses known relative transformations within image sequences to perform intra-sequence triangulation, generating 3D-2D correspondences for pose estimation and refinement. MARLOC eliminates the need for pre-built SfM maps, providing accurate and efficient localization suitable for dynamic outdoor environments. Evaluation with benchmark datasets and real-world experiments demonstrates MARLOC's state-of-the-art performance and robustness. By integrating MARLOC into an AR device, we highlight its capability to achieve precise localization in real-world outdoor scenarios, showcasing its practical effectiveness and potential to enhance visual localization in AR applications.  
  </ol>  
</details>  
**comments**: This work has been submitted to the IEEE for possible publication.
  Copyright may be transferred without notice, after which this version may no
  longer be accessible  
  
  



## Visual Localization  

### [Augmented Reality without Borders: Achieving Precise Localization Without Maps](http://arxiv.org/abs/2408.17373)  
Albert Gassol Puigjaner, Irvin Aloise, Patrik Schmuck  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual localization is crucial for Computer Vision and Augmented Reality (AR) applications, where determining the camera or device's position and orientation is essential to accurately interact with the physical environment. Traditional methods rely on detailed 3D maps constructed using Structure from Motion (SfM) or Simultaneous Localization and Mapping (SLAM), which is computationally expensive and impractical for dynamic or large-scale environments. We introduce MARLOC, a novel localization framework for AR applications that uses known relative transformations within image sequences to perform intra-sequence triangulation, generating 3D-2D correspondences for pose estimation and refinement. MARLOC eliminates the need for pre-built SfM maps, providing accurate and efficient localization suitable for dynamic outdoor environments. Evaluation with benchmark datasets and real-world experiments demonstrates MARLOC's state-of-the-art performance and robustness. By integrating MARLOC into an AR device, we highlight its capability to achieve precise localization in real-world outdoor scenarios, showcasing its practical effectiveness and potential to enhance visual localization in AR applications.  
  </ol>  
</details>  
**comments**: This work has been submitted to the IEEE for possible publication.
  Copyright may be transferred without notice, after which this version may no
  longer be accessible  
  
### [RISSOLE: Parameter-efficient Diffusion Models via Block-wise Generation and Retrieval-Guidance](http://arxiv.org/abs/2408.17095)  
Avideep Mukherjee, Soumya Banerjee, Vinay P. Namboodiri, Piyush Rai  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Diffusion-based models demonstrate impressive generation capabilities. However, they also have a massive number of parameters, resulting in enormous model sizes, thus making them unsuitable for deployment on resource-constraint devices. Block-wise generation can be a promising alternative for designing compact-sized (parameter-efficient) deep generative models since the model can generate one block at a time instead of generating the whole image at once. However, block-wise generation is also considerably challenging because ensuring coherence across generated blocks can be non-trivial. To this end, we design a retrieval-augmented generation (RAG) approach and leverage the corresponding blocks of the images retrieved by the RAG module to condition the training and generation stages of a block-wise denoising diffusion model. Our conditioning schemes ensure coherence across the different blocks during training and, consequently, during generation. While we showcase our approach using the latent diffusion model (LDM) as the base model, it can be used with other variants of denoising diffusion models. We validate the solution of the coherence problem through the proposed approach by reporting substantive experiments to demonstrate our approach's effectiveness in compact model size and excellent generation quality.  
  </ol>  
</details>  
  
  



## NeRF  

### [ConDense: Consistent 2D/3D Pre-training for Dense and Sparse Features from Multi-View Images](http://arxiv.org/abs/2408.17027)  
Xiaoshuai Zhang, Zhicheng Wang, Howard Zhou, Soham Ghosh, Danushen Gnanapragasam, Varun Jampani, Hao Su, Leonidas Guibas  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    To advance the state of the art in the creation of 3D foundation models, this paper introduces the ConDense framework for 3D pre-training utilizing existing pre-trained 2D networks and large-scale multi-view datasets. We propose a novel 2D-3D joint training scheme to extract co-embedded 2D and 3D features in an end-to-end pipeline, where 2D-3D feature consistency is enforced through a volume rendering NeRF-like ray marching process. Using dense per pixel features we are able to 1) directly distill the learned priors from 2D models to 3D models and create useful 3D backbones, 2) extract more consistent and less noisy 2D features, 3) formulate a consistent embedding space where 2D, 3D, and other modalities of data (e.g., natural language prompts) can be jointly queried. Furthermore, besides dense features, ConDense can be trained to extract sparse features (e.g., key points), also with 2D-3D consistency -- condensing 3D NeRF representations into compact sets of decorated key points. We demonstrate that our pre-trained model provides good initialization for various 3D tasks including 3D classification and segmentation, outperforming other 3D pre-training methods by a significant margin. It also enables, by exploiting our sparse features, additional useful downstream tasks, such as matching 2D images to 3D scenes, detecting duplicate 3D scenes, and querying a repository of 3D scenes through natural language -- all quite efficiently and without any per-scene fine-tuning.  
  </ol>  
</details>  
**comments**: ECCV 2024  
  
### [GameIR: A Large-Scale Synthesized Ground-Truth Dataset for Image Restoration over Gaming Content](http://arxiv.org/abs/2408.16866)  
Lebin Zhou, Kun Han, Nam Ling, Wei Wang, Wei Jiang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image restoration methods like super-resolution and image synthesis have been successfully used in commercial cloud gaming products like NVIDIA's DLSS. However, restoration over gaming content is not well studied by the general public. The discrepancy is mainly caused by the lack of ground-truth gaming training data that match the test cases. Due to the unique characteristics of gaming content, the common approach of generating pseudo training data by degrading the original HR images results in inferior restoration performance. In this work, we develop GameIR, a large-scale high-quality computer-synthesized ground-truth dataset to fill in the blanks, targeting at two different applications. The first is super-resolution with deferred rendering, to support the gaming solution of rendering and transferring LR images only and restoring HR images on the client side. We provide 19200 LR-HR paired ground-truth frames coming from 640 videos rendered at 720p and 1440p for this task. The second is novel view synthesis (NVS), to support the multiview gaming solution of rendering and transferring part of the multiview frames and generating the remaining frames on the client side. This task has 57,600 HR frames from 960 videos of 160 scenes with 6 camera views. In addition to the RGB frames, the GBuffers during the deferred rendering stage are also provided, which can be used to help restoration. Furthermore, we evaluate several SOTA super-resolution algorithms and NeRF-based NVS algorithms over our dataset, which demonstrates the effectiveness of our ground-truth GameIR data in improving restoration performance for gaming content. Also, we test the method of incorporating the GBuffers as additional input information for helping super-resolution and NVS. We release our dataset and models to the general public to facilitate research on restoration methods over gaming content.  
  </ol>  
</details>  
  
  



