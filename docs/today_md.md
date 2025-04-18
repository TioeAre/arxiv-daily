<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#SemCORE:-A-Semantic-Enhanced-Generative-Cross-Modal-Retrieval-Framework-with-MLLMs>SemCORE: A Semantic-Enhanced Generative Cross-Modal Retrieval Framework with MLLMs</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#GSAC:-Leveraging-Gaussian-Splatting-for-Photorealistic-Avatar-Creation-with-Unity-Integration>GSAC: Leveraging Gaussian Splatting for Photorealistic Avatar Creation with Unity Integration</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [SemCORE: A Semantic-Enhanced Generative Cross-Modal Retrieval Framework with MLLMs](http://arxiv.org/abs/2504.13172)  
Haoxuan Li, Yi Bin, Yunshan Ma, Guoqing Wang, Yang Yang, See-Kiong Ng, Tat-Seng Chua  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Cross-modal retrieval (CMR) is a fundamental task in multimedia research, focused on retrieving semantically relevant targets across different modalities. While traditional CMR methods match text and image via embedding-based similarity calculations, recent advancements in pre-trained generative models have established generative retrieval as a promising alternative. This paradigm assigns each target a unique identifier and leverages a generative model to directly predict identifiers corresponding to input queries without explicit indexing. Despite its great potential, current generative CMR approaches still face semantic information insufficiency in both identifier construction and generation processes. To address these limitations, we propose a novel unified Semantic-enhanced generative Cross-mOdal REtrieval framework (SemCORE), designed to unleash the semantic understanding capabilities in generative cross-modal retrieval task. Specifically, we first construct a Structured natural language IDentifier (SID) that effectively aligns target identifiers with generative models optimized for natural language comprehension and generation. Furthermore, we introduce a Generative Semantic Verification (GSV) strategy enabling fine-grained target discrimination. Additionally, to the best of our knowledge, SemCORE is the first framework to simultaneously consider both text-to-image and image-to-text retrieval tasks within generative cross-modal retrieval. Extensive experiments demonstrate that our framework outperforms state-of-the-art generative cross-modal retrieval methods. Notably, SemCORE achieves substantial improvements across benchmark datasets, with an average increase of 8.65 points in Recall@1 for text-to-image retrieval.  
  </ol>  
</details>  
  
  



## NeRF  

### [GSAC: Leveraging Gaussian Splatting for Photorealistic Avatar Creation with Unity Integration](http://arxiv.org/abs/2504.12999)  
Rendong Zhang, Alexandra Watkins, Nilanjan Sarkar  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Photorealistic avatars have become essential for immersive applications in virtual reality (VR) and augmented reality (AR), enabling lifelike interactions in areas such as training simulations, telemedicine, and virtual collaboration. These avatars bridge the gap between the physical and digital worlds, improving the user experience through realistic human representation. However, existing avatar creation techniques face significant challenges, including high costs, long creation times, and limited utility in virtual applications. Manual methods, such as MetaHuman, require extensive time and expertise, while automatic approaches, such as NeRF-based pipelines often lack efficiency, detailed facial expression fidelity, and are unable to be rendered at a speed sufficent for real-time applications. By involving several cutting-edge modern techniques, we introduce an end-to-end 3D Gaussian Splatting (3DGS) avatar creation pipeline that leverages monocular video input to create a scalable and efficient photorealistic avatar directly compatible with the Unity game engine. Our pipeline incorporates a novel Gaussian splatting technique with customized preprocessing that enables the user of "in the wild" monocular video capture, detailed facial expression reconstruction and embedding within a fully rigged avatar model. Additionally, we present a Unity-integrated Gaussian Splatting Avatar Editor, offering a user-friendly environment for VR/AR application development. Experimental results validate the effectiveness of our preprocessing pipeline in standardizing custom data for 3DGS training and demonstrate the versatility of Gaussian avatars in Unity, highlighting the scalability and practicality of our approach.  
  </ol>  
</details>  
  
  



