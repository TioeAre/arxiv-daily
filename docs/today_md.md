<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#LIR-LIVO:-A-Lightweight,Robust-LiDAR/Vision/Inertial-Odometry-with-Illumination-Resilient-Deep-Features>LIR-LIVO: A Lightweight,Robust LiDAR/Vision/Inertial Odometry with Illumination-Resilient Deep Features</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#ImageRAG:-Dynamic-Image-Retrieval-for-Reference-Guided-Image-Generation>ImageRAG: Dynamic Image Retrieval for Reference-Guided Image Generation</a></li>
        <li><a href=#SpeechCompass:-Enhancing-Mobile-Captioning-with-Diarization-and-Directional-Guidance-via-Multi-Microphone-Localization>SpeechCompass: Enhancing Mobile Captioning with Diarization and Directional Guidance via Multi-Microphone Localization</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Embed-Any-NeRF:-Graph-Meta-Networks-for-Neural-Tasks-on-Arbitrary-NeRF-Architectures>Embed Any NeRF: Graph Meta-Networks for Neural Tasks on Arbitrary NeRF Architectures</a></li>
        <li><a href=#DenseSplat:-Densifying-Gaussian-Splatting-SLAM-with-Neural-Radiance-Prior>DenseSplat: Densifying Gaussian Splatting SLAM with Neural Radiance Prior</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [LIR-LIVO: A Lightweight,Robust LiDAR/Vision/Inertial Odometry with Illumination-Resilient Deep Features](http://arxiv.org/abs/2502.08676)  
Shujie Zhou, Zihao Wang, Xinye Dai, Weiwei Song, Shengfeng Gu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we propose LIR-LIVO, a lightweight and robust LiDAR-inertial-visual odometry system designed for challenging illumination and degraded environments. The proposed method leverages deep learning-based illumination-resilient features and LiDAR-Inertial-Visual Odometry (LIVO). By incorporating advanced techniques such as uniform depth distribution of features enabled by depth association with LiDAR point clouds and adaptive feature matching utilizing Superpoint and LightGlue, LIR-LIVO achieves state-of-the-art (SOTA) accuracy and robustness with low computational cost. Experiments are conducted on benchmark datasets, including NTU-VIRAL, Hilti'22, and R3LIVE-Dataset. The corresponding results demonstrate that our proposed method outperforms other SOTA methods on both standard and challenging datasets. Particularly, the proposed method demonstrates robust pose estimation under poor ambient lighting conditions in the Hilti'22 dataset. The code of this work is publicly accessible on GitHub to facilitate advancements in the robotics community.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [ImageRAG: Dynamic Image Retrieval for Reference-Guided Image Generation](http://arxiv.org/abs/2502.09411)  
Rotem Shalev-Arkushin, Rinon Gal, Amit H. Bermano, Ohad Fried  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Diffusion models enable high-quality and diverse visual content synthesis. However, they struggle to generate rare or unseen concepts. To address this challenge, we explore the usage of Retrieval-Augmented Generation (RAG) with image generation models. We propose ImageRAG, a method that dynamically retrieves relevant images based on a given text prompt, and uses them as context to guide the generation process. Prior approaches that used retrieved images to improve generation, trained models specifically for retrieval-based generation. In contrast, ImageRAG leverages the capabilities of existing image conditioning models, and does not require RAG-specific training. Our approach is highly adaptable and can be applied across different model types, showing significant improvement in generating rare and fine-grained concepts using different base models.   Our project page is available at: https://rotem-shalev.github.io/ImageRAG  
  </ol>  
</details>  
  
### [SpeechCompass: Enhancing Mobile Captioning with Diarization and Directional Guidance via Multi-Microphone Localization](http://arxiv.org/abs/2502.08848)  
Artem Dementyev, Dimitri Kavensky, Samuel J. Yang, Mathieu Parvaix, Chiong Lai, Alex Olwal  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Speech-to-text capabilities on mobile devices have proven helpful for hearing and speech accessibility, language translation, note-taking, and meeting transcripts. However, our foundational large-scale survey (n=263) shows that the inability to distinguish and indicate speaker direction makes them challenging in group conversations. SpeechCompass addresses this limitation through real-time, multi-microphone speech localization, where the direction of speech allows visual separation and guidance (e.g., arrows) in the user interface. We introduce efficient real-time audio localization algorithms and custom sound perception hardware running on a low-power microcontroller and four integrated microphones, which we characterize in technical evaluations. Informed by a large-scale survey (n=494), we conducted an in-person study of group conversations with eight frequent users of mobile speech-to-text, who provided feedback on five visualization styles. The value of diarization and visualizing localization was consistent across participants, with everyone agreeing on the value and potential of directional guidance for group conversations.  
  </ol>  
</details>  
**comments**: Accepted to CHI 2025  
  
  



## NeRF  

### [Embed Any NeRF: Graph Meta-Networks for Neural Tasks on Arbitrary NeRF Architectures](http://arxiv.org/abs/2502.09623)  
Francesco Ballerini, Pierluigi Zama Ramirez, Samuele Salti, Luigi Di Stefano  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRFs) have emerged as a groundbreaking paradigm for representing 3D objects and scenes by encoding shape and appearance information into the weights of a neural network. Recent works have shown how such weights can be used as input to frameworks processing them to solve deep learning tasks. Yet, these frameworks can only process NeRFs with a specific, predefined architecture. In this paper, we present the first framework that can ingest NeRFs with multiple architectures and perform inference on architectures unseen at training time. We achieve this goal by training a Graph Meta-Network in a representation learning framework. Moreover, we show how a contrastive objective is conducive to obtaining an architecture-agnostic latent space. In experiments on both MLP-based and tri-planar NeRFs, our approach demonstrates robust performance in classification and retrieval tasks that either matches or exceeds that of existing frameworks constrained to single architectures, thus providing the first architecture-agnostic method to perform tasks on NeRFs by processing their weights.  
  </ol>  
</details>  
**comments**: Under review  
  
### [DenseSplat: Densifying Gaussian Splatting SLAM with Neural Radiance Prior](http://arxiv.org/abs/2502.09111)  
Mingrui Li, Shuhong Liu, Tianchen Deng, Hongyu Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Gaussian SLAM systems excel in real-time rendering and fine-grained reconstruction compared to NeRF-based systems. However, their reliance on extensive keyframes is impractical for deployment in real-world robotic systems, which typically operate under sparse-view conditions that can result in substantial holes in the map. To address these challenges, we introduce DenseSplat, the first SLAM system that effectively combines the advantages of NeRF and 3DGS. DenseSplat utilizes sparse keyframes and NeRF priors for initializing primitives that densely populate maps and seamlessly fill gaps. It also implements geometry-aware primitive sampling and pruning strategies to manage granularity and enhance rendering efficiency. Moreover, DenseSplat integrates loop closure and bundle adjustment, significantly enhancing frame-to-frame tracking accuracy. Extensive experiments on multiple large-scale datasets demonstrate that DenseSplat achieves superior performance in tracking and mapping compared to current state-of-the-art methods.  
  </ol>  
</details>  
  
  



