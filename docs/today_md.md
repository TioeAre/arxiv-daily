<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Scene-agnostic-Pose-Regression-for-Visual-Localization>Scene-agnostic Pose Regression for Visual Localization</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#CoLLM:-A-Large-Language-Model-for-Composed-Image-Retrieval>CoLLM: A Large Language Model for Composed Image Retrieval</a></li>
        <li><a href=#Scene-agnostic-Pose-Regression-for-Visual-Localization>Scene-agnostic Pose Regression for Visual Localization</a></li>
        <li><a href=#From-Sparse-to-Dense:-Camera-Relocalization-with-Scene-Specific-Detector-from-Feature-Gaussian-Splatting>From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from Feature Gaussian Splatting</a></li>
        <li><a href=#Fine-grained-Textual-Inversion-Network-for-Zero-Shot-Composed-Image-Retrieval>Fine-grained Textual Inversion Network for Zero-Shot Composed Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Multiscale-Feature-Importance-based-Bit-Allocation-for-End-to-End-Feature-Coding-for-Machines>Multiscale Feature Importance-based Bit Allocation for End-to-End Feature Coding for Machines</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#MultimodalStudio:-A-Heterogeneous-Sensor-Dataset-and-Framework-for-Neural-Rendering-across-Multiple-Imaging-Modalities>MultimodalStudio: A Heterogeneous Sensor Dataset and Framework for Neural Rendering across Multiple Imaging Modalities</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Scene-agnostic Pose Regression for Visual Localization](http://arxiv.org/abs/2503.19543)  
Junwei Zheng, Ruiping Liu, Yufan Chen, Zhenfang Chen, Kailun Yang, Jiaming Zhang, Rainer Stiefelhagen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Absolute Pose Regression (APR) predicts 6D camera poses but lacks the adaptability to unknown environments without retraining, while Relative Pose Regression (RPR) generalizes better yet requires a large image retrieval database. Visual Odometry (VO) generalizes well in unseen environments but suffers from accumulated error in open trajectories. To address this dilemma, we introduce a new task, Scene-agnostic Pose Regression (SPR), which can achieve accurate pose regression in a flexible way while eliminating the need for retraining or databases. To benchmark SPR, we created a large-scale dataset, 360SPR, with over 200K photorealistic panoramas, 3.6M pinhole images and camera poses in 270 scenes at three different sensor heights. Furthermore, a SPR-Mamba model is initially proposed to address SPR in a dual-branch manner. Extensive experiments and studies demonstrate the effectiveness of our SPR paradigm, dataset, and model. In the unknown scenes of both 360SPR and 360Loc datasets, our method consistently outperforms APR, RPR and VO. The dataset and code are available at https://junweizheng93.github.io/publications/SPR/SPR.html.  
  </ol>  
</details>  
**comments**: Accepted by CVPR 2025. Project page:
  https://junweizheng93.github.io/publications/SPR/SPR.html  
  
  



## Visual Localization  

### [CoLLM: A Large Language Model for Composed Image Retrieval](http://arxiv.org/abs/2503.19910)  
Chuong Huynh, Jinyu Yang, Ashish Tawari, Mubarak Shah, Son Tran, Raffay Hamid, Trishul Chilimbi, Abhinav Shrivastava  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed Image Retrieval (CIR) is a complex task that aims to retrieve images based on a multimodal query. Typical training data consists of triplets containing a reference image, a textual description of desired modifications, and the target image, which are expensive and time-consuming to acquire. The scarcity of CIR datasets has led to zero-shot approaches utilizing synthetic triplets or leveraging vision-language models (VLMs) with ubiquitous web-crawled image-caption pairs. However, these methods have significant limitations: synthetic triplets suffer from limited scale, lack of diversity, and unnatural modification text, while image-caption pairs hinder joint embedding learning of the multimodal query due to the absence of triplet data. Moreover, existing approaches struggle with complex and nuanced modification texts that demand sophisticated fusion and understanding of vision and language modalities. We present CoLLM, a one-stop framework that effectively addresses these limitations. Our approach generates triplets on-the-fly from image-caption pairs, enabling supervised training without manual annotation. We leverage Large Language Models (LLMs) to generate joint embeddings of reference images and modification texts, facilitating deeper multimodal fusion. Additionally, we introduce Multi-Text CIR (MTCIR), a large-scale dataset comprising 3.4M samples, and refine existing CIR benchmarks (CIRR and Fashion-IQ) to enhance evaluation reliability. Experimental results demonstrate that CoLLM achieves state-of-the-art performance across multiple CIR benchmarks and settings. MTCIR yields competitive results, with up to 15% performance improvement. Our refined benchmarks provide more reliable evaluation metrics for CIR models, contributing to the advancement of this important field.  
  </ol>  
</details>  
**comments**: CVPR 2025. Project page: https://collm-cvpr25.github.io/  
  
### [Scene-agnostic Pose Regression for Visual Localization](http://arxiv.org/abs/2503.19543)  
Junwei Zheng, Ruiping Liu, Yufan Chen, Zhenfang Chen, Kailun Yang, Jiaming Zhang, Rainer Stiefelhagen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Absolute Pose Regression (APR) predicts 6D camera poses but lacks the adaptability to unknown environments without retraining, while Relative Pose Regression (RPR) generalizes better yet requires a large image retrieval database. Visual Odometry (VO) generalizes well in unseen environments but suffers from accumulated error in open trajectories. To address this dilemma, we introduce a new task, Scene-agnostic Pose Regression (SPR), which can achieve accurate pose regression in a flexible way while eliminating the need for retraining or databases. To benchmark SPR, we created a large-scale dataset, 360SPR, with over 200K photorealistic panoramas, 3.6M pinhole images and camera poses in 270 scenes at three different sensor heights. Furthermore, a SPR-Mamba model is initially proposed to address SPR in a dual-branch manner. Extensive experiments and studies demonstrate the effectiveness of our SPR paradigm, dataset, and model. In the unknown scenes of both 360SPR and 360Loc datasets, our method consistently outperforms APR, RPR and VO. The dataset and code are available at https://junweizheng93.github.io/publications/SPR/SPR.html.  
  </ol>  
</details>  
**comments**: Accepted by CVPR 2025. Project page:
  https://junweizheng93.github.io/publications/SPR/SPR.html  
  
### [From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from Feature Gaussian Splatting](http://arxiv.org/abs/2503.19358)  
Zhiwei Huang, Hailin Yu, Yichun Shentu, Jin Yuan, Guofeng Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper presents a novel camera relocalization method, STDLoc, which leverages Feature Gaussian as scene representation. STDLoc is a full relocalization pipeline that can achieve accurate relocalization without relying on any pose prior. Unlike previous coarse-to-fine localization methods that require image retrieval first and then feature matching, we propose a novel sparse-to-dense localization paradigm. Based on this scene representation, we introduce a novel matching-oriented Gaussian sampling strategy and a scene-specific detector to achieve efficient and robust initial pose estimation. Furthermore, based on the initial localization results, we align the query feature map to the Gaussian feature field by dense feature matching to enable accurate localization. The experiments on indoor and outdoor datasets show that STDLoc outperforms current state-of-the-art localization methods in terms of localization accuracy and recall.  
  </ol>  
</details>  
**comments**: 15 pages, 12 figures, CVPR 2025  
  
### [Fine-grained Textual Inversion Network for Zero-Shot Composed Image Retrieval](http://arxiv.org/abs/2503.19296)  
Haoqiang Lin, Haokun Wen, Xuemeng Song, Meng Liu, Yupeng Hu, Liqiang Nie  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed Image Retrieval (CIR) allows users to search target images with a multimodal query, comprising a reference image and a modification text that describes the user's modification demand over the reference image. Nevertheless, due to the expensive labor cost of training data annotation, recent researchers have shifted to the challenging task of zero-shot CIR (ZS-CIR), which targets fulfilling CIR without annotated triplets. The pioneer ZS-CIR studies focus on converting the CIR task into a standard text-to-image retrieval task by pre-training a textual inversion network that can map a given image into a single pseudo-word token. Despite their significant progress, their coarse-grained textual inversion may be insufficient to capture the full content of the image accurately. To overcome this issue, in this work, we propose a novel Fine-grained Textual Inversion Network for ZS-CIR, named FTI4CIR. In particular, FTI4CIR comprises two main components: fine-grained pseudo-word token mapping and tri-wise caption-based semantic regularization. The former maps the image into a subject-oriented pseudo-word token and several attribute-oriented pseudo-word tokens to comprehensively express the image in the textual form, while the latter works on jointly aligning the fine-grained pseudo-word tokens to the real-word token embedding space based on a BLIP-generated image caption template. Extensive experiments conducted on three benchmark datasets demonstrate the superiority of our proposed method.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Multiscale Feature Importance-based Bit Allocation for End-to-End Feature Coding for Machines](http://arxiv.org/abs/2503.19278)  
Junle Liu, Yun Zhang, Zixi Guo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Feature Coding for Machines (FCM) aims to compress intermediate features effectively for remote intelligent analytics, which is crucial for future intelligent visual applications. In this paper, we propose a Multiscale Feature Importance-based Bit Allocation (MFIBA) for end-to-end FCM. First, we find that the importance of features for machine vision tasks varies with the scales, object size, and image instances. Based on this finding, we propose a Multiscale Feature Importance Prediction (MFIP) module to predict the importance weight for each scale of features. Secondly, we propose a task loss-rate model to establish the relationship between the task accuracy losses of using compressed features and the bitrate of encoding these features. Finally, we develop a MFIBA for end-to-end FCM, which is able to assign coding bits of multiscale features more reasonably based on their importance. Experimental results demonstrate that when combined with a retained Efficient Learned Image Compression (ELIC), the proposed MFIBA achieves an average of 38.202% bitrate savings in object detection compared to the anchor ELIC. Moreover, the proposed MFIBA achieves an average of 17.212% and 36.492% feature bitrate savings for instance segmentation and keypoint detection, respectively. When the proposed MFIBA is applied to the LIC-TCM, it achieves an average of 18.103%, 19.866% and 19.597% bit rate savings on three machine vision tasks, respectively, which validates the proposed MFIBA has good generalizability and adaptability to different machine vision tasks and FCM base codecs.  
  </ol>  
</details>  
  
  



## NeRF  

### [MultimodalStudio: A Heterogeneous Sensor Dataset and Framework for Neural Rendering across Multiple Imaging Modalities](http://arxiv.org/abs/2503.19673)  
Federico Lincetto, Gianluca Agresti, Mattia Rossi, Pietro Zanuttigh  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) have shown impressive performances in the rendering of 3D scenes from arbitrary viewpoints. While RGB images are widely preferred for training volume rendering models, the interest in other radiance modalities is also growing. However, the capability of the underlying implicit neural models to learn and transfer information across heterogeneous imaging modalities has seldom been explored, mostly due to the limited training data availability. For this purpose, we present MultimodalStudio (MMS): it encompasses MMS-DATA and MMS-FW. MMS-DATA is a multimodal multi-view dataset containing 32 scenes acquired with 5 different imaging modalities: RGB, monochrome, near-infrared, polarization and multispectral. MMS-FW is a novel modular multimodal NeRF framework designed to handle multimodal raw data and able to support an arbitrary number of multi-channel devices. Through extensive experiments, we demonstrate that MMS-FW trained on MMS-DATA can transfer information between different imaging modalities and produce higher quality renderings than using single modalities alone. We publicly release the dataset and the framework, to promote the research on multimodal volume rendering and beyond.  
  </ol>  
</details>  
**comments**: Accepted at CVPR 2025  
  
  



