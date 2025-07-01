<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Event-based-Stereo-Visual-Inertial-Odometry-with-Voxel-Map>Event-based Stereo Visual-Inertial Odometry with Voxel Map</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#AttentionGS:-Towards-Initialization-Free-3D-Gaussian-Splatting-via-Structural-Attention>AttentionGS: Towards Initialization-Free 3D Gaussian Splatting via Structural Attention</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Utilizing-a-Novel-Deep-Learning-Method-for-Scene-Categorization-in-Remote-Sensing-Data>Utilizing a Novel Deep Learning Method for Scene Categorization in Remote Sensing Data</a></li>
        <li><a href=#Mask-aware-Text-to-Image-Retrieval:-Referring-Expression-Segmentation-Meets-Cross-modal-Retrieval>Mask-aware Text-to-Image Retrieval: Referring Expression Segmentation Meets Cross-modal Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Efficient-and-Accurate-Image-Provenance-Analysis:-A-Scalable-Pipeline-for-Large-scale-Images>Efficient and Accurate Image Provenance Analysis: A Scalable Pipeline for Large-scale Images</a></li>
        <li><a href=#Dynamic-Contrastive-Learning-for-Hierarchical-Retrieval:-A-Case-Study-of-Distance-Aware-Cross-View-Geo-Localization>Dynamic Contrastive Learning for Hierarchical Retrieval: A Case Study of Distance-Aware Cross-View Geo-Localization</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#AttentionGS:-Towards-Initialization-Free-3D-Gaussian-Splatting-via-Structural-Attention>AttentionGS: Towards Initialization-Free 3D Gaussian Splatting via Structural Attention</a></li>
        <li><a href=#Dynamic-View-Synthesis-from-Small-Camera-Motion-Videos>Dynamic View Synthesis from Small Camera Motion Videos</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Event-based Stereo Visual-Inertial Odometry with Voxel Map](http://arxiv.org/abs/2506.23078)  
Zhaoxing Zhang, Xiaoxiang Wang, Chengliang Zhang, Yangyang Guo, Zikang Yuan, Xin Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The event camera, renowned for its high dynamic range and exceptional temporal resolution, is recognized as an important sensor for visual odometry. However, the inherent noise in event streams complicates the selection of high-quality map points, which critically determine the precision of state estimation. To address this challenge, we propose Voxel-ESVIO, an event-based stereo visual-inertial odometry system that utilizes voxel map management, which efficiently filter out high-quality 3D points. Specifically, our methodology utilizes voxel-based point selection and voxel-aware point management to collectively optimize the selection and updating of map points on a per-voxel basis. These synergistic strategies enable the efficient retrieval of noise-resilient map points with the highest observation likelihood in current frames, thereby ensureing the state estimation accuracy. Extensive evaluations on three public benchmarks demonstrate that our Voxel-ESVIO outperforms state-of-the-art methods in both accuracy and computational efficiency.  
  </ol>  
</details>  
  
  



## SFM  

### [AttentionGS: Towards Initialization-Free 3D Gaussian Splatting via Structural Attention](http://arxiv.org/abs/2506.23611)  
Ziao Liu, Zhenjia Li, Yifeng Shi, Xiangang Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D Gaussian Splatting (3DGS) is a powerful alternative to Neural Radiance Fields (NeRF), excelling in complex scene reconstruction and efficient rendering. However, it relies on high-quality point clouds from Structure-from-Motion (SfM), limiting its applicability. SfM also fails in texture-deficient or constrained-view scenarios, causing severe degradation in 3DGS reconstruction. To address this limitation, we propose AttentionGS, a novel framework that eliminates the dependency on high-quality initial point clouds by leveraging structural attention for direct 3D reconstruction from randomly initialization. In the early training stage, we introduce geometric attention to rapidly recover the global scene structure. As training progresses, we incorporate texture attention to refine fine-grained details and enhance rendering quality. Furthermore, we employ opacity-weighted gradients to guide Gaussian densification, leading to improved surface reconstruction. Extensive experiments on multiple benchmark datasets demonstrate that AttentionGS significantly outperforms state-of-the-art methods, particularly in scenarios where point cloud initialization is unreliable. Our approach paves the way for more robust and flexible 3D Gaussian Splatting in real-world applications.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Utilizing a Novel Deep Learning Method for Scene Categorization in Remote Sensing Data](http://arxiv.org/abs/2506.22939)  
Ghufran A. Omran, Wassan Saad Abduljabbar Hayale, Ahmad AbdulQadir AlRababah, Israa Ibraheem Al-Barazanchi, Ravi Sekhar, Pritesh Shah, Sushma Parihar, Harshavardhan Reddy Penubadi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Scene categorization (SC) in remotely acquired images is an important subject with broad consequences in different fields, including catastrophe control, ecological observation, architecture for cities, and more. Nevertheless, its several apps, reaching a high degree of accuracy in SC from distant observation data has demonstrated to be difficult. This is because traditional conventional deep learning models require large databases with high variety and high levels of noise to capture important visual features. To address these problems, this investigation file introduces an innovative technique referred to as the Cuttlefish Optimized Bidirectional Recurrent Neural Network (CO- BRNN) for type of scenes in remote sensing data. The investigation compares the execution of CO-BRNN with current techniques, including Multilayer Perceptron- Convolutional Neural Network (MLP-CNN), Convolutional Neural Network-Long Short Term Memory (CNN-LSTM), and Long Short Term Memory-Conditional Random Field (LSTM-CRF), Graph-Based (GB), Multilabel Image Retrieval Model (MIRM-CF), Convolutional Neural Networks Data Augmentation (CNN-DA). The results demonstrate that CO-BRNN attained the maximum accuracy of 97%, followed by LSTM-CRF with 90%, MLP-CNN with 85%, and CNN-LSTM with 80%. The study highlights the significance of physical confirmation to ensure the efficiency of satellite data.  
  </ol>  
</details>  
  
### [Mask-aware Text-to-Image Retrieval: Referring Expression Segmentation Meets Cross-modal Retrieval](http://arxiv.org/abs/2506.22864)  
Li-Cheng Shen, Jih-Kang Hsieh, Wei-Hua Li, Chu-Song Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Text-to-image retrieval (TIR) aims to find relevant images based on a textual query, but existing approaches are primarily based on whole-image captions and lack interpretability. Meanwhile, referring expression segmentation (RES) enables precise object localization based on natural language descriptions but is computationally expensive when applied across large image collections. To bridge this gap, we introduce Mask-aware TIR (MaTIR), a new task that unifies TIR and RES, requiring both efficient image search and accurate object segmentation. To address this task, we propose a two-stage framework, comprising a first stage for segmentation-aware image retrieval and a second stage for reranking and object grounding with a multimodal large language model (MLLM). We leverage SAM 2 to generate object masks and Alpha-CLIP to extract region-level embeddings offline at first, enabling effective and scalable online retrieval. Secondly, MLLM is used to refine retrieval rankings and generate bounding boxes, which are matched to segmentation masks. We evaluate our approach on COCO and D $^3$ datasets, demonstrating significant improvements in both retrieval accuracy and segmentation quality over previous methods.  
  </ol>  
</details>  
**comments**: ICMR 2025  
  
  



## Image Matching  

### [Efficient and Accurate Image Provenance Analysis: A Scalable Pipeline for Large-scale Images](http://arxiv.org/abs/2506.23707)  
Jiewei Lai, Lan Zhang, Chen Tang, Pengcheng Sun  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The rapid proliferation of modified images on social networks that are driven by widely accessible editing tools demands robust forensic tools for digital governance. Image provenance analysis, which filters various query image variants and constructs a directed graph to trace their phylogeny history, has emerged as a critical solution. However, existing methods face two fundamental limitations: First, accuracy issues arise from overlooking heavily modified images due to low similarity while failing to exclude unrelated images and determine modification directions under diverse modification scenarios. Second, scalability bottlenecks stem from pairwise image analysis incurs quadratic complexity, hindering application in large-scale scenarios. This paper presents a scalable end-to-end pipeline for image provenance analysis that achieves high precision with linear complexity. This improves filtering effectiveness through modification relationship tracing, which enables the comprehensive discovery of image variants regardless of their visual similarity to the query. In addition, the proposed pipeline integrates local features matching and compression artifact capturing, enhancing robustness against diverse modifications and enabling accurate analysis of images' relationships. This allows the generation of a directed provenance graph that accurately characterizes the image's phylogeny history. Furthermore, by optimizing similarity calculations and eliminating redundant pairwise analysis during graph construction, the pipeline achieves a linear time complexity, ensuring its scalability for large-scale scenarios. Experiments demonstrate pipeline's superior performance, achieving a 16.7-56.1% accuracy improvement. Notably, it exhibits significant scalability with an average 3.0-second response time on 10 million scale images, which is far shorter than the SOTA approach's 12-minute duration.  
  </ol>  
</details>  
**comments**: 25 pages, 6 figures  
  
### [Dynamic Contrastive Learning for Hierarchical Retrieval: A Case Study of Distance-Aware Cross-View Geo-Localization](http://arxiv.org/abs/2506.23077)  
Suofei Zhang, Xinxin Wang, Xiaofu Wu, Quan Zhou, Haifeng Hu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Existing deep learning-based cross-view geo-localization methods primarily focus on improving the accuracy of cross-domain image matching, rather than enabling models to comprehensively capture contextual information around the target and minimize the cost of localization errors. To support systematic research into this Distance-Aware Cross-View Geo-Localization (DACVGL) problem, we construct Distance-Aware Campus (DA-Campus), the first benchmark that pairs multi-view imagery with precise distance annotations across three spatial resolutions. Based on DA-Campus, we formulate DACVGL as a hierarchical retrieval problem across different domains. Our study further reveals that, due to the inherent complexity of spatial relationships among buildings, this problem can only be addressed via a contrastive learning paradigm, rather than conventional metric learning. To tackle this challenge, we propose Dynamic Contrastive Learning (DyCL), a novel framework that progressively aligns feature representations according to hierarchical spatial margins. Extensive experiments demonstrate that DyCL is highly complementary to existing multi-scale metric learning methods and yields substantial improvements in both hierarchical retrieval performance and overall cross-view geo-localization accuracy. Our code and benchmark are publicly available at https://github.com/anocodetest1/DyCL.  
  </ol>  
</details>  
  
  



## NeRF  

### [AttentionGS: Towards Initialization-Free 3D Gaussian Splatting via Structural Attention](http://arxiv.org/abs/2506.23611)  
Ziao Liu, Zhenjia Li, Yifeng Shi, Xiangang Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D Gaussian Splatting (3DGS) is a powerful alternative to Neural Radiance Fields (NeRF), excelling in complex scene reconstruction and efficient rendering. However, it relies on high-quality point clouds from Structure-from-Motion (SfM), limiting its applicability. SfM also fails in texture-deficient or constrained-view scenarios, causing severe degradation in 3DGS reconstruction. To address this limitation, we propose AttentionGS, a novel framework that eliminates the dependency on high-quality initial point clouds by leveraging structural attention for direct 3D reconstruction from randomly initialization. In the early training stage, we introduce geometric attention to rapidly recover the global scene structure. As training progresses, we incorporate texture attention to refine fine-grained details and enhance rendering quality. Furthermore, we employ opacity-weighted gradients to guide Gaussian densification, leading to improved surface reconstruction. Extensive experiments on multiple benchmark datasets demonstrate that AttentionGS significantly outperforms state-of-the-art methods, particularly in scenarios where point cloud initialization is unreliable. Our approach paves the way for more robust and flexible 3D Gaussian Splatting in real-world applications.  
  </ol>  
</details>  
  
### [Dynamic View Synthesis from Small Camera Motion Videos](http://arxiv.org/abs/2506.23153)  
Huiqiang Sun, Xingyi Li, Juewen Peng, Liao Shen, Zhiguo Cao, Ke Xian, Guosheng Lin  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Novel view synthesis for dynamic $3$ D scenes poses a significant challenge. Many notable efforts use NeRF-based approaches to address this task and yield impressive results. However, these methods rely heavily on sufficient motion parallax in the input images or videos. When the camera motion range becomes limited or even stationary (i.e., small camera motion), existing methods encounter two primary challenges: incorrect representation of scene geometry and inaccurate estimation of camera parameters. These challenges make prior methods struggle to produce satisfactory results or even become invalid. To address the first challenge, we propose a novel Distribution-based Depth Regularization (DDR) that ensures the rendering weight distribution to align with the true distribution. Specifically, unlike previous methods that use depth loss to calculate the error of the expectation, we calculate the expectation of the error by using Gumbel-softmax to differentiably sample points from discrete rendering weight distribution. Additionally, we introduce constraints that enforce the volume density of spatial points before the object boundary along the ray to be near zero, ensuring that our model learns the correct geometry of the scene. To demystify the DDR, we further propose a visualization tool that enables observing the scene geometry representation at the rendering weight level. For the second challenge, we incorporate camera parameter learning during training to enhance the robustness of our model to camera parameters. We conduct extensive experiments to demonstrate the effectiveness of our approach in representing scenes with small camera motion input, and our results compare favorably to state-of-the-art methods.  
  </ol>  
</details>  
**comments**: Accepted by TVCG  
  
  



