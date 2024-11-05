<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Exploiting-Contextual-Uncertainty-of-Visual-Data-for-Efficient-Training-of-Deep-Models>Exploiting Contextual Uncertainty of Visual Data for Efficient Training of Deep Models</a></li>
        <li><a href=#Semantic-Masking-and-Visual-Feature-Matching-for-Robust-Localization>Semantic Masking and Visual Feature Matching for Robust Localization</a></li>
        <li><a href=#Efficient-Medical-Image-Retrieval-Using-DenseNet-and-FAISS-for-BIRADS-Classification>Efficient Medical Image Retrieval Using DenseNet and FAISS for BIRADS Classification</a></li>
        <li><a href=#Identifying-Implicit-Social-Biases-in-Vision-Language-Models>Identifying Implicit Social Biases in Vision-Language Models</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Silver-medal-Solution-for-Image-Matching-Challenge-2024>Silver medal Solution for Image Matching Challenge 2024</a></li>
        <li><a href=#KptLLM:-Unveiling-the-Power-of-Large-Language-Model-for-Keypoint-Comprehension>KptLLM: Unveiling the Power of Large Language Model for Keypoint Comprehension</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Silver-medal-Solution-for-Image-Matching-Challenge-2024>Silver medal Solution for Image Matching Challenge 2024</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#FewViewGS:-Gaussian-Splatting-with-Few-View-Matching-and-Multi-stage-Training>FewViewGS: Gaussian Splatting with Few View Matching and Multi-stage Training</a></li>
        <li><a href=#GVKF:-Gaussian-Voxel-Kernel-Functions-for-Highly-Efficient-Surface-Reconstruction-in-Open-Scenes>GVKF: Gaussian Voxel Kernel Functions for Highly Efficient Surface Reconstruction in Open Scenes</a></li>
        <li><a href=#A-Probabilistic-Formulation-of-LiDAR-Mapping-with-Neural-Radiance-Fields>A Probabilistic Formulation of LiDAR Mapping with Neural Radiance Fields</a></li>
        <li><a href=#ZIM:-Zero-Shot-Image-Matting-for-Anything>ZIM: Zero-Shot Image Matting for Anything</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Exploiting Contextual Uncertainty of Visual Data for Efficient Training of Deep Models](http://arxiv.org/abs/2411.01925)  
Sharat Agarwal  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Objects, in the real world, rarely occur in isolation and exhibit typical arrangements governed by their independent utility, and their expected interaction with humans and other objects in the context. For example, a chair is expected near a table, and a computer is expected on top. Humans use this spatial context and relative placement as an important cue for visual recognition in case of ambiguities. Similar to human's, DNN's exploit contextual information from data to learn representations. Our research focuses on harnessing the contextual aspects of visual data to optimize data annotation and enhance the training of deep networks. Our contributions can be summarized as follows: (1) We introduce the notion of contextual diversity for active learning CDAL and show its applicability in three different visual tasks semantic segmentation, object detection and image classification, (2) We propose a data repair algorithm to curate contextually fair data to reduce model bias, enabling the model to detect objects out of their obvious context, (3) We propose Class-based annotation, where contextually relevant classes are selected that are complementary for model training under domain shift. Understanding the importance of well-curated data, we also emphasize the necessity of involving humans in the loop to achieve accurate annotations and to develop novel interaction strategies that allow humans to serve as fact-checkers. In line with this we are working on developing image retrieval system for wildlife camera trap images and reliable warning system for poor quality rural roads. For large-scale annotation, we are employing a strategic combination of human expertise and zero-shot models, while also integrating human input at various stages for continuous feedback.  
  </ol>  
</details>  
**comments**: ICVGIP, Young Researchers Symposium  
  
### [Semantic Masking and Visual Feature Matching for Robust Localization](http://arxiv.org/abs/2411.01804)  
Luisa Mao, Ryan Soussan, Brian Coltin, Trey Smith, Joydeep Biswas  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We are interested in long-term deployments of autonomous robots to aid astronauts with maintenance and monitoring operations in settings such as the International Space Station. Unfortunately, such environments tend to be highly dynamic and unstructured, and their frequent reconfiguration poses a challenge for robust long-term localization of robots. Many state-of-the-art visual feature-based localization algorithms are not robust towards spatial scene changes, and SLAM algorithms, while promising, cannot run within the low-compute budget available to space robots. To address this gap, we present a computationally efficient semantic masking approach for visual feature matching that improves the accuracy and robustness of visual localization systems during long-term deployment in changing environments. Our method introduces a lightweight check that enforces matches to be within long-term static objects and have consistent semantic classes. We evaluate this approach using both map-based relocalization and relative pose estimation and show that it improves Absolute Trajectory Error (ATE) and correct match ratios on the publicly available Astrobee dataset. While this approach was originally developed for microgravity robotic freeflyers, it can be applied to any visual feature matching pipeline to improve robustness.  
  </ol>  
</details>  
**comments**: 7 pages  
  
### [Efficient Medical Image Retrieval Using DenseNet and FAISS for BIRADS Classification](http://arxiv.org/abs/2411.01473)  
MD Shaikh Rahman, Feiroz Humayara, Syed Maudud E Rabbi, Muhammad Mahbubur Rashid  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    That datasets that are used in todays research are especially vast in the medical field. Different types of medical images such as X-rays, MRI, CT scan etc. take up large amounts of space. This volume of data introduces challenges like accessing and retrieving specific images due to the size of the database. An efficient image retrieval system is essential as the database continues to grow to save time and resources. In this paper, we propose an approach to medical image retrieval using DenseNet for feature extraction and use FAISS for similarity search. DenseNet is well-suited for feature extraction in complex medical images and FAISS enables efficient handling of high-dimensional data in large-scale datasets. Unlike existing methods focused solely on classification accuracy, our method prioritizes both retrieval speed and diagnostic relevance, addressing a critical gap in real-time case comparison for radiologists. We applied the classification of breast cancer images using the BIRADS system. We utilized DenseNet's powerful feature representation and FAISSs efficient indexing capabilities to achieve high precision and recall in retrieving relevant images for diagnosis. We experimented on a dataset of 2006 images from the Categorized Digital Database for Low Energy and Subtracted Contrast Enhanced Spectral Mammography (CDD-CESM) images available on The Cancer Imaging Archive (TCIA). Our method outperforms conventional retrieval techniques, achieving a precision of 80% at k=5 for BIRADS classification. The dataset includes annotated CESM images and medical reports, providing a comprehensive foundation for our research.  
  </ol>  
</details>  
**comments**: 34 pages, 5 figures  
  
### [Identifying Implicit Social Biases in Vision-Language Models](http://arxiv.org/abs/2411.00997)  
Kimia Hamidieh, Haoran Zhang, Walter Gerych, Thomas Hartvigsen, Marzyeh Ghassemi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Vision-language models, like CLIP (Contrastive Language Image Pretraining), are becoming increasingly popular for a wide range of multimodal retrieval tasks. However, prior work has shown that large language and deep vision models can learn historical biases contained in their training sets, leading to perpetuation of stereotypes and potential downstream harm. In this work, we conduct a systematic analysis of the social biases that are present in CLIP, with a focus on the interaction between image and text modalities. We first propose a taxonomy of social biases called So-B-IT, which contains 374 words categorized across ten types of bias. Each type can lead to societal harm if associated with a particular demographic group. Using this taxonomy, we examine images retrieved by CLIP from a facial image dataset using each word as part of a prompt. We find that CLIP frequently displays undesirable associations between harmful words and specific demographic groups, such as retrieving mostly pictures of Middle Eastern men when asked to retrieve images of a "terrorist". Finally, we conduct an analysis of the source of such biases, by showing that the same harmful stereotypes are also present in a large image-text dataset used to train CLIP models for examples of biases that we find. Our findings highlight the importance of evaluating and addressing bias in vision-language models, and suggest the need for transparency and fairness-aware curation of large pre-training datasets.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Silver medal Solution for Image Matching Challenge 2024](http://arxiv.org/abs/2411.01851)  
Yian Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image Matching Challenge 2024 is a competition focused on building 3D maps from diverse image sets, requiring participants to solve fundamental computer vision challenges in image matching across varying angles, lighting, and seasonal changes. This project develops a Pipeline method that combines multiple advanced techniques: using pre-trained EfficientNet-B7 for initial feature extraction and cosine distance-based image pair filtering, employing both KeyNetAffNetHardNet and SuperPoint for keypoint feature extraction, utilizing AdaLAM and SuperGlue for keypoint matching, and finally applying Pycolmap for 3D spatial analysis. The methodology achieved an excellent score of 0.167 on the private leaderboard, with experimental results demonstrating that the combination of KeyNetAffNetHardNet and SuperPoint provides significant advantages in keypoint detection and matching, particularly when dealing with challenging variations in surface texture and environmental conditions that typically degrade traditional algorithm performance.  
  </ol>  
</details>  
  
### [KptLLM: Unveiling the Power of Large Language Model for Keypoint Comprehension](http://arxiv.org/abs/2411.01846)  
Jie Yang, Wang Zeng, Sheng Jin, Lumin Xu, Wentao Liu, Chen Qian, Ruimao Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in Multimodal Large Language Models (MLLMs) have greatly improved their abilities in image understanding. However, these models often struggle with grasping pixel-level semantic details, e.g., the keypoints of an object. To bridge this gap, we introduce the novel challenge of Semantic Keypoint Comprehension, which aims to comprehend keypoints across different task scenarios, including keypoint semantic understanding, visual prompt-based keypoint detection, and textual prompt-based keypoint detection. Moreover, we introduce KptLLM, a unified multimodal model that utilizes an identify-then-detect strategy to effectively address these challenges. KptLLM underscores the initial discernment of semantics in keypoints, followed by the precise determination of their positions through a chain-of-thought process. With several carefully designed modules, KptLLM adeptly handles various modality inputs, facilitating the interpretation of both semantic contents and keypoint locations. Our extensive experiments demonstrate KptLLM's superiority in various keypoint detection benchmarks and its unique semantic capabilities in interpreting keypoints.  
  </ol>  
</details>  
**comments**: NeurIPS 2024  
  
  



## Image Matching  

### [Silver medal Solution for Image Matching Challenge 2024](http://arxiv.org/abs/2411.01851)  
Yian Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image Matching Challenge 2024 is a competition focused on building 3D maps from diverse image sets, requiring participants to solve fundamental computer vision challenges in image matching across varying angles, lighting, and seasonal changes. This project develops a Pipeline method that combines multiple advanced techniques: using pre-trained EfficientNet-B7 for initial feature extraction and cosine distance-based image pair filtering, employing both KeyNetAffNetHardNet and SuperPoint for keypoint feature extraction, utilizing AdaLAM and SuperGlue for keypoint matching, and finally applying Pycolmap for 3D spatial analysis. The methodology achieved an excellent score of 0.167 on the private leaderboard, with experimental results demonstrating that the combination of KeyNetAffNetHardNet and SuperPoint provides significant advantages in keypoint detection and matching, particularly when dealing with challenging variations in surface texture and environmental conditions that typically degrade traditional algorithm performance.  
  </ol>  
</details>  
  
  



## NeRF  

### [FewViewGS: Gaussian Splatting with Few View Matching and Multi-stage Training](http://arxiv.org/abs/2411.02229)  
Ruihong Yin, Vladimir Yugay, Yue Li, Sezer Karaoglu, Theo Gevers  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The field of novel view synthesis from images has seen rapid advancements with the introduction of Neural Radiance Fields (NeRF) and more recently with 3D Gaussian Splatting. Gaussian Splatting became widely adopted due to its efficiency and ability to render novel views accurately. While Gaussian Splatting performs well when a sufficient amount of training images are available, its unstructured explicit representation tends to overfit in scenarios with sparse input images, resulting in poor rendering performance. To address this, we present a 3D Gaussian-based novel view synthesis method using sparse input images that can accurately render the scene from the viewpoints not covered by the training images. We propose a multi-stage training scheme with matching-based consistency constraints imposed on the novel views without relying on pre-trained depth estimation or diffusion models. This is achieved by using the matches of the available training images to supervise the generation of the novel views sampled between the training frames with color, geometry, and semantic losses. In addition, we introduce a locality preserving regularization for 3D Gaussians which removes rendering artifacts by preserving the local color structure of the scene. Evaluation on synthetic and real-world datasets demonstrates competitive or superior performance of our method in few-shot novel view synthesis compared to existing state-of-the-art methods.  
  </ol>  
</details>  
**comments**: Accepted by NeurIPS2024  
  
### [GVKF: Gaussian Voxel Kernel Functions for Highly Efficient Surface Reconstruction in Open Scenes](http://arxiv.org/abs/2411.01853)  
Gaochao Song, Chong Cheng, Hao Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper we present a novel method for efficient and effective 3D surface reconstruction in open scenes. Existing Neural Radiance Fields (NeRF) based works typically require extensive training and rendering time due to the adopted implicit representations. In contrast, 3D Gaussian splatting (3DGS) uses an explicit and discrete representation, hence the reconstructed surface is built by the huge number of Gaussian primitives, which leads to excessive memory consumption and rough surface details in sparse Gaussian areas. To address these issues, we propose Gaussian Voxel Kernel Functions (GVKF), which establish a continuous scene representation based on discrete 3DGS through kernel regression. The GVKF integrates fast 3DGS rasterization and highly effective scene implicit representations, achieving high-fidelity open scene surface reconstruction. Experiments on challenging scene datasets demonstrate the efficiency and effectiveness of our proposed GVKF, featuring with high reconstruction quality, real-time rendering speed, significant savings in storage and training memory consumption.  
  </ol>  
</details>  
**comments**: NeurIPS 2024  
  
### [A Probabilistic Formulation of LiDAR Mapping with Neural Radiance Fields](http://arxiv.org/abs/2411.01725)  
[[code](https://github.com/mcdermatt/plink)]  
Matthew McDermott, Jason Rife  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper we reexamine the process through which a Neural Radiance Field (NeRF) can be trained to produce novel LiDAR views of a scene. Unlike image applications where camera pixels integrate light over time, LiDAR pulses arrive at specific times. As such, multiple LiDAR returns are possible for any given detector and the classification of these returns is inherently probabilistic. Applying a traditional NeRF training routine can result in the network learning phantom surfaces in free space between conflicting range measurements, similar to how floater aberrations may be produced by an image model. We show that by formulating loss as an integral of probability (rather than as an integral of optical density) the network can learn multiple peaks for a given ray, allowing the sampling of first, nth, or strongest returns from a single output channel. Code is available at https://github.com/mcdermatt/PLINK  
  </ol>  
</details>  
  
### [ZIM: Zero-Shot Image Matting for Anything](http://arxiv.org/abs/2411.00626)  
[[code](https://github.com/naver-ai/zim)]  
Beomyoung Kim, Chanyong Shin, Joonhyun Jeong, Hyungsik Jung, Se-Yun Lee, Sewhan Chun, Dong-Hyun Hwang, Joonsang Yu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The recent segmentation foundation model, Segment Anything Model (SAM), exhibits strong zero-shot segmentation capabilities, but it falls short in generating fine-grained precise masks. To address this limitation, we propose a novel zero-shot image matting model, called ZIM, with two key contributions: First, we develop a label converter that transforms segmentation labels into detailed matte labels, constructing the new SA1B-Matte dataset without costly manual annotations. Training SAM with this dataset enables it to generate precise matte masks while maintaining its zero-shot capability. Second, we design the zero-shot matting model equipped with a hierarchical pixel decoder to enhance mask representation, along with a prompt-aware masked attention mechanism to improve performance by enabling the model to focus on regions specified by visual prompts. We evaluate ZIM using the newly introduced MicroMat-3K test set, which contains high-quality micro-level matte labels. Experimental results show that ZIM outperforms existing methods in fine-grained mask generation and zero-shot generalization. Furthermore, we demonstrate the versatility of ZIM in various downstream tasks requiring precise masks, such as image inpainting and 3D NeRF. Our contributions provide a robust foundation for advancing zero-shot matting and its downstream applications across a wide range of computer vision tasks. The code is available at \url{https://github.com/naver-ai/ZIM}.  
  </ol>  
</details>  
**comments**: preprint (21 pages, 16 figures, and 8 tables)  
  
  



