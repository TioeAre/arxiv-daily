<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#LOCORE:-Image-Re-ranking-with-Long-Context-Sequence-Modeling>LOCORE: Image Re-ranking with Long-Context Sequence Modeling</a></li>
        <li><a href=#Fwd2Bot:-LVLM-Visual-Token-Compression-with-Double-Forward-Bottleneck>Fwd2Bot: LVLM Visual Token Compression with Double Forward Bottleneck</a></li>
        <li><a href=#UGNA-VPR:-A-Novel-Training-Paradigm-for-Visual-Place-Recognition-Based-on-Uncertainty-Guided-NeRF-Augmentation>UGNA-VPR: A Novel Training Paradigm for Visual Place Recognition Based on Uncertainty-Guided NeRF Augmentation</a></li>
        <li><a href=#FineCIR:-Explicit-Parsing-of-Fine-Grained-Modification-Semantics-for-Composed-Image-Retrieval>FineCIR: Explicit Parsing of Fine-Grained Modification Semantics for Composed Image Retrieval</a></li>
        <li><a href=#Clean-Image-May-be-Dangerous:-Data-Poisoning-Attacks-Against-Deep-Hashing>Clean Image May be Dangerous: Data Poisoning Attacks Against Deep Hashing</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Multimodal-Image-Matching-based-on-Frequency-domain-Information-of-Local-Energy-Response>Multimodal Image Matching based on Frequency-domain Information of Local Energy Response</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#HS-SLAM:-Hybrid-Representation-with-Structural-Supervision-for-Improved-Dense-SLAM>HS-SLAM: Hybrid Representation with Structural Supervision for Improved Dense SLAM</a></li>
        <li><a href=#RainyGS:-Efficient-Rain-Synthesis-with-Physically-Based-Gaussian-Splatting>RainyGS: Efficient Rain Synthesis with Physically-Based Gaussian Splatting</a></li>
        <li><a href=#LandMarkSystem-Technical-Report>LandMarkSystem Technical Report</a></li>
        <li><a href=#UGNA-VPR:-A-Novel-Training-Paradigm-for-Visual-Place-Recognition-Based-on-Uncertainty-Guided-NeRF-Augmentation>UGNA-VPR: A Novel Training Paradigm for Visual Place Recognition Based on Uncertainty-Guided NeRF Augmentation</a></li>
        <li><a href=#CoMapGS:-Covisibility-Map-based-Gaussian-Splatting-for-Sparse-Novel-View-Synthesis>CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [LOCORE: Image Re-ranking with Long-Context Sequence Modeling](http://arxiv.org/abs/2503.21772)  
Zilin Xiao, Pavel Suma, Ayush Sachdeva, Hao-Jen Wang, Giorgos Kordopatis-Zilos, Giorgos Tolias, Vicente Ordonez  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce LOCORE, Long-Context Re-ranker, a model that takes as input local descriptors corresponding to an image query and a list of gallery images and outputs similarity scores between the query and each gallery image. This model is used for image retrieval, where typically a first ranking is performed with an efficient similarity measure, and then a shortlist of top-ranked images is re-ranked based on a more fine-grained similarity measure. Compared to existing methods that perform pair-wise similarity estimation with local descriptors or list-wise re-ranking with global descriptors, LOCORE is the first method to perform list-wise re-ranking with local descriptors. To achieve this, we leverage efficient long-context sequence models to effectively capture the dependencies between query and gallery images at the local-descriptor level. During testing, we process long shortlists with a sliding window strategy that is tailored to overcome the context size limitations of sequence models. Our approach achieves superior performance compared with other re-rankers on established image retrieval benchmarks of landmarks (ROxf and RPar), products (SOP), fashion items (In-Shop), and bird species (CUB-200) while having comparable latency to the pair-wise local descriptor re-rankers.  
  </ol>  
</details>  
**comments**: CVPR 2025  
  
### [Fwd2Bot: LVLM Visual Token Compression with Double Forward Bottleneck](http://arxiv.org/abs/2503.21757)  
Adrian Bulat, Yassine Ouali, Georgios Tzimiropoulos  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this work, we aim to compress the vision tokens of a Large Vision Language Model (LVLM) into a representation that is simultaneously suitable for (a) generative and (b) discriminative tasks, (c) is nearly lossless, and (d) is storage-efficient. We propose a novel compression approach, called Fwd2Bot, that uses the LVLM itself to compress the visual information in a task-agnostic manner. At the core of Fwd2bot there exists a "double-forward pass" training strategy, whereby, during the first forward pass, the LLM (of the LVLM) creates a bottleneck by condensing the visual information into a small number of summary tokens. Then, using the same LLM, the second forward pass processes the language instruction(s) alongside the summary tokens, used as a direct replacement for the image ones. The training signal is provided by two losses: an autoregressive one applied after the second pass that provides a direct optimization objective for compression, and a contrastive loss, applied after the first pass, that further boosts the representation strength, especially for discriminative tasks. The training is further enhanced by stage-specific adapters. We accompany the proposed method by an in-depth ablation study. Overall, Fwd2Bot results in highly-informative compressed representations suitable for both generative and discriminative tasks. For generative tasks, we offer a 2x higher compression rate without compromising the generative capabilities, setting a new state-of-the-art result. For discriminative tasks, we set a new state-of-the-art on image retrieval and compositionality.  
  </ol>  
</details>  
  
### [UGNA-VPR: A Novel Training Paradigm for Visual Place Recognition Based on Uncertainty-Guided NeRF Augmentation](http://arxiv.org/abs/2503.21338)  
Yehui Shen, Lei Zhang, Qingqiu Li, Xiongwei Zhao, Yue Wang, Huimin Lu, Xieyuanli Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual place recognition (VPR) is crucial for robots to identify previously visited locations, playing an important role in autonomous navigation in both indoor and outdoor environments. However, most existing VPR datasets are limited to single-viewpoint scenarios, leading to reduced recognition accuracy, particularly in multi-directional driving or feature-sparse scenes. Moreover, obtaining additional data to mitigate these limitations is often expensive. This paper introduces a novel training paradigm to improve the performance of existing VPR networks by enhancing multi-view diversity within current datasets through uncertainty estimation and NeRF-based data augmentation. Specifically, we initially train NeRF using the existing VPR dataset. Then, our devised self-supervised uncertainty estimation network identifies places with high uncertainty. The poses of these uncertain places are input into NeRF to generate new synthetic observations for further training of VPR networks. Additionally, we propose an improved storage method for efficient organization of augmented and original training data. We conducted extensive experiments on three datasets and tested three different VPR backbone networks. The results demonstrate that our proposed training paradigm significantly improves VPR performance by fully utilizing existing data, outperforming other training approaches. We further validated the effectiveness of our approach on self-recorded indoor and outdoor datasets, consistently demonstrating superior results. Our dataset and code have been released at \href{https://github.com/nubot-nudt/UGNA-VPR}{https://github.com/nubot-nudt/UGNA-VPR}.  
  </ol>  
</details>  
**comments**: Accepted to IEEE Robotics and Automation Letters (RA-L)  
  
### [FineCIR: Explicit Parsing of Fine-Grained Modification Semantics for Composed Image Retrieval](http://arxiv.org/abs/2503.21309)  
Zixu Li, Zhiheng Fu, Yupeng Hu, Zhiwei Chen, Haokun Wen, Liqiang Nie  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed Image Retrieval (CIR) facilitates image retrieval through a multimodal query consisting of a reference image and modification text. The reference image defines the retrieval context, while the modification text specifies desired alterations. However, existing CIR datasets predominantly employ coarse-grained modification text (CoarseMT), which inadequately captures fine-grained retrieval intents. This limitation introduces two key challenges: (1) ignoring detailed differences leads to imprecise positive samples, and (2) greater ambiguity arises when retrieving visually similar images. These issues degrade retrieval accuracy, necessitating manual result filtering or repeated queries. To address these limitations, we develop a robust fine-grained CIR data annotation pipeline that minimizes imprecise positive samples and enhances CIR systems' ability to discern modification intents accurately. Using this pipeline, we refine the FashionIQ and CIRR datasets to create two fine-grained CIR datasets: Fine-FashionIQ and Fine-CIRR. Furthermore, we introduce FineCIR, the first CIR framework explicitly designed to parse the modification text. FineCIR effectively captures fine-grained modification semantics and aligns them with ambiguous visual entities, enhancing retrieval precision. Extensive experiments demonstrate that FineCIR consistently outperforms state-of-the-art CIR baselines on both fine-grained and traditional CIR benchmark datasets. Our FineCIR code and fine-grained CIR datasets are available at https://github.com/SDU-L/FineCIR.git.  
  </ol>  
</details>  
  
### [Clean Image May be Dangerous: Data Poisoning Attacks Against Deep Hashing](http://arxiv.org/abs/2503.21236)  
Shuai Li, Jie Zhang, Yuang Qi, Kejiang Chen, Tianwei Zhang, Weiming Zhang, Nenghai Yu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Large-scale image retrieval using deep hashing has become increasingly popular due to the exponential growth of image data and the remarkable feature extraction capabilities of deep neural networks (DNNs). However, deep hashing methods are vulnerable to malicious attacks, including adversarial and backdoor attacks. It is worth noting that these attacks typically involve altering the query images, which is not a practical concern in real-world scenarios. In this paper, we point out that even clean query images can be dangerous, inducing malicious target retrieval results, like undesired or illegal images. To the best of our knowledge, we are the first to study data \textbf{p}oisoning \textbf{a}ttacks against \textbf{d}eep \textbf{hash}ing \textbf{(\textit{PADHASH})}. Specifically, we first train a surrogate model to simulate the behavior of the target deep hashing model. Then, a strict gradient matching strategy is proposed to generate the poisoned images. Extensive experiments on different models, datasets, hash methods, and hash code lengths demonstrate the effectiveness and generality of our attack method.  
  </ol>  
</details>  
**comments**: Accepted by TMM  
  
  



## Image Matching  

### [Multimodal Image Matching based on Frequency-domain Information of Local Energy Response](http://arxiv.org/abs/2503.20827)  
Meng Yang, Jun Chen, Wenping Gong, Longsheng Wei, Xin Tian  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Complicated nonlinear intensity differences, nonlinear local geometric distortions, noises and rotation transformation are main challenges in multimodal image matching. In order to solve these problems, we propose a method based on Frequency-domain Information of Local Energy Response called FILER. The core of FILER is the local energy response model based on frequency-domain information, which can overcome the effect of nonlinear intensity differences. To improve the robustness to local nonlinear geometric distortions and noises, we design a new edge structure enhanced feature detector and convolutional feature weighted descriptor, respectively. In addition, FILER overcomes the sensitivity of the frequency-domain information to the rotation angle and achieves rotation invariance. Extensive experiments multimodal image pairs show that FILER outperforms other state-of-the-art algorithms and has good robustness and universality.  
  </ol>  
</details>  
**comments**: 34 pages, 11 figures  
  
  



## NeRF  

### [HS-SLAM: Hybrid Representation with Structural Supervision for Improved Dense SLAM](http://arxiv.org/abs/2503.21778)  
Ziren Gong, Fabio Tosi, Youmin Zhang, Stefano Mattoccia, Matteo Poggi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    NeRF-based SLAM has recently achieved promising results in tracking and reconstruction. However, existing methods face challenges in providing sufficient scene representation, capturing structural information, and maintaining global consistency in scenes emerging significant movement or being forgotten. To this end, we present HS-SLAM to tackle these problems. To enhance scene representation capacity, we propose a hybrid encoding network that combines the complementary strengths of hash-grid, tri-planes, and one-blob, improving the completeness and smoothness of reconstruction. Additionally, we introduce structural supervision by sampling patches of non-local pixels rather than individual rays to better capture the scene structure. To ensure global consistency, we implement an active global bundle adjustment (BA) to eliminate camera drifts and mitigate accumulative errors. Experimental results demonstrate that HS-SLAM outperforms the baselines in tracking and reconstruction accuracy while maintaining the efficiency required for robotics.  
  </ol>  
</details>  
**comments**: ICRA 2025. Project Page: https://zorangong.github.io/HS-SLAM/  
  
### [RainyGS: Efficient Rain Synthesis with Physically-Based Gaussian Splatting](http://arxiv.org/abs/2503.21442)  
Qiyu Dai, Xingyu Ni, Qianfan Shen, Wenzheng Chen, Baoquan Chen, Mengyu Chu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We consider the problem of adding dynamic rain effects to in-the-wild scenes in a physically-correct manner. Recent advances in scene modeling have made significant progress, with NeRF and 3DGS techniques emerging as powerful tools for reconstructing complex scenes. However, while effective for novel view synthesis, these methods typically struggle with challenging scene editing tasks, such as physics-based rain simulation. In contrast, traditional physics-based simulations can generate realistic rain effects, such as raindrops and splashes, but they often rely on skilled artists to carefully set up high-fidelity scenes. This process lacks flexibility and scalability, limiting its applicability to broader, open-world environments. In this work, we introduce RainyGS, a novel approach that leverages the strengths of both physics-based modeling and 3DGS to generate photorealistic, dynamic rain effects in open-world scenes with physical accuracy. At the core of our method is the integration of physically-based raindrop and shallow water simulation techniques within the fast 3DGS rendering framework, enabling realistic and efficient simulations of raindrop behavior, splashes, and reflections. Our method supports synthesizing rain effects at over 30 fps, offering users flexible control over rain intensity -- from light drizzles to heavy downpours. We demonstrate that RainyGS performs effectively for both real-world outdoor scenes and large-scale driving scenarios, delivering more photorealistic and physically-accurate rain effects compared to state-of-the-art methods. Project page can be found at https://pku-vcl-geometry.github.io/RainyGS/  
  </ol>  
</details>  
  
### [LandMarkSystem Technical Report](http://arxiv.org/abs/2503.21364)  
Zhenxiang Ma, Zhenyu Yang, Miao Tao, Yuanzhen Zhou, Zeyu He, Yuchang Zhang, Rong Fu, Hengjie Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D reconstruction is vital for applications in autonomous driving, virtual reality, augmented reality, and the metaverse. Recent advancements such as Neural Radiance Fields(NeRF) and 3D Gaussian Splatting (3DGS) have transformed the field, yet traditional deep learning frameworks struggle to meet the increasing demands for scene quality and scale. This paper introduces LandMarkSystem, a novel computing framework designed to enhance multi-scale scene reconstruction and rendering. By leveraging a componentized model adaptation layer, LandMarkSystem supports various NeRF and 3DGS structures while optimizing computational efficiency through distributed parallel computing and model parameter offloading. Our system addresses the limitations of existing frameworks, providing dedicated operators for complex 3D sparse computations, thus facilitating efficient training and rapid inference over extensive scenes. Key contributions include a modular architecture, a dynamic loading strategy for limited resources, and proven capabilities across multiple representative algorithms.This comprehensive solution aims to advance the efficiency and effectiveness of 3D reconstruction tasks.To facilitate further research and collaboration, the source code and documentation for the LandMarkSystem project are publicly available in an open-source repository, accessing the repository at: https://github.com/InternLandMark/LandMarkSystem.  
  </ol>  
</details>  
  
### [UGNA-VPR: A Novel Training Paradigm for Visual Place Recognition Based on Uncertainty-Guided NeRF Augmentation](http://arxiv.org/abs/2503.21338)  
Yehui Shen, Lei Zhang, Qingqiu Li, Xiongwei Zhao, Yue Wang, Huimin Lu, Xieyuanli Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual place recognition (VPR) is crucial for robots to identify previously visited locations, playing an important role in autonomous navigation in both indoor and outdoor environments. However, most existing VPR datasets are limited to single-viewpoint scenarios, leading to reduced recognition accuracy, particularly in multi-directional driving or feature-sparse scenes. Moreover, obtaining additional data to mitigate these limitations is often expensive. This paper introduces a novel training paradigm to improve the performance of existing VPR networks by enhancing multi-view diversity within current datasets through uncertainty estimation and NeRF-based data augmentation. Specifically, we initially train NeRF using the existing VPR dataset. Then, our devised self-supervised uncertainty estimation network identifies places with high uncertainty. The poses of these uncertain places are input into NeRF to generate new synthetic observations for further training of VPR networks. Additionally, we propose an improved storage method for efficient organization of augmented and original training data. We conducted extensive experiments on three datasets and tested three different VPR backbone networks. The results demonstrate that our proposed training paradigm significantly improves VPR performance by fully utilizing existing data, outperforming other training approaches. We further validated the effectiveness of our approach on self-recorded indoor and outdoor datasets, consistently demonstrating superior results. Our dataset and code have been released at \href{https://github.com/nubot-nudt/UGNA-VPR}{https://github.com/nubot-nudt/UGNA-VPR}.  
  </ol>  
</details>  
**comments**: Accepted to IEEE Robotics and Automation Letters (RA-L)  
  
### [CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis](http://arxiv.org/abs/2503.20998)  
Youngkyoon Jang, Eduardo Pérez-Pellitero  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose Covisibility Map-based Gaussian Splatting (CoMapGS), designed to recover underrepresented sparse regions in sparse novel view synthesis. CoMapGS addresses both high- and low-uncertainty regions by constructing covisibility maps, enhancing initial point clouds, and applying uncertainty-aware weighted supervision using a proximity classifier. Our contributions are threefold: (1) CoMapGS reframes novel view synthesis by leveraging covisibility maps as a core component to address region-specific uncertainty; (2) Enhanced initial point clouds for both low- and high-uncertainty regions compensate for sparse COLMAP-derived point clouds, improving reconstruction quality and benefiting few-shot 3DGS methods; (3) Adaptive supervision with covisibility-score-based weighting and proximity classification achieves consistent performance gains across scenes with varying sparsity scores derived from covisibility maps. Experimental results demonstrate that CoMapGS outperforms state-of-the-art methods on datasets including Mip-NeRF 360 and LLFF.  
  </ol>  
</details>  
**comments**: Accepted to CVPR 2025, Mistakenly submitted as a replacement for
  arXiv:2402.11057  
  
  



