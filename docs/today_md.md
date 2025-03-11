<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#DaD:-Distilled-Reinforcement-Learning-for-Diverse-Keypoint-Detection>DaD: Distilled Reinforcement Learning for Diverse Keypoint Detection</a></li>
        <li><a href=#Endo-FASt3r:-Endoscopic-Foundation-model-Adaptation-for-Structure-from-motion>Endo-FASt3r: Endoscopic Foundation model Adaptation for Structure from motion</a></li>
        <li><a href=#VidBot:-Learning-Generalizable-3D-Actions-from-In-the-Wild-2D-Human-Videos-for-Zero-Shot-Robotic-Manipulation>VidBot: Learning Generalizable 3D Actions from In-the-Wild 2D Human Videos for Zero-Shot Robotic Manipulation</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Find-your-Needle:-Small-Object-Image-Retrieval-via-Multi-Object-Attention-Optimization>Find your Needle: Small Object Image Retrieval via Multi-Object Attention Optimization</a></li>
        <li><a href=#Zero-Shot-Hashing-Based-on-Reconstruction-With-Part-Alignment>Zero-Shot Hashing Based on Reconstruction With Part Alignment</a></li>
        <li><a href=#Improving-Visual-Place-Recognition-with-Sequence-Matching-Receptiveness-Prediction>Improving Visual Place Recognition with Sequence-Matching Receptiveness Prediction</a></li>
        <li><a href=#RoboDesign1M:-A-Large-scale-Dataset-for-Robot-Design-Understanding>RoboDesign1M: A Large-scale Dataset for Robot Design Understanding</a></li>
        <li><a href=#StructVPR++:-Distill-Structural-and-Semantic-Knowledge-with-Weighting-Samples-for-Visual-Place-Recognition>StructVPR++: Distill Structural and Semantic Knowledge with Weighting Samples for Visual Place Recognition</a></li>
        <li><a href=#TextInPlace:-Indoor-Visual-Place-Recognition-in-Repetitive-Structures-with-Scene-Text-Spotting-and-Verification>TextInPlace: Indoor Visual Place Recognition in Repetitive Structures with Scene Text Spotting and Verification</a></li>
        <li><a href=#NeuraLoc:-Visual-Localization-in-Neural-Implicit-Map-with-Dual-Complementary-Features>NeuraLoc: Visual Localization in Neural Implicit Map with Dual Complementary Features</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#REF-VLM:-Triplet-Based-Referring-Paradigm-for-Unified-Visual-Decoding>REF-VLM: Triplet-Based Referring Paradigm for Unified Visual Decoding</a></li>
        <li><a href=#DaD:-Distilled-Reinforcement-Learning-for-Diverse-Keypoint-Detection>DaD: Distilled Reinforcement Learning for Diverse Keypoint Detection</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#CATPlan:-Loss-based-Collision-Prediction-in-End-to-End-Autonomous-Driving>CATPlan: Loss-based Collision Prediction in End-to-End Autonomous Driving</a></li>
        <li><a href=#Feature-EndoGaussian:-Feature-Distilled-Gaussian-Splatting-in-Surgical-Deformable-Scene-Reconstruction>Feature-EndoGaussian: Feature Distilled Gaussian Splatting in Surgical Deformable Scene Reconstruction</a></li>
        <li><a href=#SecureGS:-Boosting-the-Security-and-Fidelity-of-3D-Gaussian-Splatting-Steganography>SecureGS: Boosting the Security and Fidelity of 3D Gaussian Splatting Steganography</a></li>
        <li><a href=#NeuraLoc:-Visual-Localization-in-Neural-Implicit-Map-with-Dual-Complementary-Features>NeuraLoc: Visual Localization in Neural Implicit Map with Dual Complementary Features</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [DaD: Distilled Reinforcement Learning for Diverse Keypoint Detection](http://arxiv.org/abs/2503.07347)  
Johan Edstedt, Georg Bökman, Mårten Wadenbäck, Michael Felsberg  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Keypoints are what enable Structure-from-Motion (SfM) systems to scale to thousands of images. However, designing a keypoint detection objective is a non-trivial task, as SfM is non-differentiable. Typically, an auxiliary objective involving a descriptor is optimized. This however induces a dependency on the descriptor, which is undesirable. In this paper we propose a fully self-supervised and descriptor-free objective for keypoint detection, through reinforcement learning. To ensure training does not degenerate, we leverage a balanced top-K sampling strategy. While this already produces competitive models, we find that two qualitatively different types of detectors emerge, which are only able to detect light and dark keypoints respectively. To remedy this, we train a third detector, DaD, that optimizes the Kullback-Leibler divergence of the pointwise maximum of both light and dark detectors. Our approach significantly improve upon SotA across a range of benchmarks. Code and model weights are publicly available at https:github.com/parskatt/dad  
  </ol>  
</details>  
  
### [Endo-FASt3r: Endoscopic Foundation model Adaptation for Structure from motion](http://arxiv.org/abs/2503.07204)  
Mona Sheikh Zeinoddin, Mobarakol Islam, Zafer Tandogdu, Greg Shaw, Mathew J. Clarkson, Evangelos Mazomenos, Danail Stoyanov  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Accurate depth and camera pose estimation is essential for achieving high-quality 3D visualisations in robotic-assisted surgery. Despite recent advancements in foundation model adaptation to monocular depth estimation of endoscopic scenes via self-supervised learning (SSL), no prior work has explored their use for pose estimation. These methods rely on low rank-based adaptation approaches, which constrain model updates to a low-rank space. We propose Endo-FASt3r, the first monocular SSL depth and pose estimation framework that uses foundation models for both tasks. We extend the Reloc3r relative pose estimation foundation model by designing Reloc3rX, introducing modifications necessary for convergence in SSL. We also present DoMoRA, a novel adaptation technique that enables higher-rank updates and faster convergence. Experiments on the SCARED dataset show that Endo-FASt3r achieves a substantial $10\%$ improvement in pose estimation and a $2\%$ improvement in depth estimation over prior work. Similar performance gains on the Hamlyn and StereoMIS datasets reinforce the generalisability of Endo-FASt3r across different datasets.  
  </ol>  
</details>  
  
### [VidBot: Learning Generalizable 3D Actions from In-the-Wild 2D Human Videos for Zero-Shot Robotic Manipulation](http://arxiv.org/abs/2503.07135)  
Hanzhi Chen, Boyang Sun, Anran Zhang, Marc Pollefeys, Stefan Leutenegger  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Future robots are envisioned as versatile systems capable of performing a variety of household tasks. The big question remains, how can we bridge the embodiment gap while minimizing physical robot learning, which fundamentally does not scale well. We argue that learning from in-the-wild human videos offers a promising solution for robotic manipulation tasks, as vast amounts of relevant data already exist on the internet. In this work, we present VidBot, a framework enabling zero-shot robotic manipulation using learned 3D affordance from in-the-wild monocular RGB-only human videos. VidBot leverages a pipeline to extract explicit representations from them, namely 3D hand trajectories from videos, combining a depth foundation model with structure-from-motion techniques to reconstruct temporally consistent, metric-scale 3D affordance representations agnostic to embodiments. We introduce a coarse-to-fine affordance learning model that first identifies coarse actions from the pixel space and then generates fine-grained interaction trajectories with a diffusion model, conditioned on coarse actions and guided by test-time constraints for context-aware interaction planning, enabling substantial generalization to novel scenes and embodiments. Extensive experiments demonstrate the efficacy of VidBot, which significantly outperforms counterparts across 13 manipulation tasks in zero-shot settings and can be seamlessly deployed across robot systems in real-world environments. VidBot paves the way for leveraging everyday human videos to make robot learning more scalable.  
  </ol>  
</details>  
**comments**: Accepted to CVPR 2025  
  
  



## Visual Localization  

### [Find your Needle: Small Object Image Retrieval via Multi-Object Attention Optimization](http://arxiv.org/abs/2503.07038)  
Mihcael Green, Matan Levy, Issar Tzachor, Dvir Samuel, Nir Darshan, Rami Ben-Ari  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We address the challenge of Small Object Image Retrieval (SoIR), where the goal is to retrieve images containing a specific small object, in a cluttered scene. The key challenge in this setting is constructing a single image descriptor, for scalable and efficient search, that effectively represents all objects in the image. In this paper, we first analyze the limitations of existing methods on this challenging task and then introduce new benchmarks to support SoIR evaluation. Next, we introduce Multi-object Attention Optimization (MaO), a novel retrieval framework which incorporates a dedicated multi-object pre-training phase. This is followed by a refinement process that leverages attention-based feature extraction with object masks, integrating them into a single unified image descriptor. Our MaO approach significantly outperforms existing retrieval methods and strong baselines, achieving notable improvements in both zero-shot and lightweight multi-object fine-tuning. We hope this work will lay the groundwork and inspire further research to enhance retrieval performance for this highly practical task.  
  </ol>  
</details>  
  
### [Zero-Shot Hashing Based on Reconstruction With Part Alignment](http://arxiv.org/abs/2503.07037)  
Yan Jiang, Zhongmiao Qi, Jianhao Li, Jiangbo Qian, Chong Wang, Yu Xin  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Hashing algorithms have been widely used in large-scale image retrieval tasks, especially for seen class data. Zero-shot hashing algorithms have been proposed to handle unseen class data. The key technique in these algorithms involves learning features from seen classes and transferring them to unseen classes, that is, aligning the feature embeddings between the seen and unseen classes. Most existing zero-shot hashing algorithms use the shared attributes between the two classes of interest to complete alignment tasks. However, the attributes are always described for a whole image, even though they represent specific parts of the image. Hence, these methods ignore the importance of aligning attributes with the corresponding image parts, which explicitly introduces noise and reduces the accuracy achieved when aligning the features of seen and unseen classes. To address this problem, we propose a new zero-shot hashing method called RAZH. We first use a clustering algorithm to group similar patches to image parts for attribute matching and then replace the image parts with the corresponding attribute vectors, gradually aligning each part with its nearest attribute. Extensive evaluation results demonstrate the superiority of the RAZH method over several state-of-the-art methods.  
  </ol>  
</details>  
  
### [Improving Visual Place Recognition with Sequence-Matching Receptiveness Prediction](http://arxiv.org/abs/2503.06840)  
Somayeh Hussaini, Tobias Fischer, Michael Milford  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In visual place recognition (VPR), filtering and sequence-based matching approaches can improve performance by integrating temporal information across image sequences, especially in challenging conditions. While these methods are commonly applied, their effects on system behavior can be unpredictable and can actually make performance worse in certain situations. In this work, we present a new supervised learning approach that learns to predict the per-frame sequence matching receptiveness (SMR) of VPR techniques, enabling the system to selectively decide when to trust the output of a sequence matching system. The approach is agnostic to the underlying VPR technique. Our approach predicts SMR-and hence significantly improves VPR performance-across a large range of state-of-the-art and classical VPR techniques (namely CosPlace, MixVPR, EigenPlaces, SALAD, AP-GeM, NetVLAD and SAD), and across three benchmark VPR datasets (Nordland, Oxford RobotCar, and SFU-Mountain). We also provide insights into a complementary approach that uses the predictor to replace discarded matches, as well as ablation studies, including an analysis of the interactions between our SMR predictor and the selected sequence length. We will release our code upon acceptance.  
  </ol>  
</details>  
**comments**: 8 pages, 5 figures, under review  
  
### [RoboDesign1M: A Large-scale Dataset for Robot Design Understanding](http://arxiv.org/abs/2503.06796)  
Tri Le, Toan Nguyen, Quang Tran, Quang Nguyen, Baoru Huang, Hoan Nguyen, Minh Nhat Vu, Tung D. Ta, Anh Nguyen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Robot design is a complex and time-consuming process that requires specialized expertise. Gaining a deeper understanding of robot design data can enable various applications, including automated design generation, retrieving example designs from text, and developing AI-powered design assistants. While recent advancements in foundation models present promising approaches to addressing these challenges, progress in this field is hindered by the lack of large-scale design datasets. In this paper, we introduce RoboDesign1M, a large-scale dataset comprising 1 million samples. Our dataset features multimodal data collected from scientific literature, covering various robotics domains. We propose a semi-automated data collection pipeline, enabling efficient and diverse data acquisition. To assess the effectiveness of RoboDesign1M, we conduct extensive experiments across multiple tasks, including design image generation, visual question answering about designs, and design image retrieval. The results demonstrate that our dataset serves as a challenging new benchmark for design understanding tasks and has the potential to advance research in this field. RoboDesign1M will be released to support further developments in AI-driven robotic design automation.  
  </ol>  
</details>  
**comments**: 8 pages  
  
### [StructVPR++: Distill Structural and Semantic Knowledge with Weighting Samples for Visual Place Recognition](http://arxiv.org/abs/2503.06601)  
Yanqing Shen, Sanping Zhou, Jingwen Fu, Ruotong Wang, Shitao Chen, Nanning Zheng  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual place recognition is a challenging task for autonomous driving and robotics, which is usually considered as an image retrieval problem. A commonly used two-stage strategy involves global retrieval followed by re-ranking using patch-level descriptors. Most deep learning-based methods in an end-to-end manner cannot extract global features with sufficient semantic information from RGB images. In contrast, re-ranking can utilize more explicit structural and semantic information in one-to-one matching process, but it is time-consuming. To bridge the gap between global retrieval and re-ranking and achieve a good trade-off between accuracy and efficiency, we propose StructVPR++, a framework that embeds structural and semantic knowledge into RGB global representations via segmentation-guided distillation. Our key innovation lies in decoupling label-specific features from global descriptors, enabling explicit semantic alignment between image pairs without requiring segmentation during deployment. Furthermore, we introduce a sample-wise weighted distillation strategy that prioritizes reliable training pairs while suppressing noisy ones. Experiments on four benchmarks demonstrate that StructVPR++ surpasses state-of-the-art global methods by 5-23% in Recall@1 and even outperforms many two-stage approaches, achieving real-time efficiency with a single RGB input.  
  </ol>  
</details>  
  
### [TextInPlace: Indoor Visual Place Recognition in Repetitive Structures with Scene Text Spotting and Verification](http://arxiv.org/abs/2503.06501)  
Huaqi Tao, Bingxi Liu, Calvin Chen, Tingjun Huang, He Li, Jinqiang Cui, Hong Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Place Recognition (VPR) is a crucial capability for long-term autonomous robots, enabling them to identify previously visited locations using visual information. However, existing methods remain limited in indoor settings due to the highly repetitive structures inherent in such environments. We observe that scene text typically appears in indoor spaces, serving to distinguish visually similar but different places. This inspires us to propose TextInPlace, a simple yet effective VPR framework that integrates Scene Text Spotting (STS) to mitigate visual perceptual ambiguity in repetitive indoor environments. Specifically, TextInPlace adopts a dual-branch architecture within a local parameter sharing network. The VPR branch employs attention-based aggregation to extract global descriptors for coarse-grained retrieval, while the STS branch utilizes a bridging text spotter to detect and recognize scene text. Finally, the discriminative text is filtered to compute text similarity and re-rank the top-K retrieved images. To bridge the gap between current text-based repetitive indoor scene datasets and the typical scenarios encountered in robot navigation, we establish an indoor VPR benchmark dataset, called Maze-with-Text. Extensive experiments on both custom and public datasets demonstrate that TextInPlace achieves superior performance over existing methods that rely solely on appearance information. The dataset, code, and trained models are publicly available at https://github.com/HqiTao/TextInPlace.  
  </ol>  
</details>  
**comments**: 8 pages,5 figures  
  
### [NeuraLoc: Visual Localization in Neural Implicit Map with Dual Complementary Features](http://arxiv.org/abs/2503.06117)  
Hongjia Zhai, Boming Zhao, Hai Li, Xiaokun Pan, Yijia He, Zhaopeng Cui, Hujun Bao, Guofeng Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recently, neural radiance fields (NeRF) have gained significant attention in the field of visual localization. However, existing NeRF-based approaches either lack geometric constraints or require extensive storage for feature matching, limiting their practical applications. To address these challenges, we propose an efficient and novel visual localization approach based on the neural implicit map with complementary features. Specifically, to enforce geometric constraints and reduce storage requirements, we implicitly learn a 3D keypoint descriptor field, avoiding the need to explicitly store point-wise features. To further address the semantic ambiguity of descriptors, we introduce additional semantic contextual feature fields, which enhance the quality and reliability of 2D-3D correspondences. Besides, we propose descriptor similarity distribution alignment to minimize the domain gap between 2D and 3D feature spaces during matching. Finally, we construct the matching graph using both complementary descriptors and contextual features to establish accurate 2D-3D correspondences for 6-DoF pose estimation. Compared with the recent NeRF-based approaches, our method achieves a 3 $\times$ faster training speed and a 45$\times$ reduction in model storage. Extensive experiments on two widely used datasets demonstrate that our approach outperforms or is highly competitive with other state-of-the-art NeRF-based visual localization methods. Project page: \href{https://zju3dv.github.io/neuraloc}{https://zju3dv.github.io/neuraloc}  
  </ol>  
</details>  
**comments**: ICRA 2025  
  
  



## Keypoint Detection  

### [REF-VLM: Triplet-Based Referring Paradigm for Unified Visual Decoding](http://arxiv.org/abs/2503.07413)  
[[code](https://github.com/MacavityT/REF-VLM)]  
Yan Tai, Luhao Zhu, Zhiqiang Chen, Ynan Ding, Yiying Dong, Xiaohong Liu, Guodong Guo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Multimodal Large Language Models (MLLMs) demonstrate robust zero-shot capabilities across diverse vision-language tasks after training on mega-scale datasets. However, dense prediction tasks, such as semantic segmentation and keypoint detection, pose significant challenges for MLLMs when represented solely as text outputs. Simultaneously, current MLLMs utilizing latent embeddings for visual task decoding generally demonstrate limited adaptability to both multi-task learning and multi-granularity scenarios. In this work, we present REF-VLM, an end-to-end framework for unified training of various visual decoding tasks. To address complex visual decoding scenarios, we introduce the Triplet-Based Referring Paradigm (TRP), which explicitly decouples three critical dimensions in visual decoding tasks through a triplet structure: concepts, decoding types, and targets. TRP employs symbolic delimiters to enforce structured representation learning, enhancing the parsability and interpretability of model outputs. Additionally, we construct Visual-Task Instruction Following Dataset (VTInstruct), a large-scale multi-task dataset containing over 100 million multimodal dialogue samples across 25 task types. Beyond text inputs and outputs, VT-Instruct incorporates various visual prompts such as point, box, scribble, and mask, and generates outputs composed of text and visual units like box, keypoint, depth and mask. The combination of different visual prompts and visual units generates a wide variety of task types, expanding the applicability of REF-VLM significantly. Both qualitative and quantitative experiments demonstrate that our REF-VLM outperforms other MLLMs across a variety of standard benchmarks. The code, dataset, and demo available at https://github.com/MacavityT/REF-VLM.  
  </ol>  
</details>  
  
### [DaD: Distilled Reinforcement Learning for Diverse Keypoint Detection](http://arxiv.org/abs/2503.07347)  
Johan Edstedt, Georg Bökman, Mårten Wadenbäck, Michael Felsberg  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Keypoints are what enable Structure-from-Motion (SfM) systems to scale to thousands of images. However, designing a keypoint detection objective is a non-trivial task, as SfM is non-differentiable. Typically, an auxiliary objective involving a descriptor is optimized. This however induces a dependency on the descriptor, which is undesirable. In this paper we propose a fully self-supervised and descriptor-free objective for keypoint detection, through reinforcement learning. To ensure training does not degenerate, we leverage a balanced top-K sampling strategy. While this already produces competitive models, we find that two qualitatively different types of detectors emerge, which are only able to detect light and dark keypoints respectively. To remedy this, we train a third detector, DaD, that optimizes the Kullback-Leibler divergence of the pointwise maximum of both light and dark detectors. Our approach significantly improve upon SotA across a range of benchmarks. Code and model weights are publicly available at https:github.com/parskatt/dad  
  </ol>  
</details>  
  
  



## NeRF  

### [CATPlan: Loss-based Collision Prediction in End-to-End Autonomous Driving](http://arxiv.org/abs/2503.07425)  
Ziliang Xiong, Shipeng Liu, Nathaniel Helgesen, Joakim Johnander, Per-Erik Forssen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In recent years, there has been increased interest in the design, training, and evaluation of end-to-end autonomous driving (AD) systems. One often overlooked aspect is the uncertainty of planned trajectories predicted by these systems, despite awareness of their own uncertainty being key to achieve safety and robustness. We propose to estimate this uncertainty by adapting loss prediction from the uncertainty quantification literature. To this end, we introduce a novel light-weight module, dubbed CATPlan, that is trained to decode motion and planning embeddings into estimates of the collision loss used to partially supervise end-to-end AD systems. During inference, these estimates are interpreted as collision risk. We evaluate CATPlan on the safety-critical, nerf-based, closed-loop benchmark NeuroNCAP and find that it manages to detect collisions with a $54.8\%$ relative improvement to average precision over a GMM-based baseline in which the predicted trajectory is compared to the forecasted trajectories of other road users. Our findings indicate that the addition of CATPlan can lead to safer end-to-end AD systems and hope that our work will spark increased interest in uncertainty quantification for such systems.  
  </ol>  
</details>  
  
### [Feature-EndoGaussian: Feature Distilled Gaussian Splatting in Surgical Deformable Scene Reconstruction](http://arxiv.org/abs/2503.06161)  
Kai Li, Junhao Wang, William Han, Ding Zhao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Minimally invasive surgery (MIS) has transformed clinical practice by reducing recovery times, minimizing complications, and enhancing precision. Nonetheless, MIS inherently relies on indirect visualization and precise instrument control, posing unique challenges. Recent advances in artificial intelligence have enabled real-time surgical scene understanding through techniques such as image classification, object detection, and segmentation, with scene reconstruction emerging as a key element for enhanced intraoperative guidance. Although neural radiance fields (NeRFs) have been explored for this purpose, their substantial data requirements and slow rendering inhibit real-time performance. In contrast, 3D Gaussian Splatting (3DGS) offers a more efficient alternative, achieving state-of-the-art performance in dynamic surgical scene reconstruction. In this work, we introduce Feature-EndoGaussian (FEG), an extension of 3DGS that integrates 2D segmentation cues into 3D rendering to enable real-time semantic and scene reconstruction. By leveraging pretrained segmentation foundation models, FEG incorporates semantic feature distillation within the Gaussian deformation framework, thereby enhancing both reconstruction fidelity and segmentation accuracy. On the EndoNeRF dataset, FEG achieves superior performance (SSIM of 0.97, PSNR of 39.08, and LPIPS of 0.03) compared to leading methods. Additionally, on the EndoVis18 dataset, FEG demonstrates competitive class-wise segmentation metrics while balancing model size and real-time performance.  
  </ol>  
</details>  
**comments**: 14 pages, 5 figures  
  
### [SecureGS: Boosting the Security and Fidelity of 3D Gaussian Splatting Steganography](http://arxiv.org/abs/2503.06118)  
Xuanyu Zhang, Jiarui Meng, Zhipei Xu, Shuzhou Yang, Yanmin Wu, Ronggang Wang, Jian Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D Gaussian Splatting (3DGS) has emerged as a premier method for 3D representation due to its real-time rendering and high-quality outputs, underscoring the critical need to protect the privacy of 3D assets. Traditional NeRF steganography methods fail to address the explicit nature of 3DGS since its point cloud files are publicly accessible. Existing GS steganography solutions mitigate some issues but still struggle with reduced rendering fidelity, increased computational demands, and security flaws, especially in the security of the geometric structure of the visualized point cloud. To address these demands, we propose a SecureGS, a secure and efficient 3DGS steganography framework inspired by Scaffold-GS's anchor point design and neural decoding. SecureGS uses a hybrid decoupled Gaussian encryption mechanism to embed offsets, scales, rotations, and RGB attributes of the hidden 3D Gaussian points in anchor point features, retrievable only by authorized users through privacy-preserving neural networks. To further enhance security, we propose a density region-aware anchor growing and pruning strategy that adaptively locates optimal hiding regions without exposing hidden information. Extensive experiments show that SecureGS significantly surpasses existing GS steganography methods in rendering fidelity, speed, and security.  
  </ol>  
</details>  
**comments**: Accepted by ICLR 2025  
  
### [NeuraLoc: Visual Localization in Neural Implicit Map with Dual Complementary Features](http://arxiv.org/abs/2503.06117)  
Hongjia Zhai, Boming Zhao, Hai Li, Xiaokun Pan, Yijia He, Zhaopeng Cui, Hujun Bao, Guofeng Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recently, neural radiance fields (NeRF) have gained significant attention in the field of visual localization. However, existing NeRF-based approaches either lack geometric constraints or require extensive storage for feature matching, limiting their practical applications. To address these challenges, we propose an efficient and novel visual localization approach based on the neural implicit map with complementary features. Specifically, to enforce geometric constraints and reduce storage requirements, we implicitly learn a 3D keypoint descriptor field, avoiding the need to explicitly store point-wise features. To further address the semantic ambiguity of descriptors, we introduce additional semantic contextual feature fields, which enhance the quality and reliability of 2D-3D correspondences. Besides, we propose descriptor similarity distribution alignment to minimize the domain gap between 2D and 3D feature spaces during matching. Finally, we construct the matching graph using both complementary descriptors and contextual features to establish accurate 2D-3D correspondences for 6-DoF pose estimation. Compared with the recent NeRF-based approaches, our method achieves a 3 $\times$ faster training speed and a 45$\times$ reduction in model storage. Extensive experiments on two widely used datasets demonstrate that our approach outperforms or is highly competitive with other state-of-the-art NeRF-based visual localization methods. Project page: \href{https://zju3dv.github.io/neuraloc}{https://zju3dv.github.io/neuraloc}  
  </ol>  
</details>  
**comments**: ICRA 2025  
  
  



