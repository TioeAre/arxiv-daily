<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Orchestrator-Agent-Trust:-A-Modular-Agentic-AI-Visual-Classification-System-with-Trust-Aware-Orchestration-and-RAG-Based-Reasoning>Orchestrator-Agent Trust: A Modular Agentic AI Visual Classification System with Trust-Aware Orchestration and RAG-Based Reasoning</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#KptLLM++:-Towards-Generic-Keypoint-Comprehension-with-Large-Language-Model>KptLLM++: Towards Generic Keypoint Comprehension with Large Language Model</a></li>
        <li><a href=#GKNet:-Graph-based-Keypoints-Network-for-Monocular-Pose-Estimation-of-Non-cooperative-Spacecraft>GKNet: Graph-based Keypoints Network for Monocular Pose Estimation of Non-cooperative Spacecraft</a></li>
        <li><a href=#FPC-Net:-Revisiting-SuperPoint-with-Descriptor-Free-Keypoint-Detection-via-Feature-Pyramids-and-Consistency-Based-Implicit-Matching>FPC-Net: Revisiting SuperPoint with Descriptor-Free Keypoint Detection via Feature Pyramids and Consistency-Based Implicit Matching</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Orchestrator-Agent Trust: A Modular Agentic AI Visual Classification System with Trust-Aware Orchestration and RAG-Based Reasoning](http://arxiv.org/abs/2507.10571)  
Konstantinos I. Roumeliotis, Ranjan Sapkota, Manoj Karkee, Nikolaos D. Tselikas  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Modern Artificial Intelligence (AI) increasingly relies on multi-agent architectures that blend visual and language understanding. Yet, a pressing challenge remains: How can we trust these agents especially in zero-shot settings with no fine-tuning? We introduce a novel modular Agentic AI visual classification framework that integrates generalist multimodal agents with a non-visual reasoning orchestrator and a Retrieval-Augmented Generation (RAG) module. Applied to apple leaf disease diagnosis, we benchmark three configurations: (I) zero-shot with confidence-based orchestration, (II) fine-tuned agents with improved performance, and (III) trust-calibrated orchestration enhanced by CLIP-based image retrieval and re-evaluation loops. Using confidence calibration metrics (ECE, OCR, CCC), the orchestrator modulates trust across agents. Our results demonstrate a 77.94\% accuracy improvement in the zero-shot setting using trust-aware orchestration and RAG, achieving 85.63\% overall. GPT-4o showed better calibration, while Qwen-2.5-VL displayed overconfidence. Furthermore, image-RAG grounded predictions with visually similar cases, enabling correction of agent overconfidence via iterative re-evaluation. The proposed system separates perception (vision agents) from meta-reasoning (orchestrator), enabling scalable and interpretable multi-agent AI. This blueprint is extensible to diagnostics, biology, and other trust-critical domains. All models, prompts, results, and system components including the complete software source code are openly released to support reproducibility, transparency, and community benchmarking at Github: https://github.com/Applied-AI-Research-Lab/Orchestrator-Agent-Trust  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [KptLLM++: Towards Generic Keypoint Comprehension with Large Language Model](http://arxiv.org/abs/2507.11102)  
Jie Yang, Wang Zeng, Sheng Jin, Lumin Xu, Wentao Liu, Chen Qian, Zhen Li, Ruimao Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The emergence of Multimodal Large Language Models (MLLMs) has revolutionized image understanding by bridging textual and visual modalities. However, these models often struggle with capturing fine-grained semantic information, such as the precise identification and analysis of object keypoints. Keypoints, as structure-aware, pixel-level, and compact representations of objects, particularly articulated ones, play a crucial role in applications such as fine-grained image analysis, object retrieval, and behavior recognition. In this paper, we propose KptLLM++, a novel multimodal large language model that specifically designed for generic keypoint comprehension through the integration of diverse input modalities guided by user-defined instructions. By unifying keypoint detection across varied contexts, KptLLM++ establishes itself as an advanced interface, fostering more effective human-AI collaboration. The model is built upon a novel identify-then-detect paradigm, which first interprets keypoint semantics and subsequently localizes their precise positions through a structured chain-of-thought reasoning mechanism. To push the boundaries of performance, we have scaled up the training dataset to over 500K samples, encompassing diverse objects, keypoint categories, image styles, and scenarios with complex occlusions. This extensive scaling enables KptLLM++ to unlock its potential, achieving remarkable accuracy and generalization. Comprehensive experiments on multiple keypoint detection benchmarks demonstrate its state-of-the-art performance, underscoring its potential as a unified solution for fine-grained image understanding and its transformative implications for human-AI interaction.  
  </ol>  
</details>  
**comments**: Extended Version of KptLLM. arXiv admin note: text overlap with
  arXiv:2411.01846  
  
### [GKNet: Graph-based Keypoints Network for Monocular Pose Estimation of Non-cooperative Spacecraft](http://arxiv.org/abs/2507.11077)  
Weizhao Ma, Dong Zhou, Yuhui Hu, Zipeng He  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Monocular pose estimation of non-cooperative spacecraft is significant for on-orbit service (OOS) tasks, such as satellite maintenance, space debris removal, and station assembly. Considering the high demands on pose estimation accuracy, mainstream monocular pose estimation methods typically consist of keypoint detectors and PnP solver. However, current keypoint detectors remain vulnerable to structural symmetry and partial occlusion of non-cooperative spacecraft. To this end, we propose a graph-based keypoints network for the monocular pose estimation of non-cooperative spacecraft, GKNet, which leverages the geometric constraint of keypoints graph. In order to better validate keypoint detectors, we present a moderate-scale dataset for the spacecraft keypoint detection, named SKD, which consists of 3 spacecraft targets, 90,000 simulated images, and corresponding high-precise keypoint annotations. Extensive experiments and an ablation study have demonstrated the high accuracy and effectiveness of our GKNet, compared to the state-of-the-art spacecraft keypoint detectors. The code for GKNet and the SKD dataset is available at https://github.com/Dongzhou-1996/GKNet.  
  </ol>  
</details>  
  
### [FPC-Net: Revisiting SuperPoint with Descriptor-Free Keypoint Detection via Feature Pyramids and Consistency-Based Implicit Matching](http://arxiv.org/abs/2507.10770)  
Ionuţ Grigore, Călin-Adrian Popa, Claudiu Leoveanu-Condrei  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The extraction and matching of interest points are fundamental to many geometric computer vision tasks. Traditionally, matching is performed by assigning descriptors to interest points and identifying correspondences based on descriptor similarity. This work introduces a technique where interest points are inherently associated during detection, eliminating the need for computing, storing, transmitting, or matching descriptors. Although the matching accuracy is marginally lower than that of conventional approaches, our method completely eliminates the need for descriptors, leading to a drastic reduction in memory usage for localization systems. We assess its effectiveness by comparing it against both classical handcrafted methods and modern learned approaches.  
  </ol>  
</details>  
  
  



