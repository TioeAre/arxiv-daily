<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Tracking-the-Flight:-Exploring-a-Computational-Framework-for-Analyzing-Escape-Responses-in-Plains-Zebra-(Equus-quagga)>Tracking the Flight: Exploring a Computational Framework for Analyzing Escape Responses in Plains Zebra (Equus quagga)</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#TAT-VPR:-Ternary-Adaptive-Transformer-for-Dynamic-and-Efficient-Visual-Place-Recognition>TAT-VPR: Ternary Adaptive Transformer for Dynamic and Efficient Visual Place Recognition</a></li>
        <li><a href=#Highlighting-What-Matters:-Promptable-Embeddings-for-Attribute-Focused-Image-Retrieval>Highlighting What Matters: Promptable Embeddings for Attribute-Focused Image Retrieval</a></li>
        <li><a href=#SCENIR:-Visual-Semantic-Clarity-through-Unsupervised-Scene-Graph-Retrieval>SCENIR: Visual Semantic Clarity through Unsupervised Scene Graph Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#UAV-See,-UGV-Do:-Aerial-Imagery-and-Virtual-Teach-Enabling-Zero-Shot-Ground-Vehicle-Repeat>UAV See, UGV Do: Aerial Imagery and Virtual Teach Enabling Zero-Shot Ground Vehicle Repeat</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Tracking the Flight: Exploring a Computational Framework for Analyzing Escape Responses in Plains Zebra (Equus quagga)](http://arxiv.org/abs/2505.16882)  
Isla Duporge, Sofia Minano, Nikoloz Sirmpilatze, Igor Tatarnikov, Scott Wolf, Adam L. Tyson, Daniel Rubenstein  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Ethological research increasingly benefits from the growing affordability and accessibility of drones, which enable the capture of high-resolution footage of animal movement at fine spatial and temporal scales. However, analyzing such footage presents the technical challenge of separating animal movement from drone motion. While non-trivial, computer vision techniques such as image registration and Structure-from-Motion (SfM) offer practical solutions. For conservationists, open-source tools that are user-friendly, require minimal setup, and deliver timely results are especially valuable for efficient data interpretation. This study evaluates three approaches: a bioimaging-based registration technique, an SfM pipeline, and a hybrid interpolation method. We apply these to a recorded escape event involving 44 plains zebras, captured in a single drone video. Using the best-performing method, we extract individual trajectories and identify key behavioral patterns: increased alignment (polarization) during escape, a brief widening of spacing just before stopping, and tighter coordination near the group's center. These insights highlight the method's effectiveness and its potential to scale to larger datasets, contributing to broader investigations of collective animal behavior.  
  </ol>  
</details>  
**comments**: Accepted to the CV4Animals workshop at CVPR 2025  
  
  



## Visual Localization  

### [TAT-VPR: Ternary Adaptive Transformer for Dynamic and Efficient Visual Place Recognition](http://arxiv.org/abs/2505.16447)  
Oliver Grainge, Michael Milford, Indu Bodala, Sarvapali D. Ramchurn, Shoaib Ehsan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    TAT-VPR is a ternary-quantized transformer that brings dynamic accuracy-efficiency trade-offs to visual SLAM loop-closure. By fusing ternary weights with a learned activation-sparsity gate, the model can control computation by up to 40% at run-time without degrading performance (Recall@1). The proposed two-stage distillation pipeline preserves descriptor quality, letting it run on micro-UAV and embedded SLAM stacks while matching state-of-the-art localization accuracy.  
  </ol>  
</details>  
  
### [Highlighting What Matters: Promptable Embeddings for Attribute-Focused Image Retrieval](http://arxiv.org/abs/2505.15877)  
Siting Li, Xiang Gao, Simon Shaolei Du  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    While an image is worth more than a thousand words, only a few provide crucial information for a given task and thus should be focused on. In light of this, ideal text-to-image (T2I) retrievers should prioritize specific visual attributes relevant to queries. To evaluate current retrievers on handling attribute-focused queries, we build COCO-Facet, a COCO-based benchmark with 9,112 queries about diverse attributes of interest. We find that CLIP-like retrievers, which are widely adopted due to their efficiency and zero-shot ability, have poor and imbalanced performance, possibly because their image embeddings focus on global semantics and subjects while leaving out other details. Notably, we reveal that even recent Multimodal Large Language Model (MLLM)-based, stronger retrievers with a larger output dimension struggle with this limitation. Hence, we hypothesize that retrieving with general image embeddings is suboptimal for performing such queries. As a solution, we propose to use promptable image embeddings enabled by these multimodal retrievers, which boost performance by highlighting required attributes. Our pipeline for deriving such embeddings generalizes across query types, image pools, and base retriever architectures. To enhance real-world applicability, we offer two acceleration strategies: Pre-processing promptable embeddings and using linear approximations. We show that the former yields a 15% improvement in Recall@5 when prompts are predefined, while the latter achieves an 8% improvement when prompts are only available during inference.  
  </ol>  
</details>  
**comments**: 25 pages, 5 figures  
  
### [SCENIR: Visual Semantic Clarity through Unsupervised Scene Graph Retrieval](http://arxiv.org/abs/2505.15867)  
Nikolaos Chaidos, Angeliki Dimitriou, Maria Lymperaiou, Giorgos Stamou  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Despite the dominance of convolutional and transformer-based architectures in image-to-image retrieval, these models are prone to biases arising from low-level visual features, such as color. Recognizing the lack of semantic understanding as a key limitation, we propose a novel scene graph-based retrieval framework that emphasizes semantic content over superficial image characteristics. Prior approaches to scene graph retrieval predominantly rely on supervised Graph Neural Networks (GNNs), which require ground truth graph pairs driven from image captions. However, the inconsistency of caption-based supervision stemming from variable text encodings undermine retrieval reliability. To address these, we present SCENIR, a Graph Autoencoder-based unsupervised retrieval framework, which eliminates the dependence on labeled training data. Our model demonstrates superior performance across metrics and runtime efficiency, outperforming existing vision-based, multimodal, and supervised GNN approaches. We further advocate for Graph Edit Distance (GED) as a deterministic and robust ground truth measure for scene graph similarity, replacing the inconsistent caption-based alternatives for the first time in image-to-image retrieval evaluation. Finally, we validate the generalizability of our method by applying it to unannotated datasets via automated scene graph generation, while substantially contributing in advancing state-of-the-art in counterfactual image retrieval.  
  </ol>  
</details>  
  
  



## NeRF  

### [UAV See, UGV Do: Aerial Imagery and Virtual Teach Enabling Zero-Shot Ground Vehicle Repeat](http://arxiv.org/abs/2505.16912)  
Desiree Fisker, Alexander Krawciw, Sven Lilge, Melissa Greeff, Timothy D. Barfoot  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper presents Virtual Teach and Repeat (VirT&R): an extension of the Teach and Repeat (T&R) framework that enables GPS-denied, zero-shot autonomous ground vehicle navigation in untraversed environments. VirT&R leverages aerial imagery captured for a target environment to train a Neural Radiance Field (NeRF) model so that dense point clouds and photo-textured meshes can be extracted. The NeRF mesh is used to create a high-fidelity simulation of the environment for piloting an unmanned ground vehicle (UGV) to virtually define a desired path. The mission can then be executed in the actual target environment by using NeRF-derived point cloud submaps associated along the path and an existing LiDAR Teach and Repeat (LT&R) framework. We benchmark the repeatability of VirT&R on over 12 km of autonomous driving data using physical markings that allow a sim-to-real lateral path-tracking error to be obtained and compared with LT&R. VirT&R achieved measured root mean squared errors (RMSE) of 19.5 cm and 18.4 cm in two different environments, which are slightly less than one tire width (24 cm) on the robot used for testing, and respective maximum errors were 39.4 cm and 47.6 cm. This was done using only the NeRF-derived teach map, demonstrating that VirT&R has similar closed-loop path-tracking performance to LT&R but does not require a human to manually teach the path to the UGV in the actual environment.  
  </ol>  
</details>  
**comments**: 8 pages, 7 figures, submitted to IROS 2025  
  
  



