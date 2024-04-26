<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Revisiting-Relevance-Feedback-for-CLIP-based-Interactive-Image-Retrieval>Revisiting Relevance Feedback for CLIP-based Interactive Image Retrieval</a></li>
        <li><a href=#Simple-but-Effective-Raw-Data-Level-Multimodal-Fusion-for-Composed-Image-Retrieval>Simple but Effective Raw-Data Level Multimodal Fusion for Composed Image Retrieval</a></li>
        <li><a href=#DVF:-Advancing-Robust-and-Accurate-Fine-Grained-Image-Retrieval-with-Retrieval-Guidelines>DVF: Advancing Robust and Accurate Fine-Grained Image Retrieval with Retrieval Guidelines</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Transformer-Based-Local-Feature-Matching-for-Multimodal-Image-Registration>Transformer-Based Local Feature Matching for Multimodal Image Registration</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Depth-Supervised-Neural-Surface-Reconstruction-from-Airborne-Imagery>Depth Supervised Neural Surface Reconstruction from Airborne Imagery</a></li>
        <li><a href=#NeRF-XL:-Scaling-NeRFs-with-Multiple-GPUs>NeRF-XL: Scaling NeRFs with Multiple GPUs</a></li>
        <li><a href=#ESR-NeRF:-Emissive-Source-Reconstruction-Using-LDR-Multi-view-Images>ESR-NeRF: Emissive Source Reconstruction Using LDR Multi-view Images</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Revisiting Relevance Feedback for CLIP-based Interactive Image Retrieval](http://arxiv.org/abs/2404.16398)  
Ryoya Nara, Yu-Chieh Lin, Yuji Nozawa, Youyang Ng, Goh Itoh, Osamu Torii, Yusuke Matsui  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Many image retrieval studies use metric learning to train an image encoder. However, metric learning cannot handle differences in users' preferences, and requires data to train an image encoder. To overcome these limitations, we revisit relevance feedback, a classic technique for interactive retrieval systems, and propose an interactive CLIP-based image retrieval system with relevance feedback. Our retrieval system first executes the retrieval, collects each user's unique preferences through binary feedback, and returns images the user prefers. Even when users have various preferences, our retrieval system learns each user's preference through the feedback and adapts to the preference. Moreover, our retrieval system leverages CLIP's zero-shot transferability and achieves high accuracy without training. We empirically show that our retrieval system competes well with state-of-the-art metric learning in category-based image retrieval, despite not training image encoders specifically for each dataset. Furthermore, we set up two additional experimental settings where users have various preferences: one-label-based image retrieval and conditioned image retrieval. In both cases, our retrieval system effectively adapts to each user's preferences, resulting in improved accuracy compared to image retrieval without feedback. Overall, our work highlights the potential benefits of integrating CLIP with classic relevance feedback techniques to enhance image retrieval.  
  </ol>  
</details>  
**comments**: 20 pages, 8 sugures  
  
### [Simple but Effective Raw-Data Level Multimodal Fusion for Composed Image Retrieval](http://arxiv.org/abs/2404.15875)  
Haokun Wen, Xuemeng Song, Xiaolin Chen, Yinwei Wei, Liqiang Nie, Tat-Seng Chua  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed image retrieval (CIR) aims to retrieve the target image based on a multimodal query, i.e., a reference image paired with corresponding modification text. Recent CIR studies leverage vision-language pre-trained (VLP) methods as the feature extraction backbone, and perform nonlinear feature-level multimodal query fusion to retrieve the target image. Despite the promising performance, we argue that their nonlinear feature-level multimodal fusion may lead to the fused feature deviating from the original embedding space, potentially hurting the retrieval performance. To address this issue, in this work, we propose shifting the multimodal fusion from the feature level to the raw-data level to fully exploit the VLP model's multimodal encoding and cross-modal alignment abilities. In particular, we introduce a Dual Query Unification-based Composed Image Retrieval framework (DQU-CIR), whose backbone simply involves a VLP model's image encoder and a text encoder. Specifically, DQU-CIR first employs two training-free query unification components: text-oriented query unification and vision-oriented query unification, to derive a unified textual and visual query based on the raw data of the multimodal query, respectively. The unified textual query is derived by concatenating the modification text with the extracted reference image's textual description, while the unified visual query is created by writing the key modification words onto the reference image. Ultimately, to address diverse search intentions, DQU-CIR linearly combines the features of the two unified queries encoded by the VLP model to retrieve the target image. Extensive experiments on four real-world datasets validate the effectiveness of our proposed method.  
  </ol>  
</details>  
**comments**: ACM SIGIR 2024  
  
### [DVF: Advancing Robust and Accurate Fine-Grained Image Retrieval with Retrieval Guidelines](http://arxiv.org/abs/2404.15771)  
Xin Jiang, Hao Tang, Rui Yan, Jinhui Tang, Zechao Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Fine-grained image retrieval (FGIR) is to learn visual representations that distinguish visually similar objects while maintaining generalization. Existing methods propose to generate discriminative features, but rarely consider the particularity of the FGIR task itself. This paper presents a meticulous analysis leading to the proposal of practical guidelines to identify subcategory-specific discrepancies and generate discriminative features to design effective FGIR models. These guidelines include emphasizing the object (G1), highlighting subcategory-specific discrepancies (G2), and employing effective training strategy (G3). Following G1 and G2, we design a novel Dual Visual Filtering mechanism for the plain visual transformer, denoted as DVF, to capture subcategory-specific discrepancies. Specifically, the dual visual filtering mechanism comprises an object-oriented module and a semantic-oriented module. These components serve to magnify objects and identify discriminative regions, respectively. Following G3, we implement a discriminative model training strategy to improve the discriminability and generalization ability of DVF. Extensive analysis and ablation studies confirm the efficacy of our proposed guidelines. Without bells and whistles, the proposed DVF achieves state-of-the-art performance on three widely-used fine-grained datasets in closed-set and open-set settings.  
  </ol>  
</details>  
  
  



## Image Matching  

### [Transformer-Based Local Feature Matching for Multimodal Image Registration](http://arxiv.org/abs/2404.16802)  
Remi Delaunay, Ruisi Zhang, Filipe C. Pedrosa, Navid Feizi, Dianne Sacco, Rajni Patel, Jayender Jagadeesan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Ultrasound imaging is a cost-effective and radiation-free modality for visualizing anatomical structures in real-time, making it ideal for guiding surgical interventions. However, its limited field-of-view, speckle noise, and imaging artifacts make it difficult to interpret the images for inexperienced users. In this paper, we propose a new 2D ultrasound to 3D CT registration method to improve surgical guidance during ultrasound-guided interventions. Our approach adopts a dense feature matching method called LoFTR to our multimodal registration problem. We learn to predict dense coarse-to-fine correspondences using a Transformer-based architecture to estimate a robust rigid transformation between a 2D ultrasound frame and a CT scan. Additionally, a fully differentiable pose estimation method is introduced, optimizing LoFTR on pose estimation error during training. Experiments conducted on a multimodal dataset of ex vivo porcine kidneys demonstrate the method's promising results for intraoperative, trackerless ultrasound pose estimation. By mapping 2D ultrasound frames into the 3D CT volume space, the method provides intraoperative guidance, potentially improving surgical workflows and image interpretation.  
  </ol>  
</details>  
**comments**: Accepted to SPIE Medical Imaging 2024  
  
  



## NeRF  

### [Depth Supervised Neural Surface Reconstruction from Airborne Imagery](http://arxiv.org/abs/2404.16429)  
Vincent Hackstein, Paul Fauth-Mayer, Matthias Rothermel, Norbert Haala  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    While originally developed for novel view synthesis, Neural Radiance Fields (NeRFs) have recently emerged as an alternative to multi-view stereo (MVS). Triggered by a manifold of research activities, promising results have been gained especially for texture-less, transparent, and reflecting surfaces, while such scenarios remain challenging for traditional MVS-based approaches. However, most of these investigations focus on close-range scenarios, with studies for airborne scenarios still missing. For this task, NeRFs face potential difficulties at areas of low image redundancy and weak data evidence, as often found in street canyons, facades or building shadows. Furthermore, training such networks is computationally expensive. Thus, the aim of our work is twofold: First, we investigate the applicability of NeRFs for aerial image blocks representing different characteristics like nadir-only, oblique and high-resolution imagery. Second, during these investigations we demonstrate the benefit of integrating depth priors from tie-point measures, which are provided during presupposed Bundle Block Adjustment. Our work is based on the state-of-the-art framework VolSDF, which models 3D scenes by signed distance functions (SDFs), since this is more applicable for surface reconstruction compared to the standard volumetric representation in vanilla NeRFs. For evaluation, the NeRF-based reconstructions are compared to results of a publicly available benchmark dataset for airborne images.  
  </ol>  
</details>  
  
### [NeRF-XL: Scaling NeRFs with Multiple GPUs](http://arxiv.org/abs/2404.16221)  
Ruilong Li, Sanja Fidler, Angjoo Kanazawa, Francis Williams  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present NeRF-XL, a principled method for distributing Neural Radiance Fields (NeRFs) across multiple GPUs, thus enabling the training and rendering of NeRFs with an arbitrarily large capacity. We begin by revisiting existing multi-GPU approaches, which decompose large scenes into multiple independently trained NeRFs, and identify several fundamental issues with these methods that hinder improvements in reconstruction quality as additional computational resources (GPUs) are used in training. NeRF-XL remedies these issues and enables the training and rendering of NeRFs with an arbitrary number of parameters by simply using more hardware. At the core of our method lies a novel distributed training and rendering formulation, which is mathematically equivalent to the classic single-GPU case and minimizes communication between GPUs. By unlocking NeRFs with arbitrarily large parameter counts, our approach is the first to reveal multi-GPU scaling laws for NeRFs, showing improvements in reconstruction quality with larger parameter counts and speed improvements with more GPUs. We demonstrate the effectiveness of NeRF-XL on a wide variety of datasets, including the largest open-source dataset to date, MatrixCity, containing 258K images covering a 25km^2 city area.  
  </ol>  
</details>  
**comments**: Webpage: https://research.nvidia.com/labs/toronto-ai/nerfxl/  
  
### [ESR-NeRF: Emissive Source Reconstruction Using LDR Multi-view Images](http://arxiv.org/abs/2404.15707)  
Jinseo Jeong, Junseo Koo, Qimeng Zhang, Gunhee Kim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Existing NeRF-based inverse rendering methods suppose that scenes are exclusively illuminated by distant light sources, neglecting the potential influence of emissive sources within a scene. In this work, we confront this limitation using LDR multi-view images captured with emissive sources turned on and off. Two key issues must be addressed: 1) ambiguity arising from the limited dynamic range along with unknown lighting details, and 2) the expensive computational cost in volume rendering to backtrace the paths leading to final object colors. We present a novel approach, ESR-NeRF, leveraging neural networks as learnable functions to represent ray-traced fields. By training networks to satisfy light transport segments, we regulate outgoing radiances, progressively identifying emissive sources while being aware of reflection areas. The results on scenes encompassing emissive sources with various properties demonstrate the superiority of ESR-NeRF in qualitative and quantitative ways. Our approach also extends its applicability to the scenes devoid of emissive sources, achieving lower CD metrics on the DTU dataset.  
  </ol>  
</details>  
**comments**: CVPR 2024  
  
  



