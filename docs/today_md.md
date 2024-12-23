<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#A-New-Method-to-Capturing-Compositional-Knowledge-in-Linguistic-Space>A New Method to Capturing Compositional Knowledge in Linguistic Space</a></li>
        <li><a href=#Stabilizing-Laplacian-Inversion-in-Fokker-Planck-Image-Retrieval-using-the-Transport-of-Intensity-Equation>Stabilizing Laplacian Inversion in Fokker-Planck Image Retrieval using the Transport-of-Intensity Equation</a></li>
        <li><a href=#Learning-Visual-Composition-through-Improved-Semantic-Guidance>Learning Visual Composition through Improved Semantic Guidance</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#NeRF-To-Real-Tester:-Neural-Radiance-Fields-as-Test-Image-Generators-for-Vision-of-Autonomous-Systems>NeRF-To-Real Tester: Neural Radiance Fields as Test Image Generators for Vision of Autonomous Systems</a></li>
        <li><a href=#NeuroPump:-Simultaneous-Geometric-and-Color-Rectification-for-Underwater-Images>NeuroPump: Simultaneous Geometric and Color Rectification for Underwater Images</a></li>
        <li><a href=#LiHi-GS:-LiDAR-Supervised-Gaussian-Splatting-for-Highway-Driving-Scene-Reconstruction>LiHi-GS: LiDAR-Supervised Gaussian Splatting for Highway Driving Scene Reconstruction</a></li>
        <li><a href=#DreaMark:-Rooting-Watermark-in-Score-Distillation-Sampling-Generated-Neural-Radiance-Fields>DreaMark: Rooting Watermark in Score Distillation Sampling Generated Neural Radiance Fields</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [A New Method to Capturing Compositional Knowledge in Linguistic Space](http://arxiv.org/abs/2412.15632)  
Jiahe Wan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Compositional understanding allows visual language models to interpret complex relationships between objects, attributes, and relations in images and text. However, most existing methods often rely on hard negative examples and fine-tuning, which can overestimate improvements and are limited by the difficulty of obtaining hard negatives. In this work, we introduce Zero-Shot Compositional Understanding (ZS-CU), a novel task that enhances compositional understanding without requiring hard negative training data. We propose YUKINO (Yielded Compositional Understanding Knowledge via Textual Inversion with NO), which uses textual inversion to map unlabeled images to pseudo-tokens in a pre-trained CLIP model. We propose introducing "no" logical regularization to address the issue of token interaction in inversion. Additionally, we suggest using knowledge distillation to reduce the time complexity of textual inversion. Experimental results show that YUKINO outperforms the existing multi-modal SOTA models by over 8% on the SugarCREPE benchmark, and also achieves significant improvements in image retrieval tasks.  
  </ol>  
</details>  
  
### [Stabilizing Laplacian Inversion in Fokker-Planck Image Retrieval using the Transport-of-Intensity Equation](http://arxiv.org/abs/2412.15513)  
Samantha J Alloo, Kaye S Morgan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    X-ray attenuation, phase, and dark-field images provide complementary information. Different experimental techniques can capture these contrast mechanisms, and the corresponding images can be retrieved using various theoretical algorithms. Our previous works developed the Multimodal Intrinsic Speckle-Tracking (MIST) algorithm, which is suitable for multimodal image retrieval from speckle-based X-ray imaging (SBXI) data. MIST is based on the X-ray Fokker-Planck equation, requiring the inversion of derivative operators that are often numerically unstable. These instabilities can be addressed by employing regularization techniques, such as Tikhonov regularization. The regularization output is highly sensitive to the choice of the Tikhonov regularization parameter, making it crucial to select this value carefully and optimally. Here, we present an automated iterative algorithm to optimize the regularization of the inverse Laplacian operator in our most recently published MIST variant, addressing the operator's instability near the Fourier-space origin. Our algorithm leverages the inherent stability of the phase solution obtained from the transport-of-intensity equation for SBXI, using it as a reliable ground truth for the more complex Fokker-Planck-based algorithms that incorporate the dark-field signal. We applied the algorithm to an SBXI dataset collected using synchrotron light of a four-rod sample. The four-rod sample's phase and dark-field images were optimally retrieved using our developed algorithm, eliminating the tedious and subjective task of selecting a suitable Tikhonov regularization parameter. The developed regularization-optimization algorithm makes MIST more user-friendly by eliminating the need for manual parameter selection. We anticipate that our optimization algorithm can also be applied to other image retrieval approaches derived from the Fokker-Planck equation.  
  </ol>  
</details>  
  
### [Learning Visual Composition through Improved Semantic Guidance](http://arxiv.org/abs/2412.15396)  
Austin Stone, Hagen Soltau, Robert Geirhos, Xi Yi, Ye Xia, Bingyi Cao, Kaifeng Chen, Abhijit Ogale, Jonathon Shlens  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual imagery does not consist of solitary objects, but instead reflects the composition of a multitude of fluid concepts. While there have been great advances in visual representation learning, such advances have focused on building better representations for a small number of discrete objects bereft of an understanding of how these objects are interacting. One can observe this limitation in representations learned through captions or contrastive learning -- where the learned model treats an image essentially as a bag of words. Several works have attempted to address this limitation through the development of bespoke learned architectures to directly address the shortcomings in compositional learning. In this work, we focus on simple, and scalable approaches. In particular, we demonstrate that by substantially improving weakly labeled data, i.e. captions, we can vastly improve the performance of standard contrastive learning approaches. Previous CLIP models achieved near chance rate on challenging tasks probing compositional learning. However, our simple approach boosts performance of CLIP substantially and surpasses all bespoke architectures. Furthermore, we showcase our results on a relatively new captioning benchmark derived from DOCCI. We demonstrate through a series of ablations that a standard CLIP model trained with enhanced data may demonstrate impressive performance on image retrieval tasks.  
  </ol>  
</details>  
  
  



## NeRF  

### [NeRF-To-Real Tester: Neural Radiance Fields as Test Image Generators for Vision of Autonomous Systems](http://arxiv.org/abs/2412.16141)  
Laura Weihl, Bilal Wehbe, Andrzej Wąsowski  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Autonomous inspection of infrastructure on land and in water is a quickly growing market, with applications including surveying constructions, monitoring plants, and tracking environmental changes in on- and off-shore wind energy farms. For Autonomous Underwater Vehicles and Unmanned Aerial Vehicles overfitting of controllers to simulation conditions fundamentally leads to poor performance in the operation environment. There is a pressing need for more diverse and realistic test data that accurately represents the challenges faced by these systems. We address the challenge of generating perception test data for autonomous systems by leveraging Neural Radiance Fields to generate realistic and diverse test images, and integrating them into a metamorphic testing framework for vision components such as vSLAM and object detection. Our tool, N2R-Tester, allows training models of custom scenes and rendering test images from perturbed positions. An experimental evaluation of N2R-Tester on eight different vision components in AUVs and UAVs demonstrates the efficacy and versatility of the approach.  
  </ol>  
</details>  
  
### [NeuroPump: Simultaneous Geometric and Color Rectification for Underwater Images](http://arxiv.org/abs/2412.15890)  
Yue Guo, Haoxiang Liao, Haibin Ling, Bingyao Huang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Underwater image restoration aims to remove geometric and color distortions due to water refraction, absorption and scattering. Previous studies focus on restoring either color or the geometry, but to our best knowledge, not both. However, in practice it may be cumbersome to address the two rectifications one-by-one. In this paper, we propose NeuroPump, a self-supervised method to simultaneously optimize and rectify underwater geometry and color as if water were pumped out. The key idea is to explicitly model refraction, absorption and scattering in Neural Radiance Field (NeRF) pipeline, such that it not only performs simultaneous geometric and color rectification, but also enables to synthesize novel views and optical effects by controlling the decoupled parameters. In addition, to address issue of lack of real paired ground truth images, we propose an underwater 360 benchmark dataset that has real paired (i.e., with and without water) images. Our method clearly outperforms other baselines both quantitatively and qualitatively.  
  </ol>  
</details>  
  
### [LiHi-GS: LiDAR-Supervised Gaussian Splatting for Highway Driving Scene Reconstruction](http://arxiv.org/abs/2412.15447)  
Pou-Chun Kung, Xianling Zhang, Katherine A. Skinner, Nikita Jaipuria  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Photorealistic 3D scene reconstruction plays an important role in autonomous driving, enabling the generation of novel data from existing datasets to simulate safety-critical scenarios and expand training data without additional acquisition costs. Gaussian Splatting (GS) facilitates real-time, photorealistic rendering with an explicit 3D Gaussian representation of the scene, providing faster processing and more intuitive scene editing than the implicit Neural Radiance Fields (NeRFs). While extensive GS research has yielded promising advancements in autonomous driving applications, they overlook two critical aspects: First, existing methods mainly focus on low-speed and feature-rich urban scenes and ignore the fact that highway scenarios play a significant role in autonomous driving. Second, while LiDARs are commonplace in autonomous driving platforms, existing methods learn primarily from images and use LiDAR only for initial estimates or without precise sensor modeling, thus missing out on leveraging the rich depth information LiDAR offers and limiting the ability to synthesize LiDAR data. In this paper, we propose a novel GS method for dynamic scene synthesis and editing with improved scene reconstruction through LiDAR supervision and support for LiDAR rendering. Unlike prior works that are tested mostly on urban datasets, to the best of our knowledge, we are the first to focus on the more challenging and highly relevant highway scenes for autonomous driving, with sparse sensor views and monotone backgrounds.  
  </ol>  
</details>  
  
### [DreaMark: Rooting Watermark in Score Distillation Sampling Generated Neural Radiance Fields](http://arxiv.org/abs/2412.15278)  
Xingyu Zhu, Xiapu Luo, Xuetao Wei  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in text-to-3D generation can generate neural radiance fields (NeRFs) with score distillation sampling, enabling 3D asset creation without real-world data capture. With the rapid advancement in NeRF generation quality, protecting the copyright of the generated NeRF has become increasingly important. While prior works can watermark NeRFs in a post-generation way, they suffer from two vulnerabilities. First, a delay lies between NeRF generation and watermarking because the secret message is embedded into the NeRF model post-generation through fine-tuning. Second, generating a non-watermarked NeRF as an intermediate creates a potential vulnerability for theft. To address both issues, we propose Dreamark to embed a secret message by backdooring the NeRF during NeRF generation. In detail, we first pre-train a watermark decoder. Then, the Dreamark generates backdoored NeRFs in a way that the target secret message can be verified by the pre-trained watermark decoder on an arbitrary trigger viewport. We evaluate the generation quality and watermark robustness against image- and model-level attacks. Extensive experiments show that the watermarking process will not degrade the generation quality, and the watermark achieves 90+% accuracy among both image-level attacks (e.g., Gaussian noise) and model-level attacks (e.g., pruning attack).  
  </ol>  
</details>  
  
  



