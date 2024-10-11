<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#IncEventGS:-Pose-Free-Gaussian-Splatting-from-a-Single-Event-Camera>IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera</a></li>
      </ul>
    </li>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Surgical-Depth-Anything:-Depth-Estimation-for-Surgical-Scenes-using-Foundation-Models>Surgical Depth Anything: Depth Estimation for Surgical Scenes using Foundation Models</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#A-Unified-Debiasing-Approach-for-Vision-Language-Models-across-Modalities-and-Tasks>A Unified Debiasing Approach for Vision-Language Models across Modalities and Tasks</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#RGM:-Reconstructing-High-fidelity-3D-Car-Assets-with-Relightable-3D-GS-Generative-Model-from-a-Single-Image>RGM: Reconstructing High-fidelity 3D Car Assets with Relightable 3D-GS Generative Model from a Single Image</a></li>
        <li><a href=#IncEventGS:-Pose-Free-Gaussian-Splatting-from-a-Single-Event-Camera>IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera</a></li>
        <li><a href=#NeRF-Accelerated-Ecological-Monitoring-in-Mixed-Evergreen-Redwood-Forest>NeRF-Accelerated Ecological Monitoring in Mixed-Evergreen Redwood Forest</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera](http://arxiv.org/abs/2410.08107)  
[[code](https://github.com/wu-cvgl/inceventgs)]  
Jian Huang, Chengrui Dong, Peidong Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Implicit neural representation and explicit 3D Gaussian Splatting (3D-GS) for novel view synthesis have achieved remarkable progress with frame-based camera (e.g. RGB and RGB-D cameras) recently. Compared to frame-based camera, a novel type of bio-inspired visual sensor, i.e. event camera, has demonstrated advantages in high temporal resolution, high dynamic range, low power consumption and low latency. Due to its unique asynchronous and irregular data capturing process, limited work has been proposed to apply neural representation or 3D Gaussian splatting for an event camera. In this work, we present IncEventGS, an incremental 3D Gaussian Splatting reconstruction algorithm with a single event camera. To recover the 3D scene representation incrementally, we exploit the tracking and mapping paradigm of conventional SLAM pipelines for IncEventGS. Given the incoming event stream, the tracker firstly estimates an initial camera motion based on prior reconstructed 3D-GS scene representation. The mapper then jointly refines both the 3D scene representation and camera motion based on the previously estimated motion trajectory from the tracker. The experimental results demonstrate that IncEventGS delivers superior performance compared to prior NeRF-based methods and other related baselines, even we do not have the ground-truth camera poses. Furthermore, our method can also deliver better performance compared to state-of-the-art event visual odometry methods in terms of camera motion estimation. Code is publicly available at: https://github.com/wu-cvgl/IncEventGS.  
  </ol>  
</details>  
**comments**: Code Page: https://github.com/wu-cvgl/IncEventGS  
  
  



## SFM  

### [Surgical Depth Anything: Depth Estimation for Surgical Scenes using Foundation Models](http://arxiv.org/abs/2410.07434)  
Ange Lou, Yamin Li, Yike Zhang, Jack Noble  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Monocular depth estimation is crucial for tracking and reconstruction algorithms, particularly in the context of surgical videos. However, the inherent challenges in directly obtaining ground truth depth maps during surgery render supervised learning approaches impractical. While many self-supervised methods based on Structure from Motion (SfM) have shown promising results, they rely heavily on high-quality camera motion and require optimization on a per-patient basis. These limitations can be mitigated by leveraging the current state-of-the-art foundational model for depth estimation, Depth Anything. However, when directly applied to surgical scenes, Depth Anything struggles with issues such as blurring, bleeding, and reflections, resulting in suboptimal performance. This paper presents a fine-tuning of the Depth Anything model specifically for the surgical domain, aiming to deliver more accurate pixel-wise depth maps tailored to the unique requirements and challenges of surgical environments. Our fine-tuning approach significantly improves the model's performance in surgical scenes, reducing errors related to blurring and reflections, and achieving a more reliable and precise depth estimation.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [A Unified Debiasing Approach for Vision-Language Models across Modalities and Tasks](http://arxiv.org/abs/2410.07593)  
Hoin Jung, Taeuk Jang, Xiaoqian Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in Vision-Language Models (VLMs) have enabled complex multimodal tasks by processing text and image data simultaneously, significantly enhancing the field of artificial intelligence. However, these models often exhibit biases that can skew outputs towards societal stereotypes, thus necessitating debiasing strategies. Existing debiasing methods focus narrowly on specific modalities or tasks, and require extensive retraining. To address these limitations, this paper introduces Selective Feature Imputation for Debiasing (SFID), a novel methodology that integrates feature pruning and low confidence imputation (LCI) to effectively reduce biases in VLMs. SFID is versatile, maintaining the semantic integrity of outputs and costly effective by eliminating the need for retraining. Our experimental results demonstrate SFID's effectiveness across various VLMs tasks including zero-shot classification, text-to-image retrieval, image captioning, and text-to-image generation, by significantly reducing gender biases without compromising performance. This approach not only enhances the fairness of VLMs applications but also preserves their efficiency and utility across diverse scenarios.  
  </ol>  
</details>  
**comments**: NeurIPS 2024, the Thirty-Eighth Annual Conference on Neural
  Information Processing Systems  
  
  



## NeRF  

### [RGM: Reconstructing High-fidelity 3D Car Assets with Relightable 3D-GS Generative Model from a Single Image](http://arxiv.org/abs/2410.08181)  
Xiaoxue Chen, Jv Zheng, Hao Huang, Haoran Xu, Weihao Gu, Kangliang Chen, He xiang, Huan-ang Gao, Hao Zhao, Guyue Zhou, Yaqin Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The generation of high-quality 3D car assets is essential for various applications, including video games, autonomous driving, and virtual reality. Current 3D generation methods utilizing NeRF or 3D-GS as representations for 3D objects, generate a Lambertian object under fixed lighting and lack separated modelings for material and global illumination. As a result, the generated assets are unsuitable for relighting under varying lighting conditions, limiting their applicability in downstream tasks. To address this challenge, we propose a novel relightable 3D object generative framework that automates the creation of 3D car assets, enabling the swift and accurate reconstruction of a vehicle's geometry, texture, and material properties from a single input image. Our approach begins with introducing a large-scale synthetic car dataset comprising over 1,000 high-precision 3D vehicle models. We represent 3D objects using global illumination and relightable 3D Gaussian primitives integrating with BRDF parameters. Building on this representation, we introduce a feed-forward model that takes images as input and outputs both relightable 3D Gaussians and global illumination parameters. Experimental results demonstrate that our method produces photorealistic 3D car assets that can be seamlessly integrated into road scenes with different illuminations, which offers substantial practical benefits for industrial applications.  
  </ol>  
</details>  
  
### [IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera](http://arxiv.org/abs/2410.08107)  
[[code](https://github.com/wu-cvgl/inceventgs)]  
Jian Huang, Chengrui Dong, Peidong Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Implicit neural representation and explicit 3D Gaussian Splatting (3D-GS) for novel view synthesis have achieved remarkable progress with frame-based camera (e.g. RGB and RGB-D cameras) recently. Compared to frame-based camera, a novel type of bio-inspired visual sensor, i.e. event camera, has demonstrated advantages in high temporal resolution, high dynamic range, low power consumption and low latency. Due to its unique asynchronous and irregular data capturing process, limited work has been proposed to apply neural representation or 3D Gaussian splatting for an event camera. In this work, we present IncEventGS, an incremental 3D Gaussian Splatting reconstruction algorithm with a single event camera. To recover the 3D scene representation incrementally, we exploit the tracking and mapping paradigm of conventional SLAM pipelines for IncEventGS. Given the incoming event stream, the tracker firstly estimates an initial camera motion based on prior reconstructed 3D-GS scene representation. The mapper then jointly refines both the 3D scene representation and camera motion based on the previously estimated motion trajectory from the tracker. The experimental results demonstrate that IncEventGS delivers superior performance compared to prior NeRF-based methods and other related baselines, even we do not have the ground-truth camera poses. Furthermore, our method can also deliver better performance compared to state-of-the-art event visual odometry methods in terms of camera motion estimation. Code is publicly available at: https://github.com/wu-cvgl/IncEventGS.  
  </ol>  
</details>  
**comments**: Code Page: https://github.com/wu-cvgl/IncEventGS  
  
### [NeRF-Accelerated Ecological Monitoring in Mixed-Evergreen Redwood Forest](http://arxiv.org/abs/2410.07418)  
[[code](https://github.com/harelab-ucsc/redwoodnerf)]  
Adam Korycki, Cory Yeaton, Gregory S. Gilbert, Colleen Josephson, Steve McGuire  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Forest mapping provides critical observational data needed to understand the dynamics of forest environments. Notably, tree diameter at breast height (DBH) is a metric used to estimate forest biomass and carbon dioxide (CO $_2$ ) sequestration. Manual methods of forest mapping are labor intensive and time consuming, a bottleneck for large-scale mapping efforts. Automated mapping relies on acquiring dense forest reconstructions, typically in the form of point clouds. Terrestrial laser scanning (TLS) and mobile laser scanning (MLS) generate point clouds using expensive LiDAR sensing, and have been used successfully to estimate tree diameter. Neural radiance fields (NeRFs) are an emergent technology enabling photorealistic, vision-based reconstruction by training a neural network on a sparse set of input views. In this paper, we present a comparison of MLS and NeRF forest reconstructions for the purpose of trunk diameter estimation in a mixed-evergreen Redwood forest. In addition, we propose an improved DBH-estimation method using convex-hull modeling. Using this approach, we achieved 1.68 cm RMSE, which consistently outperformed standard cylinder modeling approaches. Our code contributions and forest datasets are freely available at https://github.com/harelab-ucsc/RedwoodNeRF.  
  </ol>  
</details>  
  
  



