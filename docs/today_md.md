<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#CDGS:-Confidence-Aware-Depth-Regularization-for-3D-Gaussian-Splatting>CDGS: Confidence-Aware Depth Regularization for 3D Gaussian Splatting</a></li>
        <li><a href=#Structure-from-Sherds++:-Robust-Incremental-3D-Reassembly-of-Axially-Symmetric-Pots-from-Unordered-and-Mixed-Fragment-Collections>Structure-from-Sherds++: Robust Incremental 3D Reassembly of Axially Symmetric Pots from Unordered and Mixed Fragment Collections</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Bridging-Text-and-Vision:-A-Multi-View-Text-Vision-Registration-Approach-for-Cross-Modal-Place-Recognition>Bridging Text and Vision: A Multi-View Text-Vision Registration Approach for Cross-Modal Place Recognition</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#NeRF-3DTalker:-Neural-Radiance-Field-with-3D-Prior-Aided-Audio-Disentanglement-for-Talking-Head-Synthesis>NeRF-3DTalker: Neural Radiance Field with 3D Prior Aided Audio Disentanglement for Talking Head Synthesis</a></li>
        <li><a href=#GlossGau:-Efficient-Inverse-Rendering-for-Glossy-Surface-with-Anisotropic-Spherical-Gaussian>GlossGau: Efficient Inverse Rendering for Glossy Surface with Anisotropic Spherical Gaussian</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [CDGS: Confidence-Aware Depth Regularization for 3D Gaussian Splatting](http://arxiv.org/abs/2502.14684)  
Qilin Zhang, Olaf Wysocki, Steffen Urban, Boris Jutzi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D Gaussian Splatting (3DGS) has shown significant advantages in novel view synthesis (NVS), particularly in achieving high rendering speeds and high-quality results. However, its geometric accuracy in 3D reconstruction remains limited due to the lack of explicit geometric constraints during optimization. This paper introduces CDGS, a confidence-aware depth regularization approach developed to enhance 3DGS. We leverage multi-cue confidence maps of monocular depth estimation and sparse Structure-from-Motion depth to adaptively adjust depth supervision during the optimization process. Our method demonstrates improved geometric detail preservation in early training stages and achieves competitive performance in both NVS quality and geometric accuracy. Experiments on the publicly available Tanks and Temples benchmark dataset show that our method achieves more stable convergence behavior and more accurate geometric reconstruction results, with improvements of up to 2.31 dB in PSNR for NVS and consistently lower geometric errors in M3C2 distance metrics. Notably, our method reaches comparable F-scores to the original 3DGS with only 50% of the training iterations. We expect this work will facilitate the development of efficient and accurate 3D reconstruction systems for real-world applications such as digital twin creation, heritage preservation, or forestry applications.  
  </ol>  
</details>  
  
### [Structure-from-Sherds++: Robust Incremental 3D Reassembly of Axially Symmetric Pots from Unordered and Mixed Fragment Collections](http://arxiv.org/abs/2502.13986)  
Seong Jong Yoo, Sisung Liu, Muhammad Zeeshan Arshad, Jinhyeok Kim, Young Min Kim, Yiannis Aloimonos, Cornelia Fermuller, Kyungdon Joo, Jinwook Kim, Je Hyeong Hong  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Reassembling multiple axially symmetric pots from fragmentary sherds is crucial for cultural heritage preservation, yet it poses significant challenges due to thin and sharp fracture surfaces that generate numerous false positive matches and hinder large-scale puzzle solving. Existing global approaches, which optimize all potential fragment pairs simultaneously or data-driven models, are prone to local minima and face scalability issues when multiple pots are intermixed. Motivated by Structure-from-Motion (SfM) for 3D reconstruction from multiple images, we propose an efficient reassembly method for axially symmetric pots based on iterative registration of one sherd at a time, called Structure-from-Sherds++ (SfS++). Our method extends beyond simple replication of incremental SfM and leverages multi-graph beam search to explore multiple registration paths. This allows us to effectively filter out indistinguishable false matches and simultaneously reconstruct multiple pots without requiring prior information such as base or the number of mixed objects. Our approach achieves 87% reassembly accuracy on a dataset of 142 real fragments from 10 different pots, outperforming other methods in handling complex fracture patterns with mixed datasets and achieving state-of-the-art performance. Code and results can be found in our project page https://sj-yoo.info/sfs/.  
  </ol>  
</details>  
**comments**: 24 pages  
  
  



## Visual Localization  

### [Bridging Text and Vision: A Multi-View Text-Vision Registration Approach for Cross-Modal Place Recognition](http://arxiv.org/abs/2502.14195)  
[[code](https://github.com/nuozimiaowu/Text4VPR)]  
Tianyi Shang, Zhenyu Li, Pengjie Xu, Jinwei Qiao, Gang Chen, Zihan Ruan, Weijun Hu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Mobile robots necessitate advanced natural language understanding capabilities to accurately identify locations and perform tasks such as package delivery. However, traditional visual place recognition (VPR) methods rely solely on single-view visual information and cannot interpret human language descriptions. To overcome this challenge, we bridge text and vision by proposing a multiview (360{\deg} views of the surroundings) text-vision registration approach called Text4VPR for place recognition task, which is the first method that exclusively utilizes textual descriptions to match a database of images. Text4VPR employs the frozen T5 language model to extract global textual embeddings. Additionally, it utilizes the Sinkhorn algorithm with temperature coefficient to assign local tokens to their respective clusters, thereby aggregating visual descriptors from images. During the training stage, Text4VPR emphasizes the alignment between individual text-image pairs for precise textual description. In the inference stage, Text4VPR uses the Cascaded Cross-Attention Cosine Alignment (CCCA) to address the internal mismatch between text and image groups. Subsequently, Text4VPR performs precisely place match based on the descriptions of text-image groups. On Street360Loc, the first text to image VPR dataset we created, Text4VPR builds a robust baseline, achieving a leading top-1 accuracy of 57% and a leading top-10 accuracy of 92% within a 5-meter radius on the test set, which indicates that localization from textual descriptions to images is not only feasible but also holds significant potential for further advancement, as shown in Figure 1.  
  </ol>  
</details>  
**comments**: 8 pages, 4 figures, conference  
  
  



## NeRF  

### [NeRF-3DTalker: Neural Radiance Field with 3D Prior Aided Audio Disentanglement for Talking Head Synthesis](http://arxiv.org/abs/2502.14178)  
Xiaoxing Liu, Zhilei Liu, Chongke Bi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Talking head synthesis is to synthesize a lip-synchronized talking head video using audio. Recently, the capability of NeRF to enhance the realism and texture details of synthesized talking heads has attracted the attention of researchers. However, most current NeRF methods based on audio are exclusively concerned with the rendering of frontal faces. These methods are unable to generate clear talking heads in novel views. Another prevalent challenge in current 3D talking head synthesis is the difficulty in aligning acoustic and visual spaces, which often results in suboptimal lip-syncing of the generated talking heads. To address these issues, we propose Neural Radiance Field with 3D Prior Aided Audio Disentanglement for Talking Head Synthesis (NeRF-3DTalker). Specifically, the proposed method employs 3D prior information to synthesize clear talking heads with free views. Additionally, we propose a 3D Prior Aided Audio Disentanglement module, which is designed to disentangle the audio into two distinct categories: features related to 3D awarded speech movements and features related to speaking style. Moreover, to reposition the generated frames that are distant from the speaker's motion space in the real space, we have devised a local-global Standardized Space. This method normalizes the irregular positions in the generated frames from both global and local semantic perspectives. Through comprehensive qualitative and quantitative experiments, it has been demonstrated that our NeRF-3DTalker outperforms state-of-the-art in synthesizing realistic talking head videos, exhibiting superior image quality and lip synchronization. Project page: https://nerf-3dtalker.github.io/NeRF-3Dtalker.  
  </ol>  
</details>  
**comments**: Accepted by ICASSP 2025  
  
### [GlossGau: Efficient Inverse Rendering for Glossy Surface with Anisotropic Spherical Gaussian](http://arxiv.org/abs/2502.14129)  
Bang Du, Runfa Blark Li, Chen Du, Truong Nguyen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The reconstruction of 3D objects from calibrated photographs represents a fundamental yet intricate challenge in the domains of computer graphics and vision. Although neural reconstruction approaches based on Neural Radiance Fields (NeRF) have shown remarkable capabilities, their processing costs remain substantial. Recently, the advent of 3D Gaussian Splatting (3D-GS) largely improves the training efficiency and facilitates to generate realistic rendering in real-time. However, due to the limited ability of Spherical Harmonics (SH) to represent high-frequency information, 3D-GS falls short in reconstructing glossy objects. Researchers have turned to enhance the specular expressiveness of 3D-GS through inverse rendering. Yet these methods often struggle to maintain the training and rendering efficiency, undermining the benefits of Gaussian Splatting techniques. In this paper, we introduce GlossGau, an efficient inverse rendering framework that reconstructs scenes with glossy surfaces while maintaining training and rendering speeds comparable to vanilla 3D-GS. Specifically, we explicitly model the surface normals, Bidirectional Reflectance Distribution Function (BRDF) parameters, as well as incident lights and use Anisotropic Spherical Gaussian (ASG) to approximate the per-Gaussian Normal Distribution Function under the microfacet model. We utilize 2D Gaussian Splatting (2D-GS) as foundational primitives and apply regularization to significantly alleviate the normal estimation challenge encountered in related works. Experiments demonstrate that GlossGau achieves competitive or superior reconstruction on datasets with glossy surfaces. Compared with previous GS-based works that address the specular surface, our optimization time is considerably less.  
  </ol>  
</details>  
  
  



