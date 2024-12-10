<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Doppelgangers++:-Improved-Visual-Disambiguation-with-Geometric-3D-Features>Doppelgangers++: Improved Visual Disambiguation with Geometric 3D Features</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#An-Efficient-Scene-Coordinate-Encoding-and-Relocalization-Method>An Efficient Scene Coordinate Encoding and Relocalization Method</a></li>
        <li><a href=#A-Hyperdimensional-One-Place-Signature-to-Represent-Them-All:-Stackable-Descriptors-For-Visual-Place-Recognition>A Hyperdimensional One Place Signature to Represent Them All: Stackable Descriptors For Visual Place Recognition</a></li>
        <li><a href=#Compositional-Image-Retrieval-via-Instruction-Aware-Contrastive-Learning>Compositional Image Retrieval via Instruction-Aware Contrastive Learning</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#An-Efficient-Scene-Coordinate-Encoding-and-Relocalization-Method>An Efficient Scene Coordinate Encoding and Relocalization Method</a></li>
        <li><a href=#ZeroKey:-Point-Level-Reasoning-and-Zero-Shot-3D-Keypoint-Detection-from-Large-Language-Models>ZeroKey: Point-Level Reasoning and Zero-Shot 3D Keypoint Detection from Large Language Models</a></li>
        <li><a href=#Securing-Social-Media-Against-Deepfakes-using-Identity,-Behavioral,-and-Geometric-Signatures>Securing Social Media Against Deepfakes using Identity, Behavioral, and Geometric Signatures</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Dynamic-EventNeRF:-Reconstructing-General-Dynamic-Scenes-from-Multi-view-Event-Cameras>Dynamic EventNeRF: Reconstructing General Dynamic Scenes from Multi-view Event Cameras</a></li>
        <li><a href=#Deblur4DGS:-4D-Gaussian-Splatting-from-Blurry-Monocular-Video>Deblur4DGS: 4D Gaussian Splatting from Blurry Monocular Video</a></li>
        <li><a href=#Splatter-360:-Generalizable-360$^{\circ}$-Gaussian-Splatting-for-Wide-baseline-Panoramic-Images>Splatter-360: Generalizable 360$^{\circ}$ Gaussian Splatting for Wide-baseline Panoramic Images</a></li>
        <li><a href=#WATER-GS:-Toward-Copyright-Protection-for-3D-Gaussian-Splatting-via-Universal-Watermarking>WATER-GS: Toward Copyright Protection for 3D Gaussian Splatting via Universal Watermarking</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Doppelgangers++: Improved Visual Disambiguation with Geometric 3D Features](http://arxiv.org/abs/2412.05826)  
Yuanbo Xiangli, Ruojin Cai, Hanyu Chen, Jeffrey Byrne, Noah Snavely  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Accurate 3D reconstruction is frequently hindered by visual aliasing, where visually similar but distinct surfaces (aka, doppelgangers), are incorrectly matched. These spurious matches distort the structure-from-motion (SfM) process, leading to misplaced model elements and reduced accuracy. Prior efforts addressed this with CNN classifiers trained on curated datasets, but these approaches struggle to generalize across diverse real-world scenes and can require extensive parameter tuning. In this work, we present Doppelgangers++, a method to enhance doppelganger detection and improve 3D reconstruction accuracy. Our contributions include a diversified training dataset that incorporates geo-tagged images from everyday scenes to expand robustness beyond landmark-based datasets. We further propose a Transformer-based classifier that leverages 3D-aware features from the MASt3R model, achieving superior precision and recall across both in-domain and out-of-domain tests. Doppelgangers++ integrates seamlessly into standard SfM and MASt3R-SfM pipelines, offering efficiency and adaptability across varied scenes. To evaluate SfM accuracy, we introduce an automated, geotag-based method for validating reconstructed models, eliminating the need for manual inspection. Through extensive experiments, we demonstrate that Doppelgangers++ significantly enhances pairwise visual disambiguation and improves 3D reconstruction quality in complex and diverse scenarios.  
  </ol>  
</details>  
**comments**: Project page can be found in
  https://doppelgangers25.github.io/doppelgangers_plusplus/  
  
  



## Visual Localization  

### [An Efficient Scene Coordinate Encoding and Relocalization Method](http://arxiv.org/abs/2412.06488)  
Kuan Xu, Zeyu Jiang, Haozhi Cao, Shenghai Yuan, Chen Wang, Lihua Xie  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Scene Coordinate Regression (SCR) is a visual localization technique that utilizes deep neural networks (DNN) to directly regress 2D-3D correspondences for camera pose estimation. However, current SCR methods often face challenges in handling repetitive textures and meaningless areas due to their reliance on implicit triangulation. In this paper, we propose an efficient scene coordinate encoding and relocalization method. Compared with the existing SCR methods, we design a unified architecture for both scene encoding and salient keypoint detection, enabling our system to focus on encoding informative regions, thereby significantly enhancing efficiency. Additionally, we introduce a mechanism that leverages sequential information during both map encoding and relocalization, which strengthens implicit triangulation, particularly in repetitive texture environments. Comprehensive experiments conducted across indoor and outdoor datasets demonstrate that the proposed system outperforms other state-of-the-art (SOTA) SCR methods. Our single-frame relocalization mode improves the recall rate of our baseline by 6.4% and increases the running speed from 56Hz to 90Hz. Furthermore, our sequence-based mode increases the recall rate by 11% while maintaining the original efficiency.  
  </ol>  
</details>  
**comments**: 8 pages, 6 figures  
  
### [A Hyperdimensional One Place Signature to Represent Them All: Stackable Descriptors For Visual Place Recognition](http://arxiv.org/abs/2412.06153)  
Connor Malone, Somayeh Hussaini, Tobias Fischer, Michael Milford  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Place Recognition (VPR) enables coarse localization by comparing query images to a reference database of geo-tagged images. Recent breakthroughs in deep learning architectures and training regimes have led to methods with improved robustness to factors like environment appearance change, but with the downside that the required training and/or matching compute scales with the number of distinct environmental conditions encountered. Here, we propose Hyperdimensional One Place Signatures (HOPS) to simultaneously improve the performance, compute and scalability of these state-of-the-art approaches by fusing the descriptors from multiple reference sets captured under different conditions. HOPS scales to any number of environmental conditions by leveraging the Hyperdimensional Computing framework. Extensive evaluations demonstrate that our approach is highly generalizable and consistently improves recall performance across all evaluated VPR methods and datasets by large margins. Arbitrarily fusing reference images without compute penalty enables numerous other useful possibilities, three of which we demonstrate here: descriptor dimensionality reduction with no performance penalty, stacking synthetic images, and coarse localization to an entire traverse or environmental section.  
  </ol>  
</details>  
**comments**: Under Review  
  
### [Compositional Image Retrieval via Instruction-Aware Contrastive Learning](http://arxiv.org/abs/2412.05756)  
Wenliang Zhong, Weizhi An, Feng Jiang, Hehuan Ma, Yuzhi Guo, Junzhou Huang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed Image Retrieval (CIR) involves retrieving a target image based on a composed query of an image paired with text that specifies modifications or changes to the visual reference. CIR is inherently an instruction-following task, as the model needs to interpret and apply modifications to the image. In practice, due to the scarcity of annotated data in downstream tasks, Zero-Shot CIR (ZS-CIR) is desirable. While existing ZS-CIR models based on CLIP have shown promising results, their capability in interpreting and following modification instructions remains limited. Some research attempts to address this by incorporating Large Language Models (LLMs). However, these approaches still face challenges in effectively integrating multimodal information and instruction understanding. To tackle above challenges, we propose a novel embedding method utilizing an instruction-tuned Multimodal LLM (MLLM) to generate composed representation, which significantly enhance the instruction following capability for a comprehensive integration between images and instructions. Nevertheless, directly applying MLLMs introduces a new challenge since MLLMs are primarily designed for text generation rather than embedding extraction as required in CIR. To address this, we introduce a two-stage training strategy to efficiently learn a joint multimodal embedding space and further refining the ability to follow modification instructions by tuning the model in a triplet dataset similar to the CIR format. Extensive experiments on four public datasets: FashionIQ, CIRR, GeneCIS, and CIRCO demonstrates the superior performance of our model, outperforming state-of-the-art baselines by a significant margin. Codes are available at the GitHub repository.  
  </ol>  
</details>  
**comments**: 9 pages, 8 figures  
  
  



## Keypoint Detection  

### [An Efficient Scene Coordinate Encoding and Relocalization Method](http://arxiv.org/abs/2412.06488)  
Kuan Xu, Zeyu Jiang, Haozhi Cao, Shenghai Yuan, Chen Wang, Lihua Xie  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Scene Coordinate Regression (SCR) is a visual localization technique that utilizes deep neural networks (DNN) to directly regress 2D-3D correspondences for camera pose estimation. However, current SCR methods often face challenges in handling repetitive textures and meaningless areas due to their reliance on implicit triangulation. In this paper, we propose an efficient scene coordinate encoding and relocalization method. Compared with the existing SCR methods, we design a unified architecture for both scene encoding and salient keypoint detection, enabling our system to focus on encoding informative regions, thereby significantly enhancing efficiency. Additionally, we introduce a mechanism that leverages sequential information during both map encoding and relocalization, which strengthens implicit triangulation, particularly in repetitive texture environments. Comprehensive experiments conducted across indoor and outdoor datasets demonstrate that the proposed system outperforms other state-of-the-art (SOTA) SCR methods. Our single-frame relocalization mode improves the recall rate of our baseline by 6.4% and increases the running speed from 56Hz to 90Hz. Furthermore, our sequence-based mode increases the recall rate by 11% while maintaining the original efficiency.  
  </ol>  
</details>  
**comments**: 8 pages, 6 figures  
  
### [ZeroKey: Point-Level Reasoning and Zero-Shot 3D Keypoint Detection from Large Language Models](http://arxiv.org/abs/2412.06292)  
Bingchen Gong, Diego Gomez, Abdullah Hamdi, Abdelrahman Eldesokey, Ahmed Abdelreheem, Peter Wonka, Maks Ovsjanikov  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose a novel zero-shot approach for keypoint detection on 3D shapes. Point-level reasoning on visual data is challenging as it requires precise localization capability, posing problems even for powerful models like DINO or CLIP. Traditional methods for 3D keypoint detection rely heavily on annotated 3D datasets and extensive supervised training, limiting their scalability and applicability to new categories or domains. In contrast, our method utilizes the rich knowledge embedded within Multi-Modal Large Language Models (MLLMs). Specifically, we demonstrate, for the first time, that pixel-level annotations used to train recent MLLMs can be exploited for both extracting and naming salient keypoints on 3D models without any ground truth labels or supervision. Experimental evaluations demonstrate that our approach achieves competitive performance on standard benchmarks compared to supervised methods, despite not requiring any 3D keypoint annotations during training. Our results highlight the potential of integrating language models for localized 3D shape understanding. This work opens new avenues for cross-modal learning and underscores the effectiveness of MLLMs in contributing to 3D computer vision challenges.  
  </ol>  
</details>  
**comments**: Project website is accessible at
  https://sites.google.com/view/zerokey  
  
### [Securing Social Media Against Deepfakes using Identity, Behavioral, and Geometric Signatures](http://arxiv.org/abs/2412.05487)  
Muhammad Umar Farooq, Awais Khan, Ijaz Ul Haq, Khalid Mahmood Malik  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Trust in social media is a growing concern due to its ability to influence significant societal changes. However, this space is increasingly compromised by various types of deepfake multimedia, which undermine the authenticity of shared content. Although substantial efforts have been made to address the challenge of deepfake content, existing detection techniques face a major limitation in generalization: they tend to perform well only on specific types of deepfakes they were trained on.This dependency on recognizing specific deepfake artifacts makes current methods vulnerable when applied to unseen or varied deepfakes, thereby compromising their performance in real-world applications such as social media platforms. To address the generalizability of deepfake detection, there is a need for a holistic approach that can capture a broader range of facial attributes and manipulations beyond isolated artifacts. To address this, we propose a novel deepfake detection framework featuring an effective feature descriptor that integrates Deep identity, Behavioral, and Geometric (DBaG) signatures, along with a classifier named DBaGNet. Specifically, the DBaGNet classifier utilizes the extracted DBaG signatures, leveraging a triplet loss objective to enhance generalized representation learning for improved classification. Specifically, the DBaGNet classifier utilizes the extracted DBaG signatures and applies a triplet loss objective to enhance generalized representation learning for improved classification. To test the effectiveness and generalizability of our proposed approach, we conduct extensive experiments using six benchmark deepfake datasets: WLDR, CelebDF, DFDC, FaceForensics++, DFD, and NVFAIR. Specifically, to ensure the effectiveness of our approach, we perform cross-dataset evaluations, and the results demonstrate significant performance gains over several state-of-the-art methods.  
  </ol>  
</details>  
  
  



## NeRF  

### [Dynamic EventNeRF: Reconstructing General Dynamic Scenes from Multi-view Event Cameras](http://arxiv.org/abs/2412.06770)  
Viktor Rudnev, Gereon Fox, Mohamed Elgharib, Christian Theobalt, Vladislav Golyanik  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Volumetric reconstruction of dynamic scenes is an important problem in computer vision. It is especially challenging in poor lighting and with fast motion. It is partly due to the limitations of RGB cameras: To capture fast motion without much blur, the framerate must be increased, which in turn requires more lighting. In contrast, event cameras, which record changes in pixel brightness asynchronously, are much less dependent on lighting, making them more suitable for recording fast motion. We hence propose the first method to spatiotemporally reconstruct a scene from sparse multi-view event streams and sparse RGB frames. We train a sequence of cross-faded time-conditioned NeRF models, one per short recording segment. The individual segments are supervised with a set of event- and RGB-based losses and sparse-view regularisation. We assemble a real-world multi-view camera rig with six static event cameras around the object and record a benchmark multi-view event stream dataset of challenging motions. Our work outperforms RGB-based baselines, producing state-of-the-art results, and opens up the topic of multi-view event-based reconstruction as a new path for fast scene capture beyond RGB cameras. The code and the data will be released soon at https://4dqv.mpi-inf.mpg.de/DynEventNeRF/  
  </ol>  
</details>  
**comments**: 15 pages, 11 figures, 6 tables  
  
### [Deblur4DGS: 4D Gaussian Splatting from Blurry Monocular Video](http://arxiv.org/abs/2412.06424)  
Renlong Wu, Zhilu Zhang, Mingyang Chen, Xiaopeng Fan, Zifei Yan, Wangmeng Zuo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent 4D reconstruction methods have yielded impressive results but rely on sharp videos as supervision. However, motion blur often occurs in videos due to camera shake and object movement, while existing methods render blurry results when using such videos for reconstructing 4D models. Although a few NeRF-based approaches attempted to address the problem, they struggled to produce high-quality results, due to the inaccuracy in estimating continuous dynamic representations within the exposure time. Encouraged by recent works in 3D motion trajectory modeling using 3D Gaussian Splatting (3DGS), we suggest taking 3DGS as the scene representation manner, and propose the first 4D Gaussian Splatting framework to reconstruct a high-quality 4D model from blurry monocular video, named Deblur4DGS. Specifically, we transform continuous dynamic representations estimation within an exposure time into the exposure time estimation. Moreover, we introduce exposure regularization to avoid trivial solutions, as well as multi-frame and multi-resolution consistency ones to alleviate artifacts. Furthermore, to better represent objects with large motion, we suggest blur-aware variable canonical Gaussians. Beyond novel-view synthesis, Deblur4DGS can be applied to improve blurry video from multiple perspectives, including deblurring, frame interpolation, and video stabilization. Extensive experiments on the above four tasks show that Deblur4DGS outperforms state-of-the-art 4D reconstruction methods. The codes are available at https://github.com/ZcsrenlongZ/Deblur4DGS.  
  </ol>  
</details>  
**comments**: 17 pages  
  
### [Splatter-360: Generalizable 360 $^{\circ}$ Gaussian Splatting for Wide-baseline Panoramic Images](http://arxiv.org/abs/2412.06250)  
[[code](https://github.com/thucz/splatter360)]  
Zheng Chen, Chenming Wu, Zhelun Shen, Chen Zhao, Weicai Ye, Haocheng Feng, Errui Ding, Song-Hai Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Wide-baseline panoramic images are frequently used in applications like VR and simulations to minimize capturing labor costs and storage needs. However, synthesizing novel views from these panoramic images in real time remains a significant challenge, especially due to panoramic imagery's high resolution and inherent distortions. Although existing 3D Gaussian splatting (3DGS) methods can produce photo-realistic views under narrow baselines, they often overfit the training views when dealing with wide-baseline panoramic images due to the difficulty in learning precise geometry from sparse 360$^{\circ}$ views. This paper presents \textit{Splatter-360}, a novel end-to-end generalizable 3DGS framework designed to handle wide-baseline panoramic images. Unlike previous approaches, \textit{Splatter-360} performs multi-view matching directly in the spherical domain by constructing a spherical cost volume through a spherical sweep algorithm, enhancing the network's depth perception and geometry estimation. Additionally, we introduce a 3D-aware bi-projection encoder to mitigate the distortions inherent in panoramic images and integrate cross-view attention to improve feature interactions across multiple viewpoints. This enables robust 3D-aware feature representations and real-time rendering capabilities. Experimental results on the HM3D~\cite{hm3d} and Replica~\cite{replica} demonstrate that \textit{Splatter-360} significantly outperforms state-of-the-art NeRF and 3DGS methods (e.g., PanoGRF, MVSplat, DepthSplat, and HiSplat) in both synthesis quality and generalization performance for wide-baseline panoramic images. Code and trained models are available at \url{https://3d-aigc.github.io/Splatter-360/}.  
  </ol>  
</details>  
**comments**: Project page:https://3d-aigc.github.io/Splatter-360/. Code:
  https://github.com/thucz/splatter360  
  
### [WATER-GS: Toward Copyright Protection for 3D Gaussian Splatting via Universal Watermarking](http://arxiv.org/abs/2412.05695)  
Yuqi Tan, Xiang Liu, Shuzhao Xie, Bin Chen, Shu-Tao Xia, Zhi Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D Gaussian Splatting (3DGS) has emerged as a pivotal technique for 3D scene representation, providing rapid rendering speeds and high fidelity. As 3DGS gains prominence, safeguarding its intellectual property becomes increasingly crucial since 3DGS could be used to imitate unauthorized scene creations and raise copyright issues. Existing watermarking methods for implicit NeRFs cannot be directly applied to 3DGS due to its explicit representation and real-time rendering process, leaving watermarking for 3DGS largely unexplored. In response, we propose WATER-GS, a novel method designed to protect 3DGS copyrights through a universal watermarking strategy. First, we introduce a pre-trained watermark decoder, treating raw 3DGS generative modules as potential watermark encoders to ensure imperceptibility. Additionally, we implement novel 3D distortion layers to enhance the robustness of the embedded watermark against common real-world distortions of point cloud data. Comprehensive experiments and ablation studies demonstrate that WATER-GS effectively embeds imperceptible and robust watermarks into 3DGS without compromising rendering efficiency and quality. Our experiments indicate that the 3D distortion layers can yield up to a 20% improvement in accuracy rate. Notably, our method is adaptable to different 3DGS variants, including 3DGS compression frameworks and 2D Gaussian splatting.  
  </ol>  
</details>  
  
  



