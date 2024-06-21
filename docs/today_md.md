<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#MVSBoost:-An-Efficient-Point-Cloud-based-3D-Reconstruction>MVSBoost: An Efficient Point Cloud-based 3D Reconstruction</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Towards-a-multimodal-framework-for-remote-sensing-image-change-retrieval-and-captioning>Towards a multimodal framework for remote sensing image change retrieval and captioning</a></li>
        <li><a href=#CLIP-Branches:-Interactive-Fine-Tuning-for-Text-Image-Retrieval>CLIP-Branches: Interactive Fine-Tuning for Text-Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Deblurring-Neural-Radiance-Fields-with-Event-driven-Bundle-Adjustment>Deblurring Neural Radiance Fields with Event-driven Bundle Adjustment</a></li>
        <li><a href=#NeRF-Feat:-6D-Object-Pose-Estimation-using-Feature-Rendering>NeRF-Feat: 6D Object Pose Estimation using Feature Rendering</a></li>
        <li><a href=#Style-NeRF2NeRF:-3D-Style-Transfer-From-Style-Aligned-Multi-View-Images>Style-NeRF2NeRF: 3D Style Transfer From Style-Aligned Multi-View Images</a></li>
        <li><a href=#Freq-Mip-AA-:-Frequency-Mip-Representation-for-Anti-Aliasing-Neural-Radiance-Fields>Freq-Mip-AA : Frequency Mip Representation for Anti-Aliasing Neural Radiance Fields</a></li>
        <li><a href=#Sampling-3D-Gaussian-Scenes-in-Seconds-with-Latent-Diffusion-Models>Sampling 3D Gaussian Scenes in Seconds with Latent Diffusion Models</a></li>
        <li><a href=#Head-Pose-Estimation-and-3D-Neural-Surface-Reconstruction-via-Monocular-Camera-in-situ-for-Navigation-and-Safe-Insertion-into-Natural-Openings>Head Pose Estimation and 3D Neural Surface Reconstruction via Monocular Camera in situ for Navigation and Safe Insertion into Natural Openings</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [MVSBoost: An Efficient Point Cloud-based 3D Reconstruction](http://arxiv.org/abs/2406.13515)  
Umair Haroon, Ahmad AlMughrabi, Ricardo Marques, Petia Radeva  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Efficient and accurate 3D reconstruction is crucial for various applications, including augmented and virtual reality, medical imaging, and cinematic special effects. While traditional Multi-View Stereo (MVS) systems have been fundamental in these applications, using neural implicit fields in implicit 3D scene modeling has introduced new possibilities for handling complex topologies and continuous surfaces. However, neural implicit fields often suffer from computational inefficiencies, overfitting, and heavy reliance on data quality, limiting their practical use. This paper presents an enhanced MVS framework that integrates multi-view 360-degree imagery with robust camera pose estimation via Structure from Motion (SfM) and advanced image processing for point cloud densification, mesh reconstruction, and texturing. Our approach significantly improves upon traditional MVS methods, offering superior accuracy and precision as validated using Chamfer distance metrics on the Realistic Synthetic 360 dataset. The developed MVS technique enhances the detail and clarity of 3D reconstructions and demonstrates superior computational efficiency and robustness in complex scene reconstruction, effectively handling occlusions and varying viewpoints. These improvements suggest that our MVS framework can compete with and potentially exceed current state-of-the-art neural implicit field methods, especially in scenarios requiring real-time processing and scalability.  
  </ol>  
</details>  
**comments**: The work is under review  
  
  



## Visual Localization  

### [Towards a multimodal framework for remote sensing image change retrieval and captioning](http://arxiv.org/abs/2406.13424)  
Roger Ferrod, Luigi Di Caro, Dino Ienco  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recently, there has been increasing interest in multimodal applications that integrate text with other modalities, such as images, audio and video, to facilitate natural language interactions with multimodal AI systems. While applications involving standard modalities have been extensively explored, there is still a lack of investigation into specific data modalities such as remote sensing (RS) data. Despite the numerous potential applications of RS data, including environmental protection, disaster monitoring and land planning, available solutions are predominantly focused on specific tasks like classification, captioning and retrieval. These solutions often overlook the unique characteristics of RS data, such as its capability to systematically provide information on the same geographical areas over time. This ability enables continuous monitoring of changes in the underlying landscape. To address this gap, we propose a novel foundation model for bi-temporal RS image pairs, in the context of change detection analysis, leveraging Contrastive Learning and the LEVIR-CC dataset for both captioning and text-image retrieval. By jointly training a contrastive encoder and captioning decoder, our model add text-image retrieval capabilities, in the context of bi-temporal change detection, while maintaining captioning performances that are comparable to the state of the art. We release the source code and pretrained weights at: https://github.com/rogerferrod/RSICRC.  
  </ol>  
</details>  
  
### [CLIP-Branches: Interactive Fine-Tuning for Text-Image Retrieval](http://arxiv.org/abs/2406.13322)  
Christian Lülf, Denis Mayr Lima Martins, Marcos Antonio Vaz Salles, Yongluan Zhou, Fabian Gieseke  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The advent of text-image models, most notably CLIP, has significantly transformed the landscape of information retrieval. These models enable the fusion of various modalities, such as text and images. One significant outcome of CLIP is its capability to allow users to search for images using text as a query, as well as vice versa. This is achieved via a joint embedding of images and text data that can, for instance, be used to search for similar items. Despite efficient query processing techniques such as approximate nearest neighbor search, the results may lack precision and completeness. We introduce CLIP-Branches, a novel text-image search engine built upon the CLIP architecture. Our approach enhances traditional text-image search engines by incorporating an interactive fine-tuning phase, which allows the user to further concretize the search query by iteratively defining positive and negative examples. Our framework involves training a classification model given the additional user feedback and essentially outputs all positively classified instances of the entire data catalog. By building upon recent techniques, this inference phase, however, is not implemented by scanning the entire data catalog, but by employing efficient index structures pre-built for the data. Our results show that the fine-tuned results can improve the initial search outputs in terms of relevance and accuracy while maintaining swift response times  
  </ol>  
</details>  
  
  



## NeRF  

### [Deblurring Neural Radiance Fields with Event-driven Bundle Adjustment](http://arxiv.org/abs/2406.14360)  
Yunshan Qi, Lin Zhu, Yifan Zhao, Nan Bao, Jia Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) achieve impressive 3D representation learning and novel view synthesis results with high-quality multi-view images as input. However, motion blur in images often occurs in low-light and high-speed motion scenes, which significantly degrade the reconstruction quality of NeRF. Previous deblurring NeRF methods are struggling to estimate information during the exposure time, unable to accurately model the motion blur. In contrast, the bio-inspired event camera measuring intensity changes with high temporal resolution makes up this information deficiency. In this paper, we propose Event-driven Bundle Adjustment for Deblurring Neural Radiance Fields (EBAD-NeRF) to jointly optimize the learnable poses and NeRF parameters by leveraging the hybrid event-RGB data. An intensity-change-metric event loss and a photo-metric blur loss are introduced to strengthen the explicit modeling of camera motion blur. Experiment results on both synthetic data and real captured data demonstrate that EBAD-NeRF can obtain accurate camera poses during the exposure time and learn sharper 3D representations compared to prior works.  
  </ol>  
</details>  
  
### [NeRF-Feat: 6D Object Pose Estimation using Feature Rendering](http://arxiv.org/abs/2406.13796)  
Shishir Reddy Vutukur, Heike Brock, Benjamin Busam, Tolga Birdal, Andreas Hutter, Slobodan Ilic  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Object Pose Estimation is a crucial component in robotic grasping and augmented reality. Learning based approaches typically require training data from a highly accurate CAD model or labeled training data acquired using a complex setup. We address this by learning to estimate pose from weakly labeled data without a known CAD model. We propose to use a NeRF to learn object shape implicitly which is later used to learn view-invariant features in conjunction with CNN using a contrastive loss. While NeRF helps in learning features that are view-consistent, CNN ensures that the learned features respect symmetry. During inference, CNN is used to predict view-invariant features which can be used to establish correspondences with the implicit 3d model in NeRF. The correspondences are then used to estimate the pose in the reference frame of NeRF. Our approach can also handle symmetric objects unlike other approaches using a similar training setup. Specifically, we learn viewpoint invariant, discriminative features using NeRF which are later used for pose estimation. We evaluated our approach on LM, LM-Occlusion, and T-Less dataset and achieved benchmark accuracy despite using weakly labeled data.  
  </ol>  
</details>  
**comments**: 3DV 2024  
  
### [Style-NeRF2NeRF: 3D Style Transfer From Style-Aligned Multi-View Images](http://arxiv.org/abs/2406.13393)  
Haruo Fujiwara, Yusuke Mukuta, Tatsuya Harada  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose a simple yet effective pipeline for stylizing a 3D scene, harnessing the power of 2D image diffusion models. Given a NeRF model reconstructed from a set of multi-view images, we perform 3D style transfer by refining the source NeRF model using stylized images generated by a style-aligned image-to-image diffusion model. Given a target style prompt, we first generate perceptually similar multi-view images by leveraging a depth-conditioned diffusion model with an attention-sharing mechanism. Next, based on the stylized multi-view images, we propose to guide the style transfer process with the sliced Wasserstein loss based on the feature maps extracted from a pre-trained CNN model. Our pipeline consists of decoupled steps, allowing users to test various prompt ideas and preview the stylized 3D result before proceeding to the NeRF fine-tuning stage. We demonstrate that our method can transfer diverse artistic styles to real-world 3D scenes with competitive quality.  
  </ol>  
</details>  
**comments**: 16 pages, 9 figures  
  
### [Freq-Mip-AA : Frequency Mip Representation for Anti-Aliasing Neural Radiance Fields](http://arxiv.org/abs/2406.13251)  
[[code](https://github.com/yi0109/freqmipaa)]  
Youngin Park, Seungtae Nam, Cheul-hee Hahm, Eunbyung Park  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) have shown remarkable success in representing 3D scenes and generating novel views. However, they often struggle with aliasing artifacts, especially when rendering images from different camera distances from the training views. To address the issue, Mip-NeRF proposed using volumetric frustums to render a pixel and suggested integrated positional encoding (IPE). While effective, this approach requires long training times due to its reliance on MLP architecture. In this work, we propose a novel anti-aliasing technique that utilizes grid-based representations, usually showing significantly faster training time. In addition, we exploit frequency-domain representation to handle the aliasing problem inspired by the sampling theorem. The proposed method, FreqMipAA, utilizes scale-specific low-pass filtering (LPF) and learnable frequency masks. Scale-specific low-pass filters (LPF) prevent aliasing and prioritize important image details, and learnable masks effectively remove problematic high-frequency elements while retaining essential information. By employing a scale-specific LPF and trainable masks, FreqMipAA can effectively eliminate the aliasing factor while retaining important details. We validated the proposed technique by incorporating it into a widely used grid-based method. The experimental results have shown that the FreqMipAA effectively resolved the aliasing issues and achieved state-of-the-art results in the multi-scale Blender dataset. Our code is available at https://github.com/yi0109/FreqMipAA .  
  </ol>  
</details>  
**comments**: Accepted to ICIP 2024, 7 pages, 3 figures  
  
### [Sampling 3D Gaussian Scenes in Seconds with Latent Diffusion Models](http://arxiv.org/abs/2406.13099)  
Paul Henderson, Melonie de Almeida, Daniela Ivanova, Titas Anciukevičius  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present a latent diffusion model over 3D scenes, that can be trained using only 2D image data. To achieve this, we first design an autoencoder that maps multi-view images to 3D Gaussian splats, and simultaneously builds a compressed latent representation of these splats. Then, we train a multi-view diffusion model over the latent space to learn an efficient generative model. This pipeline does not require object masks nor depths, and is suitable for complex scenes with arbitrary camera positions. We conduct careful experiments on two large-scale datasets of complex real-world scenes -- MVImgNet and RealEstate10K. We show that our approach enables generating 3D scenes in as little as 0.2 seconds, either from scratch, from a single input view, or from sparse input views. It produces diverse and high-quality results while running an order of magnitude faster than non-latent diffusion models and earlier NeRF-based generative models  
  </ol>  
</details>  
  
### [Head Pose Estimation and 3D Neural Surface Reconstruction via Monocular Camera in situ for Navigation and Safe Insertion into Natural Openings](http://arxiv.org/abs/2406.13048)  
Ruijie Tang, Beilei Cui, Hongliang Ren  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    As the significance of simulation in medical care and intervention continues to grow, it is anticipated that a simplified and low-cost platform can be set up to execute personalized diagnoses and treatments. 3D Slicer can not only perform medical image analysis and visualization but can also provide surgical navigation and surgical planning functions. In this paper, we have chosen 3D Slicer as our base platform and monocular cameras are used as sensors. Then, We used the neural radiance fields (NeRF) algorithm to complete the 3D model reconstruction of the human head. We compared the accuracy of the NeRF algorithm in generating 3D human head scenes and utilized the MarchingCube algorithm to generate corresponding 3D mesh models. The individual's head pose, obtained through single-camera vision, is transmitted in real-time to the scene created within 3D Slicer. The demonstrations presented in this paper include real-time synchronization of transformations between the human head model in the 3D Slicer scene and the detected head posture. Additionally, we tested a scene where a tool, marked with an ArUco Maker tracked by a single camera, synchronously points to the real-time transformation of the head posture. These demos indicate that our methodology can provide a feasible real-time simulation platform for nasopharyngeal swab collection or intubation.  
  </ol>  
</details>  
**comments**: Accepted by ICBIR 2024  
  
  



