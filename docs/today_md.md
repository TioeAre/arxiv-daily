<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#To-Glue-or-Not-to-Glue?-Classical-vs-Learned-Image-Matching-for-Mobile-Mapping-Cameras-to-Textured-Semantic-3D-Building-Models>To Glue or Not to Glue? Classical vs Learned Image Matching for Mobile Mapping Cameras to Textured Semantic 3D Building Models</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#To-Glue-or-Not-to-Glue?-Classical-vs-Learned-Image-Matching-for-Mobile-Mapping-Cameras-to-Textured-Semantic-3D-Building-Models>To Glue or Not to Glue? Classical vs Learned Image Matching for Mobile Mapping Cameras to Textured Semantic 3D Building Models</a></li>
        <li><a href=#DetailFusion:-A-Dual-branch-Framework-with-Detail-Enhancement-for-Composed-Image-Retrieval>DetailFusion: A Dual-branch Framework with Detail Enhancement for Composed Image Retrieval</a></li>
        <li><a href=#CU-Multi:-A-Dataset-for-Multi-Robot-Data-Association>CU-Multi: A Dataset for Multi-Robot Data Association</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#To-Glue-or-Not-to-Glue?-Classical-vs-Learned-Image-Matching-for-Mobile-Mapping-Cameras-to-Textured-Semantic-3D-Building-Models>To Glue or Not to Glue? Classical vs Learned Image Matching for Mobile Mapping Cameras to Textured Semantic 3D Building Models</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [To Glue or Not to Glue? Classical vs Learned Image Matching for Mobile Mapping Cameras to Textured Semantic 3D Building Models](http://arxiv.org/abs/2505.17973)  
Simone Gaisbauer, Prabin Gyawali, Qilin Zhang, Olaf Wysocki, Boris Jutzi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Feature matching is a necessary step for many computer vision and photogrammetry applications such as image registration, structure-from-motion, and visual localization. Classical handcrafted methods such as SIFT feature detection and description combined with nearest neighbour matching and RANSAC outlier removal have been state-of-the-art for mobile mapping cameras. With recent advances in deep learning, learnable methods have been introduced and proven to have better robustness and performance under complex conditions. Despite their growing adoption, a comprehensive comparison between classical and learnable feature matching methods for the specific task of semantic 3D building camera-to-model matching is still missing. This submission systematically evaluates the effectiveness of different feature-matching techniques in visual localization using textured CityGML LoD2 models. We use standard benchmark datasets (HPatches, MegaDepth-1500) and custom datasets consisting of facade textures and corresponding camera images (terrestrial and drone). For the latter, we evaluate the achievable accuracy of the absolute pose estimated using a Perspective-n-Point (PnP) algorithm, with geometric ground truth derived from geo-referenced trajectory data. The results indicate that the learnable feature matching methods vastly outperform traditional approaches regarding accuracy and robustness on our challenging custom datasets with zero to 12 RANSAC-inliers and zero to 0.16 area under the curve. We believe that this work will foster the development of model-based visual localization methods. Link to the code: https://github.com/simBauer/To\_Glue\_or\_not\_to\_Glue  
  </ol>  
</details>  
**comments**: Accepted to MMT, Xiamen, China; ISPRS Annals  
  
  



## Visual Localization  

### [To Glue or Not to Glue? Classical vs Learned Image Matching for Mobile Mapping Cameras to Textured Semantic 3D Building Models](http://arxiv.org/abs/2505.17973)  
Simone Gaisbauer, Prabin Gyawali, Qilin Zhang, Olaf Wysocki, Boris Jutzi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Feature matching is a necessary step for many computer vision and photogrammetry applications such as image registration, structure-from-motion, and visual localization. Classical handcrafted methods such as SIFT feature detection and description combined with nearest neighbour matching and RANSAC outlier removal have been state-of-the-art for mobile mapping cameras. With recent advances in deep learning, learnable methods have been introduced and proven to have better robustness and performance under complex conditions. Despite their growing adoption, a comprehensive comparison between classical and learnable feature matching methods for the specific task of semantic 3D building camera-to-model matching is still missing. This submission systematically evaluates the effectiveness of different feature-matching techniques in visual localization using textured CityGML LoD2 models. We use standard benchmark datasets (HPatches, MegaDepth-1500) and custom datasets consisting of facade textures and corresponding camera images (terrestrial and drone). For the latter, we evaluate the achievable accuracy of the absolute pose estimated using a Perspective-n-Point (PnP) algorithm, with geometric ground truth derived from geo-referenced trajectory data. The results indicate that the learnable feature matching methods vastly outperform traditional approaches regarding accuracy and robustness on our challenging custom datasets with zero to 12 RANSAC-inliers and zero to 0.16 area under the curve. We believe that this work will foster the development of model-based visual localization methods. Link to the code: https://github.com/simBauer/To\_Glue\_or\_not\_to\_Glue  
  </ol>  
</details>  
**comments**: Accepted to MMT, Xiamen, China; ISPRS Annals  
  
### [DetailFusion: A Dual-branch Framework with Detail Enhancement for Composed Image Retrieval](http://arxiv.org/abs/2505.17796)  
Yuxin Yang, Yinan Zhou, Yuxin Chen, Ziqi Zhang, Zongyang Ma, Chunfeng Yuan, Bing Li, Lin Song, Jun Gao, Peng Li, Weiming Hu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed Image Retrieval (CIR) aims to retrieve target images from a gallery based on a reference image and modification text as a combined query. Recent approaches focus on balancing global information from two modalities and encode the query into a unified feature for retrieval. However, due to insufficient attention to fine-grained details, these coarse fusion methods often struggle with handling subtle visual alterations or intricate textual instructions. In this work, we propose DetailFusion, a novel dual-branch framework that effectively coordinates information across global and detailed granularities, thereby enabling detail-enhanced CIR. Our approach leverages atomic detail variation priors derived from an image editing dataset, supplemented by a detail-oriented optimization strategy to develop a Detail-oriented Inference Branch. Furthermore, we design an Adaptive Feature Compositor that dynamically fuses global and detailed features based on fine-grained information of each unique multimodal query. Extensive experiments and ablation analyses not only demonstrate that our method achieves state-of-the-art performance on both CIRR and FashionIQ datasets but also validate the effectiveness and cross-domain adaptability of detail enhancement for CIR.  
  </ol>  
</details>  
**comments**: 20 pages, 6 figures  
  
### [CU-Multi: A Dataset for Multi-Robot Data Association](http://arxiv.org/abs/2505.17576)  
Doncey Albin, Miles Mena, Annika Thomas, Harel Biggie, Xuefei Sun, Dusty Woods, Steve McGuire, Christoffer Heckman  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Multi-robot systems (MRSs) are valuable for tasks such as search and rescue due to their ability to coordinate over shared observations. A central challenge in these systems is aligning independently collected perception data across space and time, i.e., multi-robot data association. While recent advances in collaborative SLAM (C-SLAM), map merging, and inter-robot loop closure detection have significantly progressed the field, evaluation strategies still predominantly rely on splitting a single trajectory from single-robot SLAM datasets into multiple segments to simulate multiple robots. Without careful consideration to how a single trajectory is split, this approach will fail to capture realistic pose-dependent variation in observations of a scene inherent to multi-robot systems. To address this gap, we present CU-Multi, a multi-robot dataset collected over multiple days at two locations on the University of Colorado Boulder campus. Using a single robotic platform, we generate four synchronized runs with aligned start times and deliberate percentages of trajectory overlap. CU-Multi includes RGB-D, GPS with accurate geospatial heading, and semantically annotated LiDAR data. By introducing controlled variations in trajectory overlap and dense lidar annotations, CU-Multi offers a compelling alternative for evaluating methods in multi-robot data association. Instructions on accessing the dataset, support code, and the latest updates are publicly available at https://arpg.github.io/cumulti  
  </ol>  
</details>  
**comments**: 8 pages, 6 figures, 4 tables  
  
  



## Image Matching  

### [To Glue or Not to Glue? Classical vs Learned Image Matching for Mobile Mapping Cameras to Textured Semantic 3D Building Models](http://arxiv.org/abs/2505.17973)  
Simone Gaisbauer, Prabin Gyawali, Qilin Zhang, Olaf Wysocki, Boris Jutzi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Feature matching is a necessary step for many computer vision and photogrammetry applications such as image registration, structure-from-motion, and visual localization. Classical handcrafted methods such as SIFT feature detection and description combined with nearest neighbour matching and RANSAC outlier removal have been state-of-the-art for mobile mapping cameras. With recent advances in deep learning, learnable methods have been introduced and proven to have better robustness and performance under complex conditions. Despite their growing adoption, a comprehensive comparison between classical and learnable feature matching methods for the specific task of semantic 3D building camera-to-model matching is still missing. This submission systematically evaluates the effectiveness of different feature-matching techniques in visual localization using textured CityGML LoD2 models. We use standard benchmark datasets (HPatches, MegaDepth-1500) and custom datasets consisting of facade textures and corresponding camera images (terrestrial and drone). For the latter, we evaluate the achievable accuracy of the absolute pose estimated using a Perspective-n-Point (PnP) algorithm, with geometric ground truth derived from geo-referenced trajectory data. The results indicate that the learnable feature matching methods vastly outperform traditional approaches regarding accuracy and robustness on our challenging custom datasets with zero to 12 RANSAC-inliers and zero to 0.16 area under the curve. We believe that this work will foster the development of model-based visual localization methods. Link to the code: https://github.com/simBauer/To\_Glue\_or\_not\_to\_Glue  
  </ol>  
</details>  
**comments**: Accepted to MMT, Xiamen, China; ISPRS Annals  
  
  



