<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Camera-Motion-Estimation-from-RGB-D-Inertial-Scene-Flow>Camera Motion Estimation from RGB-D-Inertial Scene Flow</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Learning-text-to-video-retrieval-from-image-captioning>Learning text-to-video retrieval from image captioning</a></li>
        <li><a href=#CriSp:-Leveraging-Tread-Depth-Maps-for-Enhanced-Crime-Scene-Shoeprint-Matching>CriSp: Leveraging Tread Depth Maps for Enhanced Crime-Scene Shoeprint Matching</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Geometry-aware-Reconstruction-and-Fusion-refined-Rendering-for-Generalizable-Neural-Radiance-Fields>Geometry-aware Reconstruction and Fusion-refined Rendering for Generalizable Neural Radiance Fields</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Camera Motion Estimation from RGB-D-Inertial Scene Flow](http://arxiv.org/abs/2404.17251)  
Samuel Cerezo, Javier Civera  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we introduce a novel formulation for camera motion estimation that integrates RGB-D images and inertial data through scene flow. Our goal is to accurately estimate the camera motion in a rigid 3D environment, along with the state of the inertial measurement unit (IMU). Our proposed method offers the flexibility to operate as a multi-frame optimization or to marginalize older data, thus effectively utilizing past measurements. To assess the performance of our method, we conducted evaluations using both synthetic data from the ICL-NUIM dataset and real data sequences from the OpenLORIS-Scene dataset. Our results show that the fusion of these two sensors enhances the accuracy of camera motion estimation when compared to using only visual data.  
  </ol>  
</details>  
**comments**: Accepted to CVPR2024 Workshop on Visual Odometry and Computer Vision
  Applications  
  
  



## Visual Localization  

### [Learning text-to-video retrieval from image captioning](http://arxiv.org/abs/2404.17498)  
Lucas Ventura, Cordelia Schmid, Gül Varol  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We describe a protocol to study text-to-video retrieval training with unlabeled videos, where we assume (i) no access to labels for any videos, i.e., no access to the set of ground-truth captions, but (ii) access to labeled images in the form of text. Using image expert models is a realistic scenario given that annotating images is cheaper therefore scalable, in contrast to expensive video labeling schemes. Recently, zero-shot image experts such as CLIP have established a new strong baseline for video understanding tasks. In this paper, we make use of this progress and instantiate the image experts from two types of models: a text-to-image retrieval model to provide an initial backbone, and image captioning models to provide supervision signal into unlabeled videos. We show that automatically labeling video frames with image captioning allows text-to-video retrieval training. This process adapts the features to the target domain at no manual annotation cost, consequently outperforming the strong zero-shot CLIP baseline. During training, we sample captions from multiple video frames that best match the visual content, and perform a temporal pooling over frame representations by scoring frames according to their relevance to each caption. We conduct extensive ablations to provide insights and demonstrate the effectiveness of this simple framework by outperforming the CLIP zero-shot baselines on text-to-video retrieval on three standard datasets, namely ActivityNet, MSR-VTT, and MSVD.  
  </ol>  
</details>  
**comments**: A short version of this work appeared at CVPR 2023 Workshops. Project
  page: https://imagine.enpc.fr/~ventural/multicaps/  
  
### [CriSp: Leveraging Tread Depth Maps for Enhanced Crime-Scene Shoeprint Matching](http://arxiv.org/abs/2404.16972)  
Samia Shafique, Shu Kong, Charless Fowlkes  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Shoeprints are a common type of evidence found at crime scenes and are used regularly in forensic investigations. However, existing methods cannot effectively employ deep learning techniques to match noisy and occluded crime-scene shoeprints to a shoe database due to a lack of training data. Moreover, all existing methods match crime-scene shoeprints to clean reference prints, yet our analysis shows matching to more informative tread depth maps yields better retrieval results. The matching task is further complicated by the necessity to identify similarities only in corresponding regions (heels, toes, etc) of prints and shoe treads. To overcome these challenges, we leverage shoe tread images from online retailers and utilize an off-the-shelf predictor to estimate depth maps and clean prints. Our method, named CriSp, matches crime-scene shoeprints to tread depth maps by training on this data. CriSp incorporates data augmentation to simulate crime-scene shoeprints, an encoder to learn spatially-aware features, and a masking module to ensure only visible regions of crime-scene prints affect retrieval results. To validate our approach, we introduce two validation sets by reprocessing existing datasets of crime-scene shoeprints and establish a benchmarking protocol for comparison. On this benchmark, CriSp significantly outperforms state-of-the-art methods in both automated shoeprint matching and image retrieval tailored to this task.  
  </ol>  
</details>  
  
  



## NeRF  

### [Geometry-aware Reconstruction and Fusion-refined Rendering for Generalizable Neural Radiance Fields](http://arxiv.org/abs/2404.17528)  
Tianqi Liu, Xinyi Ye, Min Shi, Zihao Huang, Zhiyu Pan, Zhan Peng, Zhiguo Cao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Generalizable NeRF aims to synthesize novel views for unseen scenes. Common practices involve constructing variance-based cost volumes for geometry reconstruction and encoding 3D descriptors for decoding novel views. However, existing methods show limited generalization ability in challenging conditions due to inaccurate geometry, sub-optimal descriptors, and decoding strategies. We address these issues point by point. First, we find the variance-based cost volume exhibits failure patterns as the features of pixels corresponding to the same point can be inconsistent across different views due to occlusions or reflections. We introduce an Adaptive Cost Aggregation (ACA) approach to amplify the contribution of consistent pixel pairs and suppress inconsistent ones. Unlike previous methods that solely fuse 2D features into descriptors, our approach introduces a Spatial-View Aggregator (SVA) to incorporate 3D context into descriptors through spatial and inter-view interaction. When decoding the descriptors, we observe the two existing decoding strategies excel in different areas, which are complementary. A Consistency-Aware Fusion (CAF) strategy is proposed to leverage the advantages of both. We incorporate the above ACA, SVA, and CAF into a coarse-to-fine framework, termed Geometry-aware Reconstruction and Fusion-refined Rendering (GeFu). GeFu attains state-of-the-art performance across multiple datasets. Code is available at https://github.com/TQTQliu/GeFu .  
  </ol>  
</details>  
**comments**: Accepted by CVPR 2024. Project page: https://gefucvpr24.github.io  
  
  



