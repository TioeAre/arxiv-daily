<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Content-Based-Image-Retrieval-for-Multi-Class-Volumetric-Radiology-Images:-A-Benchmark-Study>Content-Based Image Retrieval for Multi-Class Volumetric Radiology Images: A Benchmark Study</a></li>
        <li><a href=#BEVRender:-Vision-based-Cross-view-Vehicle-Registration-in-Off-road-GNSS-denied-Environment>BEVRender: Vision-based Cross-view Vehicle Registration in Off-road GNSS-denied Environment</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Content-Based Image Retrieval for Multi-Class Volumetric Radiology Images: A Benchmark Study](http://arxiv.org/abs/2405.09334)  
Farnaz Khun Jush, Steffen Vogler, Tuan Truong, Matthias Lenga  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    While content-based image retrieval (CBIR) has been extensively studied in natural image retrieval, its application to medical images presents ongoing challenges, primarily due to the 3D nature of medical images. Recent studies have shown the potential use of pre-trained vision embeddings for CBIR in the context of radiology image retrieval. However, a benchmark for the retrieval of 3D volumetric medical images is still lacking, hindering the ability to objectively evaluate and compare the efficiency of proposed CBIR approaches in medical imaging. In this study, we extend previous work and establish a benchmark for region-based and multi-organ retrieval using the TotalSegmentator dataset (TS) with detailed multi-organ annotations. We benchmark embeddings derived from pre-trained supervised models on medical images against embeddings derived from pre-trained unsupervised models on non-medical images for 29 coarse and 104 detailed anatomical structures in volume and region levels. We adopt a late interaction re-ranking method inspired by text matching for image retrieval, comparing it against the original method proposed for volume and region retrieval achieving retrieval recall of 1.0 for diverse anatomical regions with a wide size range. The findings and methodologies presented in this paper provide essential insights and benchmarks for the development and evaluation of CBIR approaches in the context of medical imaging.  
  </ol>  
</details>  
**comments**: 23 pages, 9 Figures, 13 Tables  
  
### [BEVRender: Vision-based Cross-view Vehicle Registration in Off-road GNSS-denied Environment](http://arxiv.org/abs/2405.09001)  
Lihong Jin, Wei Dong, Michael Kaess  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce BEVRender, a novel learning-based approach for the localization of ground vehicles in Global Navigation Satellite System (GNSS)-denied off-road scenarios. These environments are typically challenging for conventional vision-based state estimation due to the lack of distinct visual landmarks and the instability of vehicle poses. To address this, BEVRender generates high-quality local bird's eye view (BEV) images of the local terrain. Subsequently, these images are aligned with a geo-referenced aerial map via template-matching to achieve accurate cross-view registration. Our approach overcomes the inherent limitations of visual inertial odometry systems and the substantial storage requirements of image-retrieval localization strategies, which are susceptible to drift and scalability issues, respectively. Extensive experimentation validates BEVRender's advancement over existing GNSS-denied visual localization methods, demonstrating notable enhancements in both localization accuracy and update frequency. The code for BEVRender will be made available soon.  
  </ol>  
</details>  
**comments**: 8 pages, 6 figures  
  
  



