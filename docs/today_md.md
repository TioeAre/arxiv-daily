<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Implementation-of-a-Zed-2i-Stereo-Camera-for-High-Frequency-Shoreline-Change-and-Coastal-Elevation-Monitoring>Implementation of a Zed 2i Stereo Camera for High-Frequency Shoreline Change and Coastal Elevation Monitoring</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Patch-Matters:-Training-free-Fine-grained-Image-Caption-Enhancement-via-Local-Perception>Patch Matters: Training-free Fine-grained Image Caption Enhancement via Local Perception</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Image-registration-of-2D-optical-thin-sections-in-a-3D-porous-medium:-Application-to-a-Berea-sandstone-digital-rock-image>Image registration of 2D optical thin sections in a 3D porous medium: Application to a Berea sandstone digital rock image</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Wheat3DGS:-In-field-3D-Reconstruction,-Instance-Segmentation-and-Phenotyping-of-Wheat-Heads-with-Gaussian-Splatting>Wheat3DGS: In-field 3D Reconstruction, Instance Segmentation and Phenotyping of Wheat Heads with Gaussian Splatting</a></li>
        <li><a href=#S-EO:-A-Large-Scale-Dataset-for-Geometry-Aware-Shadow-Detection-in-Remote-Sensing-Applications>S-EO: A Large-Scale Dataset for Geometry-Aware Shadow Detection in Remote Sensing Applications</a></li>
        <li><a href=#SVG-IR:-Spatially-Varying-Gaussian-Splatting-for-Inverse-Rendering>SVG-IR: Spatially-Varying Gaussian Splatting for Inverse Rendering</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Implementation of a Zed 2i Stereo Camera for High-Frequency Shoreline Change and Coastal Elevation Monitoring](http://arxiv.org/abs/2504.06464)  
José A. Pilartes-Congo, Matthew Kastl, Michael J. Starek, Marina Vicens-Miquel, Philippe Tissot  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The increasing population, thus financial interests, in coastal areas have increased the need to monitor coastal elevation and shoreline change. Though several resources exist to obtain this information, they often lack the required temporal resolution for short-term monitoring (e.g., every hour). To address this issue, this study implements a low-cost ZED 2i stereo camera system and close-range photogrammetry to collect images for generating 3D point clouds, digital surface models (DSMs) of beach elevation, and georectified imagery at a localized scale and high temporal resolution. The main contributions of this study are (i) intrinsic camera calibration, (ii) georectification and registration of acquired imagery and point cloud, (iii) generation of the DSM of the beach elevation, and (iv) a comparison of derived products against those from uncrewed aircraft system structure-from-motion photogrammetry. Preliminary results show that despite its limitations, the ZED 2i can provide the desired mapping products at localized and high temporal scales. The system achieved a mean reprojection error of 0.20 px, a point cloud registration of 27 cm, a vertical error of 37.56 cm relative to ground truth, and georectification root mean square errors of 2.67 cm and 2.81 cm for x and y.  
  </ol>  
</details>  
**comments**: Published in IGARSS 2023 - 2023 IEEE International Geoscience and
  Remote Sensing Symposium  
  
  



## Visual Localization  

### [Patch Matters: Training-free Fine-grained Image Caption Enhancement via Local Perception](http://arxiv.org/abs/2504.06666)  
Ruotian Peng, Haiying He, Yake Wei, Yandong Wen, Di Hu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    High-quality image captions play a crucial role in improving the performance of cross-modal applications such as text-to-image generation, text-to-video generation, and text-image retrieval. To generate long-form, high-quality captions, many recent studies have employed multimodal large language models (MLLMs). However, current MLLMs often produce captions that lack fine-grained details or suffer from hallucinations, a challenge that persists in both open-source and closed-source models. Inspired by Feature-Integration theory, which suggests that attention must focus on specific regions to integrate visual information effectively, we propose a \textbf{divide-then-aggregate} strategy. Our method first divides the image into semantic and spatial patches to extract fine-grained details, enhancing the model's local perception of the image. These local details are then hierarchically aggregated to generate a comprehensive global description. To address hallucinations and inconsistencies in the generated captions, we apply a semantic-level filtering process during hierarchical aggregation. This training-free pipeline can be applied to both open-source models (LLaVA-1.5, LLaVA-1.6, Mini-Gemini) and closed-source models (Claude-3.5-Sonnet, GPT-4o, GLM-4V-Plus). Extensive experiments demonstrate that our method generates more detailed, reliable captions, advancing multimodal description generation without requiring model retraining. The source code are available at https://github.com/GeWu-Lab/Patch-Matters  
  </ol>  
</details>  
  
  



## Image Matching  

### [Image registration of 2D optical thin sections in a 3D porous medium: Application to a Berea sandstone digital rock image](http://arxiv.org/abs/2504.06604)  
Jaehong Chung, Wei Cai, Tapan Mukerji  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This study proposes a systematic image registration approach to align 2D optical thin-section images within a 3D digital rock volume. Using template image matching with differential evolution optimization, we identify the most similar 2D plane in 3D. The method is validated on a synthetic porous medium, achieving exact registration, and applied to Berea sandstone, where it achieves a structural similarity index (SSIM) of 0.990. With the registered images, we explore upscaling properties based on paired multimodal images, focusing on pore characteristics and effective elastic moduli. The thin-section image reveals 50 % more porosity and submicron pores than the registered CT plane. In addition, bulk and shear moduli from thin sections are 25 % and 30 % lower, respectively, than those derived from CT images. Beyond numerical comparisons, thin sections provide additional geological insights, including cementation, mineral phases, and weathering effects, which are not clear in CT images. This study demonstrates the potential of multimodal image registration to improve computed rock properties in digital rock physics by integrating complementary imaging modalities.  
  </ol>  
</details>  
  
  



## NeRF  

### [Wheat3DGS: In-field 3D Reconstruction, Instance Segmentation and Phenotyping of Wheat Heads with Gaussian Splatting](http://arxiv.org/abs/2504.06978)  
Daiwei Zhang, Joaquin Gajardo, Tomislav Medic, Isinsu Katircioglu, Mike Boss, Norbert Kirchgessner, Achim Walter, Lukas Roth  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Automated extraction of plant morphological traits is crucial for supporting crop breeding and agricultural management through high-throughput field phenotyping (HTFP). Solutions based on multi-view RGB images are attractive due to their scalability and affordability, enabling volumetric measurements that 2D approaches cannot directly capture. While advanced methods like Neural Radiance Fields (NeRFs) have shown promise, their application has been limited to counting or extracting traits from only a few plants or organs. Furthermore, accurately measuring complex structures like individual wheat heads-essential for studying crop yields-remains particularly challenging due to occlusions and the dense arrangement of crop canopies in field conditions. The recent development of 3D Gaussian Splatting (3DGS) offers a promising alternative for HTFP due to its high-quality reconstructions and explicit point-based representation. In this paper, we present Wheat3DGS, a novel approach that leverages 3DGS and the Segment Anything Model (SAM) for precise 3D instance segmentation and morphological measurement of hundreds of wheat heads automatically, representing the first application of 3DGS to HTFP. We validate the accuracy of wheat head extraction against high-resolution laser scan data, obtaining per-instance mean absolute percentage errors of 15.1%, 18.3%, and 40.2% for length, width, and volume. We provide additional comparisons to NeRF-based approaches and traditional Muti-View Stereo (MVS), demonstrating superior results. Our approach enables rapid, non-destructive measurements of key yield-related traits at scale, with significant implications for accelerating crop breeding and improving our understanding of wheat development.  
  </ol>  
</details>  
**comments**: Copyright 2025 IEEE. This is the author's version of the work. It is
  posted here for your personal use. Not for redistribution. The definitive
  version is published in the 2025 IEEE/CVF Conference on Computer Vision and
  Pattern Recognition Workshops (CVPRW)  
  
### [S-EO: A Large-Scale Dataset for Geometry-Aware Shadow Detection in Remote Sensing Applications](http://arxiv.org/abs/2504.06920)  
Masquil Elías, Marí Roger, Ehret Thibaud, Meinhardt-Llopis Enric, Musé Pablo, Facciolo Gabriele  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce the S-EO dataset: a large-scale, high-resolution dataset, designed to advance geometry-aware shadow detection. Collected from diverse public-domain sources, including challenge datasets and government providers such as USGS, our dataset comprises 702 georeferenced tiles across the USA, each covering 500x500 m. Each tile includes multi-date, multi-angle WorldView-3 pansharpened RGB images, panchromatic images, and a ground-truth DSM of the area obtained from LiDAR scans. For each image, we provide a shadow mask derived from geometry and sun position, a vegetation mask based on the NDVI index, and a bundle-adjusted RPC model. With approximately 20,000 images, the S-EO dataset establishes a new public resource for shadow detection in remote sensing imagery and its applications to 3D reconstruction. To demonstrate the dataset's impact, we train and evaluate a shadow detector, showcasing its ability to generalize, even to aerial images. Finally, we extend EO-NeRF - a state-of-the-art NeRF approach for satellite imagery - to leverage our shadow predictions for improved 3D reconstructions.  
  </ol>  
</details>  
**comments**: Accepted at Earthvision 2025 (CVPR Workshop)  
  
### [SVG-IR: Spatially-Varying Gaussian Splatting for Inverse Rendering](http://arxiv.org/abs/2504.06815)  
Hanxiao Sun, YuPeng Gao, Jin Xie, Jian Yang, Beibei Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Reconstructing 3D assets from images, known as inverse rendering (IR), remains a challenging task due to its ill-posed nature. 3D Gaussian Splatting (3DGS) has demonstrated impressive capabilities for novel view synthesis (NVS) tasks. Methods apply it to relighting by separating radiance into BRDF parameters and lighting, yet produce inferior relighting quality with artifacts and unnatural indirect illumination due to the limited capability of each Gaussian, which has constant material parameters and normal, alongside the absence of physical constraints for indirect lighting. In this paper, we present a novel framework called Spatially-vayring Gaussian Inverse Rendering (SVG-IR), aimed at enhancing both NVS and relighting quality. To this end, we propose a new representation-Spatially-varying Gaussian (SVG)-that allows per-Gaussian spatially varying parameters. This enhanced representation is complemented by a SVG splatting scheme akin to vertex/fragment shading in traditional graphics pipelines. Furthermore, we integrate a physically-based indirect lighting model, enabling more realistic relighting. The proposed SVG-IR framework significantly improves rendering quality, outperforming state-of-the-art NeRF-based methods by 2.5 dB in peak signal-to-noise ratio (PSNR) and surpassing existing Gaussian-based techniques by 3.5 dB in relighting tasks, all while maintaining a real-time rendering speed.  
  </ol>  
</details>  
  
  



