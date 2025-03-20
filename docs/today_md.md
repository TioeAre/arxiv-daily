<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#PAPI-Reg:-Patch-to-Pixel-Solution-for-Efficient-Cross-Modal-Registration-between-LiDAR-Point-Cloud-and-Camera-Image>PAPI-Reg: Patch-to-Pixel Solution for Efficient Cross-Modal Registration between LiDAR Point Cloud and Camera Image</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#GO-N3RDet:-Geometry-Optimized-NeRF-enhanced-3D-Object-Detector>GO-N3RDet: Geometry Optimized NeRF-enhanced 3D Object Detector</a></li>
        <li><a href=#MultiBARF:-Integrating-Imagery-of-Different-Wavelength-Regions-by-Using-Neural-Radiance-Fields>MultiBARF: Integrating Imagery of Different Wavelength Regions by Using Neural Radiance Fields</a></li>
        <li><a href=#3D-Engine-ready-Photorealistic-Avatars-via-Dynamic-Textures>3D Engine-ready Photorealistic Avatars via Dynamic Textures</a></li>
        <li><a href=#ClimateGS:-Real-Time-Climate-Simulation-with-3D-Gaussian-Style-Transfer>ClimateGS: Real-Time Climate Simulation with 3D Gaussian Style Transfer</a></li>
      </ul>
    </li>
  </ol>
</details>

## Image Matching  

### [PAPI-Reg: Patch-to-Pixel Solution for Efficient Cross-Modal Registration between LiDAR Point Cloud and Camera Image](http://arxiv.org/abs/2503.15285)  
Yuanchao Yue, Zhengxin Li, Wei Zhang, Hui Yuan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The primary requirement for cross-modal data fusion is the precise alignment of data from different sensors. However, the calibration between LiDAR point clouds and camera images is typically time-consuming and needs external calibration board or specific environmental features. Cross-modal registration effectively solves this problem by aligning the data directly without requiring external calibration. However, due to the domain gap between the point cloud and the image, existing methods rarely achieve satisfactory registration accuracy while maintaining real-time performance. To address this issue, we propose a framework that projects point clouds into several 2D representations for matching with camera images, which not only leverages the geometric characteristic of LiDAR point clouds more effectively but also bridge the domain gap between the point cloud and image. Moreover, to tackle the challenges of cross modal differences and the limited overlap between LiDAR point clouds and images in the image matching task, we introduce a multi-scale feature extraction network to effectively extract features from both camera images and the projection maps of LiDAR point cloud. Additionally, we propose a patch-to-pixel matching network to provide more effective supervision and achieve higher accuracy. We validate the performance of our model through experiments on the KITTI and nuScenes datasets. Our network achieves real-time performance and extremely high registration accuracy. On the KITTI dataset, our model achieves a registration accuracy rate of over 99\%.  
  </ol>  
</details>  
  
  



## NeRF  

### [GO-N3RDet: Geometry Optimized NeRF-enhanced 3D Object Detector](http://arxiv.org/abs/2503.15211)  
Zechuan Li, Hongshan Yu, Yihao Ding, Jinhao Qiao, Basim Azam, Naveed Akhtar  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose GO-N3RDet, a scene-geometry optimized multi-view 3D object detector enhanced by neural radiance fields. The key to accurate 3D object detection is in effective voxel representation. However, due to occlusion and lack of 3D information, constructing 3D features from multi-view 2D images is challenging. Addressing that, we introduce a unique 3D positional information embedded voxel optimization mechanism to fuse multi-view features. To prioritize neural field reconstruction in object regions, we also devise a double importance sampling scheme for the NeRF branch of our detector. We additionally propose an opacity optimization module for precise voxel opacity prediction by enforcing multi-view consistency constraints. Moreover, to further improve voxel density consistency across multiple perspectives, we incorporate ray distance as a weighting factor to minimize cumulative ray errors. Our unique modules synergetically form an end-to-end neural model that establishes new state-of-the-art in NeRF-based multi-view 3D detection, verified with extensive experiments on ScanNet and ARKITScenes. Code will be available at https://github.com/ZechuanLi/GO-N3RDet.  
  </ol>  
</details>  
**comments**: Accepted by CVPR2025  
  
### [MultiBARF: Integrating Imagery of Different Wavelength Regions by Using Neural Radiance Fields](http://arxiv.org/abs/2503.15070)  
Kana Kurata, Hitoshi Niigaki, Xiaojun Wu, Ryuichi Tanida  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Optical sensor applications have become popular through digital transformation. Linking observed data to real-world locations and combining different image sensors is essential to make the applications practical and efficient. However, data preparation to try different sensor combinations requires high sensing and image processing expertise. To make data preparation easier for users unfamiliar with sensing and image processing, we have developed MultiBARF. This method replaces the co-registration and geometric calibration by synthesizing pairs of two different sensor images and depth images at assigned viewpoints. Our method extends Bundle Adjusting Neural Radiance Fields(BARF), a deep neural network-based novel view synthesis method, for the two imagers. Through experiments on visible light and thermographic images, we demonstrate that our method superimposes two color channels of those sensor images on NeRF.  
  </ol>  
</details>  
  
### [3D Engine-ready Photorealistic Avatars via Dynamic Textures](http://arxiv.org/abs/2503.14943)  
Yifan Wang, Ivan Molodetskikh, Ondrej Texler, Dimitar Dinev  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    As the digital and physical worlds become more intertwined, there has been a lot of interest in digital avatars that closely resemble their real-world counterparts. Current digitization methods used in 3D production pipelines require costly capture setups, making them impractical for mass usage among common consumers. Recent academic literature has found success in reconstructing humans from limited data using implicit representations (e.g., voxels used in NeRFs), which are able to produce impressive videos. However, these methods are incompatible with traditional rendering pipelines, making it difficult to use them in applications such as games. In this work, we propose an end-to-end pipeline that builds explicitly-represented photorealistic 3D avatars using standard 3D assets. Our key idea is the use of dynamically-generated textures to enhance the realism and visually mask deficiencies in the underlying mesh geometry. This allows for seamless integration with current graphics pipelines while achieving comparable visual quality to state-of-the-art 3D avatar generation methods.  
  </ol>  
</details>  
  
### [ClimateGS: Real-Time Climate Simulation with 3D Gaussian Style Transfer](http://arxiv.org/abs/2503.14845)  
Yuezhen Xie, Meiying Zhang, Qi Hao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Adverse climate conditions pose significant challenges for autonomous systems, demanding reliable perception and decision-making across diverse environments. To better simulate these conditions, physically-based NeRF rendering methods have been explored for their ability to generate realistic scene representations. However, these methods suffer from slow rendering speeds and long preprocessing times, making them impractical for real-time testing and user interaction. This paper presents ClimateGS, a novel framework integrating 3D Gaussian representations with physical simulation to enable real-time climate effects rendering. The novelty of this work is threefold: 1) developing a linear transformation for 3D Gaussian photorealistic style transfer, enabling direct modification of spherical harmonics across bands for efficient and consistent style adaptation; 2) developing a joint training strategy for 3D style transfer, combining supervised and self-supervised learning to accelerate convergence while preserving original scene details; 3) developing a real-time rendering method for climate simulation, integrating physics-based effects with 3D Gaussian to achieve efficient and realistic rendering. We evaluate ClimateGS on MipNeRF360 and Tanks and Temples, demonstrating real-time rendering with comparable or superior visual quality to SOTA 2D/3D methods, making it suitable for interactive applications.  
  </ol>  
</details>  
  
  



