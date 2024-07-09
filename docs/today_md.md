<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Enhancing-Neural-Radiance-Fields-with-Depth-and-Normal-Completion-Priors-from-Sparse-Views>Enhancing Neural Radiance Fields with Depth and Normal Completion Priors from Sparse Views</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Pseudo-triplet-Guided-Few-shot-Composed-Image-Retrieval>Pseudo-triplet Guided Few-shot Composed Image Retrieval</a></li>
        <li><a href=#HyCIR:-Boosting-Zero-Shot-Composed-Image-Retrieval-with-Synthetic-Labels>HyCIR: Boosting Zero-Shot Composed Image Retrieval with Synthetic Labels</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#PanDORA:-Casual-HDR-Radiance-Acquisition-for-Indoor-Scenes>PanDORA: Casual HDR Radiance Acquisition for Indoor Scenes</a></li>
        <li><a href=#Enhancing-Neural-Radiance-Fields-with-Depth-and-Normal-Completion-Priors-from-Sparse-Views>Enhancing Neural Radiance Fields with Depth and Normal Completion Priors from Sparse Views</a></li>
        <li><a href=#GeoNLF:-Geometry-guided-Pose-Free-Neural-LiDAR-Fields>GeoNLF: Geometry guided Pose-Free Neural LiDAR Fields</a></li>
        <li><a href=#Dynamic-Neural-Radiance-Field-From-Defocused-Monocular-Video>Dynamic Neural Radiance Field From Defocused Monocular Video</a></li>
        <li><a href=#GaussReg:-Fast-3D-Registration-with-Gaussian-Splatting>GaussReg: Fast 3D Registration with Gaussian Splatting</a></li>
        <li><a href=#SurgicalGaussian:-Deformable-3D-Gaussians-for-High-Fidelity-Surgical-Scene-Reconstruction>SurgicalGaussian: Deformable 3D Gaussians for High-Fidelity Surgical Scene Reconstruction</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Enhancing Neural Radiance Fields with Depth and Normal Completion Priors from Sparse Views](http://arxiv.org/abs/2407.05666)  
Jiawei Guo, HungChyun Chou, Ning Ding  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) are an advanced technology that creates highly realistic images by learning about scenes through a neural network model. However, NeRF often encounters issues when there are not enough images to work with, leading to problems in accurately rendering views. The main issue is that NeRF lacks sufficient structural details to guide the rendering process accurately. To address this, we proposed a Depth and Normal Dense Completion Priors for NeRF (CP\_NeRF) framework. This framework enhances view rendering by adding depth and normal dense completion priors to the NeRF optimization process. Before optimizing NeRF, we obtain sparse depth maps using the Structure from Motion (SfM) technique used to get camera poses. Based on the sparse depth maps and a normal estimator, we generate sparse normal maps for training a normal completion prior with precise standard deviations. During optimization, we apply depth and normal completion priors to transform sparse data into dense depth and normal maps with their standard deviations. We use these dense maps to guide ray sampling, assist distance sampling and construct a normal loss function for better training accuracy. To improve the rendering of NeRF's normal outputs, we incorporate an optical centre position embedder that helps synthesize more accurate normals through volume rendering. Additionally, we employ a normal patch matching technique to choose accurate rendered normal maps, ensuring more precise supervision for the model. Our method is superior to leading techniques in rendering detailed indoor scenes, even with limited input views.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Pseudo-triplet Guided Few-shot Composed Image Retrieval](http://arxiv.org/abs/2407.06001)  
Bohan Hou, Haoqiang Lin, Haokun Wen, Meng Liu, Xuemeng Song  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed Image Retrieval (CIR) is a challenging task that aims to retrieve the target image based on a multimodal query, i.e., a reference image and its corresponding modification text. While previous supervised or zero-shot learning paradigms all fail to strike a good trade-off between time-consuming annotation cost and retrieval performance, recent researchers introduced the task of few-shot CIR (FS-CIR) and proposed a textual inversion-based network based on pretrained CLIP model to realize it. Despite its promising performance, the approach suffers from two key limitations: insufficient multimodal query composition training and indiscriminative training triplet selection. To address these two limitations, in this work, we propose a novel two-stage pseudo triplet guided few-shot CIR scheme, dubbed PTG-FSCIR. In the first stage, we employ a masked training strategy and advanced image caption generator to construct pseudo triplets from pure image data to enable the model to acquire primary knowledge related to multimodal query composition. In the second stage, based on active learning, we design a pseudo modification text-based query-target distance metric to evaluate the challenging score for each unlabeled sample. Meanwhile, we propose a robust top range-based random sampling strategy according to the 3- $\sigma$ rule in statistics, to sample the challenging samples for fine-tuning the pretrained model. Notably, our scheme is plug-and-play and compatible with any existing supervised CIR models. We tested our scheme across three backbones on three public datasets (i.e., FashionIQ, CIRR, and Birds-to-Words), achieving maximum improvements of 26.4%, 25.5% and 21.6% respectively, demonstrating our scheme's effectiveness.  
  </ol>  
</details>  
**comments**: 15 pages, 5 figures,  
  
### [HyCIR: Boosting Zero-Shot Composed Image Retrieval with Synthetic Labels](http://arxiv.org/abs/2407.05795)  
Yingying Jiang, Hanchao Jia, Xiaobing Wang, Peng Hao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed Image Retrieval (CIR) aims to retrieve images based on a query image with text. Current Zero-Shot CIR (ZS-CIR) methods try to solve CIR tasks without using expensive triplet-labeled training datasets. However, the gap between ZS-CIR and triplet-supervised CIR is still large. In this work, we propose Hybrid CIR (HyCIR), which uses synthetic labels to boost the performance of ZS-CIR. A new label Synthesis pipeline for CIR (SynCir) is proposed, in which only unlabeled images are required. First, image pairs are extracted based on visual similarity. Second, query text is generated for each image pair based on vision-language model and LLM. Third, the data is further filtered in language space based on semantic similarity. To improve ZS-CIR performance, we propose a hybrid training strategy to work with both ZS-CIR supervision and synthetic CIR triplets. Two kinds of contrastive learning are adopted. One is to use large-scale unlabeled image dataset to learn an image-to-text mapping with good generalization. The other is to use synthetic CIR triplets to learn a better mapping for CIR tasks. Our approach achieves SOTA zero-shot performance on the common CIR benchmarks: CIRR and CIRCO.  
  </ol>  
</details>  
**comments**: 8 pages, 5 figures  
  
  



## NeRF  

### [PanDORA: Casual HDR Radiance Acquisition for Indoor Scenes](http://arxiv.org/abs/2407.06150)  
Mohammad Reza Karimi Dastjerdi, Frédéric Fortier-Chouinard, Yannick Hold-Geoffroy, Marc Hébert, Claude Demers, Nima Kalantari, Jean-François Lalonde  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Most novel view synthesis methods such as NeRF are unable to capture the true high dynamic range (HDR) radiance of scenes since they are typically trained on photos captured with standard low dynamic range (LDR) cameras. While the traditional exposure bracketing approach which captures several images at different exposures has recently been adapted to the multi-view case, we find such methods to fall short of capturing the full dynamic range of indoor scenes, which includes very bright light sources. In this paper, we present PanDORA: a PANoramic Dual-Observer Radiance Acquisition system for the casual capture of indoor scenes in high dynamic range. Our proposed system comprises two 360{\deg} cameras rigidly attached to a portable tripod. The cameras simultaneously acquire two 360{\deg} videos: one at a regular exposure and the other at a very fast exposure, allowing a user to simply wave the apparatus casually around the scene in a matter of minutes. The resulting images are fed to a NeRF-based algorithm that reconstructs the scene's full high dynamic range. Compared to HDR baselines from previous work, our approach reconstructs the full HDR radiance of indoor scenes without sacrificing the visual quality while retaining the ease of capture from recent NeRF-like approaches.  
  </ol>  
</details>  
**comments**: 10 pages, 8 figures  
  
### [Enhancing Neural Radiance Fields with Depth and Normal Completion Priors from Sparse Views](http://arxiv.org/abs/2407.05666)  
Jiawei Guo, HungChyun Chou, Ning Ding  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) are an advanced technology that creates highly realistic images by learning about scenes through a neural network model. However, NeRF often encounters issues when there are not enough images to work with, leading to problems in accurately rendering views. The main issue is that NeRF lacks sufficient structural details to guide the rendering process accurately. To address this, we proposed a Depth and Normal Dense Completion Priors for NeRF (CP\_NeRF) framework. This framework enhances view rendering by adding depth and normal dense completion priors to the NeRF optimization process. Before optimizing NeRF, we obtain sparse depth maps using the Structure from Motion (SfM) technique used to get camera poses. Based on the sparse depth maps and a normal estimator, we generate sparse normal maps for training a normal completion prior with precise standard deviations. During optimization, we apply depth and normal completion priors to transform sparse data into dense depth and normal maps with their standard deviations. We use these dense maps to guide ray sampling, assist distance sampling and construct a normal loss function for better training accuracy. To improve the rendering of NeRF's normal outputs, we incorporate an optical centre position embedder that helps synthesize more accurate normals through volume rendering. Additionally, we employ a normal patch matching technique to choose accurate rendered normal maps, ensuring more precise supervision for the model. Our method is superior to leading techniques in rendering detailed indoor scenes, even with limited input views.  
  </ol>  
</details>  
  
### [GeoNLF: Geometry guided Pose-Free Neural LiDAR Fields](http://arxiv.org/abs/2407.05597)  
Weiyi Xue, Zehan Zheng, Fan Lu, Haiyun Wei, Guang Chen, Changjun Jiang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Although recent efforts have extended Neural Radiance Fields (NeRF) into LiDAR point cloud synthesis, the majority of existing works exhibit a strong dependence on precomputed poses. However, point cloud registration methods struggle to achieve precise global pose estimation, whereas previous pose-free NeRFs overlook geometric consistency in global reconstruction. In light of this, we explore the geometric insights of point clouds, which provide explicit registration priors for reconstruction. Based on this, we propose Geometry guided Neural LiDAR Fields(GeoNLF), a hybrid framework performing alternately global neural reconstruction and pure geometric pose optimization. Furthermore, NeRFs tend to overfit individual frames and easily get stuck in local minima under sparse-view inputs. To tackle this issue, we develop a selective-reweighting strategy and introduce geometric constraints for robust optimization. Extensive experiments on NuScenes and KITTI-360 datasets demonstrate the superiority of GeoNLF in both novel view synthesis and multi-view registration of low-frequency large-scale point clouds.  
  </ol>  
</details>  
  
### [Dynamic Neural Radiance Field From Defocused Monocular Video](http://arxiv.org/abs/2407.05586)  
Xianrui Luo, Huiqiang Sun, Juewen Peng, Zhiguo Cao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Dynamic Neural Radiance Field (NeRF) from monocular videos has recently been explored for space-time novel view synthesis and achieved excellent results. However, defocus blur caused by depth variation often occurs in video capture, compromising the quality of dynamic reconstruction because the lack of sharp details interferes with modeling temporal consistency between input views. To tackle this issue, we propose D2RF, the first dynamic NeRF method designed to restore sharp novel views from defocused monocular videos. We introduce layered Depth-of-Field (DoF) volume rendering to model the defocus blur and reconstruct a sharp NeRF supervised by defocused views. The blur model is inspired by the connection between DoF rendering and volume rendering. The opacity in volume rendering aligns with the layer visibility in DoF rendering.To execute the blurring, we modify the layered blur kernel to the ray-based kernel and employ an optimized sparse kernel to gather the input rays efficiently and render the optimized rays with our layered DoF volume rendering. We synthesize a dataset with defocused dynamic scenes for our task, and extensive experiments on our dataset show that our method outperforms existing approaches in synthesizing all-in-focus novel views from defocus blur while maintaining spatial-temporal consistency in the scene.  
  </ol>  
</details>  
**comments**: Accepted by ECCV 2024  
  
### [GaussReg: Fast 3D Registration with Gaussian Splatting](http://arxiv.org/abs/2407.05254)  
Jiahao Chang, Yinglin Xu, Yihao Li, Yuantao Chen, Xiaoguang Han  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Point cloud registration is a fundamental problem for large-scale 3D scene scanning and reconstruction. With the help of deep learning, registration methods have evolved significantly, reaching a nearly-mature stage. As the introduction of Neural Radiance Fields (NeRF), it has become the most popular 3D scene representation as its powerful view synthesis capabilities. Regarding NeRF representation, its registration is also required for large-scale scene reconstruction. However, this topic extremly lacks exploration. This is due to the inherent challenge to model the geometric relationship among two scenes with implicit representations. The existing methods usually convert the implicit representation to explicit representation for further registration. Most recently, Gaussian Splatting (GS) is introduced, employing explicit 3D Gaussian. This method significantly enhances rendering speed while maintaining high rendering quality. Given two scenes with explicit GS representations, in this work, we explore the 3D registration task between them. To this end, we propose GaussReg, a novel coarse-to-fine framework, both fast and accurate. The coarse stage follows existing point cloud registration methods and estimates a rough alignment for point clouds from GS. We further newly present an image-guided fine registration approach, which renders images from GS to provide more detailed geometric information for precise alignment. To support comprehensive evaluation, we carefully build a scene-level dataset called ScanNet-GSReg with 1379 scenes obtained from the ScanNet dataset and collect an in-the-wild dataset called GSReg. Experimental results demonstrate our method achieves state-of-the-art performance on multiple datasets. Our GaussReg is 44 times faster than HLoc (SuperPoint as the feature extractor and SuperGlue as the matcher) with comparable accuracy.  
  </ol>  
</details>  
**comments**: ECCV 2024  
  
### [SurgicalGaussian: Deformable 3D Gaussians for High-Fidelity Surgical Scene Reconstruction](http://arxiv.org/abs/2407.05023)  
Weixing Xie, Junfeng Yao, Xianpeng Cao, Qiqin Lin, Zerui Tang, Xiao Dong, Xiaohu Guo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Dynamic reconstruction of deformable tissues in endoscopic video is a key technology for robot-assisted surgery. Recent reconstruction methods based on neural radiance fields (NeRFs) have achieved remarkable results in the reconstruction of surgical scenes. However, based on implicit representation, NeRFs struggle to capture the intricate details of objects in the scene and cannot achieve real-time rendering. In addition, restricted single view perception and occluded instruments also propose special challenges in surgical scene reconstruction. To address these issues, we develop SurgicalGaussian, a deformable 3D Gaussian Splatting method to model dynamic surgical scenes. Our approach models the spatio-temporal features of soft tissues at each time stamp via a forward-mapping deformation MLP and regularization to constrain local 3D Gaussians to comply with consistent movement. With the depth initialization strategy and tool mask-guided training, our method can remove surgical instruments and reconstruct high-fidelity surgical scenes. Through experiments on various surgical videos, our network outperforms existing method on many aspects, including rendering quality, rendering speed and GPU usage. The project page can be found at https://surgicalgaussian.github.io.  
  </ol>  
</details>  
  
  



