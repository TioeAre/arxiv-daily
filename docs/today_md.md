<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#IM360:-Textured-Mesh-Reconstruction-for-Large-scale-Indoor-Mapping-with-360$^\circ$-Cameras>IM360: Textured Mesh Reconstruction for Large-scale Indoor Mapping with 360$^\circ$ Cameras</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Re-Align:-Aligning-Vision-Language-Models-via-Retrieval-Augmented-Direct-Preference-Optimization>Re-Align: Aligning Vision Language Models via Retrieval-Augmented Direct Preference Optimization</a></li>
        <li><a href=#IM360:-Textured-Mesh-Reconstruction-for-Large-scale-Indoor-Mapping-with-360$^\circ$-Cameras>IM360: Textured Mesh Reconstruction for Large-scale Indoor Mapping with 360$^\circ$ Cameras</a></li>
        <li><a href=#From-Gaming-to-Research:-GTA-V-for-Synthetic-Data-Generation-for-Robotics-and-Navigations>From Gaming to Research: GTA V for Synthetic Data Generation for Robotics and Navigations</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#ROI-NeRFs:-Hi-Fi-Visualization-of-Objects-of-Interest-within-a-Scene-by-NeRFs-Composition>ROI-NeRFs: Hi-Fi Visualization of Objects of Interest within a Scene by NeRFs Composition</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [IM360: Textured Mesh Reconstruction for Large-scale Indoor Mapping with 360 $^\circ$ Cameras](http://arxiv.org/abs/2502.12545)  
Dongki Jung, Jaehoon Choi, Yonghan Lee, Dinesh Manocha  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present a novel 3D reconstruction pipeline for 360$^\circ$ cameras for 3D mapping and rendering of indoor environments. Traditional Structure-from-Motion (SfM) methods may not work well in large-scale indoor scenes due to the prevalence of textureless and repetitive regions. To overcome these challenges, our approach (IM360) leverages the wide field of view of omnidirectional images and integrates the spherical camera model into every core component of the SfM pipeline. In order to develop a comprehensive 3D reconstruction solution, we integrate a neural implicit surface reconstruction technique to generate high-quality surfaces from sparse input data. Additionally, we utilize a mesh-based neural rendering approach to refine texture maps and accurately capture view-dependent properties by combining diffuse and specular components. We evaluate our pipeline on large-scale indoor scenes from the Matterport3D and Stanford2D3D datasets. In practice, IM360 demonstrate superior performance in terms of textured mesh reconstruction over SOTA. We observe accuracy improvements in terms of camera localization and registration as well as rendering high frequency details.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Re-Align: Aligning Vision Language Models via Retrieval-Augmented Direct Preference Optimization](http://arxiv.org/abs/2502.13146)  
Shuo Xing, Yuping Wang, Peiran Li, Ruizheng Bai, Yueqi Wang, Chengxuan Qian, Huaxiu Yao, Zhengzhong Tu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The emergence of large Vision Language Models (VLMs) has broadened the scope and capabilities of single-modal Large Language Models (LLMs) by integrating visual modalities, thereby unlocking transformative cross-modal applications in a variety of real-world scenarios. Despite their impressive performance, VLMs are prone to significant hallucinations, particularly in the form of cross-modal inconsistencies. Building on the success of Reinforcement Learning from Human Feedback (RLHF) in aligning LLMs, recent advancements have focused on applying direct preference optimization (DPO) on carefully curated datasets to mitigate these issues. Yet, such approaches typically introduce preference signals in a brute-force manner, neglecting the crucial role of visual information in the alignment process. In this paper, we introduce Re-Align, a novel alignment framework that leverages image retrieval to construct a dual-preference dataset, effectively incorporating both textual and visual preference signals. We further introduce rDPO, an extension of the standard direct preference optimization that incorporates an additional visual preference objective during fine-tuning. Our experimental results demonstrate that Re-Align not only mitigates hallucinations more effectively than previous methods but also yields significant performance gains in general visual question-answering (VQA) tasks. Moreover, we show that Re-Align maintains robustness and scalability across a wide range of VLM sizes and architectures. This work represents a significant step forward in aligning multimodal LLMs, paving the way for more reliable and effective cross-modal applications. We release all the code in https://github.com/taco-group/Re-Align.  
  </ol>  
</details>  
**comments**: 15 pages  
  
### [IM360: Textured Mesh Reconstruction for Large-scale Indoor Mapping with 360 $^\circ$ Cameras](http://arxiv.org/abs/2502.12545)  
Dongki Jung, Jaehoon Choi, Yonghan Lee, Dinesh Manocha  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present a novel 3D reconstruction pipeline for 360$^\circ$ cameras for 3D mapping and rendering of indoor environments. Traditional Structure-from-Motion (SfM) methods may not work well in large-scale indoor scenes due to the prevalence of textureless and repetitive regions. To overcome these challenges, our approach (IM360) leverages the wide field of view of omnidirectional images and integrates the spherical camera model into every core component of the SfM pipeline. In order to develop a comprehensive 3D reconstruction solution, we integrate a neural implicit surface reconstruction technique to generate high-quality surfaces from sparse input data. Additionally, we utilize a mesh-based neural rendering approach to refine texture maps and accurately capture view-dependent properties by combining diffuse and specular components. We evaluate our pipeline on large-scale indoor scenes from the Matterport3D and Stanford2D3D datasets. In practice, IM360 demonstrate superior performance in terms of textured mesh reconstruction over SOTA. We observe accuracy improvements in terms of camera localization and registration as well as rendering high frequency details.  
  </ol>  
</details>  
  
### [From Gaming to Research: GTA V for Synthetic Data Generation for Robotics and Navigations](http://arxiv.org/abs/2502.12303)  
Matteo Scucchia, Matteo Ferrara, Davide Maltoni  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In computer vision, the development of robust algorithms capable of generalizing effectively in real-world scenarios more and more often requires large-scale datasets collected under diverse environmental conditions. However, acquiring such datasets is time-consuming, costly, and sometimes unfeasible. To address these limitations, the use of synthetic data has gained attention as a viable alternative, allowing researchers to generate vast amounts of data while simulating various environmental contexts in a controlled setting. In this study, we investigate the use of synthetic data in robotics and navigation, specifically focusing on Simultaneous Localization and Mapping (SLAM) and Visual Place Recognition (VPR). In particular, we introduce a synthetic dataset created using the virtual environment of the video game Grand Theft Auto V (GTA V), along with an algorithm designed to generate a VPR dataset, without human supervision. Through a series of experiments centered on SLAM and VPR, we demonstrate that synthetic data derived from GTA V are qualitatively comparable to real-world data. Furthermore, these synthetic data can complement or even substitute real-world data in these applications. This study sets the stage for the creation of large-scale synthetic datasets, offering a cost-effective and scalable solution for future research and development.  
  </ol>  
</details>  
  
  



## NeRF  

### [ROI-NeRFs: Hi-Fi Visualization of Objects of Interest within a Scene by NeRFs Composition](http://arxiv.org/abs/2502.12673)  
Quoc-Anh Bui, Gilles Rougeron, Géraldine Morin, Simone Gasparini  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Efficient and accurate 3D reconstruction is essential for applications in cultural heritage. This study addresses the challenge of visualizing objects within large-scale scenes at a high level of detail (LOD) using Neural Radiance Fields (NeRFs). The aim is to improve the visual fidelity of chosen objects while maintaining the efficiency of the computations by focusing on details only for relevant content. The proposed ROI-NeRFs framework divides the scene into a Scene NeRF, which represents the overall scene at moderate detail, and multiple ROI NeRFs that focus on user-defined objects of interest. An object-focused camera selection module automatically groups relevant cameras for each NeRF training during the decomposition phase. In the composition phase, a Ray-level Compositional Rendering technique combines information from the Scene NeRF and ROI NeRFs, allowing simultaneous multi-object rendering composition. Quantitative and qualitative experiments conducted on two real-world datasets, including one on a complex eighteen's century cultural heritage room, demonstrate superior performance compared to baseline methods, improving LOD for object regions, minimizing artifacts, and without significantly increasing inference time.  
  </ol>  
</details>  
**comments**: 17 pages including appendix, 16 figures, 8 tables  
  
  



