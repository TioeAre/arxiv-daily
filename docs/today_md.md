<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Voyager:-Long-Range-and-World-Consistent-Video-Diffusion-for-Explorable-3D-Scene-Generation>Voyager: Long-Range and World-Consistent Video Diffusion for Explorable 3D Scene Generation</a></li>
        <li><a href=#Accelerating-SfM-based-Pose-Estimation-with-Dominating-Set>Accelerating SfM-based Pose Estimation with Dominating Set</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Voyager: Long-Range and World-Consistent Video Diffusion for Explorable 3D Scene Generation](http://arxiv.org/abs/2506.04225)  
Tianyu Huang, Wangguandong Zheng, Tengfei Wang, Yuhao Liu, Zhenwei Wang, Junta Wu, Jie Jiang, Hui Li, Rynson W. H. Lau, Wangmeng Zuo, Chunchao Guo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Real-world applications like video gaming and virtual reality often demand the ability to model 3D scenes that users can explore along custom camera trajectories. While significant progress has been made in generating 3D objects from text or images, creating long-range, 3D-consistent, explorable 3D scenes remains a complex and challenging problem. In this work, we present Voyager, a novel video diffusion framework that generates world-consistent 3D point-cloud sequences from a single image with user-defined camera path. Unlike existing approaches, Voyager achieves end-to-end scene generation and reconstruction with inherent consistency across frames, eliminating the need for 3D reconstruction pipelines (e.g., structure-from-motion or multi-view stereo). Our method integrates three key components: 1) World-Consistent Video Diffusion: A unified architecture that jointly generates aligned RGB and depth video sequences, conditioned on existing world observation to ensure global coherence 2) Long-Range World Exploration: An efficient world cache with point culling and an auto-regressive inference with smooth video sampling for iterative scene extension with context-aware consistency, and 3) Scalable Data Engine: A video reconstruction pipeline that automates camera pose estimation and metric depth prediction for arbitrary videos, enabling large-scale, diverse training data curation without manual 3D annotations. Collectively, these designs result in a clear improvement over existing methods in visual quality and geometric accuracy, with versatile applications.  
  </ol>  
</details>  
  
### [Accelerating SfM-based Pose Estimation with Dominating Set](http://arxiv.org/abs/2506.03667)  
Joji Joseph, Bharadwaj Amrutur, Shalabh Bhatnagar  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper introduces a preprocessing technique to speed up Structure-from-Motion (SfM) based pose estimation, which is critical for real-time applications like augmented reality (AR), virtual reality (VR), and robotics. Our method leverages the concept of a dominating set from graph theory to preprocess SfM models, significantly enhancing the speed of the pose estimation process without losing significant accuracy. Using the OnePose dataset, we evaluated our method across various SfM-based pose estimation techniques. The results demonstrate substantial improvements in processing speed, ranging from 1.5 to 14.48 times, and a reduction in reference images and point cloud size by factors of 17-23 and 2.27-4, respectively. This work offers a promising solution for efficient and accurate 3D pose estimation, balancing speed and accuracy in real-time applications.  
  </ol>  
</details>  
  
  



