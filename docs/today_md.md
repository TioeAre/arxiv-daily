<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Direct-Sparse-Odometry-with-Continuous-3D-Gaussian-Maps-for-Indoor-Environments>Direct Sparse Odometry with Continuous 3D Gaussian Maps for Indoor Environments</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#JamMa:-Ultra-lightweight-Local-Feature-Matching-with-Joint-Mamba>JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Direct Sparse Odometry with Continuous 3D Gaussian Maps for Indoor Environments](http://arxiv.org/abs/2503.03373)  
Jie Deng, Fengtian Lang, Zikang Yuan, Xin Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Accurate localization is essential for robotics and augmented reality applications such as autonomous navigation. Vision-based methods combining prior maps aim to integrate LiDAR-level accuracy with camera cost efficiency for robust pose estimation. Existing approaches, however, often depend on unreliable interpolation procedures when associating discrete point cloud maps with dense image pixels, which inevitably introduces depth errors and degrades pose estimation accuracy. We propose a monocular visual odometry framework utilizing a continuous 3D Gaussian map, which directly assigns geometrically consistent depth values to all extracted high-gradient points without interpolation. Evaluations on two public datasets demonstrate superior tracking accuracy compared to existing methods. We have released the source code of this work for the development of the community.  
  </ol>  
</details>  
**comments**: 7 pages,5 figures  
  
  



## Image Matching  

### [JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba](http://arxiv.org/abs/2503.03437)  
Xiaoyong Lu, Songlin Du  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Existing state-of-the-art feature matchers capture long-range dependencies with Transformers but are hindered by high spatial complexity, leading to demanding training and highlatency inference. Striking a better balance between performance and efficiency remains a challenge in feature matching. Inspired by the linear complexity O(N) of Mamba, we propose an ultra-lightweight Mamba-based matcher, named JamMa, which converges on a single GPU and achieves an impressive performance-efficiency balance in inference. To unlock the potential of Mamba for feature matching, we propose Joint Mamba with a scan-merge strategy named JEGO, which enables: (1) Joint scan of two images to achieve high-frequency mutual interaction, (2) Efficient scan with skip steps to reduce sequence length, (3) Global receptive field, and (4) Omnidirectional feature representation. With the above properties, the JEGO strategy significantly outperforms the scan-merge strategies proposed in VMamba and EVMamba in the feature matching task. Compared to attention-based sparse and semi-dense matchers, JamMa demonstrates a superior balance between performance and efficiency, delivering better performance with less than 50% of the parameters and FLOPs.  
  </ol>  
</details>  
**comments**: CVPR 2025, Project page: https://leoluxxx.github.io/JamMa-page/  
  
  



