<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Sparse-View-3D-Reconstruction:-Recent-Advances-and-Open-Challenges>Sparse-View 3D Reconstruction: Recent Advances and Open Challenges</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#A-Single-step-Accurate-Fingerprint-Registration-Method-Based-on-Local-Feature-Matching>A Single-step Accurate Fingerprint Registration Method Based on Local Feature Matching</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Sparse-View-3D-Reconstruction:-Recent-Advances-and-Open-Challenges>Sparse-View 3D Reconstruction: Recent Advances and Open Challenges</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Sparse-View 3D Reconstruction: Recent Advances and Open Challenges](http://arxiv.org/abs/2507.16406)  
Tanveer Younis, Zhanglin Cheng  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Sparse-view 3D reconstruction is essential for applications in which dense image acquisition is impractical, such as robotics, augmented/virtual reality (AR/VR), and autonomous systems. In these settings, minimal image overlap prevents reliable correspondence matching, causing traditional methods, such as structure-from-motion (SfM) and multiview stereo (MVS), to fail. This survey reviews the latest advances in neural implicit models (e.g., NeRF and its regularized versions), explicit point-cloud-based approaches (e.g., 3D Gaussian Splatting), and hybrid frameworks that leverage priors from diffusion and vision foundation models (VFMs).We analyze how geometric regularization, explicit shape modeling, and generative inference are used to mitigate artifacts such as floaters and pose ambiguities in sparse-view settings. Comparative results on standard benchmarks reveal key trade-offs between the reconstruction accuracy, efficiency, and generalization. Unlike previous reviews, our survey provides a unified perspective on geometry-based, neural implicit, and generative (diffusion-based) methods. We highlight the persistent challenges in domain generalization and pose-free reconstruction and outline future directions for developing 3D-native generative priors and achieving real-time, unconstrained sparse-view reconstruction.  
  </ol>  
</details>  
**comments**: 30 pages, 6 figures  
  
  



## Image Matching  

### [A Single-step Accurate Fingerprint Registration Method Based on Local Feature Matching](http://arxiv.org/abs/2507.16201)  
Yuwei Jia, Zhe Cui, Fei Su  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Distortion of the fingerprint images leads to a decline in fingerprint recognition performance, and fingerprint registration can mitigate this distortion issue by accurately aligning two fingerprint images. Currently, fingerprint registration methods often consist of two steps: an initial registration based on minutiae, and a dense registration based on matching points. However, when the quality of fingerprint image is low, the number of detected minutiae is reduced, leading to frequent failures in the initial registration, which ultimately causes the entire fingerprint registration process to fail. In this study, we propose an end-to-end single-step fingerprint registration algorithm that aligns two fingerprints by directly predicting the semi-dense matching points correspondences between two fingerprints. Thus, our method minimizes the risk of minutiae registration failure and also leverages global-local attentions to achieve end-to-end pixel-level alignment between the two fingerprints. Experiment results prove that our method can achieve the state-of-the-art matching performance with only single-step registration, and it can also be used in conjunction with dense registration algorithms for further performance improvements.  
  </ol>  
</details>  
  
  



## NeRF  

### [Sparse-View 3D Reconstruction: Recent Advances and Open Challenges](http://arxiv.org/abs/2507.16406)  
Tanveer Younis, Zhanglin Cheng  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Sparse-view 3D reconstruction is essential for applications in which dense image acquisition is impractical, such as robotics, augmented/virtual reality (AR/VR), and autonomous systems. In these settings, minimal image overlap prevents reliable correspondence matching, causing traditional methods, such as structure-from-motion (SfM) and multiview stereo (MVS), to fail. This survey reviews the latest advances in neural implicit models (e.g., NeRF and its regularized versions), explicit point-cloud-based approaches (e.g., 3D Gaussian Splatting), and hybrid frameworks that leverage priors from diffusion and vision foundation models (VFMs).We analyze how geometric regularization, explicit shape modeling, and generative inference are used to mitigate artifacts such as floaters and pose ambiguities in sparse-view settings. Comparative results on standard benchmarks reveal key trade-offs between the reconstruction accuracy, efficiency, and generalization. Unlike previous reviews, our survey provides a unified perspective on geometry-based, neural implicit, and generative (diffusion-based) methods. We highlight the persistent challenges in domain generalization and pose-free reconstruction and outline future directions for developing 3D-native generative priors and achieving real-time, unconstrained sparse-view reconstruction.  
  </ol>  
</details>  
**comments**: 30 pages, 6 figures  
  
  



