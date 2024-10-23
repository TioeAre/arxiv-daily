<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#LVSM:-A-Large-View-Synthesis-Model-with-Minimal-3D-Inductive-Bias>LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias</a></li>
        <li><a href=#E-3DGS:-Gaussian-Splatting-with-Exposure-and-Motion-Events>E-3DGS: Gaussian Splatting with Exposure and Motion Events</a></li>
        <li><a href=#Joker:-Conditional-3D-Head-Synthesis-with-Extreme-Facial-Expressions>Joker: Conditional 3D Head Synthesis with Extreme Facial Expressions</a></li>
        <li><a href=#GS-LIVM:-Real-Time-Photo-Realistic-LiDAR-Inertial-Visual-Mapping-with-Gaussian-Splatting>GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting</a></li>
      </ul>
    </li>
  </ol>
</details>

## NeRF  

### [LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias](http://arxiv.org/abs/2410.17242)  
Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang, Fujun Luan, Noah Snavely, Zexiang Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose the Large View Synthesis Model (LVSM), a novel transformer-based approach for scalable and generalizable novel view synthesis from sparse-view inputs. We introduce two architectures: (1) an encoder-decoder LVSM, which encodes input image tokens into a fixed number of 1D latent tokens, functioning as a fully learned scene representation, and decodes novel-view images from them; and (2) a decoder-only LVSM, which directly maps input images to novel-view outputs, completely eliminating intermediate scene representations. Both models bypass the 3D inductive biases used in previous methods -- from 3D representations (e.g., NeRF, 3DGS) to network designs (e.g., epipolar projections, plane sweeps) -- addressing novel view synthesis with a fully data-driven approach. While the encoder-decoder model offers faster inference due to its independent latent representation, the decoder-only LVSM achieves superior quality, scalability, and zero-shot generalization, outperforming previous state-of-the-art methods by 1.5 to 3.5 dB PSNR. Comprehensive evaluations across multiple datasets demonstrate that both LVSM variants achieve state-of-the-art novel view synthesis quality. Notably, our models surpass all previous methods even with reduced computational resources (1-2 GPUs). Please see our website for more details: https://haian-jin.github.io/projects/LVSM/ .  
  </ol>  
</details>  
**comments**: project page: https://haian-jin.github.io/projects/LVSM/  
  
### [GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting](http://arxiv.org/abs/2410.17084)  
Yusen Xie, Zhenmin Huang, Jin Wu, Jun Ma  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we introduce GS-LIVM, a real-time photo-realistic LiDAR-Inertial-Visual mapping framework with Gaussian Splatting tailored for outdoor scenes. Compared to existing methods based on Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), our approach enables real-time photo-realistic mapping while ensuring high-quality image rendering in large-scale unbounded outdoor environments. In this work, Gaussian Process Regression (GPR) is employed to mitigate the issues resulting from sparse and unevenly distributed LiDAR observations. The voxel-based 3D Gaussians map representation facilitates real-time dense mapping in large outdoor environments with acceleration governed by custom CUDA kernels. Moreover, the overall framework is designed in a covariance-centered manner, where the estimated covariance is used to initialize the scale and rotation of 3D Gaussians, as well as update the parameters of the GPR. We evaluate our algorithm on several outdoor datasets, and the results demonstrate that our method achieves state-of-the-art performance in terms of mapping efficiency and rendering quality. The source code is available on GitHub.  
  </ol>  
</details>  
**comments**: 15 pages, 13 figures  
  
### [E-3DGS: Gaussian Splatting with Exposure and Motion Events](http://arxiv.org/abs/2410.16995)  
Xiaoting Yin, Hao Shi, Yuhan Bao, Zhenshan Bing, Yiyi Liao, Kailun Yang, Kaiwei Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Estimating Neural Radiance Fields (NeRFs) from images captured under optimal conditions has been extensively explored in the vision community. However, robotic applications often face challenges such as motion blur, insufficient illumination, and high computational overhead, which adversely affect downstream tasks like navigation, inspection, and scene visualization. To address these challenges, we propose E-3DGS, a novel event-based approach that partitions events into motion (from camera or object movement) and exposure (from camera exposure), using the former to handle fast-motion scenes and using the latter to reconstruct grayscale images for high-quality training and optimization of event-based 3D Gaussian Splatting (3DGS). We introduce a novel integration of 3DGS with exposure events for high-quality reconstruction of explicit scene representations. Our versatile framework can operate on motion events alone for 3D reconstruction, enhance quality using exposure events, or adopt a hybrid mode that balances quality and effectiveness by optimizing with initial exposure events followed by high-speed motion events. We also introduce EME-3D, a real-world 3D dataset with exposure events, motion events, camera calibration parameters, and sparse point clouds. Our method is faster and delivers better reconstruction quality than event-based NeRF while being more cost-effective than NeRF methods that combine event and RGB data by using a single event sensor. By combining motion and exposure events, E-3DGS sets a new benchmark for event-based 3D reconstruction with robust performance in challenging conditions and lower hardware demands. The source code and dataset will be available at https://github.com/MasterHow/E-3DGS.  
  </ol>  
</details>  
**comments**: The source code and dataset will be available at
  https://github.com/MasterHow/E-3DGS  
  
### [Joker: Conditional 3D Head Synthesis with Extreme Facial Expressions](http://arxiv.org/abs/2410.16395)  
Malte Prinzler, Egor Zakharov, Vanessa Sklyarova, Berna Kabadayi, Justus Thies  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce Joker, a new method for the conditional synthesis of 3D human heads with extreme expressions. Given a single reference image of a person, we synthesize a volumetric human head with the reference identity and a new expression. We offer control over the expression via a 3D morphable model (3DMM) and textual inputs. This multi-modal conditioning signal is essential since 3DMMs alone fail to define subtle emotional changes and extreme expressions, including those involving the mouth cavity and tongue articulation. Our method is built upon a 2D diffusion-based prior that generalizes well to out-of-domain samples, such as sculptures, heavy makeup, and paintings while achieving high levels of expressiveness. To improve view consistency, we propose a new 3D distillation technique that converts predictions of our 2D prior into a neural radiance field (NeRF). Both the 2D prior and our distillation technique produce state-of-the-art results, which are confirmed by our extensive evaluations. Also, to the best of our knowledge, our method is the first to achieve view-consistent extreme tongue articulation.  
  </ol>  
</details>  
**comments**: Project Page: https://malteprinzler.github.io/projects/joker/  
  
  



