<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#A-Review-of-3D-Reconstruction-Techniques-for-Deformable-Tissues-in-Robotic-Surgery>A Review of 3D Reconstruction Techniques for Deformable Tissues in Robotic Surgery</a></li>
        <li><a href=#Evaluating-Modern-Approaches-in-3D-Scene-Reconstruction:-NeRF-vs-Gaussian-Based-Methods>Evaluating Modern Approaches in 3D Scene Reconstruction: NeRF vs Gaussian-Based Methods</a></li>
        <li><a href=#LumiGauss:-High-Fidelity-Outdoor-Relighting-with-2D-Gaussian-Splatting>LumiGauss: High-Fidelity Outdoor Relighting with 2D Gaussian Splatting</a></li>
      </ul>
    </li>
  </ol>
</details>

## NeRF  

### [LumiGauss: High-Fidelity Outdoor Relighting with 2D Gaussian Splatting](http://arxiv.org/abs/2408.04474)  
Joanna Kaleta, Kacper Kania, Tomasz Trzcinski, Marek Kowalski  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Decoupling lighting from geometry using unconstrained photo collections is notoriously challenging. Solving it would benefit many users, as creating complex 3D assets takes days of manual labor. Many previous works have attempted to address this issue, often at the expense of output fidelity, which questions the practicality of such methods.   We introduce LumiGauss, a technique that tackles 3D reconstruction of scenes and environmental lighting through 2D Gaussian Splatting. Our approach yields high-quality scene reconstructions and enables realistic lighting synthesis under novel environment maps. We also propose a method for enhancing the quality of shadows, common in outdoor scenes, by exploiting spherical harmonics properties. Our approach facilitates seamless integration with game engines and enables the use of fast precomputed radiance transfer.   We validate our method on the NeRF-OSR dataset, demonstrating superior performance over baseline methods. Moreover, LumiGauss can synthesize realistic images when applying novel environment maps.  
  </ol>  
</details>  
**comments**: Includes video files in src  
  
### [A Review of 3D Reconstruction Techniques for Deformable Tissues in Robotic Surgery](http://arxiv.org/abs/2408.04426)  
[[code](https://github.com/epsilon404/surgicalnerf)]  
Mengya Xu, Ziqi Guo, An Wang, Long Bai, Hongliang Ren  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    As a crucial and intricate task in robotic minimally invasive surgery, reconstructing surgical scenes using stereo or monocular endoscopic video holds immense potential for clinical applications. NeRF-based techniques have recently garnered attention for the ability to reconstruct scenes implicitly. On the other hand, Gaussian splatting-based 3D-GS represents scenes explicitly using 3D Gaussians and projects them onto a 2D plane as a replacement for the complex volume rendering in NeRF. However, these methods face challenges regarding surgical scene reconstruction, such as slow inference, dynamic scenes, and surgical tool occlusion. This work explores and reviews state-of-the-art (SOTA) approaches, discussing their innovations and implementation principles. Furthermore, we replicate the models and conduct testing and evaluation on two datasets. The test results demonstrate that with advancements in these techniques, achieving real-time, high-quality reconstructions becomes feasible.  
  </ol>  
</details>  
**comments**: To appear in MICCAI 2024 EARTH Workshop. Code availability:
  https://github.com/Epsilon404/surgicalnerf  
  
### [Evaluating Modern Approaches in 3D Scene Reconstruction: NeRF vs Gaussian-Based Methods](http://arxiv.org/abs/2408.04268)  
Yiming Zhou, Zixuan Zeng, Andi Chen, Xiaofan Zhou, Haowei Ni, Shiyao Zhang, Panfeng Li, Liangxi Liu, Mengyao Zheng, Xupeng Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Exploring the capabilities of Neural Radiance Fields (NeRF) and Gaussian-based methods in the context of 3D scene reconstruction, this study contrasts these modern approaches with traditional Simultaneous Localization and Mapping (SLAM) systems. Utilizing datasets such as Replica and ScanNet, we assess performance based on tracking accuracy, mapping fidelity, and view synthesis. Findings reveal that NeRF excels in view synthesis, offering unique capabilities in generating new perspectives from existing data, albeit at slower processing speeds. Conversely, Gaussian-based methods provide rapid processing and significant expressiveness but lack comprehensive scene completion. Enhanced by global optimization and loop closure techniques, newer methods like NICE-SLAM and SplaTAM not only surpass older frameworks such as ORB-SLAM2 in terms of robustness but also demonstrate superior performance in dynamic and complex environments. This comparative analysis bridges theoretical research with practical implications, shedding light on future developments in robust 3D scene reconstruction across various real-world applications.  
  </ol>  
</details>  
**comments**: Accepted by 2024 6th International Conference on Data-driven
  Optimization of Complex Systems  
  
  



