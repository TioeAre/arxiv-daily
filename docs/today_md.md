<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Visualizing-intercalation-effects-in-2D-materials-using-AFM-based-techniques>Visualizing intercalation effects in 2D materials using AFM based techniques</a></li>
        <li><a href=#On-the-Burstiness-of-Faces-in-Set>On the Burstiness of Faces in Set</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Fast-entropy-regularized-SDP-relaxations-for-permutation-synchronization>Fast entropy-regularized SDP relaxations for permutation synchronization</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Joint-attitude-estimation-and-3D-neural-reconstruction-of-non-cooperative-space-objects>Joint attitude estimation and 3D neural reconstruction of non-cooperative space objects</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Visualizing intercalation effects in 2D materials using AFM based techniques](http://arxiv.org/abs/2506.20467)  
Karmen Kapustić, Cosme G. Ayani, Borna Pielić, Kateřina Plevová, Šimun Mandić, Iva Šrut Rakić  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Intercalation of two dimensional materials, particularly transition metal dichalcogenides, is a noninvasive way to modify electronic, optical and structural properties of these materials. However, research of these atomic-scale phenomena usually relies on using Ultra High Vacuum techniques which is time consuming, expensive and spatially limited. Here we utilize Atomic Force Microscopy (AFM) based techniques to visualize local structural and electronic changes of the MoS2 on graphene on Ir(111), caused by sulfur intercalation. AFM topography reveals structural changes, while phase imaging and mechanical measurements show reduced Young's modulus and adhesion. Kelvin Probe Force Microscopy highlights variations in surface potential and work function, aligning with intercalation signatures, while Photoinduced Force Microscopy detects enhanced optical response in intercalated regions. These results demonstrate the efficacy of AFM based techniques in mapping intercalation, offering insights into tailoring 2D materials electronic and optical properties. This work underscores the potential of AFM techniques for advanced material characterization and the development of 2D material applications.  
  </ol>  
</details>  
  
### [On the Burstiness of Faces in Set](http://arxiv.org/abs/2506.20312)  
Jiong Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Burstiness, a phenomenon observed in text and image retrieval, refers to that particular elements appear more times in a set than a statistically independent model assumes. We argue that in the context of set-based face recognition (SFR), burstiness exists widely and degrades the performance in two aspects: Firstly, the bursty faces, where faces with particular attributes %exist frequently in a face set, dominate the training instances and dominate the training face sets and lead to poor generalization ability to unconstrained scenarios. Secondly, the bursty faces %dominating the evaluation sets interfere with the similarity comparison in set verification and identification when evaluation. To detect the bursty faces in a set, we propose three strategies based on Quickshift++, feature self-similarity, and generalized max-pooling (GMP). We apply the burst detection results on training and evaluation stages to enhance the sampling ratios or contributions of the infrequent faces. When evaluation, we additionally propose the quality-aware GMP that enables awareness of the face quality and robustness to the low-quality faces for the original GMP. We give illustrations and extensive experiments on the SFR benchmarks to demonstrate that burstiness is widespread and suppressing burstiness considerably improves the recognition performance.  
  </ol>  
</details>  
**comments**: 18 pages, 5 figures  
  
  



## Image Matching  

### [Fast entropy-regularized SDP relaxations for permutation synchronization](http://arxiv.org/abs/2506.20191)  
Michael Lindsey, Yunpeng Shi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce fast randomized algorithms for solving semidefinite programming (SDP) relaxations of the partial permutation synchronization (PPS) problem, a core task in multi-image matching with significant relevance to 3D reconstruction. Our methods build on recent advances in entropy-regularized semidefinite programming and are tailored to the unique structure of PPS, in which the unknowns are partial permutation matrices aligning sparse and noisy pairwise correspondences across images. We prove that entropy regularization resolves optimizer non-uniqueness in standard relaxations, and we develop a randomized solver with nearly optimal scaling in the number of observed correspondences. We also develop several rounding procedures for recovering combinatorial solutions from the implicitly represented primal solution variable, maintaining cycle consistency if desired without harming computational scaling. We demonstrate that our approach achieves state-of-the-art performance on synthetic and real-world datasets in terms of speed and accuracy. Our results highlight PPS as a paradigmatic setting in which entropy-regularized SDP admits both theoretical and practical advantages over traditional low-rank or spectral techniques.  
  </ol>  
</details>  
  
  



## NeRF  

### [Joint attitude estimation and 3D neural reconstruction of non-cooperative space objects](http://arxiv.org/abs/2506.20638)  
Clément Forray, Pauline Delporte, Nicolas Delaygue, Florence Genin, Dawa Derksen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Obtaining a better knowledge of the current state and behavior of objects orbiting Earth has proven to be essential for a range of applications such as active debris removal, in-orbit maintenance, or anomaly detection. 3D models represent a valuable source of information in the field of Space Situational Awareness (SSA). In this work, we leveraged Neural Radiance Fields (NeRF) to perform 3D reconstruction of non-cooperative space objects from simulated images. This scenario is challenging for NeRF models due to unusual camera characteristics and environmental conditions : mono-chromatic images, unknown object orientation, limited viewing angles, absence of diffuse lighting etc. In this work we focus primarly on the joint optimization of camera poses alongside the NeRF. Our experimental results show that the most accurate 3D reconstruction is achieved when training with successive images one-by-one. We estimate camera poses by optimizing an uniform rotation and use regularization to prevent successive poses from being too far apart.  
  </ol>  
</details>  
**comments**: accepted for CVPR 2025 NFBCC workshop  
  
  



