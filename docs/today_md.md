<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Power-Variable-Projection-for-Initialization-Free-Large-Scale-Bundle-Adjustment>Power Variable Projection for Initialization-Free Large-Scale Bundle Adjustment</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Unsupervised-Skin-Feature-Tracking-with-Deep-Neural-Networks>Unsupervised Skin Feature Tracking with Deep Neural Networks</a></li>
        <li><a href=#A-Self-Supervised-Method-for-Body-Part-Segmentation-and-Keypoint-Detection-of-Rat-Images>A Self-Supervised Method for Body Part Segmentation and Keypoint Detection of Rat Images</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#${M^2D}$NeRF:-Multi-Modal-Decomposition-NeRF-with-3D-Feature-Fields>${M^2D}$NeRF: Multi-Modal Decomposition NeRF with 3D Feature Fields</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Power Variable Projection for Initialization-Free Large-Scale Bundle Adjustment](http://arxiv.org/abs/2405.05079)  
Simon Weber, Je Hyeong Hong, Daniel Cremers  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Initialization-free bundle adjustment (BA) remains largely uncharted. While Levenberg-Marquardt algorithm is the golden method to solve the BA problem, it generally relies on a good initialization. In contrast, the under-explored Variable Projection algorithm (VarPro) exhibits a wide convergence basin even without initialization. Coupled with object space error formulation, recent works have shown its ability to solve (small-scale) initialization-free bundle adjustment problem. We introduce Power Variable Projection (PoVar), extending a recent inverse expansion method based on power series. Importantly, we link the power series expansion to Riemannian manifold optimization. This projective framework is crucial to solve large-scale bundle adjustment problem without initialization. Using the real-world BAL dataset, we experimentally demonstrate that our solver achieves state-of-the-art results in terms of speed and accuracy. In particular, our work is the first, to our knowledge, that addresses the scalability of BA without initialization and opens new venues for initialization-free Structure-from-Motion.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Unsupervised Skin Feature Tracking with Deep Neural Networks](http://arxiv.org/abs/2405.04943)  
Jose Chang, Torbjörn E. M. Nordling  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Facial feature tracking is essential in imaging ballistocardiography for accurate heart rate estimation and enables motor degradation quantification in Parkinson's disease through skin feature tracking. While deep convolutional neural networks have shown remarkable accuracy in tracking tasks, they typically require extensive labeled data for supervised training. Our proposed pipeline employs a convolutional stacked autoencoder to match image crops with a reference crop containing the target feature, learning deep feature encodings specific to the object category in an unsupervised manner, thus reducing data requirements. To overcome edge effects making the performance dependent on crop size, we introduced a Gaussian weight on the residual errors of the pixels when calculating the loss function. Training the autoencoder on facial images and validating its performance on manually labeled face and hand videos, our Deep Feature Encodings (DFE) method demonstrated superior tracking accuracy with a mean error ranging from 0.6 to 3.3 pixels, outperforming traditional methods like SIFT, SURF, Lucas Kanade, and the latest transformers like PIPs++ and CoTracker. Overall, our unsupervised learning approach excels in tracking various skin features under significant motion conditions, providing superior feature descriptors for tracking, matching, and image registration compared to both traditional and state-of-the-art supervised learning methods.  
  </ol>  
</details>  
**comments**: arXiv admin note: text overlap with arXiv:2112.14159  
  
### [A Self-Supervised Method for Body Part Segmentation and Keypoint Detection of Rat Images](http://arxiv.org/abs/2405.04650)  
László Kopácsi, Áron Fóthi, András Lőrincz  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recognition of individual components and keypoint detection supported by instance segmentation is crucial to analyze the behavior of agents on the scene. Such systems could be used for surveillance, self-driving cars, and also for medical research, where behavior analysis of laboratory animals is used to confirm the aftereffects of a given medicine. A method capable of solving the aforementioned tasks usually requires a large amount of high-quality hand-annotated data, which takes time and money to produce. In this paper, we propose a method that alleviates the need for manual labeling of laboratory rats. To do so, first, we generate initial annotations with a computer vision-based approach, then through extensive augmentation, we train a deep neural network on the generated data. The final system is capable of instance segmentation, keypoint detection, and body part segmentation even when the objects are heavily occluded.  
  </ol>  
</details>  
  
  



## NeRF  

### [ ${M^2D}$ NeRF: Multi-Modal Decomposition NeRF with 3D Feature Fields](http://arxiv.org/abs/2405.05010)  
Ning Wang, Lefei Zhang, Angel X Chang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural fields (NeRF) have emerged as a promising approach for representing continuous 3D scenes. Nevertheless, the lack of semantic encoding in NeRFs poses a significant challenge for scene decomposition. To address this challenge, we present a single model, Multi-Modal Decomposition NeRF (${M^2D}$NeRF), that is capable of both text-based and visual patch-based edits. Specifically, we use multi-modal feature distillation to integrate teacher features from pretrained visual and language models into 3D semantic feature volumes, thereby facilitating consistent 3D editing. To enforce consistency between the visual and language features in our 3D feature volumes, we introduce a multi-modal similarity constraint. We also introduce a patch-based joint contrastive loss that helps to encourage object-regions to coalesce in the 3D feature space, resulting in more precise boundaries. Experiments on various real-world scenes show superior performance in 3D scene decomposition tasks compared to prior NeRF-based methods.  
  </ol>  
</details>  
  
  



