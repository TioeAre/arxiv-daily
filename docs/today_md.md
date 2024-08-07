<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#AMES:-Asymmetric-and-Memory-Efficient-Similarity-Estimation-for-Instance-level-Retrieval>AMES: Asymmetric and Memory-Efficient Similarity Estimation for Instance-level Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#ConDL:-Detector-Free-Dense-Image-Matching>ConDL: Detector-Free Dense Image Matching</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Efficient-NeRF-Optimization----Not-All-Samples-Remain-Equally-Hard>Efficient NeRF Optimization -- Not All Samples Remain Equally Hard</a></li>
        <li><a href=#MGFs:-Masked-Gaussian-Fields-for-Meshing-Building-based-on-Multi-View-Images>MGFs: Masked Gaussian Fields for Meshing Building based on Multi-View Images</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [AMES: Asymmetric and Memory-Efficient Similarity Estimation for Instance-level Retrieval](http://arxiv.org/abs/2408.03282)  
Pavel Suma, Giorgos Kordopatis-Zilos, Ahmet Iscen, Giorgos Tolias  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This work investigates the problem of instance-level image retrieval re-ranking with the constraint of memory efficiency, ultimately aiming to limit memory usage to 1KB per image. Departing from the prevalent focus on performance enhancements, this work prioritizes the crucial trade-off between performance and memory requirements. The proposed model uses a transformer-based architecture designed to estimate image-to-image similarity by capturing interactions within and across images based on their local descriptors. A distinctive property of the model is the capability for asymmetric similarity estimation. Database images are represented with a smaller number of descriptors compared to query images, enabling performance improvements without increasing memory consumption. To ensure adaptability across different applications, a universal model is introduced that adjusts to a varying number of local descriptors during the testing phase. Results on standard benchmarks demonstrate the superiority of our approach over both hand-crafted and learned models. In particular, compared with current state-of-the-art methods that overlook their memory footprint, our approach not only attains superior performance but does so with a significantly reduced memory footprint. The code and pretrained models are publicly available at: https://github.com/pavelsuma/ames  
  </ol>  
</details>  
**comments**: ECCV 2024  
  
  



## Image Matching  

### [ConDL: Detector-Free Dense Image Matching](http://arxiv.org/abs/2408.02766)  
Monika Kwiatkowski, Simon Matern, Olaf Hellwich  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this work, we introduce a deep-learning framework designed for estimating dense image correspondences. Our fully convolutional model generates dense feature maps for images, where each pixel is associated with a descriptor that can be matched across multiple images. Unlike previous methods, our model is trained on synthetic data that includes significant distortions, such as perspective changes, illumination variations, shadows, and specular highlights. Utilizing contrastive learning, our feature maps achieve greater invariance to these distortions, enabling robust matching. Notably, our method eliminates the need for a keypoint detector, setting it apart from many existing image-matching techniques.  
  </ol>  
</details>  
  
  



## NeRF  

### [Efficient NeRF Optimization -- Not All Samples Remain Equally Hard](http://arxiv.org/abs/2408.03193)  
Juuso Korhonen, Goutham Rangu, Hamed R. Tavakoli, Juho Kannala  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose an application of online hard sample mining for efficient training of Neural Radiance Fields (NeRF). NeRF models produce state-of-the-art quality for many 3D reconstruction and rendering tasks but require substantial computational resources. The encoding of the scene information within the NeRF network parameters necessitates stochastic sampling. We observe that during the training, a major part of the compute time and memory usage is spent on processing already learnt samples, which no longer affect the model update significantly. We identify the backward pass on the stochastic samples as the computational bottleneck during the optimization. We thus perform the first forward pass in inference mode as a relatively low-cost search for hard samples. This is followed by building the computational graph and updating the NeRF network parameters using only the hard samples. To demonstrate the effectiveness of the proposed approach, we apply our method to Instant-NGP, resulting in significant improvements of the view-synthesis quality over the baseline (1 dB improvement on average per training time, or 2x speedup to reach the same PSNR level) along with approx. 40% memory savings coming from using only the hard samples to build the computational graph. As our method only interfaces with the network module, we expect it to be widely applicable.  
  </ol>  
</details>  
  
### [MGFs: Masked Gaussian Fields for Meshing Building based on Multi-View Images](http://arxiv.org/abs/2408.03060)  
Tengfei Wang, Zongqian Zhan, Rui Xia, Linxia Ji, Xin Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Over the last few decades, image-based building surface reconstruction has garnered substantial research interest and has been applied across various fields, such as heritage preservation, architectural planning, etc. Compared to the traditional photogrammetric and NeRF-based solutions, recently, Gaussian fields-based methods have exhibited significant potential in generating surface meshes due to their time-efficient training and detailed 3D information preservation. However, most gaussian fields-based methods are trained with all image pixels, encompassing building and nonbuilding areas, which results in a significant noise for building meshes and degeneration in time efficiency. This paper proposes a novel framework, Masked Gaussian Fields (MGFs), designed to generate accurate surface reconstruction for building in a time-efficient way. The framework first applies EfficientSAM and COLMAP to generate multi-level masks of building and the corresponding masked point clouds. Subsequently, the masked gaussian fields are trained by integrating two innovative losses: a multi-level perceptual masked loss focused on constructing building regions and a boundary loss aimed at enhancing the details of the boundaries between different masks. Finally, we improve the tetrahedral surface mesh extraction method based on the masked gaussian spheres. Comprehensive experiments on UAV images demonstrate that, compared to the traditional method and several NeRF-based and Gaussian-based SOTA solutions, our approach significantly improves both the accuracy and efficiency of building surface reconstruction. Notably, as a byproduct, there is an additional gain in the novel view synthesis of building.  
  </ol>  
</details>  
  
  



