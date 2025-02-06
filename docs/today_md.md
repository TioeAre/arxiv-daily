<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#SiLVR:-Scalable-Lidar-Visual-Radiance-Field-Reconstruction-with-Uncertainty-Quantification>SiLVR: Scalable Lidar-Visual Radiance Field Reconstruction with Uncertainty Quantification</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Human-Aligned-Image-Models-Improve-Visual-Decoding-from-the-Brain>Human-Aligned Image Models Improve Visual Decoding from the Brain</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Muographic-Image-Upsampling-with-Machine-Learning-for-Built-Infrastructure-Applications>Muographic Image Upsampling with Machine Learning for Built Infrastructure Applications</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#SiLVR:-Scalable-Lidar-Visual-Radiance-Field-Reconstruction-with-Uncertainty-Quantification>SiLVR: Scalable Lidar-Visual Radiance Field Reconstruction with Uncertainty Quantification</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [SiLVR: Scalable Lidar-Visual Radiance Field Reconstruction with Uncertainty Quantification](http://arxiv.org/abs/2502.02657)  
Yifu Tao, Maurice Fallon  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present a neural radiance field (NeRF) based large-scale reconstruction system that fuses lidar and vision data to generate high-quality reconstructions that are geometrically accurate and capture photorealistic texture. Our system adopts the state-of-the-art NeRF representation to additionally incorporate lidar. Adding lidar data adds strong geometric constraints on the depth and surface normals, which is particularly useful when modelling uniform texture surfaces which contain ambiguous visual reconstruction cues. Furthermore, we estimate the epistemic uncertainty of the reconstruction as the spatial variance of each point location in the radiance field given the sensor observations from camera and lidar. This enables the identification of areas that are reliably reconstructed by each sensor modality, allowing the map to be filtered according to the estimated uncertainty. Our system can also exploit the trajectory produced by a real-time pose-graph lidar SLAM system during online mapping to bootstrap a (post-processed) Structure-from-Motion (SfM) reconstruction procedure reducing SfM training time by up to 70%. It also helps to properly constrain the overall metric scale which is essential for the lidar depth loss. The globally-consistent trajectory can then be divided into submaps using Spectral Clustering to group sets of co-visible images together. This submapping approach is more suitable for visual reconstruction than distance-based partitioning. Each submap is filtered according to point-wise uncertainty estimates and merged to obtain the final large-scale 3D reconstruction. We demonstrate the reconstruction system using a multi-camera, lidar sensor suite in experiments involving both robot-mounted and handheld scanning. Our test datasets cover a total area of more than 20,000 square metres, including multiple university buildings and an aerial survey of a multi-storey.  
  </ol>  
</details>  
**comments**: webpage: https://dynamic.robots.ox.ac.uk/projects/silvr/  
  
  



## Visual Localization  

### [Human-Aligned Image Models Improve Visual Decoding from the Brain](http://arxiv.org/abs/2502.03081)  
Nona Rajabi, Antônio H. Ribeiro, Miguel Vasco, Farzaneh Taleb, Mårten Björkman, Danica Kragic  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Decoding visual images from brain activity has significant potential for advancing brain-computer interaction and enhancing the understanding of human perception. Recent approaches align the representation spaces of images and brain activity to enable visual decoding. In this paper, we introduce the use of human-aligned image encoders to map brain signals to images. We hypothesize that these models more effectively capture perceptual attributes associated with the rapid visual stimuli presentations commonly used in visual brain data recording experiments. Our empirical results support this hypothesis, demonstrating that this simple modification improves image retrieval accuracy by up to 21% compared to state-of-the-art methods. Comprehensive experiments confirm consistent performance improvements across diverse EEG architectures, image encoders, alignment methods, participants, and brain imaging modalities.  
  </ol>  
</details>  
  
  



## Image Matching  

### [Muographic Image Upsampling with Machine Learning for Built Infrastructure Applications](http://arxiv.org/abs/2502.02624)  
William O'Donnell, David Mahon, Guangliang Yang, Simon Gardner  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The civil engineering industry faces a critical need for innovative non-destructive evaluation methods, particularly for ageing critical infrastructure, such as bridges, where current techniques fall short. Muography, a non-invasive imaging technique, constructs three-dimensional density maps by detecting interactions of naturally occurring cosmic-ray muons within the scanned volume. Cosmic-ray muons provide deep penetration and inherent safety due to their high momenta and natural source. However, the technology's reliance on this source results in constrained muon flux, leading to prolonged acquisition times, noisy reconstructions and image interpretation challenges. To address these limitations, we developed a two-model deep learning approach. First, we employed a conditional Wasserstein generative adversarial network with gradient penalty (cWGAN-GP) to perform predictive upsampling of undersampled muography images. Using the structural similarity index measure (SSIM), 1-day sampled images matched the perceptual qualities of a 21-day image, while the peak signal-to-noise ratio (PSNR) indicated noise improvement equivalent to 31 days of sampling. A second cWGAN-GP model, trained for semantic segmentation, quantitatively assessed the upsampling model's impact on concrete sample features. This model achieved segmentation of rebar grids and tendon ducts, with Dice-S{\o}rensen accuracy coefficients of 0.8174 and 0.8663. Notably, it could mitigate or remove z-plane smearing artifacts caused by muography's inverse imaging problem. Both models were trained on a comprehensive Geant4 Monte-Carlo simulation dataset reflecting realistic civil infrastructure scenarios. Our results demonstrate significant improvements in acquisition speed and image quality, marking a substantial step toward making muography more practical for reinforced concrete infrastructure monitoring applications.  
  </ol>  
</details>  
  
  



## NeRF  

### [SiLVR: Scalable Lidar-Visual Radiance Field Reconstruction with Uncertainty Quantification](http://arxiv.org/abs/2502.02657)  
Yifu Tao, Maurice Fallon  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present a neural radiance field (NeRF) based large-scale reconstruction system that fuses lidar and vision data to generate high-quality reconstructions that are geometrically accurate and capture photorealistic texture. Our system adopts the state-of-the-art NeRF representation to additionally incorporate lidar. Adding lidar data adds strong geometric constraints on the depth and surface normals, which is particularly useful when modelling uniform texture surfaces which contain ambiguous visual reconstruction cues. Furthermore, we estimate the epistemic uncertainty of the reconstruction as the spatial variance of each point location in the radiance field given the sensor observations from camera and lidar. This enables the identification of areas that are reliably reconstructed by each sensor modality, allowing the map to be filtered according to the estimated uncertainty. Our system can also exploit the trajectory produced by a real-time pose-graph lidar SLAM system during online mapping to bootstrap a (post-processed) Structure-from-Motion (SfM) reconstruction procedure reducing SfM training time by up to 70%. It also helps to properly constrain the overall metric scale which is essential for the lidar depth loss. The globally-consistent trajectory can then be divided into submaps using Spectral Clustering to group sets of co-visible images together. This submapping approach is more suitable for visual reconstruction than distance-based partitioning. Each submap is filtered according to point-wise uncertainty estimates and merged to obtain the final large-scale 3D reconstruction. We demonstrate the reconstruction system using a multi-camera, lidar sensor suite in experiments involving both robot-mounted and handheld scanning. Our test datasets cover a total area of more than 20,000 square metres, including multiple university buildings and an aerial survey of a multi-storey.  
  </ol>  
</details>  
**comments**: webpage: https://dynamic.robots.ox.ac.uk/projects/silvr/  
  
  



