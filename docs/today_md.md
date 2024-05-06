<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#HoloGS:-Instant-Depth-based-3D-Gaussian-Splatting-with-Microsoft-HoloLens-2>HoloGS: Instant Depth-based 3D Gaussian Splatting with Microsoft HoloLens 2</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#WateRF:-Robust-Watermarks-in-Radiance-Fields-for-Protection-of-Copyrights>WateRF: Robust Watermarks in Radiance Fields for Protection of Copyrights</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [HoloGS: Instant Depth-based 3D Gaussian Splatting with Microsoft HoloLens 2](http://arxiv.org/abs/2405.02005)  
Miriam Jäger, Theodor Kapler, Michael Feßenbecker, Felix Birkelbach, Markus Hillemann, Boris Jutzi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In the fields of photogrammetry, computer vision and computer graphics, the task of neural 3D scene reconstruction has led to the exploration of various techniques. Among these, 3D Gaussian Splatting stands out for its explicit representation of scenes using 3D Gaussians, making it appealing for tasks like 3D point cloud extraction and surface reconstruction. Motivated by its potential, we address the domain of 3D scene reconstruction, aiming to leverage the capabilities of the Microsoft HoloLens 2 for instant 3D Gaussian Splatting. We present HoloGS, a novel workflow utilizing HoloLens sensor data, which bypasses the need for pre-processing steps like Structure from Motion by instantly accessing the required input data i.e. the images, camera poses and the point cloud from depth sensing. We provide comprehensive investigations, including the training process and the rendering quality, assessed through the Peak Signal-to-Noise Ratio, and the geometric 3D accuracy of the densified point cloud from Gaussian centers, measured by Chamfer Distance. We evaluate our approach on two self-captured scenes: An outdoor scene of a cultural heritage statue and an indoor scene of a fine-structured plant. Our results show that the HoloLens data, including RGB images, corresponding camera poses, and depth sensing based point clouds to initialize the Gaussians, are suitable as input for 3D Gaussian Splatting.  
  </ol>  
</details>  
**comments**: 8 pages, 9 figures, 2 tables. Will be published in the ISPRS The
  International Archives of Photogrammetry, Remote Sensing and Spatial
  Information Sciences  
  
  



## NeRF  

### [WateRF: Robust Watermarks in Radiance Fields for Protection of Copyrights](http://arxiv.org/abs/2405.02066)  
Youngdong Jang, Dong In Lee, MinHyuk Jang, Jong Wook Kim, Feng Yang, Sangpil Kim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The advances in the Neural Radiance Fields (NeRF) research offer extensive applications in diverse domains, but protecting their copyrights has not yet been researched in depth. Recently, NeRF watermarking has been considered one of the pivotal solutions for safely deploying NeRF-based 3D representations. However, existing methods are designed to apply only to implicit or explicit NeRF representations. In this work, we introduce an innovative watermarking method that can be employed in both representations of NeRF. This is achieved by fine-tuning NeRF to embed binary messages in the rendering process. In detail, we propose utilizing the discrete wavelet transform in the NeRF space for watermarking. Furthermore, we adopt a deferred back-propagation technique and introduce a combination with the patch-wise loss to improve rendering quality and bit accuracy with minimum trade-offs. We evaluate our method in three different aspects: capacity, invisibility, and robustness of the embedded watermarks in the 2D-rendered images. Our method achieves state-of-the-art performance with faster training speed over the compared state-of-the-art methods.  
  </ol>  
</details>  
  
  



