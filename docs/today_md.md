<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#GenN2N:-Generative-NeRF2NeRF-Translation>GenN2N: Generative NeRF2NeRF Translation</a></li>
        <li><a href=#LiDAR4D:-Dynamic-Neural-Fields-for-Novel-Space-time-View-LiDAR-Synthesis>LiDAR4D: Dynamic Neural Fields for Novel Space-time View LiDAR Synthesis</a></li>
        <li><a href=#Neural-Radiance-Fields-with-Torch-Units>Neural Radiance Fields with Torch Units</a></li>
        <li><a href=#Freditor:-High-Fidelity-and-Transferable-NeRF-Editing-by-Frequency-Decomposition>Freditor: High-Fidelity and Transferable NeRF Editing by Frequency Decomposition</a></li>
        <li><a href=#NeRFCodec:-Neural-Feature-Compression-Meets-Neural-Radiance-Fields-for-Memory-Efficient-Scene-Representation>NeRFCodec: Neural Feature Compression Meets Neural Radiance Fields for Memory-Efficient Scene Representation</a></li>
      </ul>
    </li>
  </ol>
</details>

## NeRF  

### [GenN2N: Generative NeRF2NeRF Translation](http://arxiv.org/abs/2404.02788)  
Xiangyue Liu, Han Xue, Kunming Luo, Ping Tan, Li Yi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present GenN2N, a unified NeRF-to-NeRF translation framework for various NeRF translation tasks such as text-driven NeRF editing, colorization, super-resolution, inpainting, etc. Unlike previous methods designed for individual translation tasks with task-specific schemes, GenN2N achieves all these NeRF editing tasks by employing a plug-and-play image-to-image translator to perform editing in the 2D domain and lifting 2D edits into the 3D NeRF space. Since the 3D consistency of 2D edits may not be assured, we propose to model the distribution of the underlying 3D edits through a generative model that can cover all possible edited NeRFs. To model the distribution of 3D edited NeRFs from 2D edited images, we carefully design a VAE-GAN that encodes images while decoding NeRFs. The latent space is trained to align with a Gaussian distribution and the NeRFs are supervised through an adversarial loss on its renderings. To ensure the latent code does not depend on 2D viewpoints but truly reflects the 3D edits, we also regularize the latent code through a contrastive learning scheme. Extensive experiments on various editing tasks show GenN2N, as a universal framework, performs as well or better than task-specific specialists while possessing flexible generative power. More results on our project page: https://xiangyueliu.github.io/GenN2N/  
  </ol>  
</details>  
**comments**: Accepted to CVPR 2024. Project page:
  https://xiangyueliu.github.io/GenN2N/  
  
### [LiDAR4D: Dynamic Neural Fields for Novel Space-time View LiDAR Synthesis](http://arxiv.org/abs/2404.02742)  
[[code](https://github.com/ispc-lab/lidar4d)]  
Zehan Zheng, Fan Lu, Weiyi Xue, Guang Chen, Changjun Jiang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Although neural radiance fields (NeRFs) have achieved triumphs in image novel view synthesis (NVS), LiDAR NVS remains largely unexplored. Previous LiDAR NVS methods employ a simple shift from image NVS methods while ignoring the dynamic nature and the large-scale reconstruction problem of LiDAR point clouds. In light of this, we propose LiDAR4D, a differentiable LiDAR-only framework for novel space-time LiDAR view synthesis. In consideration of the sparsity and large-scale characteristics, we design a 4D hybrid representation combined with multi-planar and grid features to achieve effective reconstruction in a coarse-to-fine manner. Furthermore, we introduce geometric constraints derived from point clouds to improve temporal consistency. For the realistic synthesis of LiDAR point clouds, we incorporate the global optimization of ray-drop probability to preserve cross-region patterns. Extensive experiments on KITTI-360 and NuScenes datasets demonstrate the superiority of our method in accomplishing geometry-aware and time-consistent dynamic reconstruction. Codes are available at https://github.com/ispc-lab/LiDAR4D.  
  </ol>  
</details>  
**comments**: Accepted by CVPR 2024. Project Page:
  https://dyfcalid.github.io/LiDAR4D  
  
### [Neural Radiance Fields with Torch Units](http://arxiv.org/abs/2404.02617)  
Bingnan Ni, Huanyu Wang, Dongfeng Bai, Minghe Weng, Dexin Qi, Weichao Qiu, Bingbing Liu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) give rise to learning-based 3D reconstruction methods widely used in industrial applications. Although prevalent methods achieve considerable improvements in small-scale scenes, accomplishing reconstruction in complex and large-scale scenes is still challenging. First, the background in complex scenes shows a large variance among different views. Second, the current inference pattern, $i.e.$ , a pixel only relies on an individual camera ray, fails to capture contextual information. To solve these problems, we propose to enlarge the ray perception field and build up the sample points interactions. In this paper, we design a novel inference pattern that encourages a single camera ray possessing more contextual information, and models the relationship among sample points on each camera ray. To hold contextual information,a camera ray in our proposed method can render a patch of pixels simultaneously. Moreover, we replace the MLP in neural radiance field models with distance-aware convolutions to enhance the feature propagation among sample points from the same camera ray. To summarize, as a torchlight, a ray in our proposed method achieves rendering a patch of image. Thus, we call the proposed method, Torch-NeRF. Extensive experiments on KITTI-360 and LLFF show that the Torch-NeRF exhibits excellent performance.  
  </ol>  
</details>  
  
### [Freditor: High-Fidelity and Transferable NeRF Editing by Frequency Decomposition](http://arxiv.org/abs/2404.02514)  
Yisheng He, Weihao Yuan, Siyu Zhu, Zilong Dong, Liefeng Bo, Qixing Huang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper enables high-fidelity, transferable NeRF editing by frequency decomposition. Recent NeRF editing pipelines lift 2D stylization results to 3D scenes while suffering from blurry results, and fail to capture detailed structures caused by the inconsistency between 2D editings. Our critical insight is that low-frequency components of images are more multiview-consistent after editing compared with their high-frequency parts. Moreover, the appearance style is mainly exhibited on the low-frequency components, and the content details especially reside in high-frequency parts. This motivates us to perform editing on low-frequency components, which results in high-fidelity edited scenes. In addition, the editing is performed in the low-frequency feature space, enabling stable intensity control and novel scene transfer. Comprehensive experiments conducted on photorealistic datasets demonstrate the superior performance of high-fidelity and transferable NeRF editing. The project page is at \url{https://aigc3d.github.io/freditor}.  
  </ol>  
</details>  
  
### [NeRFCodec: Neural Feature Compression Meets Neural Radiance Fields for Memory-Efficient Scene Representation](http://arxiv.org/abs/2404.02185)  
Sicheng Li, Hao Li, Yiyi Liao, Lu Yu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The emergence of Neural Radiance Fields (NeRF) has greatly impacted 3D scene modeling and novel-view synthesis. As a kind of visual media for 3D scene representation, compression with high rate-distortion performance is an eternal target. Motivated by advances in neural compression and neural field representation, we propose NeRFCodec, an end-to-end NeRF compression framework that integrates non-linear transform, quantization, and entropy coding for memory-efficient scene representation. Since training a non-linear transform directly on a large scale of NeRF feature planes is impractical, we discover that pre-trained neural 2D image codec can be utilized for compressing the features when adding content-specific parameters. Specifically, we reuse neural 2D image codec but modify its encoder and decoder heads, while keeping the other parts of the pre-trained decoder frozen. This allows us to train the full pipeline via supervision of rendering loss and entropy loss, yielding the rate-distortion balance by updating the content-specific parameters. At test time, the bitstreams containing latent code, feature decoder head, and other side information are transmitted for communication. Experimental results demonstrate our method outperforms existing NeRF compression methods, enabling high-quality novel view synthesis with a memory budget of 0.5 MB.  
  </ol>  
</details>  
**comments**: Accepted at CVPR2024. The source code will be released  
  
  



