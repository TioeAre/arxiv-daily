<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#SPVLoc:-Semantic-Panoramic-Viewport-Matching-for-6D-Camera-Localization-in-Unseen-Environments>SPVLoc: Semantic Panoramic Viewport Matching for 6D Camera Localization in Unseen Environments</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Exploring-selective-image-matching-methods-for-zero-shot-and-few-sample-unsupervised-domain-adaptation-of-urban-canopy-prediction>Exploring selective image matching methods for zero-shot and few-sample unsupervised domain adaptation of urban canopy prediction</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Gaussian-Splatting-Decoder-for-3D-aware-Generative-Adversarial-Networks>Gaussian Splatting Decoder for 3D-aware Generative Adversarial Networks</a></li>
        <li><a href=#Enhancing-3D-Fidelity-of-Text-to-3D-using-Cross-View-Correspondences>Enhancing 3D Fidelity of Text-to-3D using Cross-View Correspondences</a></li>
        <li><a href=#1st-Place-Solution-for-ICCV-2023-OmniObject3D-Challenge:-Sparse-View-Reconstruction>1st Place Solution for ICCV 2023 OmniObject3D Challenge: Sparse-View Reconstruction</a></li>
        <li><a href=#SRGS:-Super-Resolution-3D-Gaussian-Splatting>SRGS: Super-Resolution 3D Gaussian Splatting</a></li>
        <li><a href=#Plug-and-Play-Acceleration-of-Occupancy-Grid-based-NeRF-Rendering-using-VDB-Grid-and-Hierarchical-Ray-Traversal>Plug-and-Play Acceleration of Occupancy Grid-based NeRF Rendering using VDB Grid and Hierarchical Ray Traversal</a></li>
        <li><a href=#Taming-Latent-Diffusion-Model-for-Neural-Radiance-Field-Inpainting>Taming Latent Diffusion Model for Neural Radiance Field Inpainting</a></li>
        <li><a href=#Video2Game:-Real-time,-Interactive,-Realistic-and-Browser-Compatible-Environment-from-a-Single-Video>Video2Game: Real-time, Interactive, Realistic and Browser-Compatible Environment from a Single Video</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [SPVLoc: Semantic Panoramic Viewport Matching for 6D Camera Localization in Unseen Environments](http://arxiv.org/abs/2404.10527)  
Niklas Gard, Anna Hilsmann, Peter Eisert  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we present SPVLoc, a global indoor localization method that accurately determines the six-dimensional (6D) camera pose of a query image and requires minimal scene-specific prior knowledge and no scene-specific training. Our approach employs a novel matching procedure to localize the perspective camera's viewport, given as an RGB image, within a set of panoramic semantic layout representations of the indoor environment. The panoramas are rendered from an untextured 3D reference model, which only comprises approximate structural information about room shapes, along with door and window annotations. We demonstrate that a straightforward convolutional network structure can successfully achieve image-to-panorama and ultimately image-to-model matching. Through a viewport classification score, we rank reference panoramas and select the best match for the query image. Then, a 6D relative pose is estimated between the chosen panorama and query image. Our experiments demonstrate that this approach not only efficiently bridges the domain gap but also generalizes well to previously unseen scenes that are not part of the training data. Moreover, it achieves superior localization accuracy compared to the state of the art methods and also estimates more degrees of freedom of the camera pose. We will make our source code publicly available at https://github.com/fraunhoferhhi/spvloc .  
  </ol>  
</details>  
**comments**: This submission includes the paper and supplementary material. 24
  pages, 11 figures  
  
  



## Image Matching  

### [Exploring selective image matching methods for zero-shot and few-sample unsupervised domain adaptation of urban canopy prediction](http://arxiv.org/abs/2404.10626)  
John Francis, Stephen Law  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We explore simple methods for adapting a trained multi-task UNet which predicts canopy cover and height to a new geographic setting using remotely sensed data without the need of training a domain-adaptive classifier and extensive fine-tuning. Extending previous research, we followed a selective alignment process to identify similar images in the two geographical domains and then tested an array of data-based unsupervised domain adaptation approaches in a zero-shot setting as well as with a small amount of fine-tuning. We find that the selective aligned data-based image matching methods produce promising results in a zero-shot setting, and even more so with a small amount of fine-tuning. These methods outperform both an untransformed baseline and a popular data-based image-to-image translation model. The best performing methods were pixel distribution adaptation and fourier domain adaptation on the canopy cover and height tasks respectively.  
  </ol>  
</details>  
**comments**: ICLR 2024 Machine Learning for Remote Sensing (ML4RS) Workshop  
  
  



## NeRF  

### [Gaussian Splatting Decoder for 3D-aware Generative Adversarial Networks](http://arxiv.org/abs/2404.10625)  
Florian Barthel, Arian Beckmann, Wieland Morgenstern, Anna Hilsmann, Peter Eisert  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    NeRF-based 3D-aware Generative Adversarial Networks (GANs) like EG3D or GIRAFFE have shown very high rendering quality under large representational variety. However, rendering with Neural Radiance Fields poses challenges for 3D applications: First, the significant computational demands of NeRF rendering preclude its use on low-power devices, such as mobiles and VR/AR headsets. Second, implicit representations based on neural networks are difficult to incorporate into explicit 3D scenes, such as VR environments or video games. 3D Gaussian Splatting (3DGS) overcomes these limitations by providing an explicit 3D representation that can be rendered efficiently at high frame rates. In this work, we present a novel approach that combines the high rendering quality of NeRF-based 3D-aware GANs with the flexibility and computational advantages of 3DGS. By training a decoder that maps implicit NeRF representations to explicit 3D Gaussian Splatting attributes, we can integrate the representational diversity and quality of 3D GANs into the ecosystem of 3D Gaussian Splatting for the first time. Additionally, our approach allows for a high resolution GAN inversion and real-time GAN editing with 3D Gaussian Splatting scenes.  
  </ol>  
</details>  
**comments**: CVPRW  
  
### [Enhancing 3D Fidelity of Text-to-3D using Cross-View Correspondences](http://arxiv.org/abs/2404.10603)  
Seungwook Kim, Kejie Li, Xueqing Deng, Yichun Shi, Minsu Cho, Peng Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Leveraging multi-view diffusion models as priors for 3D optimization have alleviated the problem of 3D consistency, e.g., the Janus face problem or the content drift problem, in zero-shot text-to-3D models. However, the 3D geometric fidelity of the output remains an unresolved issue; albeit the rendered 2D views are realistic, the underlying geometry may contain errors such as unreasonable concavities. In this work, we propose CorrespondentDream, an effective method to leverage annotation-free, cross-view correspondences yielded from the diffusion U-Net to provide additional 3D prior to the NeRF optimization process. We find that these correspondences are strongly consistent with human perception, and by adopting it in our loss design, we are able to produce NeRF models with geometries that are more coherent with common sense, e.g., more smoothed object surface, yielding higher 3D fidelity. We demonstrate the efficacy of our approach through various comparative qualitative results and a solid user study.  
  </ol>  
</details>  
**comments**: 25 pages, 22 figures, accepted to CVPR 2024  
  
### [1st Place Solution for ICCV 2023 OmniObject3D Challenge: Sparse-View Reconstruction](http://arxiv.org/abs/2404.10441)  
Hang Du, Yaping Xue, Weidong Dai, Xuejun Yan, Jingjing Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this report, we present the 1st place solution for ICCV 2023 OmniObject3D Challenge: Sparse-View Reconstruction. The challenge aims to evaluate approaches for novel view synthesis and surface reconstruction using only a few posed images of each object. We utilize Pixel-NeRF as the basic model, and apply depth supervision as well as coarse-to-fine positional encoding. The experiments demonstrate the effectiveness of our approach in improving sparse-view reconstruction quality. We ranked first in the final test with a PSNR of 25.44614.  
  </ol>  
</details>  
  
### [SRGS: Super-Resolution 3D Gaussian Splatting](http://arxiv.org/abs/2404.10318)  
Xiang Feng, Yongbo He, Yubo Wang, Yan Yang, Zhenzhong Kuang, Yu Jun, Jianping Fan, Jiajun ding  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recently, 3D Gaussian Splatting (3DGS) has gained popularity as a novel explicit 3D representation. This approach relies on the representation power of Gaussian primitives to provide a high-quality rendering. However, primitives optimized at low resolution inevitably exhibit sparsity and texture deficiency, posing a challenge for achieving high-resolution novel view synthesis (HRNVS). To address this problem, we propose Super-Resolution 3D Gaussian Splatting (SRGS) to perform the optimization in a high-resolution (HR) space. The sub-pixel constraint is introduced for the increased viewpoints in HR space, exploiting the sub-pixel cross-view information of the multiple low-resolution (LR) views. The gradient accumulated from more viewpoints will facilitate the densification of primitives. Furthermore, a pre-trained 2D super-resolution model is integrated with the sub-pixel constraint, enabling these dense primitives to learn faithful texture features. In general, our method focuses on densification and texture learning to effectively enhance the representation ability of primitives. Experimentally, our method achieves high rendering quality on HRNVS only with LR inputs, outperforming state-of-the-art methods on challenging datasets such as Mip-NeRF 360 and Tanks & Temples. Related codes will be released upon acceptance.  
  </ol>  
</details>  
**comments**: submit ACM MM 2024  
  
### [Plug-and-Play Acceleration of Occupancy Grid-based NeRF Rendering using VDB Grid and Hierarchical Ray Traversal](http://arxiv.org/abs/2404.10272)  
[[code](https://github.com/yosshi999/faster-occgrid)]  
Yoshio Kato, Shuhei Tarashima  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Transmittance estimators such as Occupancy Grid (OG) can accelerate the training and rendering of Neural Radiance Field (NeRF) by predicting important samples that contributes much to the generated image. However, OG manages occupied regions in the form of the dense binary grid, in which there are many blocks with the same values that cause redundant examination of voxels' emptiness in ray-tracing. In our work, we introduce two techniques to improve the efficiency of ray-tracing in trained OG without fine-tuning. First, we replace the dense grids with VDB grids to reduce the spatial redundancy. Second, we use hierarchical digital differential analyzer (HDDA) to efficiently trace voxels in the VDB grids. Our experiments on NeRF-Synthetic and Mip-NeRF 360 datasets show that our proposed method successfully accelerates rendering NeRF-Synthetic dataset by 12% in average and Mip-NeRF 360 dataset by 4% in average, compared to a fast implementation of OG, NerfAcc, without losing the quality of rendered images.  
  </ol>  
</details>  
**comments**: Short paper for CVPR Neural Rendering Intelligence Workshop 2024.
  Code: https://github.com/Yosshi999/faster-occgrid  
  
### [Taming Latent Diffusion Model for Neural Radiance Field Inpainting](http://arxiv.org/abs/2404.09995)  
Chieh Hubert Lin, Changil Kim, Jia-Bin Huang, Qinbo Li, Chih-Yao Ma, Johannes Kopf, Ming-Hsuan Yang, Hung-Yu Tseng  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Field (NeRF) is a representation for 3D reconstruction from multi-view images. Despite some recent work showing preliminary success in editing a reconstructed NeRF with diffusion prior, they remain struggling to synthesize reasonable geometry in completely uncovered regions. One major reason is the high diversity of synthetic contents from the diffusion model, which hinders the radiance field from converging to a crisp and deterministic geometry. Moreover, applying latent diffusion models on real data often yields a textural shift incoherent to the image condition due to auto-encoding errors. These two problems are further reinforced with the use of pixel-distance losses. To address these issues, we propose tempering the diffusion model's stochasticity with per-scene customization and mitigating the textural shift with masked adversarial training. During the analyses, we also found the commonly used pixel and perceptual losses are harmful in the NeRF inpainting task. Through rigorous experiments, our framework yields state-of-the-art NeRF inpainting results on various real-world scenes. Project page: https://hubert0527.github.io/MALD-NeRF  
  </ol>  
</details>  
**comments**: Project page: https://hubert0527.github.io/MALD-NeRF  
  
### [Video2Game: Real-time, Interactive, Realistic and Browser-Compatible Environment from a Single Video](http://arxiv.org/abs/2404.09833)  
Hongchi Xia, Zhi-Hao Lin, Wei-Chiu Ma, Shenlong Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Creating high-quality and interactive virtual environments, such as games and simulators, often involves complex and costly manual modeling processes. In this paper, we present Video2Game, a novel approach that automatically converts videos of real-world scenes into realistic and interactive game environments. At the heart of our system are three core components:(i) a neural radiance fields (NeRF) module that effectively captures the geometry and visual appearance of the scene; (ii) a mesh module that distills the knowledge from NeRF for faster rendering; and (iii) a physics module that models the interactions and physical dynamics among the objects. By following the carefully designed pipeline, one can construct an interactable and actionable digital replica of the real world. We benchmark our system on both indoor and large-scale outdoor scenes. We show that we can not only produce highly-realistic renderings in real-time, but also build interactive games on top.  
  </ol>  
</details>  
**comments**: CVPR 2024. Project page (with code): https://video2game.github.io/  
  
  



