<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Fashion-Image-to-Image-Translation-for-Complementary-Item-Retrieval>Fashion Image-to-Image Translation for Complementary Item Retrieval</a></li>
        <li><a href=#MambaLoc:-Efficient-Camera-Localisation-via-State-Space-Model>MambaLoc: Efficient Camera Localisation via State Space Model</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#$R^2$-Mesh:-Reinforcement-Learning-Powered-Mesh-Reconstruction-via-Geometry-and-Appearance-Refinement>$R^2$-Mesh: Reinforcement Learning Powered Mesh Reconstruction via Geometry and Appearance Refinement</a></li>
        <li><a href=#DiscoNeRF:-Class-Agnostic-Object-Field-for-3D-Object-Discovery>DiscoNeRF: Class-Agnostic Object Field for 3D Object Discovery</a></li>
        <li><a href=#CHASE:-3D-Consistent-Human-Avatars-with-Sparse-Inputs-via-Gaussian-Splatting-and-Contrastive-Learning>CHASE: 3D-Consistent Human Avatars with Sparse Inputs via Gaussian Splatting and Contrastive Learning</a></li>
        <li><a href=#S^3D-NeRF:-Single-Shot-Speech-Driven-Neural-Radiance-Field-for-High-Fidelity-Talking-Head-Synthesis>S^3D-NeRF: Single-Shot Speech-Driven Neural Radiance Field for High Fidelity Talking Head Synthesis</a></li>
        <li><a href=#SSNeRF:-Sparse-View-Semi-supervised-Neural-Radiance-Fields-with-Augmentation>SSNeRF: Sparse View Semi-supervised Neural Radiance Fields with Augmentation</a></li>
        <li><a href=#HybridOcc:-NeRF-Enhanced-Transformer-based-Multi-Camera-3D-Occupancy-Prediction>HybridOcc: NeRF Enhanced Transformer-based Multi-Camera 3D Occupancy Prediction</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Fashion Image-to-Image Translation for Complementary Item Retrieval](http://arxiv.org/abs/2408.09847)  
Matteo Attimonelli, Claudio Pomo, Dietmar Jannach, Tommaso Di Noia  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The increasing demand for online fashion retail has boosted research in fashion compatibility modeling and item retrieval, focusing on matching user queries (textual descriptions or reference images) with compatible fashion items. A key challenge is top-bottom retrieval, where precise compatibility modeling is essential. Traditional methods, often based on Bayesian Personalized Ranking (BPR), have shown limited performance. Recent efforts have explored using generative models in compatibility modeling and item retrieval, where generated images serve as additional inputs. However, these approaches often overlook the quality of generated images, which could be crucial for model performance. Additionally, generative models typically require large datasets, posing challenges when such data is scarce.   To address these issues, we introduce the Generative Compatibility Model (GeCo), a two-stage approach that improves fashion image retrieval through paired image-to-image translation. First, the Complementary Item Generation Model (CIGM), built on Conditional Generative Adversarial Networks (GANs), generates target item images (e.g., bottoms) from seed items (e.g., tops), offering conditioning signals for retrieval. These generated samples are then integrated into GeCo, enhancing compatibility modeling and retrieval accuracy. Evaluations on three datasets show that GeCo outperforms state-of-the-art baselines. Key contributions include: (i) the GeCo model utilizing paired image-to-image translation within the Composed Image Retrieval framework, (ii) comprehensive evaluations on benchmark datasets, and (iii) the release of a new Fashion Taobao dataset designed for top-bottom retrieval, promoting further research.  
  </ol>  
</details>  
  
### [MambaLoc: Efficient Camera Localisation via State Space Model](http://arxiv.org/abs/2408.09680)  
Jialu Wang, Kaichen Zhou, Andrew Markham, Niki Trigoni  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Location information is pivotal for the automation and intelligence of terminal devices and edge-cloud IoT systems, such as autonomous vehicles and augmented reality. However, achieving reliable positioning across diverse IoT applications remains challenging due to significant training costs and the necessity of densely collected data. To tackle these issues, we have innovatively applied the selective state space (SSM) model to visual localization, introducing a new model named MambaLoc. The proposed model demonstrates exceptional training efficiency by capitalizing on the SSM model's strengths in efficient feature extraction, rapid computation, and memory optimization, and it further ensures robustness in sparse data environments due to its parameter sparsity. Additionally, we propose the Global Information Selector (GIS), which leverages selective SSM to implicitly achieve the efficient global feature extraction capabilities of Non-local Neural Networks. This design leverages the computational efficiency of the SSM model alongside the Non-local Neural Networks' capacity to capture long-range dependencies with minimal layers. Consequently, the GIS enables effective global information capture while significantly accelerating convergence. Our extensive experimental validation using public indoor and outdoor datasets first demonstrates our model's effectiveness, followed by evidence of its versatility with various existing localization models.  
  </ol>  
</details>  
  
  



## NeRF  

### [ $R^2$ -Mesh: Reinforcement Learning Powered Mesh Reconstruction via Geometry and Appearance Refinement](http://arxiv.org/abs/2408.10135)  
Haoyang Wang, Liming Liu, Quanlu Jia, Jiangkai Wu, Haodan Zhang, Peiheng Wang, Xinggong Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Mesh reconstruction based on Neural Radiance Fields (NeRF) is popular in a variety of applications such as computer graphics, virtual reality, and medical imaging due to its efficiency in handling complex geometric structures and facilitating real-time rendering. However, existing works often fail to capture fine geometric details accurately and struggle with optimizing rendering quality. To address these challenges, we propose a novel algorithm that progressively generates and optimizes meshes from multi-view images. Our approach initiates with the training of a NeRF model to establish an initial Signed Distance Field (SDF) and a view-dependent appearance field. Subsequently, we iteratively refine the SDF through a differentiable mesh extraction method, continuously updating both the vertex positions and their connectivity based on the loss from mesh differentiable rasterization, while also optimizing the appearance representation. To further leverage high-fidelity and detail-rich representations from NeRF, we propose an online-learning strategy based on Upper Confidence Bound (UCB) to enhance viewpoints by adaptively incorporating images rendered by the initial NeRF model into the training dataset. Through extensive experiments, we demonstrate that our method delivers highly competitive and robust performance in both mesh rendering quality and geometric quality.  
  </ol>  
</details>  
  
### [DiscoNeRF: Class-Agnostic Object Field for 3D Object Discovery](http://arxiv.org/abs/2408.09928)  
Corentin Dumery, Aoxiang Fan, Ren Li, Nicolas Talabot, Pascal Fua  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRFs) have become a powerful tool for modeling 3D scenes from multiple images. However, NeRFs remain difficult to segment into semantically meaningful regions. Previous approaches to 3D segmentation of NeRFs either require user interaction to isolate a single object, or they rely on 2D semantic masks with a limited number of classes for supervision. As a consequence, they generalize poorly to class-agnostic masks automatically generated in real scenes. This is attributable to the ambiguity arising from zero-shot segmentation, yielding inconsistent masks across views. In contrast, we propose a method that is robust to inconsistent segmentations and successfully decomposes the scene into a set of objects of any class. By introducing a limited number of competing object slots against which masks are matched, a meaningful object representation emerges that best explains the 2D supervision and minimizes an additional regularization term. Our experiments demonstrate the ability of our method to generate 3D panoptic segmentations on complex scenes, and extract high-quality 3D assets from NeRFs that can then be used in virtual 3D environments.  
  </ol>  
</details>  
  
### [CHASE: 3D-Consistent Human Avatars with Sparse Inputs via Gaussian Splatting and Contrastive Learning](http://arxiv.org/abs/2408.09663)  
Haoyu Zhao, Hao Wang, Chen Yang, Wei Shen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in human avatar synthesis have utilized radiance fields to reconstruct photo-realistic animatable human avatars. However, both NeRFs-based and 3DGS-based methods struggle with maintaining 3D consistency and exhibit suboptimal detail reconstruction, especially with sparse inputs. To address this challenge, we propose CHASE, which introduces supervision from intrinsic 3D consistency across poses and 3D geometry contrastive learning, achieving performance comparable with sparse inputs to that with full inputs. Following previous work, we first integrate a skeleton-driven rigid deformation and a non-rigid cloth dynamics deformation to coordinate the movements of individual Gaussians during animation, reconstructing basic avatar with coarse 3D consistency. To improve 3D consistency under sparse inputs, we design Dynamic Avatar Adjustment(DAA) to adjust deformed Gaussians based on a selected similar pose/image from the dataset. Minimizing the difference between the image rendered by adjusted Gaussians and the image with the similar pose serves as an additional form of supervision for avatar. Furthermore, we propose a 3D geometry contrastive learning strategy to maintain the 3D global consistency of generated avatars. Though CHASE is designed for sparse inputs, it surprisingly outperforms current SOTA methods \textbf{in both full and sparse settings} on the ZJU-MoCap and H36M datasets, demonstrating that our CHASE successfully maintains avatar's 3D consistency, hence improving rendering quality.  
  </ol>  
</details>  
**comments**: 13 pages, 6 figures  
  
### [S^3D-NeRF: Single-Shot Speech-Driven Neural Radiance Field for High Fidelity Talking Head Synthesis](http://arxiv.org/abs/2408.09347)  
Dongze Li, Kang Zhao, Wei Wang, Yifeng Ma, Bo Peng, Yingya Zhang, Jing Dong  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Talking head synthesis is a practical technique with wide applications. Current Neural Radiance Field (NeRF) based approaches have shown their superiority on driving one-shot talking heads with videos or signals regressed from audio. However, most of them failed to take the audio as driven information directly, unable to enjoy the flexibility and availability of speech. Since mapping audio signals to face deformation is non-trivial, we design a Single-Shot Speech-Driven Neural Radiance Field (S^3D-NeRF) method in this paper to tackle the following three difficulties: learning a representative appearance feature for each identity, modeling motion of different face regions with audio, and keeping the temporal consistency of the lip area. To this end, we introduce a Hierarchical Facial Appearance Encoder to learn multi-scale representations for catching the appearance of different speakers, and elaborate a Cross-modal Facial Deformation Field to perform speech animation according to the relationship between the audio signal and different face regions. Moreover, to enhance the temporal consistency of the important lip area, we introduce a lip-sync discriminator to penalize the out-of-sync audio-visual sequences. Extensive experiments have shown that our S^3D-NeRF surpasses previous arts on both video fidelity and audio-lip synchronization.  
  </ol>  
</details>  
**comments**: ECCV 2024  
  
### [SSNeRF: Sparse View Semi-supervised Neural Radiance Fields with Augmentation](http://arxiv.org/abs/2408.09144)  
Xiao Cao, Beibei Lin, Bo Wang, Zhiyong Huang, Robby T. Tan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Sparse view NeRF is challenging because limited input images lead to an under constrained optimization problem for volume rendering. Existing methods address this issue by relying on supplementary information, such as depth maps. However, generating this supplementary information accurately remains problematic and often leads to NeRF producing images with undesired artifacts. To address these artifacts and enhance robustness, we propose SSNeRF, a sparse view semi supervised NeRF method based on a teacher student framework. Our key idea is to challenge the NeRF module with progressively severe sparse view degradation while providing high confidence pseudo labels. This approach helps the NeRF model become aware of noise and incomplete information associated with sparse views, thus improving its robustness. The novelty of SSNeRF lies in its sparse view specific augmentations and semi supervised learning mechanism. In this approach, the teacher NeRF generates novel views along with confidence scores, while the student NeRF, perturbed by the augmented input, learns from the high confidence pseudo labels. Our sparse view degradation augmentation progressively injects noise into volume rendering weights, perturbs feature maps in vulnerable layers, and simulates sparse view blurriness. These augmentation strategies force the student NeRF to recognize degradation and produce clearer rendered views. By transferring the student's parameters to the teacher, the teacher gains increased robustness in subsequent training iterations. Extensive experiments demonstrate the effectiveness of our SSNeRF in generating novel views with less sparse view degradation. We will release code upon acceptance.  
  </ol>  
</details>  
  
### [HybridOcc: NeRF Enhanced Transformer-based Multi-Camera 3D Occupancy Prediction](http://arxiv.org/abs/2408.09104)  
Xiao Zhao, Bo Chen, Mingyang Sun, Dingkang Yang, Youxing Wang, Xukun Zhang, Mingcheng Li, Dongliang Kou, Xiaoyi Wei, Lihua Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Vision-based 3D semantic scene completion (SSC) describes autonomous driving scenes through 3D volume representations. However, the occlusion of invisible voxels by scene surfaces poses challenges to current SSC methods in hallucinating refined 3D geometry. This paper proposes HybridOcc, a hybrid 3D volume query proposal method generated by Transformer framework and NeRF representation and refined in a coarse-to-fine SSC prediction framework. HybridOcc aggregates contextual features through the Transformer paradigm based on hybrid query proposals while combining it with NeRF representation to obtain depth supervision. The Transformer branch contains multiple scales and uses spatial cross-attention for 2D to 3D transformation. The newly designed NeRF branch implicitly infers scene occupancy through volume rendering, including visible and invisible voxels, and explicitly captures scene depth rather than generating RGB color. Furthermore, we present an innovative occupancy-aware ray sampling method to orient the SSC task instead of focusing on the scene surface, further improving the overall performance. Extensive experiments on nuScenes and SemanticKITTI datasets demonstrate the effectiveness of our HybridOcc on the SSC task.  
  </ol>  
</details>  
**comments**: Accepted to IEEE RAL  
  
  



