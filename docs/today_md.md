<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#TAMBRIDGE:-Bridging-Frame-Centered-Tracking-and-3D-Gaussian-Splatting-for-Enhanced-SLAM>TAMBRIDGE: Bridging Frame-Centered Tracking and 3D Gaussian Splatting for Enhanced SLAM</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#$\textit{S}^3$Gaussian:-Self-Supervised-Street-Gaussians-for-Autonomous-Driving>$\textit{S}^3$Gaussian: Self-Supervised Street Gaussians for Autonomous Driving</a></li>
        <li><a href=#TetSphere-Splatting:-Representing-High-Quality-Geometry-with-Lagrangian-Volumetric-Meshes>TetSphere Splatting: Representing High-Quality Geometry with Lagrangian Volumetric Meshes</a></li>
        <li><a href=#NeRF-View-Synthesis:-Subjective-Quality-Assessment-and-Objective-Metrics-Evaluation>NeRF View Synthesis: Subjective Quality Assessment and Objective Metrics Evaluation</a></li>
        <li><a href=#IReNe:-Instant-Recoloring-in-Neural-Radiance-Fields>IReNe: Instant Recoloring in Neural Radiance Fields</a></li>
        <li><a href=#HINT:-Learning-Complete-Human-Neural-Representations-from-Limited-Viewpoints>HINT: Learning Complete Human Neural Representations from Limited Viewpoints</a></li>
        <li><a href=#View-Consistent-Hierarchical-3D-SegmentationUsing-Ultrametric-Feature-Fields>View-Consistent Hierarchical 3D SegmentationUsing Ultrametric Feature Fields</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [TAMBRIDGE: Bridging Frame-Centered Tracking and 3D Gaussian Splatting for Enhanced SLAM](http://arxiv.org/abs/2405.19614)  
Peifeng Jiang, Hong Liu, Xia Li, Ti Wang, Fabian Zhang, Joachim M. Buhmann  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The limited robustness of 3D Gaussian Splatting (3DGS) to motion blur and camera noise, along with its poor real-time performance, restricts its application in robotic SLAM tasks. Upon analysis, the primary causes of these issues are the density of views with motion blur and the cumulative errors in dense pose estimation from calculating losses based on noisy original images and rendering results, which increase the difficulty of 3DGS rendering convergence. Thus, a cutting-edge 3DGS-based SLAM system is introduced, leveraging the efficiency and flexibility of 3DGS to achieve real-time performance while remaining robust against sensor noise, motion blur, and the challenges posed by long-session SLAM. Central to this approach is the Fusion Bridge module, which seamlessly integrates tracking-centered ORB Visual Odometry with mapping-centered online 3DGS. Precise pose initialization is enabled by this module through joint optimization of re-projection and rendering loss, as well as strategic view selection, enhancing rendering convergence in large-scale scenes. Extensive experiments demonstrate state-of-the-art rendering quality and localization accuracy, positioning this system as a promising solution for real-world robotics applications that require stable, near-real-time performance. Our project is available at https://ZeldaFromHeaven.github.io/TAMBRIDGE/  
  </ol>  
</details>  
  
  



## NeRF  

### [ $\textit{S}^3$ Gaussian: Self-Supervised Street Gaussians for Autonomous Driving](http://arxiv.org/abs/2405.20323)  
[[code](https://github.com/nnanhuang/s3gaussian)]  
Nan Huang, Xiaobao Wei, Wenzhao Zheng, Pengju An, Ming Lu, Wei Zhan, Masayoshi Tomizuka, Kurt Keutzer, Shanghang Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Photorealistic 3D reconstruction of street scenes is a critical technique for developing real-world simulators for autonomous driving. Despite the efficacy of Neural Radiance Fields (NeRF) for driving scenes, 3D Gaussian Splatting (3DGS) emerges as a promising direction due to its faster speed and more explicit representation. However, most existing street 3DGS methods require tracked 3D vehicle bounding boxes to decompose the static and dynamic elements for effective reconstruction, limiting their applications for in-the-wild scenarios. To facilitate efficient 3D scene reconstruction without costly annotations, we propose a self-supervised street Gaussian ($\textit{S}^3$Gaussian) method to decompose dynamic and static elements from 4D consistency. We represent each scene with 3D Gaussians to preserve the explicitness and further accompany them with a spatial-temporal field network to compactly model the 4D dynamics. We conduct extensive experiments on the challenging Waymo-Open dataset to evaluate the effectiveness of our method. Our $\textit{S}^3$Gaussian demonstrates the ability to decompose static and dynamic scenes and achieves the best performance without using 3D annotations. Code is available at: https://github.com/nnanhuang/S3Gaussian/.  
  </ol>  
</details>  
**comments**: Code is available at: https://github.com/nnanhuang/S3Gaussian/  
  
### [TetSphere Splatting: Representing High-Quality Geometry with Lagrangian Volumetric Meshes](http://arxiv.org/abs/2405.20283)  
Minghao Guo, Bohan Wang, Kaiming He, Wojciech Matusik  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present TetSphere splatting, an explicit, Lagrangian representation for reconstructing 3D shapes with high-quality geometry. In contrast to conventional object reconstruction methods which predominantly use Eulerian representations, including both neural implicit (e.g., NeRF, NeuS) and explicit representations (e.g., DMTet), and often struggle with high computational demands and suboptimal mesh quality, TetSphere splatting utilizes an underused but highly effective geometric primitive -- tetrahedral meshes. This approach directly yields superior mesh quality without relying on neural networks or post-processing. It deforms multiple initial tetrahedral spheres to accurately reconstruct the 3D shape through a combination of differentiable rendering and geometric energy optimization, resulting in significant computational efficiency. Serving as a robust and versatile geometry representation, Tet-Sphere splatting seamlessly integrates into diverse applications, including single-view 3D reconstruction, image-/text-to-3D content generation. Experimental results demonstrate that TetSphere splatting outperforms existing representations, delivering faster optimization speed, enhanced mesh quality, and reliable preservation of thin structures.  
  </ol>  
</details>  
  
### [NeRF View Synthesis: Subjective Quality Assessment and Objective Metrics Evaluation](http://arxiv.org/abs/2405.20078)  
Pedro Martin, Antonio Rodrigues, Joao Ascenso, Maria Paula Queluz  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural radiance fields (NeRF) are a groundbreaking computer vision technology that enables the generation of high-quality, immersive visual content from multiple viewpoints. This capability holds significant advantages for applications such as virtual/augmented reality, 3D modelling and content creation for the film and entertainment industry. However, the evaluation of NeRF methods poses several challenges, including a lack of comprehensive datasets, reliable assessment methodologies, and objective quality metrics. This paper addresses the problem of NeRF quality assessment thoroughly, by conducting a rigorous subjective quality assessment test that considers several scene classes and recently proposed NeRF view synthesis methods. Additionally, the performance of a wide range of state-of-the-art conventional and learning-based full-reference 2D image and video quality assessment metrics is evaluated against the subjective scores of the subjective study. The experimental results are analyzed in depth, providing a comparative evaluation of several NeRF methods and objective quality metrics, across different classes of visual scenes, including real and synthetic content for front-face and 360-degree camera trajectories.  
  </ol>  
</details>  
  
### [IReNe: Instant Recoloring in Neural Radiance Fields](http://arxiv.org/abs/2405.19876)  
Alessio Mazzucchelli, Adrian Garcia-Garcia, Elena Garces, Fernando Rivas-Manzaneque, Francesc Moreno-Noguer, Adrian Penate-Sanchez  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Advances in NERFs have allowed for 3D scene reconstructions and novel view synthesis. Yet, efficiently editing these representations while retaining photorealism is an emerging challenge. Recent methods face three primary limitations: they're slow for interactive use, lack precision at object boundaries, and struggle to ensure multi-view consistency. We introduce IReNe to address these limitations, enabling swift, near real-time color editing in NeRF. Leveraging a pre-trained NeRF model and a single training image with user-applied color edits, IReNe swiftly adjusts network parameters in seconds. This adjustment allows the model to generate new scene views, accurately representing the color changes from the training image while also controlling object boundaries and view-specific effects. Object boundary control is achieved by integrating a trainable segmentation module into the model. The process gains efficiency by retraining only the weights of the last network layer. We observed that neurons in this layer can be classified into those responsible for view-dependent appearance and those contributing to diffuse appearance. We introduce an automated classification approach to identify these neuron types and exclusively fine-tune the weights of the diffuse neurons. This further accelerates training and ensures consistent color edits across different views. A thorough validation on a new dataset, with edited object colors, shows significant quantitative and qualitative advancements over competitors, accelerating speeds by 5x to 500x.  
  </ol>  
</details>  
  
### [HINT: Learning Complete Human Neural Representations from Limited Viewpoints](http://arxiv.org/abs/2405.19712)  
Alessandro Sanvito, Andrea Ramazzina, Stefanie Walz, Mario Bijelic, Felix Heide  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    No augmented application is possible without animated humanoid avatars. At the same time, generating human replicas from real-world monocular hand-held or robotic sensor setups is challenging due to the limited availability of views. Previous work showed the feasibility of virtual avatars but required the presence of 360 degree views of the targeted subject. To address this issue, we propose HINT, a NeRF-based algorithm able to learn a detailed and complete human model from limited viewing angles. We achieve this by introducing a symmetry prior, regularization constraints, and training cues from large human datasets. In particular, we introduce a sagittal plane symmetry prior to the appearance of the human, directly supervise the density function of the human model using explicit 3D body modeling, and leverage a co-learned human digitization network as additional supervision for the unseen angles. As a result, our method can reconstruct complete humans even from a few viewing angles, increasing performance by more than 15% PSNR compared to previous state-of-the-art algorithms.  
  </ol>  
</details>  
  
### [View-Consistent Hierarchical 3D SegmentationUsing Ultrametric Feature Fields](http://arxiv.org/abs/2405.19678)  
Haodi He, Colton Stearns, Adam W. Harley, Leonidas J. Guibas  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Large-scale vision foundation models such as Segment Anything (SAM) demonstrate impressive performance in zero-shot image segmentation at multiple levels of granularity. However, these zero-shot predictions are rarely 3D-consistent. As the camera viewpoint changes in a scene, so do the segmentation predictions, as well as the characterizations of ``coarse" or ``fine" granularity. In this work, we address the challenging task of lifting multi-granular and view-inconsistent image segmentations into a hierarchical and 3D-consistent representation. We learn a novel feature field within a Neural Radiance Field (NeRF) representing a 3D scene, whose segmentation structure can be revealed at different scales by simply using different thresholds on feature distance. Our key idea is to learn an ultrametric feature space, which unlike a Euclidean space, exhibits transitivity in distance-based grouping, naturally leading to a hierarchical clustering. Put together, our method takes view-inconsistent multi-granularity 2D segmentations as input and produces a hierarchy of 3D-consistent segmentations as output. We evaluate our method and several baselines on synthetic datasets with multi-view images and multi-granular segmentation, showcasing improved accuracy and viewpoint-consistency. We additionally provide qualitative examples of our model's 3D hierarchical segmentations in real world scenes.\footnote{The code and dataset are available at:  
  </ol>  
</details>  
  
  



