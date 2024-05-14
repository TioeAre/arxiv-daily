<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#OverlapMamba:-Novel-Shift-State-Space-Model-for-LiDAR-based-Place-Recognition>OverlapMamba: Novel Shift State Space Model for LiDAR-based Place Recognition</a></li>
        <li><a href=#HybridHash:-Hybrid-Convolutional-and-Self-Attention-Deep-Hashing-for-Image-Retrieval>HybridHash: Hybrid Convolutional and Self-Attention Deep Hashing for Image Retrieval</a></li>
        <li><a href=#JointLoc:-A-Real-time-Visual-Localization-Framework-for-Planetary-UAVs-Based-on-Joint-Relative-and-Absolute-Pose-Estimation>JointLoc: A Real-time Visual Localization Framework for Planetary UAVs Based on Joint Relative and Absolute Pose Estimation</a></li>
        <li><a href=#BoQ:-A-Place-is-Worth-a-Bag-of-Learnable-Queries>BoQ: A Place is Worth a Bag of Learnable Queries</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#RGBD-Glue:-General-Feature-Combination-for-Robust-RGB-D-Point-Cloud-Registration>RGBD-Glue: General Feature Combination for Robust RGB-D Point Cloud Registration</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Authentic-Hand-Avatar-from-a-Phone-Scan-via-Universal-Hand-Model>Authentic Hand Avatar from a Phone Scan via Universal Hand Model</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Synergistic-Integration-of-Coordinate-Network-and-Tensorial-Feature-for-Improving-Neural-Radiance-Fields-from-Sparse-Inputs>Synergistic Integration of Coordinate Network and Tensorial Feature for Improving Neural Radiance Fields from Sparse Inputs</a></li>
        <li><a href=#Point-Resampling-and-Ray-Transformation-Aid-to-Editable-NeRF-Models>Point Resampling and Ray Transformation Aid to Editable NeRF Models</a></li>
        <li><a href=#Hologram:-Realtime-Holographic-Overlays-via-LiDAR-Augmented-Reconstruction>Hologram: Realtime Holographic Overlays via LiDAR Augmented Reconstruction</a></li>
        <li><a href=#TD-NeRF:-Novel-Truncated-Depth-Prior-for-Joint-Camera-Pose-and-Neural-Radiance-Field-Optimization>TD-NeRF: Novel Truncated Depth Prior for Joint Camera Pose and Neural Radiance Field Optimization</a></li>
        <li><a href=#LIVE:-LaTex-Interactive-Visual-Editing>LIVE: LaTex Interactive Visual Editing</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [OverlapMamba: Novel Shift State Space Model for LiDAR-based Place Recognition](http://arxiv.org/abs/2405.07966)  
[[code](https://github.com/scnu-rislab/overlapmamba)]  
Qiuchi Xiang, Jintao Cheng, Jiehao Luo, Jin Wu, Rui Fan, Xieyuanli Chen, Xiaoyu Tang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Place recognition is the foundation for enabling autonomous systems to achieve independent decision-making and safe operations. It is also crucial in tasks such as loop closure detection and global localization within SLAM. Previous methods utilize mundane point cloud representations as input and deep learning-based LiDAR-based Place Recognition (LPR) approaches employing different point cloud image inputs with convolutional neural networks (CNNs) or transformer architectures. However, the recently proposed Mamba deep learning model, combined with state space models (SSMs), holds great potential for long sequence modeling. Therefore, we developed OverlapMamba, a novel network for place recognition, which represents input range views (RVs) as sequences. In a novel way, we employ a stochastic reconstruction approach to build shift state space models, compressing the visual representation. Evaluated on three different public datasets, our method effectively detects loop closures, showing robustness even when traversing previously visited locations from different directions. Relying on raw range view inputs, it outperforms typical LiDAR and multi-view combination methods in time complexity and speed, indicating strong place recognition capabilities and real-time efficiency.  
  </ol>  
</details>  
  
### [HybridHash: Hybrid Convolutional and Self-Attention Deep Hashing for Image Retrieval](http://arxiv.org/abs/2405.07524)  
[[code](https://github.com/shuaichaochao/hybridhash)]  
Chao He, Hongxi Wei  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Deep image hashing aims to map input images into simple binary hash codes via deep neural networks and thus enable effective large-scale image retrieval. Recently, hybrid networks that combine convolution and Transformer have achieved superior performance on various computer tasks and have attracted extensive attention from researchers. Nevertheless, the potential benefits of such hybrid networks in image retrieval still need to be verified. To this end, we propose a hybrid convolutional and self-attention deep hashing method known as HybridHash. Specifically, we propose a backbone network with stage-wise architecture in which the block aggregation function is introduced to achieve the effect of local self-attention and reduce the computational complexity. The interaction module has been elaborately designed to promote the communication of information between image blocks and to enhance the visual representations. We have conducted comprehensive experiments on three widely used datasets: CIFAR-10, NUS-WIDE and IMAGENET. The experimental results demonstrate that the method proposed in this paper has superior performance with respect to state-of-the-art deep hashing methods. Source code is available https://github.com/shuaichaochao/HybridHash.  
  </ol>  
</details>  
  
### [JointLoc: A Real-time Visual Localization Framework for Planetary UAVs Based on Joint Relative and Absolute Pose Estimation](http://arxiv.org/abs/2405.07429)  
Xubo Luo, Xue Wan, Yixing Gao, Yaolin Tian, Wei Zhang, Leizheng Shu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Unmanned aerial vehicles (UAVs) visual localization in planetary aims to estimate the absolute pose of the UAV in the world coordinate system through satellite maps and images captured by on-board cameras. However, since planetary scenes often lack significant landmarks and there are modal differences between satellite maps and UAV images, the accuracy and real-time performance of UAV positioning will be reduced. In order to accurately determine the position of the UAV in a planetary scene in the absence of the global navigation satellite system (GNSS), this paper proposes JointLoc, which estimates the real-time UAV position in the world coordinate system by adaptively fusing the absolute 2-degree-of-freedom (2-DoF) pose and the relative 6-degree-of-freedom (6-DoF) pose. Extensive comparative experiments were conducted on a proposed planetary UAV image cross-modal localization dataset, which contains three types of typical Martian topography generated via a simulation engine as well as real Martian UAV images from the Ingenuity helicopter. JointLoc achieved a root-mean-square error of 0.237m in the trajectories of up to 1,000m, compared to 0.594m and 0.557m for ORB-SLAM2 and ORB-SLAM3 respectively. The source code will be available at https://github.com/LuoXubo/JointLoc.  
  </ol>  
</details>  
**comments**: 8 pages  
  
### [BoQ: A Place is Worth a Bag of Learnable Queries](http://arxiv.org/abs/2405.07364)  
[[code](https://github.com/amaralibey/bag-of-queries)]  
Amar Ali-bey, Brahim Chaib-draa, Philippe Giguère  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In visual place recognition, accurately identifying and matching images of locations under varying environmental conditions and viewpoints remains a significant challenge. In this paper, we introduce a new technique, called Bag-of-Queries (BoQ), which learns a set of global queries designed to capture universal place-specific attributes. Unlike existing methods that employ self-attention and generate the queries directly from the input features, BoQ employs distinct learnable global queries, which probe the input features via cross-attention, ensuring consistent information aggregation. In addition, our technique provides an interpretable attention mechanism and integrates with both CNN and Vision Transformer backbones. The performance of BoQ is demonstrated through extensive experiments on 14 large-scale benchmarks. It consistently outperforms current state-of-the-art techniques including NetVLAD, MixVPR and EigenPlaces. Moreover, as a global retrieval technique (one-stage), BoQ surpasses two-stage retrieval methods, such as Patch-NetVLAD, TransVPR and R2Former, all while being orders of magnitude faster and more efficient. The code and model weights are publicly available at https://github.com/amaralibey/Bag-of-Queries.  
  </ol>  
</details>  
**comments**: Accepted at CVPR 2024  
  
  



## Keypoint Detection  

### [RGBD-Glue: General Feature Combination for Robust RGB-D Point Cloud Registration](http://arxiv.org/abs/2405.07594)  
Congjia Chen, Xiaoyu Jia, Yanhong Zheng, Yufu Qu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Point cloud registration is a fundamental task for estimating rigid transformations between point clouds. Previous studies have used geometric information for extracting features, matching and estimating transformation. Recently, owing to the advancement of RGB-D sensors, researchers have attempted to utilize visual information to improve registration performance. However, these studies focused on extracting distinctive features by deep feature fusion, which cannot effectively solve the negative effects of each feature's weakness, and cannot sufficiently leverage the valid information. In this paper, we propose a new feature combination framework, which applies a looser but more effective fusion and can achieve better performance. An explicit filter based on transformation consistency is designed for the combination framework, which can overcome each feature's weakness. And an adaptive threshold determined by the error distribution is proposed to extract more valid information from the two types of features. Owing to the distinctive design, our proposed framework can estimate more accurate correspondences and is applicable to both hand-crafted and learning-based feature descriptors. Experiments on ScanNet show that our method achieves a state-of-the-art performance and the rotation accuracy of 99.1%.  
  </ol>  
</details>  
  
  



## Image Matching  

### [Authentic Hand Avatar from a Phone Scan via Universal Hand Model](http://arxiv.org/abs/2405.07933)  
Gyeongsik Moon, Weipeng Xu, Rohan Joshi, Chenglei Wu, Takaaki Shiratori  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The authentic 3D hand avatar with every identifiable information, such as hand shapes and textures, is necessary for immersive experiences in AR/VR. In this paper, we present a universal hand model (UHM), which 1) can universally represent high-fidelity 3D hand meshes of arbitrary identities (IDs) and 2) can be adapted to each person with a short phone scan for the authentic hand avatar. For effective universal hand modeling, we perform tracking and modeling at the same time, while previous 3D hand models perform them separately. The conventional separate pipeline suffers from the accumulated errors from the tracking stage, which cannot be recovered in the modeling stage. On the other hand, ours does not suffer from the accumulated errors while having a much more concise overall pipeline. We additionally introduce a novel image matching loss function to address a skin sliding during the tracking and modeling, while existing works have not focused on it much. Finally, using learned priors from our UHM, we effectively adapt our UHM to each person's short phone scan for the authentic hand avatar.  
  </ol>  
</details>  
**comments**: Accepted to CVPR 2024  
  
  



## NeRF  

### [Synergistic Integration of Coordinate Network and Tensorial Feature for Improving Neural Radiance Fields from Sparse Inputs](http://arxiv.org/abs/2405.07857)  
[[code](https://github.com/mingyukim87/synergynerf)]  
Mingyu Kim, Jun-Seong Kim, Se-Young Yun, Jin-Hwa Kim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The multi-plane representation has been highlighted for its fast training and inference across static and dynamic neural radiance fields. This approach constructs relevant features via projection onto learnable grids and interpolating adjacent vertices. However, it has limitations in capturing low-frequency details and tends to overuse parameters for low-frequency features due to its bias toward fine details, despite its multi-resolution concept. This phenomenon leads to instability and inefficiency when training poses are sparse. In this work, we propose a method that synergistically integrates multi-plane representation with a coordinate-based network known for strong bias toward low-frequency signals. The coordinate-based network is responsible for capturing low-frequency details, while the multi-plane representation focuses on capturing fine-grained details. We demonstrate that using residual connections between them seamlessly preserves their own inherent properties. Additionally, the proposed progressive training scheme accelerates the disentanglement of these two features. We empirically show that the proposed method achieves comparable results to explicit encoding with fewer parameters, and particularly, it outperforms others for the static and dynamic NeRFs under sparse inputs.  
  </ol>  
</details>  
**comments**: ICML2024 ; Project page is accessible at
  https://mingyukim87.github.io/SynergyNeRF ; Code is available at
  https://github.com/MingyuKim87/SynergyNeRF  
  
### [Point Resampling and Ray Transformation Aid to Editable NeRF Models](http://arxiv.org/abs/2405.07306)  
Zhenyang Li, Zilong Chen, Feifan Qu, Mingqing Wang, Yizhou Zhao, Kai Zhang, Yifan Peng  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In NeRF-aided editing tasks, object movement presents difficulties in supervision generation due to the introduction of variability in object positions. Moreover, the removal operations of certain scene objects often lead to empty regions, presenting challenges for NeRF models in inpainting them effectively. We propose an implicit ray transformation strategy, allowing for direct manipulation of the 3D object's pose by operating on the neural-point in NeRF rays. To address the challenge of inpainting potential empty regions, we present a plug-and-play inpainting module, dubbed differentiable neural-point resampling (DNR), which interpolates those regions in 3D space at the original ray locations within the implicit space, thereby facilitating object removal & scene inpainting tasks. Importantly, employing DNR effectively narrows the gap between ground truth and predicted implicit features, potentially increasing the mutual information (MI) of the features across rays. Then, we leverage DNR and ray transformation to construct a point-based editable NeRF pipeline PR^2T-NeRF. Results primarily evaluated on 3D object removal & inpainting tasks indicate that our pipeline achieves state-of-the-art performance. In addition, our pipeline supports high-quality rendering visualization for diverse editing operations without necessitating extra supervision.  
  </ol>  
</details>  
  
### [Hologram: Realtime Holographic Overlays via LiDAR Augmented Reconstruction](http://arxiv.org/abs/2405.07178)  
Ekansh Agrawal  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Guided by the hologram technology of the infamous Star Wars franchise, I present an application that creates real-time holographic overlays using LiDAR augmented 3D reconstruction. Prior attempts involve SLAM or NeRFs which either require highly calibrated scenes, incur steep computation costs, or fail to render dynamic scenes. I propose 3 high-fidelity reconstruction tools that can run on a portable device, such as a iPhone 14 Pro, which can allow for metric accurate facial reconstructions. My systems enable interactive and immersive holographic experiences that can be used for a wide range of applications, including augmented reality, telepresence, and entertainment.  
  </ol>  
</details>  
  
### [TD-NeRF: Novel Truncated Depth Prior for Joint Camera Pose and Neural Radiance Field Optimization](http://arxiv.org/abs/2405.07027)  
Zhen Tan, Zongtan Zhou, Yangbing Ge, Zi Wang, Xieyuanli Chen, Dewen Hu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The reliance on accurate camera poses is a significant barrier to the widespread deployment of Neural Radiance Fields (NeRF) models for 3D reconstruction and SLAM tasks. The existing method introduces monocular depth priors to jointly optimize the camera poses and NeRF, which fails to fully exploit the depth priors and neglects the impact of their inherent noise. In this paper, we propose Truncated Depth NeRF (TD-NeRF), a novel approach that enables training NeRF from unknown camera poses - by jointly optimizing learnable parameters of the radiance field and camera poses. Our approach explicitly utilizes monocular depth priors through three key advancements: 1) we propose a novel depth-based ray sampling strategy based on the truncated normal distribution, which improves the convergence speed and accuracy of pose estimation; 2) to circumvent local minima and refine depth geometry, we introduce a coarse-to-fine training strategy that progressively improves the depth precision; 3) we propose a more robust inter-frame point constraint that enhances robustness against depth noise during training. The experimental results on three datasets demonstrate that TD-NeRF achieves superior performance in the joint optimization of camera pose and NeRF, surpassing prior works, and generates more accurate depth geometry. The implementation of our method has been released at https://github.com/nubot-nudt/TD-NeRF.  
  </ol>  
</details>  
  
### [LIVE: LaTex Interactive Visual Editing](http://arxiv.org/abs/2405.06762)  
Jinwei Lin  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    LaTex coding is one of the main methods of writing an academic paper. When writing a paper, abundant proper visual or graphic components will represent more information volume than the textual data. However, most of the implementation of LaTex graphic items are designed as static items that have some weaknesses in representing more informative figures or tables with an interactive reading experience. To address this problem, we propose LIVE, a novel design methods idea to design interactive LaTex graphic items. To make a lucid representation of the main idea of LIVE, we designed several novels representing implementations that are interactive and enough explanation for the basic level principles. Using LIVE can design more graphic items, which we call the Gitems, and easily and automatically get the relationship of the mutual application of a specific range of papers, which will add more vitality and performance factors into writing of traditional papers especially the review papers. For vividly representing the functions of LIVE, we use the papers from NeRF as the example reference papers. The code of the implementation project is open source.  
  </ol>  
</details>  
**comments**: 8 pages, double column, ieee  
  
  



