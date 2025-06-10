<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#SurGSplat:-Progressive-Geometry-Constrained-Gaussian-Splatting-for-Surgical-Scene-Reconstruction>SurGSplat: Progressive Geometry-Constrained Gaussian Splatting for Surgical Scene Reconstruction</a></li>
        <li><a href=#On-the-fly-Reconstruction-for-Large-Scale-Novel-View-Synthesis-from-Unposed-Images>On-the-fly Reconstruction for Large-Scale Novel View Synthesis from Unposed Images</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#GenIR:-Generative-Visual-Feedback-for-Mental-Image-Retrieval>GenIR: Generative Visual Feedback for Mental Image Retrieval</a></li>
        <li><a href=#Astra:-Toward-General-Purpose-Mobile-Robots-via-Hierarchical-Multimodal-Learning>Astra: Toward General-Purpose Mobile Robots via Hierarchical Multimodal Learning</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Dy3DGS-SLAM:-Monocular-3D-Gaussian-Splatting-SLAM-for-Dynamic-Environments>Dy3DGS-SLAM: Monocular 3D Gaussian Splatting SLAM for Dynamic Environments</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [SurGSplat: Progressive Geometry-Constrained Gaussian Splatting for Surgical Scene Reconstruction](http://arxiv.org/abs/2506.05935)  
Yuchao Zheng, Jianing Zhang, Guochen Ning, Hongen Liao  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Intraoperative navigation relies heavily on precise 3D reconstruction to ensure accuracy and safety during surgical procedures. However, endoscopic scenarios present unique challenges, including sparse features and inconsistent lighting, which render many existing Structure-from-Motion (SfM)-based methods inadequate and prone to reconstruction failure. To mitigate these constraints, we propose SurGSplat, a novel paradigm designed to progressively refine 3D Gaussian Splatting (3DGS) through the integration of geometric constraints. By enabling the detailed reconstruction of vascular structures and other critical features, SurGSplat provides surgeons with enhanced visual clarity, facilitating precise intraoperative decision-making. Experimental evaluations demonstrate that SurGSplat achieves superior performance in both novel view synthesis (NVS) and pose estimation accuracy, establishing it as a high-fidelity and efficient solution for surgical scene reconstruction. More information and results can be found on the page https://surgsplat.github.io/.  
  </ol>  
</details>  
  
### [On-the-fly Reconstruction for Large-Scale Novel View Synthesis from Unposed Images](http://arxiv.org/abs/2506.05558)  
Andreas Meuleman, Ishaan Shah, Alexandre Lanvin, Bernhard Kerbl, George Drettakis  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Radiance field methods such as 3D Gaussian Splatting (3DGS) allow easy reconstruction from photos, enabling free-viewpoint navigation. Nonetheless, pose estimation using Structure from Motion and 3DGS optimization can still each take between minutes and hours of computation after capture is complete. SLAM methods combined with 3DGS are fast but struggle with wide camera baselines and large scenes. We present an on-the-fly method to produce camera poses and a trained 3DGS immediately after capture. Our method can handle dense and wide-baseline captures of ordered photo sequences and large-scale scenes. To do this, we first introduce fast initial pose estimation, exploiting learned features and a GPU-friendly mini bundle adjustment. We then introduce direct sampling of Gaussian primitive positions and shapes, incrementally spawning primitives where required, significantly accelerating training. These two efficient steps allow fast and robust joint optimization of poses and Gaussian primitives. Our incremental approach handles large-scale scenes by introducing scalable radiance field construction, progressively clustering 3DGS primitives, storing them in anchors, and offloading them from the GPU. Clustered primitives are progressively merged, keeping the required scale of 3DGS at any viewpoint. We evaluate our solution on a variety of datasets and show that our solution can provide on-the-fly processing of all the capture scenarios and scene sizes we target while remaining competitive with other methods that only handle specific capture styles or scene sizes in speed, image quality, or both.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [GenIR: Generative Visual Feedback for Mental Image Retrieval](http://arxiv.org/abs/2506.06220)  
Diji Yang, Minghao Liu, Chung-Hsiang Lo, Yi Zhang, James Davis  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Vision-language models (VLMs) have shown strong performance on text-to-image retrieval benchmarks. However, bridging this success to real-world applications remains a challenge. In practice, human search behavior is rarely a one-shot action. Instead, it is often a multi-round process guided by clues in mind, that is, a mental image ranging from vague recollections to vivid mental representations of the target image. Motivated by this gap, we study the task of Mental Image Retrieval (MIR), which targets the realistic yet underexplored setting where users refine their search for a mentally envisioned image through multi-round interactions with an image search engine. Central to successful interactive retrieval is the capability of machines to provide users with clear, actionable feedback; however, existing methods rely on indirect or abstract verbal feedback, which can be ambiguous, misleading, or ineffective for users to refine the query. To overcome this, we propose GenIR, a generative multi-round retrieval paradigm leveraging diffusion-based image generation to explicitly reify the AI system's understanding at each round. These synthetic visual representations provide clear, interpretable feedback, enabling users to refine their queries intuitively and effectively. We further introduce a fully automated pipeline to generate a high-quality multi-round MIR dataset. Experimental results demonstrate that GenIR significantly outperforms existing interactive methods in the MIR scenario. This work establishes a new task with a dataset and an effective generative retrieval method, providing a foundation for future research in this direction.  
  </ol>  
</details>  
  
### [Astra: Toward General-Purpose Mobile Robots via Hierarchical Multimodal Learning](http://arxiv.org/abs/2506.06205)  
Sheng Chen, Peiyu He, Jiaxin Hu, Ziyang Liu, Yansheng Wang, Tao Xu, Chi Zhang, Chongchong Zhang, Chao An, Shiyu Cai, Duo Cao, Kangping Chen, Shuai Chu, Tianwei Chu, Mingdi Dan, Min Du, Weiwei Fang, Pengyou Fu, Junkai Hu, Xiaowei Jiang, Zhaodi Jiang, Fuxuan Li, Jun Li, Minghui Li, Mingyao Li, Yanchang Li, Zhibin Li, Guangming Liu, Kairui Liu, Lihao Liu, Weizhi Liu, Xiaoshun Liu, Yufei Liu, Yunfei Liu, Qiang Lu, Yuanfei Luo, Xiang Lv, Hongying Ma, Sai Ma, Lingxian Mi, Sha Sa, Hongxiang Shu, Lei Tian, Chengzhi Wang, Jiayu Wang, Kaijie Wang, Qingyi Wang, Renwen Wang, Tao Wang, Wei Wang, Xirui Wang, Chao Wei, Xuguang Wei, Zijun Xia, Zhaohao Xiao, Tingshuai Yan, Liyan Yang, Yifan Yang, Zhikai Yang, Zhong Yin, Li Yuan, Liuchun Yuan, Chi Zhang, Jinyang Zhang, Junhui Zhang, Linge Zhang, Zhenyi Zhang, Zheyu Zhang, Dongjie Zhu, Hang Li, Yangang Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Modern robot navigation systems encounter difficulties in diverse and complex indoor environments. Traditional approaches rely on multiple modules with small models or rule-based systems and thus lack adaptability to new environments. To address this, we developed Astra, a comprehensive dual-model architecture, Astra-Global and Astra-Local, for mobile robot navigation. Astra-Global, a multimodal LLM, processes vision and language inputs to perform self and goal localization using a hybrid topological-semantic graph as the global map, and outperforms traditional visual place recognition methods. Astra-Local, a multitask network, handles local path planning and odometry estimation. Its 4D spatial-temporal encoder, trained through self-supervised learning, generates robust 4D features for downstream tasks. The planning head utilizes flow matching and a novel masked ESDF loss to minimize collision risks for generating local trajectories, and the odometry head integrates multi-sensor inputs via a transformer encoder to predict the relative pose of the robot. Deployed on real in-house mobile robots, Astra achieves high end-to-end mission success rate across diverse indoor environments.  
  </ol>  
</details>  
**comments**: Astra Technical Report  
  
  



## NeRF  

### [Dy3DGS-SLAM: Monocular 3D Gaussian Splatting SLAM for Dynamic Environments](http://arxiv.org/abs/2506.05965)  
Mingrui Li, Yiming Zhou, Hongxing Zhou, Xinggang Hu, Florian Roemer, Hongyu Wang, Ahmad Osman  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Current Simultaneous Localization and Mapping (SLAM) methods based on Neural Radiance Fields (NeRF) or 3D Gaussian Splatting excel in reconstructing static 3D scenes but struggle with tracking and reconstruction in dynamic environments, such as real-world scenes with moving elements. Existing NeRF-based SLAM approaches addressing dynamic challenges typically rely on RGB-D inputs, with few methods accommodating pure RGB input. To overcome these limitations, we propose Dy3DGS-SLAM, the first 3D Gaussian Splatting (3DGS) SLAM method for dynamic scenes using monocular RGB input. To address dynamic interference, we fuse optical flow masks and depth masks through a probabilistic model to obtain a fused dynamic mask. With only a single network iteration, this can constrain tracking scales and refine rendered geometry. Based on the fused dynamic mask, we designed a novel motion loss to constrain the pose estimation network for tracking. In mapping, we use the rendering loss of dynamic pixels, color, and depth to eliminate transient interference and occlusion caused by dynamic objects. Experimental results demonstrate that Dy3DGS-SLAM achieves state-of-the-art tracking and rendering in dynamic environments, outperforming or matching existing RGB-D methods.  
  </ol>  
</details>  
  
  



