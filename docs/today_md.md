<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Experimental-Assessment-of-Neural-3D-Reconstruction-for-Small-UAV-based-Applications>Experimental Assessment of Neural 3D Reconstruction for Small UAV-based Applications</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#NeRF-based-CBCT-Reconstruction-needs-Normalization-and-Initialization>NeRF-based CBCT Reconstruction needs Normalization and Initialization</a></li>
        <li><a href=#Self-Supervised-Multimodal-NeRF-for-Autonomous-Driving>Self-Supervised Multimodal NeRF for Autonomous Driving</a></li>
        <li><a href=#HoliGS:-Holistic-Gaussian-Splatting-for-Embodied-View-Synthesis>HoliGS: Holistic Gaussian Splatting for Embodied View Synthesis</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Experimental Assessment of Neural 3D Reconstruction for Small UAV-based Applications](http://arxiv.org/abs/2506.19491)  
Genís Castillo Gómez-Raya, Álmos Veres-Vitályos, Filip Lemic, Pablo Royo, Mario Montagud, Sergi Fernández, Sergi Abadal, Xavier Costa-Pérez  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The increasing miniaturization of Unmanned Aerial Vehicles (UAVs) has expanded their deployment potential to indoor and hard-to-reach areas. However, this trend introduces distinct challenges, particularly in terms of flight dynamics and power consumption, which limit the UAVs' autonomy and mission capabilities. This paper presents a novel approach to overcoming these limitations by integrating Neural 3D Reconstruction (N3DR) with small UAV systems for fine-grained 3-Dimensional (3D) digital reconstruction of small static objects. Specifically, we design, implement, and evaluate an N3DR-based pipeline that leverages advanced models, i.e., Instant-ngp, Nerfacto, and Splatfacto, to improve the quality of 3D reconstructions using images of the object captured by a fleet of small UAVs. We assess the performance of the considered models using various imagery and pointcloud metrics, comparing them against the baseline Structure from Motion (SfM) algorithm. The experimental results demonstrate that the N3DR-enhanced pipeline significantly improves reconstruction quality, making it feasible for small UAVs to support high-precision 3D mapping and anomaly detection in constrained environments. In more general terms, our results highlight the potential of N3DR in advancing the capabilities of miniaturized UAV systems.  
  </ol>  
</details>  
**comments**: 6 pages, 7 figures, 2 tables, accepted at IEEE International
  Symposium on Personal, Indoor and Mobile Radio Communications 2025  
  
  



## NeRF  

### [NeRF-based CBCT Reconstruction needs Normalization and Initialization](http://arxiv.org/abs/2506.19742)  
Zhuowei Xu, Han Li, Dai Sun, Zhicheng Li, Yujia Li, Qingpeng Kong, Zhiwei Cheng, Nassir Navab, S. Kevin Zhou  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Cone Beam Computed Tomography (CBCT) is widely used in medical imaging. However, the limited number and intensity of X-ray projections make reconstruction an ill-posed problem with severe artifacts. NeRF-based methods have achieved great success in this task. However, they suffer from a local-global training mismatch between their two key components: the hash encoder and the neural network. Specifically, in each training step, only a subset of the hash encoder's parameters is used (local sparse), whereas all parameters in the neural network participate (global dense). Consequently, hash features generated in each step are highly misaligned, as they come from different subsets of the hash encoder. These misalignments from different training steps are then fed into the neural network, causing repeated inconsistent global updates in training, which leads to unstable training, slower convergence, and degraded reconstruction quality. Aiming to alleviate the impact of this local-global optimization mismatch, we introduce a Normalized Hash Encoder, which enhances feature consistency and mitigates the mismatch. Additionally, we propose a Mapping Consistency Initialization(MCI) strategy that initializes the neural network before training by leveraging the global mapping property from a well-trained model. The initialized neural network exhibits improved stability during early training, enabling faster convergence and enhanced reconstruction performance. Our method is simple yet effective, requiring only a few lines of code while substantially improving training efficiency on 128 CT cases collected from 4 different datasets, covering 7 distinct anatomical regions.  
  </ol>  
</details>  
  
### [Self-Supervised Multimodal NeRF for Autonomous Driving](http://arxiv.org/abs/2506.19615)  
Gaurav Sharma, Ravi Kothari, Josef Schmid  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this paper, we propose a Neural Radiance Fields (NeRF) based framework, referred to as Novel View Synthesis Framework (NVSF). It jointly learns the implicit neural representation of space and time-varying scene for both LiDAR and Camera. We test this on a real-world autonomous driving scenario containing both static and dynamic scenes. Compared to existing multimodal dynamic NeRFs, our framework is self-supervised, thus eliminating the need for 3D labels. For efficient training and faster convergence, we introduce heuristic-based image pixel sampling to focus on pixels with rich information. To preserve the local features of LiDAR points, a Double Gradient based mask is employed. Extensive experiments on the KITTI-360 dataset show that, compared to the baseline models, our framework has reported best performance on both LiDAR and Camera domain. Code of the model is available at https://github.com/gaurav00700/Selfsupervised-NVSF  
  </ol>  
</details>  
  
### [HoliGS: Holistic Gaussian Splatting for Embodied View Synthesis](http://arxiv.org/abs/2506.19291)  
Xiaoyuan Wang, Yizhou Zhao, Botao Ye, Xiaojun Shan, Weijie Lyu, Lu Qi, Kelvin C. K. Chan, Yinxiao Li, Ming-Hsuan Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose HoliGS, a novel deformable Gaussian splatting framework that addresses embodied view synthesis from long monocular RGB videos. Unlike prior 4D Gaussian splatting and dynamic NeRF pipelines, which struggle with training overhead in minute-long captures, our method leverages invertible Gaussian Splatting deformation networks to reconstruct large-scale, dynamic environments accurately. Specifically, we decompose each scene into a static background plus time-varying objects, each represented by learned Gaussian primitives undergoing global rigid transformations, skeleton-driven articulation, and subtle non-rigid deformations via an invertible neural flow. This hierarchical warping strategy enables robust free-viewpoint novel-view rendering from various embodied camera trajectories by attaching Gaussians to a complete canonical foreground shape (\eg, egocentric or third-person follow), which may involve substantial viewpoint changes and interactions between multiple actors. Our experiments demonstrate that \ourmethod~ achieves superior reconstruction quality on challenging datasets while significantly reducing both training and rendering time compared to state-of-the-art monocular deformable NeRFs. These results highlight a practical and scalable solution for EVS in real-world scenarios. The source code will be released.  
  </ol>  
</details>  
  
  



