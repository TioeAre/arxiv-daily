<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Targetless-Intrinsics-and-Extrinsic-Calibration-of-Multiple-LiDARs-and-Cameras-with-IMU-using-Continuous-Time-Estimation>Targetless Intrinsics and Extrinsic Calibration of Multiple LiDARs and Cameras with IMU using Continuous-Time Estimation</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Integrating-Language-Image-Prior-into-EEG-Decoding-for-Cross-Task-Zero-Calibration-RSVP-BCI>Integrating Language-Image Prior into EEG Decoding for Cross-Task Zero-Calibration RSVP-BCI</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#AE-NeRF:-Augmenting-Event-Based-Neural-Radiance-Fields-for-Non-ideal-Conditions-and-Larger-Scene>AE-NeRF: Augmenting Event-Based Neural Radiance Fields for Non-ideal Conditions and Larger Scene</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Targetless Intrinsics and Extrinsic Calibration of Multiple LiDARs and Cameras with IMU using Continuous-Time Estimation](http://arxiv.org/abs/2501.02821)  
Yuezhang Lv, Yunzhou Zhang, Chao Lu, Jiajun Zhu, Song Wu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Accurate spatiotemporal calibration is a prerequisite for multisensor fusion. However, sensors are typically asynchronous, and there is no overlap between the fields of view of cameras and LiDARs, posing challenges for intrinsic and extrinsic parameter calibration. To address this, we propose a calibration pipeline based on continuous-time and bundle adjustment (BA) capable of simultaneous intrinsic and extrinsic calibration (6 DOF transformation and time offset). We do not require overlapping fields of view or any calibration board. Firstly, we establish data associations between cameras using Structure from Motion (SFM) and perform self-calibration of camera intrinsics. Then, we establish data associations between LiDARs through adaptive voxel map construction, optimizing for extrinsic calibration within the map. Finally, by matching features between the intensity projection of LiDAR maps and camera images, we conduct joint optimization for intrinsic and extrinsic parameters. This pipeline functions in texture-rich structured environments, allowing simultaneous calibration of any number of cameras and LiDARs without the need for intricate sensor synchronization triggers. Experimental results demonstrate our method's ability to fulfill co-visibility and motion constraints between sensors without accumulating errors.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Integrating Language-Image Prior into EEG Decoding for Cross-Task Zero-Calibration RSVP-BCI](http://arxiv.org/abs/2501.02841)  
Xujin Li, Wei Wei, Shuang Qiu, Xinyi Zhang, Fu Li, Huiguang He  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Rapid Serial Visual Presentation (RSVP)-based Brain-Computer Interface (BCI) is an effective technology used for information detection by detecting Event-Related Potentials (ERPs). The current RSVP decoding methods can perform well in decoding EEG signals within a single RSVP task, but their decoding performance significantly decreases when directly applied to different RSVP tasks without calibration data from the new tasks. This limits the rapid and efficient deployment of RSVP-BCI systems for detecting different categories of targets in various scenarios. To overcome this limitation, this study aims to enhance the cross-task zero-calibration RSVP decoding performance. First, we design three distinct RSVP tasks for target image retrieval and build an open-source dataset containing EEG signals and corresponding stimulus images. Then we propose an EEG with Language-Image Prior fusion Transformer (ELIPformer) for cross-task zero-calibration RSVP decoding. Specifically, we propose a prompt encoder based on the language-image pre-trained model to extract language-image features from task-specific prompts and stimulus images as prior knowledge for enhancing EEG decoding. A cross bidirectional attention mechanism is also adopted to facilitate the effective feature fusion and alignment between the EEG and language-image features. Extensive experiments demonstrate that the proposed model achieves superior performance in cross-task zero-calibration RSVP decoding, which promotes the RSVP-BCI system from research to practical application.  
  </ol>  
</details>  
**comments**: 15 pages, 11 figures  
  
  



## NeRF  

### [AE-NeRF: Augmenting Event-Based Neural Radiance Fields for Non-ideal Conditions and Larger Scene](http://arxiv.org/abs/2501.02807)  
Chaoran Feng, Wangbo Yu, Xinhua Cheng, Zhenyu Tang, Junwu Zhang, Li Yuan, Yonghong Tian  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Compared to frame-based methods, computational neuromorphic imaging using event cameras offers significant advantages, such as minimal motion blur, enhanced temporal resolution, and high dynamic range. The multi-view consistency of Neural Radiance Fields combined with the unique benefits of event cameras, has spurred recent research into reconstructing NeRF from data captured by moving event cameras. While showing impressive performance, existing methods rely on ideal conditions with the availability of uniform and high-quality event sequences and accurate camera poses, and mainly focus on the object level reconstruction, thus limiting their practical applications. In this work, we propose AE-NeRF to address the challenges of learning event-based NeRF from non-ideal conditions, including non-uniform event sequences, noisy poses, and various scales of scenes. Our method exploits the density of event streams and jointly learn a pose correction module with an event-based NeRF (e-NeRF) framework for robust 3D reconstruction from inaccurate camera poses. To generalize to larger scenes, we propose hierarchical event distillation with a proposal e-NeRF network and a vanilla e-NeRF network to resample and refine the reconstruction process. We further propose an event reconstruction loss and a temporal loss to improve the view consistency of the reconstructed scene. We established a comprehensive benchmark that includes large-scale scenes to simulate practical non-ideal conditions, incorporating both synthetic and challenging real-world event datasets. The experimental results show that our method achieves a new state-of-the-art in event-based 3D reconstruction.  
  </ol>  
</details>  
**comments**: Accepted by AAAI 2025. https://github.com/SuperFCR/AE-NeRF  
  
  



