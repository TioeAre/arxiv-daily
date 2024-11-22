<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#InCrowd-VI:-A-Realistic-Visual-Inertial-Dataset-for-Evaluating-SLAM-in-Indoor-Pedestrian-Rich-Spaces-for-Human-Navigation>InCrowd-VI: A Realistic Visual-Inertial Dataset for Evaluating SLAM in Indoor Pedestrian-Rich Spaces for Human Navigation</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Robust-SG-NeRF:-Robust-Scene-Graph-Aided-Neural-Surface-Reconstruction>Robust SG-NeRF: Robust Scene Graph Aided Neural Surface Reconstruction</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [InCrowd-VI: A Realistic Visual-Inertial Dataset for Evaluating SLAM in Indoor Pedestrian-Rich Spaces for Human Navigation](http://arxiv.org/abs/2411.14358)  
Marziyeh Bamdad, Hans-Peter Hutter, Alireza Darvishy  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Simultaneous localization and mapping (SLAM) techniques can be used to navigate the visually impaired, but the development of robust SLAM solutions for crowded spaces is limited by the lack of realistic datasets. To address this, we introduce InCrowd-VI, a novel visual-inertial dataset specifically designed for human navigation in indoor pedestrian-rich environments. Recorded using Meta Aria Project glasses, it captures realistic scenarios without environmental control. InCrowd-VI features 58 sequences totaling a 5 km trajectory length and 1.5 hours of recording time, including RGB, stereo images, and IMU measurements. The dataset captures important challenges such as pedestrian occlusions, varying crowd densities, complex layouts, and lighting changes. Ground-truth trajectories, accurate to approximately 2 cm, are provided in the dataset, originating from the Meta Aria project machine perception SLAM service. In addition, a semi-dense 3D point cloud of scenes is provided for each sequence. The evaluation of state-of-the-art visual odometry (VO) and SLAM algorithms on InCrowd-VI revealed severe performance limitations in these realistic scenarios, demonstrating the need and value of the new dataset to advance SLAM research for visually impaired navigation in complex indoor environments.  
  </ol>  
</details>  
**comments**: 18 pages, 7 figures, 5 tabels  
  
  



## NeRF  

### [Robust SG-NeRF: Robust Scene Graph Aided Neural Surface Reconstruction](http://arxiv.org/abs/2411.13620)  
Yi Gu, Dongjun Ye, Zhaorui Wang, Jiaxu Wang, Jiahang Cao, Renjing Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural surface reconstruction relies heavily on accurate camera poses as input. Despite utilizing advanced pose estimators like COLMAP or ARKit, camera poses can still be noisy. Existing pose-NeRF joint optimization methods handle poses with small noise (inliers) effectively but struggle with large noise (outliers), such as mirrored poses. In this work, we focus on mitigating the impact of outlier poses. Our method integrates an inlier-outlier confidence estimation scheme, leveraging scene graph information gathered during the data preparation phase. Unlike previous works directly using rendering metrics as the reference, we employ a detached color network that omits the viewing direction as input to minimize the impact caused by shape-radiance ambiguities. This enhanced confidence updating strategy effectively differentiates between inlier and outlier poses, allowing us to sample more rays from inlier poses to construct more reliable radiance fields. Additionally, we introduce a re-projection loss based on the current Signed Distance Function (SDF) and pose estimations, strengthening the constraints between matching image pairs. For outlier poses, we adopt a Monte Carlo re-localization method to find better solutions. We also devise a scene graph updating strategy to provide more accurate information throughout the training process. We validate our approach on the SG-NeRF and DTU datasets. Experimental results on various datasets demonstrate that our methods can consistently improve the reconstruction qualities and pose accuracies.  
  </ol>  
</details>  
**comments**: https://rsg-nerf.github.io/RSG-NeRF/  
  
  



