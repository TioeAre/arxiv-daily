<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#CodedVO:-Coded-Visual-Odometry>CodedVO: Coded Visual Odometry</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#LION:-Linear-Group-RNN-for-3D-Object-Detection-in-Point-Clouds>LION: Linear Group RNN for 3D Object Detection in Point Clouds</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [CodedVO: Coded Visual Odometry](http://arxiv.org/abs/2407.18240)  
Sachin Shah, Naitri Rajyaguru, Chahat Deep Singh, Christopher Metzler, Yiannis Aloimonos  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Autonomous robots often rely on monocular cameras for odometry estimation and navigation. However, the scale ambiguity problem presents a critical barrier to effective monocular visual odometry. In this paper, we present CodedVO, a novel monocular visual odometry method that overcomes the scale ambiguity problem by employing custom optics to physically encode metric depth information into imagery. By incorporating this information into our odometry pipeline, we achieve state-of-the-art performance in monocular visual odometry with a known scale. We evaluate our method in diverse indoor environments and demonstrate its robustness and adaptability. We achieve a 0.08m average trajectory error in odometry evaluation on the ICL-NUIM indoor odometry dataset.  
  </ol>  
</details>  
**comments**: 7 pages, 4 figures, IEEE ROBOTICS AND AUTOMATION LETTERS  
  
  



## Keypoint Detection  

### [LION: Linear Group RNN for 3D Object Detection in Point Clouds](http://arxiv.org/abs/2407.18232)  
[[code](https://github.com/happinesslz/LION)]  
Zhe Liu, Jinghua Hou, Xinyu Wang, Xiaoqing Ye, Jingdong Wang, Hengshuang Zhao, Xiang Bai  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The benefit of transformers in large-scale 3D point cloud perception tasks, such as 3D object detection, is limited by their quadratic computation cost when modeling long-range relationships. In contrast, linear RNNs have low computational complexity and are suitable for long-range modeling. Toward this goal, we propose a simple and effective window-based framework built on LInear grOup RNN (i.e., perform linear RNN for grouped features) for accurate 3D object detection, called LION. The key property is to allow sufficient feature interaction in a much larger group than transformer-based methods. However, effectively applying linear group RNN to 3D object detection in highly sparse point clouds is not trivial due to its limitation in handling spatial modeling. To tackle this problem, we simply introduce a 3D spatial feature descriptor and integrate it into the linear group RNN operators to enhance their spatial features rather than blindly increasing the number of scanning orders for voxel features. To further address the challenge in highly sparse point clouds, we propose a 3D voxel generation strategy to densify foreground features thanks to linear group RNN as a natural property of auto-regressive models. Extensive experiments verify the effectiveness of the proposed components and the generalization of our LION on different linear group RNN operators including Mamba, RWKV, and RetNet. Furthermore, it is worth mentioning that our LION-Mamba achieves state-of-the-art on Waymo, nuScenes, Argoverse V2, and ONCE dataset. Last but not least, our method supports kinds of advanced linear RNN operators (e.g., RetNet, RWKV, Mamba, xLSTM and TTT) on small but popular KITTI dataset for a quick experience with our linear RNN-based framework.  
  </ol>  
</details>  
**comments**: Project page: https://happinesslz.github.io/projects/LION/  
  
  



