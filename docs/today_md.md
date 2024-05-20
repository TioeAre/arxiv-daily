<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#CCTNet:-A-Circular-Convolutional-Transformer-Network-for-LiDAR-based-Place-Recognition-Handling-Movable-Objects-Occlusion>CCTNet: A Circular Convolutional Transformer Network for LiDAR-based Place Recognition Handling Movable Objects Occlusion</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [CCTNet: A Circular Convolutional Transformer Network for LiDAR-based Place Recognition Handling Movable Objects Occlusion](http://arxiv.org/abs/2405.10793)  
Gang Wang, Chaoran Zhu, Qian Xu, Tongzhou Zhang, Hai Zhang, XiaoPeng Fan, Jue Hu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Place recognition is a fundamental task for robotic application, allowing robots to perform loop closure detection within simultaneous localization and mapping (SLAM), and achieve relocalization on prior maps. Current range image-based networks use single-column convolution to maintain feature invariance to shifts in image columns caused by LiDAR viewpoint change.However, this raises the issues such as "restricted receptive fields" and "excessive focus on local regions", degrading the performance of networks. To address the aforementioned issues, we propose a lightweight circular convolutional Transformer network denoted as CCTNet, which boosts performance by capturing structural information in point clouds and facilitating crossdimensional interaction of spatial and channel information. Initially, a Circular Convolution Module (CCM) is introduced, expanding the network's perceptual field while maintaining feature consistency across varying LiDAR perspectives. Then, a Range Transformer Module (RTM) is proposed, which enhances place recognition accuracy in scenarios with movable objects by employing a combination of channel and spatial attention mechanisms. Furthermore, we propose an Overlap-based loss function, transforming the place recognition task from a binary loop closure classification into a regression problem linked to the overlap between LiDAR frames. Through extensive experiments on the KITTI and Ford Campus datasets, CCTNet surpasses comparable methods, achieving Recall@1 of 0.924 and 0.965, and Recall@1% of 0.990 and 0.993 on the test set, showcasing a superior performance. Results on the selfcollected dataset further demonstrate the proposed method's potential for practical implementation in complex scenarios to handle movable objects, showing improved generalization in various datasets.  
  </ol>  
</details>  
  
  



