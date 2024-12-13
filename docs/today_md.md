<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#eCARLA-scenes:-A-synthetically-generated-dataset-for-event-based-optical-flow-prediction>eCARLA-scenes: A synthetically generated dataset for event-based optical flow prediction</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#MVC-VPR:-Mutual-Learning-of-Viewpoint-Classification-and-Visual-Place-Recognition>MVC-VPR: Mutual Learning of Viewpoint Classification and Visual Place Recognition</a></li>
        <li><a href=#A-Flexible-Plug-and-Play-Module-for-Generating-Variable-Length>A Flexible Plug-and-Play Module for Generating Variable-Length</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [eCARLA-scenes: A synthetically generated dataset for event-based optical flow prediction](http://arxiv.org/abs/2412.09209)  
Jad Mansour, Hayat Rajani, Rafael Garcia, Nuno Gracias  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The joint use of event-based vision and Spiking Neural Networks (SNNs) is expected to have a large impact in robotics in the near future, in tasks such as, visual odometry and obstacle avoidance. While researchers have used real-world event datasets for optical flow prediction (mostly captured with Unmanned Aerial Vehicles (UAVs)), these datasets are limited in diversity, scalability, and are challenging to collect. Thus, synthetic datasets offer a scalable alternative by bridging the gap between reality and simulation. In this work, we address the lack of datasets by introducing eWiz, a comprehensive library for processing event-based data. It includes tools for data loading, augmentation, visualization, encoding, and generation of training data, along with loss functions and performance metrics. We further present a synthetic event-based datasets and data generation pipelines for optical flow prediction tasks. Built on top of eWiz, eCARLA-scenes makes use of the CARLA simulator to simulate self-driving car scenarios. The ultimate goal of this dataset is the depiction of diverse environments while laying a foundation for advancing event-based camera applications in autonomous field vehicle navigation, paving the way for using SNNs on neuromorphic hardware such as the Intel Loihi.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [MVC-VPR: Mutual Learning of Viewpoint Classification and Visual Place Recognition](http://arxiv.org/abs/2412.09199)  
Qiwen Gu, Xufei Wang, Fenglin Zhang, Junqiao Zhao, Siyue Tao, Chen Ye, Tiantian Feng, Changjun Jiang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Place Recognition (VPR) aims to robustly identify locations by leveraging image retrieval based on descriptors encoded from environmental images. However, drastic appearance changes of images captured from different viewpoints at the same location pose incoherent supervision signals for descriptor learning, which severely hinder the performance of VPR. Previous work proposes classifying images based on manually defined rules or ground truth labels for viewpoints, followed by descriptor training based on the classification results. However, not all datasets have ground truth labels of viewpoints and manually defined rules may be suboptimal, leading to degraded descriptor performance.To address these challenges, we introduce the mutual learning of viewpoint self-classification and VPR. Starting from coarse classification based on geographical coordinates, we progress to finer classification of viewpoints using simple clustering techniques. The dataset is partitioned in an unsupervised manner while simultaneously training a descriptor extractor for place recognition. Experimental results show that this approach almost perfectly partitions the dataset based on viewpoints, thus achieving mutually reinforcing effects. Our method even excels state-of-the-art (SOTA) methods that partition datasets using ground truth labels.  
  </ol>  
</details>  
**comments**: 8 pages  
  
### [A Flexible Plug-and-Play Module for Generating Variable-Length](http://arxiv.org/abs/2412.08922)  
[[code](https://github.com/hly1998/nhl)]  
Liyang He, Yuren Zhang, Rui Li, Zhenya Huang, Runze Wu, Enhong Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Deep supervised hashing has become a pivotal technique in large-scale image retrieval, offering significant benefits in terms of storage and search efficiency. However, existing deep supervised hashing models predominantly focus on generating fixed-length hash codes. This approach fails to address the inherent trade-off between efficiency and effectiveness when using hash codes of varying lengths. To determine the optimal hash code length for a specific task, multiple models must be trained for different lengths, leading to increased training time and computational overhead. Furthermore, the current paradigm overlooks the potential relationships between hash codes of different lengths, limiting the overall effectiveness of the models. To address these challenges, we propose the Nested Hash Layer (NHL), a plug-and-play module designed for existing deep supervised hashing models. The NHL framework introduces a novel mechanism to simultaneously generate hash codes of varying lengths in a nested manner. To tackle the optimization conflicts arising from the multiple learning objectives associated with different code lengths, we further propose an adaptive weights strategy that dynamically monitors and adjusts gradients during training. Additionally, recognizing that the structural information in longer hash codes can provide valuable guidance for shorter hash codes, we develop a long-short cascade self-distillation method within the NHL to enhance the overall quality of the generated hash codes. Extensive experiments demonstrate that NHL not only accelerates the training process but also achieves superior retrieval performance across various deep hashing models. Our code is publicly available at https://github.com/hly1998/NHL.  
  </ol>  
</details>  
  
  



