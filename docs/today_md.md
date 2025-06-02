<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#SORCE:-Small-Object-Retrieval-in-Complex-Environments>SORCE: Small Object Retrieval in Complex Environments</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#SR3D:-Unleashing-Single-view-3D-Reconstruction-for-Transparent-and-Specular-Object-Grasping>SR3D: Unleashing Single-view 3D Reconstruction for Transparent and Specular Object Grasping</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [SORCE: Small Object Retrieval in Complex Environments](http://arxiv.org/abs/2505.24441)  
Chunxu Liu, Chi Xie, Xiaxu Chen, Wei Li, Feng Zhu, Rui Zhao, Limin Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Text-to-Image Retrieval (T2IR) is a highly valuable task that aims to match a given textual query to images in a gallery. Existing benchmarks primarily focus on textual queries describing overall image semantics or foreground salient objects, possibly overlooking inconspicuous small objects, especially in complex environments. Such small object retrieval is crucial, as in real-world applications, the targets of interest are not always prominent in the image. Thus, we introduce SORCE (Small Object Retrieval in Complex Environments), a new subfield of T2IR, focusing on retrieving small objects in complex images with textual queries. We propose a new benchmark, SORCE-1K, consisting of images with complex environments and textual queries describing less conspicuous small objects with minimal contextual cues from other salient objects. Preliminary analysis on SORCE-1K finds that existing T2IR methods struggle to capture small objects and encode all the semantics into a single embedding, leading to poor retrieval performance on SORCE-1K. Therefore, we propose to represent each image with multiple distinctive embeddings. We leverage Multimodal Large Language Models (MLLMs) to extract multiple embeddings for each image instructed by a set of Regional Prompts (ReP). Experimental results show that our multi-embedding approach through MLLM and ReP significantly outperforms existing T2IR methods on SORCE-1K. Our experiments validate the effectiveness of SORCE-1K for benchmarking SORCE performances, highlighting the potential of multi-embedding representation and text-customized MLLM features for addressing this task.  
  </ol>  
</details>  
**comments**: Project Page: https://github.com/MCG-NJU/SORCE  
  
  



## Image Matching  

### [SR3D: Unleashing Single-view 3D Reconstruction for Transparent and Specular Object Grasping](http://arxiv.org/abs/2505.24305)  
Mingxu Zhang, Xiaoqi Li, Jiahui Xu, Kaichen Zhou, Hojin Bae, Yan Shen, Chuyan Xiong, Jiaming Liu, Hao Dong  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in 3D robotic manipulation have improved grasping of everyday objects, but transparent and specular materials remain challenging due to depth sensing limitations. While several 3D reconstruction and depth completion approaches address these challenges, they suffer from setup complexity or limited observation information utilization. To address this, leveraging the power of single view 3D object reconstruction approaches, we propose a training free framework SR3D that enables robotic grasping of transparent and specular objects from a single view observation. Specifically, given single view RGB and depth images, SR3D first uses the external visual models to generate 3D reconstructed object mesh based on RGB image. Then, the key idea is to determine the 3D object's pose and scale to accurately localize the reconstructed object back into its original depth corrupted 3D scene. Therefore, we propose view matching and keypoint matching mechanisms,which leverage both the 2D and 3D's inherent semantic and geometric information in the observation to determine the object's 3D state within the scene, thereby reconstructing an accurate 3D depth map for effective grasp detection. Experiments in both simulation and real world show the reconstruction effectiveness of SR3D.  
  </ol>  
</details>  
  
  



