<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#HARMONY:-A-Scalable-Distributed-Vector-Database-for-High-Throughput-Approximate-Nearest-Neighbor-Search>HARMONY: A Scalable Distributed Vector Database for High-Throughput Approximate Nearest Neighbor Search</a></li>
        <li><a href=#TACS-Graphs:-Traversability-Aware-Consistent-Scene-Graphs-for-Ground-Robot-Indoor-Localization-and-Mapping>TACS-Graphs: Traversability-Aware Consistent Scene Graphs for Ground Robot Indoor Localization and Mapping</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [HARMONY: A Scalable Distributed Vector Database for High-Throughput Approximate Nearest Neighbor Search](http://arxiv.org/abs/2506.14707)  
Qian Xu, Feng Zhang, Chengxi Li, Lei Cao, Zheng Chen, Jidong Zhai, Xiaoyong Du  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Approximate Nearest Neighbor Search (ANNS) is essential for various data-intensive applications, including recommendation systems, image retrieval, and machine learning. Scaling ANNS to handle billions of high-dimensional vectors on a single machine presents significant challenges in memory capacity and processing efficiency. To address these challenges, distributed vector databases leverage multiple nodes for the parallel storage and processing of vectors. However, existing solutions often suffer from load imbalance and high communication overhead, primarily due to traditional partition strategies that fail to effectively distribute the workload. In this paper, we introduce Harmony, a distributed ANNS system that employs a novel multi-granularity partition strategy, combining dimension-based and vector-based partition. This strategy ensures a balanced distribution of computational load across all nodes while effectively minimizing communication costs. Furthermore, Harmony incorporates an early-stop pruning mechanism that leverages the monotonicity of distance computations in dimension-based partition, resulting in significant reductions in both computational and communication overhead. We conducted extensive experiments on diverse real-world datasets, demonstrating that Harmony outperforms leading distributed vector databases, achieving 4.63 times throughput on average in four nodes and 58% performance improvement over traditional distribution for skewed workloads.  
  </ol>  
</details>  
  
### [TACS-Graphs: Traversability-Aware Consistent Scene Graphs for Ground Robot Indoor Localization and Mapping](http://arxiv.org/abs/2506.14178)  
Jeewon Kim, Minho Oh, Hyun Myung  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Scene graphs have emerged as a powerful tool for robots, providing a structured representation of spatial and semantic relationships for advanced task planning. Despite their potential, conventional 3D indoor scene graphs face critical limitations, particularly under- and over-segmentation of room layers in structurally complex environments. Under-segmentation misclassifies non-traversable areas as part of a room, often in open spaces, while over-segmentation fragments a single room into overlapping segments in complex environments. These issues stem from naive voxel-based map representations that rely solely on geometric proximity, disregarding the structural constraints of traversable spaces and resulting in inconsistent room layers within scene graphs. To the best of our knowledge, this work is the first to tackle segmentation inconsistency as a challenge and address it with Traversability-Aware Consistent Scene Graphs (TACS-Graphs), a novel framework that integrates ground robot traversability with room segmentation. By leveraging traversability as a key factor in defining room boundaries, the proposed method achieves a more semantically meaningful and topologically coherent segmentation, effectively mitigating the inaccuracies of voxel-based scene graph approaches in complex environments. Furthermore, the enhanced segmentation consistency improves loop closure detection efficiency in the proposed Consistent Scene Graph-leveraging Loop Closure Detection (CoSG-LCD) leading to higher pose estimation accuracy. Experimental results confirm that the proposed approach outperforms state-of-the-art methods in terms of scene graph consistency and pose graph optimization performance.  
  </ol>  
</details>  
**comments**: Accepted by IROS 2025  
  
  



