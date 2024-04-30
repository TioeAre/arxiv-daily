<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#An-Attention-Based-Deep-Learning-Architecture-for-Real-Time-Monocular-Visual-Odometry:-Applications-to-GPS-free-Drone-Navigation>An Attention-Based Deep Learning Architecture for Real-Time Monocular Visual Odometry: Applications to GPS-free Drone Navigation</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Enhancing-Interactive-Image-Retrieval-With-Query-Rewriting-Using-Large-Language-Models-and-Vision-Language-Models>Enhancing Interactive Image Retrieval With Query Rewriting Using Large Language Models and Vision Language Models</a></li>
        <li><a href=#Dual-Modal-Prompting-for-Sketch-Based-Image-Retrieval>Dual-Modal Prompting for Sketch-Based Image Retrieval</a></li>
        <li><a href=#Semantic-Line-Combination-Detector>Semantic Line Combination Detector</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#MinBackProp----Backpropagating-through-Minimal-Solvers>MinBackProp -- Backpropagating through Minimal Solvers</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#S3-SLAM:-Sparse-Tri-plane-Encoding-for-Neural-Implicit-SLAM>S3-SLAM: Sparse Tri-plane Encoding for Neural Implicit SLAM</a></li>
        <li><a href=#DPER:-Diffusion-Prior-Driven-Neural-Representation-for-Limited-Angle-and-Sparse-View-CT-Reconstruction>DPER: Diffusion Prior Driven Neural Representation for Limited Angle and Sparse View CT Reconstruction</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [An Attention-Based Deep Learning Architecture for Real-Time Monocular Visual Odometry: Applications to GPS-free Drone Navigation](http://arxiv.org/abs/2404.17745)  
Olivier Brochu Dufour, Abolfazl Mohebbi, Sofiane Achiche  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Drones are increasingly used in fields like industry, medicine, research, disaster relief, defense, and security. Technical challenges, such as navigation in GPS-denied environments, hinder further adoption. Research in visual odometry is advancing, potentially solving GPS-free navigation issues. Traditional visual odometry methods use geometry-based pipelines which, while popular, often suffer from error accumulation and high computational demands. Recent studies utilizing deep neural networks (DNNs) have shown improved performance, addressing these drawbacks. Deep visual odometry typically employs convolutional neural networks (CNNs) and sequence modeling networks like recurrent neural networks (RNNs) to interpret scenes and deduce visual odometry from video sequences. This paper presents a novel real-time monocular visual odometry model for drones, using a deep neural architecture with a self-attention module. It estimates the ego-motion of a camera on a drone, using consecutive video frames. An inference utility processes the live video feed, employing deep learning to estimate the drone's trajectory. The architecture combines a CNN for image feature extraction and a long short-term memory (LSTM) network with a multi-head attention module for video sequence modeling. Tested on two visual odometry datasets, this model converged 48% faster than a previous RNN model and showed a 22% reduction in mean translational drift and a 12% improvement in mean translational absolute trajectory error, demonstrating enhanced robustness to noise.  
  </ol>  
</details>  
**comments**: 22 Pages, 3 Tables, 9 Figures  
  
  



## Visual Localization  

### [Enhancing Interactive Image Retrieval With Query Rewriting Using Large Language Models and Vision Language Models](http://arxiv.org/abs/2404.18746)  
Hongyi Zhu, Jia-Hong Huang, Stevan Rudinac, Evangelos Kanoulas  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Image search stands as a pivotal task in multimedia and computer vision, finding applications across diverse domains, ranging from internet search to medical diagnostics. Conventional image search systems operate by accepting textual or visual queries, retrieving the top-relevant candidate results from the database. However, prevalent methods often rely on single-turn procedures, introducing potential inaccuracies and limited recall. These methods also face the challenges, such as vocabulary mismatch and the semantic gap, constraining their overall effectiveness. To address these issues, we propose an interactive image retrieval system capable of refining queries based on user relevance feedback in a multi-turn setting. This system incorporates a vision language model (VLM) based image captioner to enhance the quality of text-based queries, resulting in more informative queries with each iteration. Moreover, we introduce a large language model (LLM) based denoiser to refine text-based query expansions, mitigating inaccuracies in image descriptions generated by captioning models. To evaluate our system, we curate a new dataset by adapting the MSR-VTT video retrieval dataset to the image retrieval task, offering multiple relevant ground truth images for each query. Through comprehensive experiments, we validate the effectiveness of our proposed system against baseline methods, achieving state-of-the-art performance with a notable 10\% improvement in terms of recall. Our contributions encompass the development of an innovative interactive image retrieval system, the integration of an LLM-based denoiser, the curation of a meticulously designed evaluation dataset, and thorough experimental validation.  
  </ol>  
</details>  
  
### [Dual-Modal Prompting for Sketch-Based Image Retrieval](http://arxiv.org/abs/2404.18695)  
Liying Gao, Bingliang Jiao, Peng Wang, Shizhou Zhang, Hanwang Zhang, Yanning Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Sketch-based image retrieval (SBIR) associates hand-drawn sketches with their corresponding realistic images. In this study, we aim to tackle two major challenges of this task simultaneously: i) zero-shot, dealing with unseen categories, and ii) fine-grained, referring to intra-category instance-level retrieval. Our key innovation lies in the realization that solely addressing this cross-category and fine-grained recognition task from the generalization perspective may be inadequate since the knowledge accumulated from limited seen categories might not be fully valuable or transferable to unseen target categories. Inspired by this, in this work, we propose a dual-modal prompting CLIP (DP-CLIP) network, in which an adaptive prompting strategy is designed. Specifically, to facilitate the adaptation of our DP-CLIP toward unpredictable target categories, we employ a set of images within the target category and the textual category label to respectively construct a set of category-adaptive prompt tokens and channel scales. By integrating the generated guidance, DP-CLIP could gain valuable category-centric insights, efficiently adapting to novel categories and capturing unique discriminative clues for effective retrieval within each target category. With these designs, our DP-CLIP outperforms the state-of-the-art fine-grained zero-shot SBIR method by 7.3% in Acc.@1 on the Sketchy dataset. Meanwhile, in the other two category-level zero-shot SBIR benchmarks, our method also achieves promising performance.  
  </ol>  
</details>  
  
### [Semantic Line Combination Detector](http://arxiv.org/abs/2404.18399)  
[[code](https://github.com/jinwon-ko/slcd)]  
Jinwon Ko, Dongkwon Jin, Chang-Su Kim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    A novel algorithm, called semantic line combination detector (SLCD), to find an optimal combination of semantic lines is proposed in this paper. It processes all lines in each line combination at once to assess the overall harmony of the lines. First, we generate various line combinations from reliable lines. Second, we estimate the score of each line combination and determine the best one. Experimental results demonstrate that the proposed SLCD outperforms existing semantic line detectors on various datasets. Moreover, it is shown that SLCD can be applied effectively to three vision tasks of vanishing point detection, symmetry axis detection, and composition-based image retrieval. Our codes are available at https://github.com/Jinwon-Ko/SLCD.  
  </ol>  
</details>  
  
  



## Image Matching  

### [MinBackProp -- Backpropagating through Minimal Solvers](http://arxiv.org/abs/2404.17993)  
Diana Sungatullina, Tomas Pajdla  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We present an approach to backpropagating through minimal problem solvers in end-to-end neural network training. Traditional methods relying on manually constructed formulas, finite differences, and autograd are laborious, approximate, and unstable for complex minimal problem solvers. We show that using the Implicit function theorem to calculate derivatives to backpropagate through the solution of a minimal problem solver is simple, fast, and stable. We compare our approach to (i) using the standard autograd on minimal problem solvers and relate it to existing backpropagation formulas through SVD-based and Eig-based solvers and (ii) implementing the backprop with an existing PyTorch Deep Declarative Networks (DDN) framework. We demonstrate our technique on a toy example of training outlier-rejection weights for 3D point registration and on a real application of training an outlier-rejection and RANSAC sampling network in image matching. Our method provides $100\%$ stability and is 10 times faster compared to autograd, which is unstable and slow, and compared to DDN, which is stable but also slow.  
  </ol>  
</details>  
  
  



## NeRF  

### [S3-SLAM: Sparse Tri-plane Encoding for Neural Implicit SLAM](http://arxiv.org/abs/2404.18284)  
Zhiyao Zhang, Yunzhou Zhang, Yanmin Wu, Bin Zhao, Xingshuo Wang, Rui Tian  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    With the emergence of Neural Radiance Fields (NeRF), neural implicit representations have gained widespread applications across various domains, including simultaneous localization and mapping. However, current neural implicit SLAM faces a challenging trade-off problem between performance and the number of parameters. To address this problem, we propose sparse tri-plane encoding, which efficiently achieves scene reconstruction at resolutions up to 512 using only 2~4% of the commonly used tri-plane parameters (reduced from 100MB to 2~4MB). On this basis, we design S3-SLAM to achieve rapid and high-quality tracking and mapping through sparsifying plane parameters and integrating orthogonal features of tri-plane. Furthermore, we develop hierarchical bundle adjustment to achieve globally consistent geometric structures and reconstruct high-resolution appearance. Experimental results demonstrate that our approach achieves competitive tracking and scene reconstruction with minimal parameters on three datasets. Source code will soon be available.  
  </ol>  
</details>  
  
### [DPER: Diffusion Prior Driven Neural Representation for Limited Angle and Sparse View CT Reconstruction](http://arxiv.org/abs/2404.17890)  
Chenhe Du, Xiyue Lin, Qing Wu, Xuanyu Tian, Ying Su, Zhe Luo, Hongjiang Wei, S. Kevin Zhou, Jingyi Yu, Yuyao Zhang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Limited-angle and sparse-view computed tomography (LACT and SVCT) are crucial for expanding the scope of X-ray CT applications. However, they face challenges due to incomplete data acquisition, resulting in diverse artifacts in the reconstructed CT images. Emerging implicit neural representation (INR) techniques, such as NeRF, NeAT, and NeRP, have shown promise in under-determined CT imaging reconstruction tasks. However, the unsupervised nature of INR architecture imposes limited constraints on the solution space, particularly for the highly ill-posed reconstruction task posed by LACT and ultra-SVCT. In this study, we introduce the Diffusion Prior Driven Neural Representation (DPER), an advanced unsupervised framework designed to address the exceptionally ill-posed CT reconstruction inverse problems. DPER adopts the Half Quadratic Splitting (HQS) algorithm to decompose the inverse problem into data fidelity and distribution prior sub-problems. The two sub-problems are respectively addressed by INR reconstruction scheme and pre-trained score-based diffusion model. This combination initially preserves the implicit image local consistency prior from INR. Additionally, it effectively augments the feasibility of the solution space for the inverse problem through the generative diffusion model, resulting in increased stability and precision in the solutions. We conduct comprehensive experiments to evaluate the performance of DPER on LACT and ultra-SVCT reconstruction with two public datasets (AAPM and LIDC). The results show that our method outperforms the state-of-the-art reconstruction methods on in-domain datasets, while achieving significant performance improvements on out-of-domain datasets.  
  </ol>  
</details>  
**comments**: 15 pages, 10 figures  
  
  



