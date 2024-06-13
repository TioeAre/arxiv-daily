<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#From-Variance-to-Veracity:-Unbundling-and-Mitigating-Gradient-Variance-in-Differentiable-Bundle-Adjustment-Layers>From Variance to Veracity: Unbundling and Mitigating Gradient Variance in Differentiable Bundle Adjustment Layers</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Self-supervised-Learning-of-Neural-Implicit-Feature-Fields-for-Camera-Pose-Refinement>Self-supervised Learning of Neural Implicit Feature Fields for Camera Pose Refinement</a></li>
        <li><a href=#ConceptHash:-Interpretable-Fine-Grained-Hashing-via-Concept-Discovery>ConceptHash: Interpretable Fine-Grained Hashing via Concept Discovery</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#ICE-G:-Image-Conditional-Editing-of-3D-Gaussian-Splats>ICE-G: Image Conditional Editing of 3D Gaussian Splats</a></li>
        <li><a href=#OpenObj:-Open-Vocabulary-Object-Level-Neural-Radiance-Fields-with-Fine-Grained-Understanding>OpenObj: Open-Vocabulary Object-Level Neural Radiance Fields with Fine-Grained Understanding</a></li>
        <li><a href=#Spatial-Annealing-Smoothing-for-Efficient-Few-shot-Neural-Rendering>Spatial Annealing Smoothing for Efficient Few-shot Neural Rendering</a></li>
        <li><a href=#C3DAG:-Controlled-3D-Animal-Generation-using-3D-pose-guidance>C3DAG: Controlled 3D Animal Generation using 3D pose guidance</a></li>
        <li><a href=#M-LRM:-Multi-view-Large-Reconstruction-Model>M-LRM: Multi-view Large Reconstruction Model</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [From Variance to Veracity: Unbundling and Mitigating Gradient Variance in Differentiable Bundle Adjustment Layers](http://arxiv.org/abs/2406.07785)  
Swaminathan Gurumurthy, Karnik Ram, Bingqing Chen, Zachary Manchester, Zico Kolter  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Various pose estimation and tracking problems in robotics can be decomposed into a correspondence estimation problem (often computed using a deep network) followed by a weighted least squares optimization problem to solve for the poses. Recent work has shown that coupling the two problems by iteratively refining one conditioned on the other's output yields SOTA results across domains. However, training these models has proved challenging, requiring a litany of tricks to stabilize and speed up training. In this work, we take the visual odometry problem as an example and identify three plausible causes: (1) flow loss interference, (2) linearization errors in the bundle adjustment (BA) layer, and (3) dependence of weight gradients on the BA residual. We show how these issues result in noisy and higher variance gradients, potentially leading to a slow down in training and instabilities. We then propose a simple, yet effective solution to reduce the gradient variance by using the weights predicted by the network in the inner optimization loop to weight the correspondence objective in the training problem. This helps the training objective `focus' on the more important points, thereby reducing the variance and mitigating the influence of outliers. We show that the resulting method leads to faster training and can be more flexibly trained in varying training setups without sacrificing performance. In particular we show $2$--$2.5\times$ training speedups over a baseline visual odometry model we modify.  
  </ol>  
</details>  
**comments**: Accepted at CVPR 2024  
  
  



## Visual Localization  

### [Self-supervised Learning of Neural Implicit Feature Fields for Camera Pose Refinement](http://arxiv.org/abs/2406.08463)  
Maxime Pietrantoni, Gabriela Csurka, Martin Humenberger, Torsten Sattler  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual localization techniques rely upon some underlying scene representation to localize against. These representations can be explicit such as 3D SFM map or implicit, such as a neural network that learns to encode the scene. The former requires sparse feature extractors and matchers to build the scene representation. The latter might lack geometric grounding not capturing the 3D structure of the scene well enough. This paper proposes to jointly learn the scene representation along with a 3D dense feature field and a 2D feature extractor whose outputs are embedded in the same metric space. Through a contrastive framework we align this volumetric field with the image-based extractor and regularize the latter with a ranking loss from learned surface information. We learn the underlying geometry of the scene with an implicit field through volumetric rendering and design our feature field to leverage intermediate geometric information encoded in the implicit field. The resulting features are discriminative and robust to viewpoint change while maintaining rich encoded information. Visual localization is then achieved by aligning the image-based features and the rendered volumetric features. We show the effectiveness of our approach on real-world scenes, demonstrating that our approach outperforms prior and concurrent work on leveraging implicit scene representations for localization.  
  </ol>  
</details>  
**comments**: Published in 3DV24 (highlight)  
  
### [ConceptHash: Interpretable Fine-Grained Hashing via Concept Discovery](http://arxiv.org/abs/2406.08457)  
[[code](https://github.com/kamwoh/concepthash)]  
Kam Woh Ng, Xiatian Zhu, Yi-Zhe Song, Tao Xiang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Existing fine-grained hashing methods typically lack code interpretability as they compute hash code bits holistically using both global and local features. To address this limitation, we propose ConceptHash, a novel method that achieves sub-code level interpretability. In ConceptHash, each sub-code corresponds to a human-understandable concept, such as an object part, and these concepts are automatically discovered without human annotations. Specifically, we leverage a Vision Transformer architecture and introduce concept tokens as visual prompts, along with image patch tokens as model inputs. Each concept is then mapped to a specific sub-code at the model output, providing natural sub-code interpretability. To capture subtle visual differences among highly similar sub-categories (e.g., bird species), we incorporate language guidance to ensure that the learned hash codes are distinguishable within fine-grained object classes while maintaining semantic alignment. This approach allows us to develop hash codes that exhibit similarity within families of species while remaining distinct from species in other families. Extensive experiments on four fine-grained image retrieval benchmarks demonstrate that ConceptHash outperforms previous methods by a significant margin, offering unique sub-code interpretability as an additional benefit. Code at: https://github.com/kamwoh/concepthash.  
  </ol>  
</details>  
**comments**: CVPRW 2024 - FGVC11 best paper award  
  
  



## NeRF  

### [ICE-G: Image Conditional Editing of 3D Gaussian Splats](http://arxiv.org/abs/2406.08488)  
Vishnu Jaganathan, Hannah Hanyun Huang, Muhammad Zubair Irshad, Varun Jampani, Amit Raj, Zsolt Kira  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recently many techniques have emerged to create high quality 3D assets and scenes. When it comes to editing of these objects, however, existing approaches are either slow, compromise on quality, or do not provide enough customization. We introduce a novel approach to quickly edit a 3D model from a single reference view. Our technique first segments the edit image, and then matches semantically corresponding regions across chosen segmented dataset views using DINO features. A color or texture change from a particular region of the edit image can then be applied to other views automatically in a semantically sensible manner. These edited views act as an updated dataset to further train and re-style the 3D scene. The end-result is therefore an edited 3D model. Our framework enables a wide variety of editing tasks such as manual local edits, correspondence based style transfer from any example image, and a combination of different styles from multiple example images. We use Gaussian Splats as our primary 3D representation due to their speed and ease of local editing, but our technique works for other methods such as NeRFs as well. We show through multiple examples that our method produces higher quality results while offering fine-grained control of editing. Project page: ice-gaussian.github.io  
  </ol>  
</details>  
**comments**: Accepted to CVPR AI4CC Workshop 2024. Project page:
  https://ice-gaussian.github.io  
  
### [OpenObj: Open-Vocabulary Object-Level Neural Radiance Fields with Fine-Grained Understanding](http://arxiv.org/abs/2406.08009)  
[[code](https://github.com/BIT-DYN/OpenObj)]  
Yinan Deng, Jiahui Wang, Jingyu Zhao, Jianyu Dou, Yi Yang, Yufeng Yue  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In recent years, there has been a surge of interest in open-vocabulary 3D scene reconstruction facilitated by visual language models (VLMs), which showcase remarkable capabilities in open-set retrieval. However, existing methods face some limitations: they either focus on learning point-wise features, resulting in blurry semantic understanding, or solely tackle object-level reconstruction, thereby overlooking the intricate details of the object's interior. To address these challenges, we introduce OpenObj, an innovative approach to build open-vocabulary object-level Neural Radiance Fields (NeRF) with fine-grained understanding. In essence, OpenObj establishes a robust framework for efficient and watertight scene modeling and comprehension at the object-level. Moreover, we incorporate part-level features into the neural fields, enabling a nuanced representation of object interiors. This approach captures object-level instances while maintaining a fine-grained understanding. The results on multiple datasets demonstrate that OpenObj achieves superior performance in zero-shot semantic segmentation and retrieval tasks. Additionally, OpenObj supports real-world robotics tasks at multiple scales, including global movement and local manipulation.  
  </ol>  
</details>  
**comments**: 8 pages, 7figures. Project Url: https://openobj.github.io/  
  
### [Spatial Annealing Smoothing for Efficient Few-shot Neural Rendering](http://arxiv.org/abs/2406.07828)  
Yuru Xiao, Xianming Liu, Deming Zhai, Kui Jiang, Junjun Jiang, Xiangyang Ji  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) with hybrid representations have shown impressive capabilities in reconstructing scenes for view synthesis, delivering high efficiency. Nonetheless, their performance significantly drops with sparse view inputs, due to the issue of overfitting. While various regularization strategies have been devised to address these challenges, they often depend on inefficient assumptions or are not compatible with hybrid models. There is a clear need for a method that maintains efficiency and improves resilience to sparse views within a hybrid framework. In this paper, we introduce an accurate and efficient few-shot neural rendering method named Spatial Annealing smoothing regularized NeRF (SANeRF), which is specifically designed for a pre-filtering-driven hybrid representation architecture. We implement an exponential reduction of the sample space size from an initially large value. This methodology is crucial for stabilizing the early stages of the training phase and significantly contributes to the enhancement of the subsequent process of detail refinement. Our extensive experiments reveal that, by adding merely one line of code, SANeRF delivers superior rendering quality and much faster reconstruction speed compared to current few-shot NeRF methods. Notably, SANeRF outperforms FreeNeRF by 0.3 dB in PSNR on the Blender dataset, while achieving 700x faster reconstruction speed.  
  </ol>  
</details>  
  
### [C3DAG: Controlled 3D Animal Generation using 3D pose guidance](http://arxiv.org/abs/2406.07742)  
Sandeep Mishra, Oindrila Saha, Alan C. Bovik  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent advancements in text-to-3D generation have demonstrated the ability to generate high quality 3D assets. However while generating animals these methods underperform, often portraying inaccurate anatomy and geometry. Towards ameliorating this defect, we present C3DAG, a novel pose-Controlled text-to-3D Animal Generation framework which generates a high quality 3D animal consistent with a given pose. We also introduce an automatic 3D shape creator tool, that allows dynamic pose generation and modification via a web-based tool, and that generates a 3D balloon animal using simple geometries. A NeRF is then initialized using this 3D shape using depth-controlled SDS. In the next stage, the pre-trained NeRF is fine-tuned using quadruped-pose-controlled SDS. The pipeline that we have developed not only produces geometrically and anatomically consistent results, but also renders highly controlled 3D animals, unlike prior methods which do not allow fine-grained pose control.  
  </ol>  
</details>  
  
### [M-LRM: Multi-view Large Reconstruction Model](http://arxiv.org/abs/2406.07648)  
Mengfei Li, Xiaoxiao Long, Yixun Liang, Weiyu Li, Yuan Liu, Peng Li, Xiaowei Chi, Xingqun Qi, Wei Xue, Wenhan Luo, Qifeng Liu, Yike Guo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Despite recent advancements in the Large Reconstruction Model (LRM) demonstrating impressive results, when extending its input from single image to multiple images, it exhibits inefficiencies, subpar geometric and texture quality, as well as slower convergence speed than expected.   It is attributed to that, LRM formulates 3D reconstruction as a naive images-to-3D translation problem, ignoring the strong 3D coherence among the input images. In this paper, we propose a Multi-view Large Reconstruction Model (M-LRM) designed to efficiently reconstruct high-quality 3D shapes from multi-views in a 3D-aware manner. Specifically, we introduce a multi-view consistent cross-attention scheme to enable M-LRM to accurately query information from the input images. Moreover, we employ the 3D priors of the input multi-view images to initialize the tri-plane tokens. Compared to LRM, the proposed M-LRM can produce a tri-plane NeRF with $128 \times 128$ resolution and generate 3D shapes of high fidelity. Experimental studies demonstrate that our model achieves a significant performance gain and faster training convergence than LRM. Project page: https://murphylmf.github.io/M-LRM/  
  </ol>  
</details>  
  
  



