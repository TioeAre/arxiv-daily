<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#TripletCLIP:-Improving-Compositional-Reasoning-of-CLIP-via-Synthetic-Vision-Language-Negatives>TripletCLIP: Improving Compositional Reasoning of CLIP via Synthetic Vision-Language Negatives</a></li>
        <li><a href=#INQUIRE:-A-Natural-World-Text-to-Image-Retrieval-Benchmark>INQUIRE: A Natural World Text-to-Image Retrieval Benchmark</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#CAD-NeRF:-Learning-NeRFs-from-Uncalibrated-Few-view-Images-by-CAD-Model-Retrieval>CAD-NeRF: Learning NeRFs from Uncalibrated Few-view Images by CAD Model Retrieval</a></li>
        <li><a href=#Exploring-Seasonal-Variability-in-the-Context-of-Neural-Radiance-Fields-for-3D-Reconstruction-on-Satellite-Imagery>Exploring Seasonal Variability in the Context of Neural Radiance Fields for 3D Reconstruction on Satellite Imagery</a></li>
        <li><a href=#Multi-modal-NeRF-Self-Supervision-for-LiDAR-Semantic-Segmentation>Multi-modal NeRF Self-Supervision for LiDAR Semantic Segmentation</a></li>
        <li><a href=#NeRF-Aug:-Data-Augmentation-for-Robotics-with-Neural-Radiance-Fields>NeRF-Aug: Data Augmentation for Robotics with Neural Radiance Fields</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [TripletCLIP: Improving Compositional Reasoning of CLIP via Synthetic Vision-Language Negatives](http://arxiv.org/abs/2411.02545)  
Maitreya Patel, Abhiram Kusumba, Sheng Cheng, Changhoon Kim, Tejas Gokhale, Chitta Baral, Yezhou Yang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Contrastive Language-Image Pretraining (CLIP) models maximize the mutual information between text and visual modalities to learn representations. This makes the nature of the training data a significant factor in the efficacy of CLIP for downstream tasks. However, the lack of compositional diversity in contemporary image-text datasets limits the compositional reasoning ability of CLIP. We show that generating ``hard'' negative captions via in-context learning and synthesizing corresponding negative images with text-to-image generators offers a solution. We introduce a novel contrastive pre-training strategy that leverages these hard negative captions and images in an alternating fashion to train CLIP. We demonstrate that our method, named TripletCLIP, when applied to existing datasets such as CC3M and CC12M, enhances the compositional capabilities of CLIP, resulting in an absolute improvement of over 9% on the SugarCrepe benchmark on an equal computational budget, as well as improvements in zero-shot image classification and image retrieval. Our code, models, and data are available at: https://tripletclip.github.io  
  </ol>  
</details>  
**comments**: Accepted at: NeurIPS 2024 | Project Page:
  https://tripletclip.github.io  
  
### [INQUIRE: A Natural World Text-to-Image Retrieval Benchmark](http://arxiv.org/abs/2411.02537)  
[[code](https://github.com/inquire-benchmark/INQUIRE)]  
Edward Vendrow, Omiros Pantazis, Alexander Shepard, Gabriel Brostow, Kate E. Jones, Oisin Mac Aodha, Sara Beery, Grant Van Horn  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce INQUIRE, a text-to-image retrieval benchmark designed to challenge multimodal vision-language models on expert-level queries. INQUIRE includes iNaturalist 2024 (iNat24), a new dataset of five million natural world images, along with 250 expert-level retrieval queries. These queries are paired with all relevant images comprehensively labeled within iNat24, comprising 33,000 total matches. Queries span categories such as species identification, context, behavior, and appearance, emphasizing tasks that require nuanced image understanding and domain expertise. Our benchmark evaluates two core retrieval tasks: (1) INQUIRE-Fullrank, a full dataset ranking task, and (2) INQUIRE-Rerank, a reranking task for refining top-100 retrievals. Detailed evaluation of a range of recent multimodal models demonstrates that INQUIRE poses a significant challenge, with the best models failing to achieve an mAP@50 above 50%. In addition, we show that reranking with more powerful multimodal models can enhance retrieval performance, yet there remains a significant margin for improvement. By focusing on scientifically-motivated ecological challenges, INQUIRE aims to bridge the gap between AI capabilities and the needs of real-world scientific inquiry, encouraging the development of retrieval systems that can assist with accelerating ecological and biodiversity research. Our dataset and code are available at https://inquire-benchmark.github.io  
  </ol>  
</details>  
**comments**: Published in NeurIPS 2024, Datasets and Benchmarks Track  
  
  



## NeRF  

### [CAD-NeRF: Learning NeRFs from Uncalibrated Few-view Images by CAD Model Retrieval](http://arxiv.org/abs/2411.02979)  
Xin Wen, Xuening Zhu, Renjiao Yi, Zhifeng Wang, Chenyang Zhu, Kai Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Reconstructing from multi-view images is a longstanding problem in 3D vision, where neural radiance fields (NeRFs) have shown great potential and get realistic rendered images of novel views. Currently, most NeRF methods either require accurate camera poses or a large number of input images, or even both. Reconstructing NeRF from few-view images without poses is challenging and highly ill-posed. To address this problem, we propose CAD-NeRF, a method reconstructed from less than 10 images without any known poses. Specifically, we build a mini library of several CAD models from ShapeNet and render them from many random views. Given sparse-view input images, we run a model and pose retrieval from the library, to get a model with similar shapes, serving as the density supervision and pose initializations. Here we propose a multi-view pose retrieval method to avoid pose conflicts among views, which is a new and unseen problem in uncalibrated NeRF methods. Then, the geometry of the object is trained by the CAD guidance. The deformation of the density field and camera poses are optimized jointly. Then texture and density are trained and fine-tuned as well. All training phases are in self-supervised manners. Comprehensive evaluations of synthetic and real images show that CAD-NeRF successfully learns accurate densities with a large deformation from retrieved CAD models, showing the generalization abilities.  
  </ol>  
</details>  
**comments**: The article has been accepted by Frontiers of Computer Science (FCS)  
  
### [Exploring Seasonal Variability in the Context of Neural Radiance Fields for 3D Reconstruction on Satellite Imagery](http://arxiv.org/abs/2411.02972)  
Liv Kåreborn, Erica Ingerstad, Amanda Berg, Justus Karlsson, Leif Haglund  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    In this work, the seasonal predictive capabilities of Neural Radiance Fields (NeRF) applied to satellite images are investigated. Focusing on the utilization of satellite data, the study explores how Sat-NeRF, a novel approach in computer vision, performs in predicting seasonal variations across different months. Through comprehensive analysis and visualization, the study examines the model's ability to capture and predict seasonal changes, highlighting specific challenges and strengths. Results showcase the impact of the sun direction on predictions, revealing nuanced details in seasonal transitions, such as snow cover, color accuracy, and texture representation in different landscapes. Given these results, we propose Planet-NeRF, an extension to Sat-NeRF capable of incorporating seasonal variability through a set of month embedding vectors. Comparative evaluations reveal that Planet-NeRF outperforms prior models in the case where seasonal changes are present. The extensive evaluation combined with the proposed method offers promising avenues for future research in this domain.  
  </ol>  
</details>  
  
### [Multi-modal NeRF Self-Supervision for LiDAR Semantic Segmentation](http://arxiv.org/abs/2411.02969)  
Xavier Timoneda, Markus Herb, Fabian Duerr, Daniel Goehring, Fisher Yu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    LiDAR Semantic Segmentation is a fundamental task in autonomous driving perception consisting of associating each LiDAR point to a semantic label. Fully-supervised models have widely tackled this task, but they require labels for each scan, which either limits their domain or requires impractical amounts of expensive annotations. Camera images, which are generally recorded alongside LiDAR pointclouds, can be processed by the widely available 2D foundation models, which are generic and dataset-agnostic. However, distilling knowledge from 2D data to improve LiDAR perception raises domain adaptation challenges. For example, the classical perspective projection suffers from the parallax effect produced by the position shift between both sensors at their respective capture times. We propose a Semi-Supervised Learning setup to leverage unlabeled LiDAR pointclouds alongside distilled knowledge from the camera images. To self-supervise our model on the unlabeled scans, we add an auxiliary NeRF head and cast rays from the camera viewpoint over the unlabeled voxel features. The NeRF head predicts densities and semantic logits at each sampled ray location which are used for rendering pixel semantics. Concurrently, we query the Segment-Anything (SAM) foundation model with the camera image to generate a set of unlabeled generic masks. We fuse the masks with the rendered pixel semantics from LiDAR to produce pseudo-labels that supervise the pixel predictions. During inference, we drop the NeRF head and run our model with only LiDAR. We show the effectiveness of our approach in three public LiDAR Semantic Segmentation benchmarks: nuScenes, SemanticKITTI and ScribbleKITTI.  
  </ol>  
</details>  
**comments**: IEEE/RSJ International Conference on Intelligent Robots and Systems
  (IROS) 2024  
  
### [NeRF-Aug: Data Augmentation for Robotics with Neural Radiance Fields](http://arxiv.org/abs/2411.02482)  
Eric Zhu, Mara Levy, Matthew Gwilliam, Abhinav Shrivastava  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Training a policy that can generalize to unknown objects is a long standing challenge within the field of robotics. The performance of a policy often drops significantly in situations where an object in the scene was not seen during training. To solve this problem, we present NeRF-Aug, a novel method that is capable of teaching a policy to interact with objects that are not present in the dataset. This approach differs from existing approaches by leveraging the speed and photorealism of a neural radiance field for augmentation. NeRF- Aug both creates more photorealistic data and runs 3.83 times faster than existing methods. We demonstrate the effectiveness of our method on 4 tasks with 11 novel objects that have no expert demonstration data. We achieve an average 69.1% success rate increase over existing methods. See video results at https://nerf-aug.github.io.  
  </ol>  
</details>  
  
  



