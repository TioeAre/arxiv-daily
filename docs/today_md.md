<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Rotation-Averaging:-A-Primal-Dual-Method-and-Closed-Forms-in-Cycle-Graphs>Rotation Averaging: A Primal-Dual Method and Closed-Forms in Cycle Graphs</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#360-in-the-Wild:-Dataset-for-Depth-Prediction-and-View-Synthesis>360 in the Wild: Dataset for Depth Prediction and View Synthesis</a></li>
        <li><a href=#Zero-shot-Composed-Image-Retrieval-Considering-Query-target-Relationship-Leveraging-Masked-Image-text-Pairs>Zero-shot Composed Image Retrieval Considering Query-target Relationship Leveraging Masked Image-text Pairs</a></li>
        <li><a href=#WV-Net:-A-foundation-model-for-SAR-WV-mode-satellite-imagery-trained-using-contrastive-self-supervised-learning-on-10-million-images>WV-Net: A foundation model for SAR WV-mode satellite imagery trained using contrastive self-supervised learning on 10 million images</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Shorter-SPECT-Scans-Using-Self-supervised-Coordinate-Learning-to-Synthesize-Skipped-Projection-Views>Shorter SPECT Scans Using Self-supervised Coordinate Learning to Synthesize Skipped Projection Views</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Rotation Averaging: A Primal-Dual Method and Closed-Forms in Cycle Graphs](http://arxiv.org/abs/2406.18564)  
Gabriel Moreira, Manuel Marques, João Paulo Costeira  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    A cornerstone of geometric reconstruction, rotation averaging seeks the set of absolute rotations that optimally explains a set of measured relative orientations between them. In addition to being an integral part of bundle adjustment and structure-from-motion, the problem of synchronizing rotations also finds applications in visual simultaneous localization and mapping, where it is used as an initialization for iterative solvers, and camera network calibration. Nevertheless, this optimization problem is both non-convex and high-dimensional. In this paper, we address it from a maximum likelihood estimation standpoint and make a twofold contribution. Firstly, we set forth a novel primal-dual method, motivated by the widely accepted spectral initialization. Further, we characterize stationary points of rotation averaging in cycle graphs topologies and contextualize this result within spectral graph theory. We benchmark the proposed method in multiple settings and certify our solution via duality theory, achieving a significant gain in precision and performance.  
  </ol>  
</details>  
**comments**: arXiv admin note: text overlap with arXiv:2109.08046  
  
  



## Visual Localization  

### [360 in the Wild: Dataset for Depth Prediction and View Synthesis](http://arxiv.org/abs/2406.18898)  
Kibaek Park, Francois Rameau, Jaesik Park, In So Kweon  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The large abundance of perspective camera datasets facilitated the emergence of novel learning-based strategies for various tasks, such as camera localization, single image depth estimation, or view synthesis. However, panoramic or omnidirectional image datasets, including essential information, such as pose and depth, are mostly made with synthetic scenes. In this work, we introduce a large scale 360 $^{\circ}$ videos dataset in the wild. This dataset has been carefully scraped from the Internet and has been captured from various locations worldwide. Hence, this dataset exhibits very diversified environments (e.g., indoor and outdoor) and contexts (e.g., with and without moving objects). Each of the 25K images constituting our dataset is provided with its respective camera's pose and depth map. We illustrate the relevance of our dataset for two main tasks, namely, single image depth estimation and view synthesis.  
  </ol>  
</details>  
  
### [Zero-shot Composed Image Retrieval Considering Query-target Relationship Leveraging Masked Image-text Pairs](http://arxiv.org/abs/2406.18836)  
Huaying Zhang, Rintaro Yanagi, Ren Togo, Takahiro Ogawa, Miki Haseyama  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper proposes a novel zero-shot composed image retrieval (CIR) method considering the query-target relationship by masked image-text pairs. The objective of CIR is to retrieve the target image using a query image and a query text. Existing methods use a textual inversion network to convert the query image into a pseudo word to compose the image and text and use a pre-trained visual-language model to realize the retrieval. However, they do not consider the query-target relationship to train the textual inversion network to acquire information for retrieval. In this paper, we propose a novel zero-shot CIR method that is trained end-to-end using masked image-text pairs. By exploiting the abundant image-text pairs that are convenient to obtain with a masking strategy for learning the query-target relationship, it is expected that accurate zero-shot CIR using a retrieval-focused textual inversion network can be realized. Experimental results show the effectiveness of the proposed method.  
  </ol>  
</details>  
**comments**: Accepted as a conference paper in IEEE ICIP 2024  
  
### [WV-Net: A foundation model for SAR WV-mode satellite imagery trained using contrastive self-supervised learning on 10 million images](http://arxiv.org/abs/2406.18765)  
Yannik Glaser, Justin E. Stopa, Linnea M. Wolniewicz, Ralph Foster, Doug Vandemark, Alexis Mouche, Bertrand Chapron, Peter Sadowski  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The European Space Agency's Copernicus Sentinel-1 (S-1) mission is a constellation of C-band synthetic aperture radar (SAR) satellites that provide unprecedented monitoring of the world's oceans. S-1's wave mode (WV) captures 20x20 km image patches at 5 m pixel resolution and is unaffected by cloud cover or time-of-day. The mission's open data policy has made SAR data easily accessible for a range of applications, but the need for manual image annotations is a bottleneck that hinders the use of machine learning methods. This study uses nearly 10 million WV-mode images and contrastive self-supervised learning to train a semantic embedding model called WV-Net. In multiple downstream tasks, WV-Net outperforms a comparable model that was pre-trained on natural images (ImageNet) with supervised learning. Experiments show improvements for estimating wave height (0.50 vs 0.60 RMSE using linear probing), estimating near-surface air temperature (0.90 vs 0.97 RMSE), and performing multilabel-classification of geophysical and atmospheric phenomena (0.96 vs 0.95 micro-averaged AUROC). WV-Net embeddings are also superior in an unsupervised image-retrieval task and scale better in data-sparse settings. Together, these results demonstrate that WV-Net embeddings can support geophysical research by providing a convenient foundation model for a variety of data analysis and exploration tasks.  
  </ol>  
</details>  
**comments**: 20 pages, 9 figures, submitted to NeurIPS 2024  
  
  



## NeRF  

### [Shorter SPECT Scans Using Self-supervised Coordinate Learning to Synthesize Skipped Projection Views](http://arxiv.org/abs/2406.18840)  
Zongyu Li, Yixuan Jia, Xiaojian Xu, Jason Hu, Jeffrey A. Fessler, Yuni K. Dewaraja  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Purpose: This study addresses the challenge of extended SPECT imaging duration under low-count conditions, as encountered in Lu-177 SPECT imaging, by developing a self-supervised learning approach to synthesize skipped SPECT projection views, thus shortening scan times in clinical settings. Methods: We employed a self-supervised coordinate-based learning technique, adapting the neural radiance field (NeRF) concept in computer vision to synthesize under-sampled SPECT projection views. For each single scan, we used self-supervised coordinate learning to estimate skipped SPECT projection views. The method was tested with various down-sampling factors (DFs=2, 4, 8) on both Lu-177 phantom SPECT/CT measurements and clinical SPECT/CT datasets, from 11 patients undergoing Lu-177 DOTATATE and 6 patients undergoing Lu-177 PSMA-617 radiopharmaceutical therapy. Results: For SPECT reconstructions, our method outperformed the use of linearly interpolated projections and partial projection views in relative contrast-to-noise-ratios (RCNR) averaged across different downsampling factors: 1) DOTATATE: 83% vs. 65% vs. 67% for lesions and 86% vs. 70% vs. 67% for kidney, 2) PSMA: 76% vs. 69% vs. 68% for lesions and 75% vs. 55% vs. 66% for organs, including kidneys, lacrimal glands, parotid glands, and submandibular glands. Conclusion: The proposed method enables reduction in acquisition time (by factors of 2, 4, or 8) while maintaining quantitative accuracy in clinical SPECT protocols by allowing for the collection of fewer projections. Importantly, the self-supervised nature of this NeRF-based approach eliminates the need for extensive training data, instead learning from each patient's projection data alone. The reduction in acquisition time is particularly relevant for imaging under low-count conditions and for protocols that require multiple-bed positions such as whole-body imaging.  
  </ol>  
</details>  
**comments**: 25 pages, 5568 words  
  
  



