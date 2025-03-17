<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Exploring-the-best-way-for-UAV-visual-localization-under-Low-altitude-Multi-view-Observation-Condition:-a-Benchmark>Exploring the best way for UAV visual localization under Low-altitude Multi-view Observation Condition: a Benchmark</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#Refining-Image-Edge-Detection-via-Linear-Canonical-Riesz-Transforms>Refining Image Edge Detection via Linear Canonical Riesz Transforms</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Exploring the best way for UAV visual localization under Low-altitude Multi-view Observation Condition: a Benchmark](http://arxiv.org/abs/2503.10692)  
Yibin Ye, Xichao Teng, Shuo Chen, Zhang Li, Leqi Liu, Qifeng Yu, Tao Tan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Absolute Visual Localization (AVL) enables Unmanned Aerial Vehicle (UAV) to determine its position in GNSS-denied environments by establishing geometric relationships between UAV images and geo-tagged reference maps. While many previous works have achieved AVL with image retrieval and matching techniques, research in low-altitude multi-view scenarios still remains limited. Low-altitude Multi-view condition presents greater challenges due to extreme viewpoint changes. To explore the best UAV AVL approach in such condition, we proposed this benchmark. Firstly, a large-scale Low-altitude Multi-view dataset called AnyVisLoc was constructed. This dataset includes 18,000 images captured at multiple scenes and altitudes, along with 2.5D reference maps containing aerial photogrammetry maps and historical satellite maps. Secondly, a unified framework was proposed to integrate the state-of-the-art AVL approaches and comprehensively test their performance. The best combined method was chosen as the baseline and the key factors that influencing localization accuracy are thoroughly analyzed based on it. This baseline achieved a 74.1% localization accuracy within 5m under Low-altitude, Multi-view conditions. In addition, a novel retrieval metric called PDM@K was introduced to better align with the characteristics of the UAV AVL task. Overall, this benchmark revealed the challenges of Low-altitude, Multi-view UAV AVL and provided valuable guidance for future research. The dataset and codes are available at https://github.com/UAV-AVL/Benchmark  
  </ol>  
</details>  
  
  



## Image Matching  

### [Refining Image Edge Detection via Linear Canonical Riesz Transforms](http://arxiv.org/abs/2503.11148)  
Shuhui Yang, Zunwei Fu, Dachun Yang, Yan Lin, Zhen Li  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Combining the linear canonical transform and the Riesz transform, we introduce the linear canonical Riesz transform (for short, LCRT), which is further proved to be a linear canonical multiplier. Using this LCRT multiplier, we conduct numerical simulations on images. Notably, the LCRT multiplier significantly reduces the complexity of the algorithm. Based on these we introduce the new concept of the sharpness $R^{\rm E}_{\rm sc}$ of the edge strength and continuity of images associated with the LCRT and, using it, we propose a new LCRT image edge detection method (for short, LCRT-IED method) and provide its mathematical foundation. Our experiments indicate that this sharpness $R^{\rm E}_{\rm sc}$ characterizes the macroscopic trend of edge variations of the image under consideration, while this new LCRT-IED method not only controls the overall edge strength and continuity of the image, but also excels in feature extraction in some local regions. These highlight the fundamental differences between the LCRT and the Riesz transform, which are precisely due to the multiparameter of the former. This new LCRT-IED method might be of significant importance for image feature extraction, image matching, and image refinement.  
  </ol>  
</details>  
**comments**: 30 pages, 60 figures,  
  
  



