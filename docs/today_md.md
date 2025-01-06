<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#A-Minimal-Subset-Approach-for-Efficient-and-Scalable-Loop-Closure>A Minimal Subset Approach for Efficient and Scalable Loop Closure</a></li>
        <li><a href=#iCBIR-Sli:-Interpretable-Content-Based-Image-Retrieval-with-2D-Slice-Embeddings>iCBIR-Sli: Interpretable Content-Based Image Retrieval with 2D Slice Embeddings</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [A Minimal Subset Approach for Efficient and Scalable Loop Closure](http://arxiv.org/abs/2501.01791)  
Nikolaos Stathoulopoulos, Christoforos Kanellakis, George Nikolakopoulos  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Loop closure detection in large-scale and long-term missions can be computationally demanding due to the need to identify, verify, and process numerous candidate pairs to establish edge connections for the pose graph optimization. Keyframe sampling mitigates this by reducing the number of frames stored and processed in the back-end system. In this article, we address the gap in optimized keyframe sampling for the combined problem of pose graph optimization and loop closure detection. Our Minimal Subset Approach (MSA) employs an optimization strategy with two key factors, redundancy minimization and information preservation, within a sliding window framework to efficiently reduce redundant keyframes, while preserving essential information. This method delivers comparable performance to baseline approaches, while enhancing scalability and reducing computational overhead. Finally, we evaluate MSA on relevant publicly available datasets, showcasing that it consistently performs across a wide range of environments, without requiring any manual parameter tuning.  
  </ol>  
</details>  
**comments**: 7 pages, 8 Figures, 2 Tables. Submitted  
  
### [iCBIR-Sli: Interpretable Content-Based Image Retrieval with 2D Slice Embeddings](http://arxiv.org/abs/2501.01642)  
Shuhei Tomoshige, Hayato Muraki, Kenichi Oishi, Hitoshi Iyatomi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Current methods for searching brain MR images rely on text-based approaches, highlighting a significant need for content-based image retrieval (CBIR) systems. Directly applying 3D brain MR images to machine learning models offers the benefit of effectively learning the brain's structure; however, building the generalized model necessitates a large amount of training data. While models that consider depth direction and utilize continuous 2D slices have demonstrated success in segmentation and classification tasks involving 3D data, concerns remain. Specifically, using general 2D slices may lead to the oversight of pathological features and discontinuities in depth direction information. Furthermore, to the best of the authors' knowledge, there have been no attempts to develop a practical CBIR system that preserves the entire brain's structural information. In this study, we propose an interpretable CBIR method for brain MR images, named iCBIR-Sli (Interpretable CBIR with 2D Slice Embedding), which, for the first time globally, utilizes a series of 2D slices. iCBIR-Sli addresses the challenges associated with using 2D slices by effectively aggregating slice information, thereby achieving low-dimensional representations with high completeness, usability, robustness, and interoperability, which are qualities essential for effective CBIR. In retrieval evaluation experiments utilizing five publicly available brain MR datasets (ADNI2/3, OASIS3/4, AIBL) for Alzheimer's disease and cognitively normal, iCBIR-Sli demonstrated top-1 retrieval performance (macro F1 = 0.859), comparable to existing deep learning models explicitly designed for classification, without the need for an external classifier. Additionally, the method provided high interpretability by clearly identifying the brain regions indicative of the searched-for disease.  
  </ol>  
</details>  
**comments**: 8 pages, 2 figures. Accepted at the SPIE Medical Imaging  
  
  



