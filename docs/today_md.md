<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Composite-Sketch+Text-Queries-for-Retrieving-Objects-with-Elusive-Names-and-Complex-Interactions>Composite Sketch+Text Queries for Retrieving Objects with Elusive Names and Complex Interactions</a></li>
        <li><a href=#Captured-by-Captions:-On-Memorization-and-its-Mitigation-in-CLIP-Models>Captured by Captions: On Memorization and its Mitigation in CLIP Models</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Sat-DN:-Implicit-Surface-Reconstruction-from-Multi-View-Satellite-Images-with-Depth-and-Normal-Supervision>Sat-DN: Implicit Surface Reconstruction from Multi-View Satellite Images with Depth and Normal Supervision</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Composite Sketch+Text Queries for Retrieving Objects with Elusive Names and Complex Interactions](http://arxiv.org/abs/2502.08438)  
Prajwal Gatti, Kshitij Parikh, Dhriti Prasanna Paul, Manish Gupta, Anand Mishra  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Non-native speakers with limited vocabulary often struggle to name specific objects despite being able to visualize them, e.g., people outside Australia searching for numbats. Further, users may want to search for such elusive objects with difficult-to-sketch interactions, e.g., numbat digging in the ground. In such common but complex situations, users desire a search interface that accepts composite multimodal queries comprising hand-drawn sketches of difficult-to-name but easy-to-draw objects and text describing difficult-to-sketch but easy-to-verbalize object attributes or interaction with the scene. This novel problem statement distinctly differs from the previously well-researched TBIR (text-based image retrieval) and SBIR (sketch-based image retrieval) problems. To study this under-explored task, we curate a dataset, CSTBIR (Composite Sketch+Text Based Image Retrieval), consisting of approx. 2M queries and 108K natural scene images. Further, as a solution to this problem, we propose a pretrained multimodal transformer-based baseline, STNET (Sketch+Text Network), that uses a hand-drawn sketch to localize relevant objects in the natural scene image, and encodes the text and image to perform image retrieval. In addition to contrastive learning, we propose multiple training objectives that improve the performance of our model. Extensive experiments show that our proposed method outperforms several state-of-the-art retrieval methods for text-only, sketch-only, and composite query modalities. We make the dataset and code available at our project website.  
  </ol>  
</details>  
**comments**: Accepted at AAAI 2024, 9 pages. Project Website:
  https://vl2g.github.io/projects/cstbir  
  
### [Captured by Captions: On Memorization and its Mitigation in CLIP Models](http://arxiv.org/abs/2502.07830)  
Wenhao Wang, Adam Dziedzic, Grace C. Kim, Michael Backes, Franziska Boenisch  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Multi-modal models, such as CLIP, have demonstrated strong performance in aligning visual and textual representations, excelling in tasks like image retrieval and zero-shot classification. Despite this success, the mechanisms by which these models utilize training data, particularly the role of memorization, remain unclear. In uni-modal models, both supervised and self-supervised, memorization has been shown to be essential for generalization. However, it is not well understood how these findings would apply to CLIP, which incorporates elements from both supervised learning via captions that provide a supervisory signal similar to labels, and from self-supervised learning via the contrastive objective. To bridge this gap in understanding, we propose a formal definition of memorization in CLIP (CLIPMem) and use it to quantify memorization in CLIP models. Our results indicate that CLIP's memorization behavior falls between the supervised and self-supervised paradigms, with "mis-captioned" samples exhibiting highest levels of memorization. Additionally, we find that the text encoder contributes more to memorization than the image encoder, suggesting that mitigation strategies should focus on the text domain. Building on these insights, we propose multiple strategies to reduce memorization while at the same time improving utility--something that had not been shown before for traditional learning paradigms where reducing memorization typically results in utility decrease.  
  </ol>  
</details>  
**comments**: Accepted at ICLR 2025  
  
  



## NeRF  

### [Sat-DN: Implicit Surface Reconstruction from Multi-View Satellite Images with Depth and Normal Supervision](http://arxiv.org/abs/2502.08352)  
Tianle Liu, Shuangming Zhao, Wanshou Jiang, Bingxuan Guo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    With advancements in satellite imaging technology, acquiring high-resolution multi-view satellite imagery has become increasingly accessible, enabling rapid and location-independent ground model reconstruction. However, traditional stereo matching methods struggle to capture fine details, and while neural radiance fields (NeRFs) achieve high-quality reconstructions, their training time is prohibitively long. Moreover, challenges such as low visibility of building facades, illumination and style differences between pixels, and weakly textured regions in satellite imagery further make it hard to reconstruct reasonable terrain geometry and detailed building facades. To address these issues, we propose Sat-DN, a novel framework leveraging a progressively trained multi-resolution hash grid reconstruction architecture with explicit depth guidance and surface normal consistency constraints to enhance reconstruction quality. The multi-resolution hash grid accelerates training, while the progressive strategy incrementally increases the learning frequency, using coarse low-frequency geometry to guide the reconstruction of fine high-frequency details. The depth and normal constraints ensure a clear building outline and correct planar distribution. Extensive experiments on the DFC2019 dataset demonstrate that Sat-DN outperforms existing methods, achieving state-of-the-art results in both qualitative and quantitative evaluations. The code is available at https://github.com/costune/SatDN.  
  </ol>  
</details>  
  
  



