<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#FastMap:-Revisiting-Dense-and-Scalable-Structure-from-Motion>FastMap: Revisiting Dense and Scalable Structure from Motion</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#OBD-Finder:-Explainable-Coarse-to-Fine-Text-Centric-Oracle-Bone-Duplicates-Discovery>OBD-Finder: Explainable Coarse-to-Fine Text-Centric Oracle Bone Duplicates Discovery</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#OBD-Finder:-Explainable-Coarse-to-Fine-Text-Centric-Oracle-Bone-Duplicates-Discovery>OBD-Finder: Explainable Coarse-to-Fine Text-Centric Oracle Bone Duplicates Discovery</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [FastMap: Revisiting Dense and Scalable Structure from Motion](http://arxiv.org/abs/2505.04612)  
Jiahao Li, Haochen Wang, Muhammad Zubair Irshad, Igor Vasiljevic, Matthew R. Walter, Vitor Campagnolo Guizilini, Greg Shakhnarovich  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We propose FastMap, a new global structure from motion method focused on speed and simplicity. Previous methods like COLMAP and GLOMAP are able to estimate high-precision camera poses, but suffer from poor scalability when the number of matched keypoint pairs becomes large. We identify two key factors leading to this problem: poor parallelization and computationally expensive optimization steps. To overcome these issues, we design an SfM framework that relies entirely on GPU-friendly operations, making it easily parallelizable. Moreover, each optimization step runs in time linear to the number of image pairs, independent of keypoint pairs or 3D points. Through extensive experiments, we show that FastMap is one to two orders of magnitude faster than COLMAP and GLOMAP on large-scale scenes with comparable pose accuracy.  
  </ol>  
</details>  
**comments**: Project webpage: https://jiahao.ai/fastmap  
  
  



## Visual Localization  

### [OBD-Finder: Explainable Coarse-to-Fine Text-Centric Oracle Bone Duplicates Discovery](http://arxiv.org/abs/2505.03836)  
[[code](https://github.com/cszhanglmu/obd-finder)]  
Chongsheng Zhang, Shuwen Wu, Yingqi Chen, Matthias Aßenmacher, Christian Heumann, Yi Men, Gaojuan Fan, João Gama  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Oracle Bone Inscription (OBI) is the earliest systematic writing system in China, while the identification of Oracle Bone (OB) duplicates is a fundamental issue in OBI research. In this work, we design a progressive OB duplicate discovery framework that combines unsupervised low-level keypoints matching with high-level text-centric content-based matching to refine and rank the candidate OB duplicates with semantic awareness and interpretability. We compare our approach with state-of-the-art content-based image retrieval and image matching methods, showing that our approach yields comparable recall performance and the highest simplified mean reciprocal rank scores for both Top-5 and Top-15 retrieval results, and with significantly accelerated computation efficiency. We have discovered over 60 pairs of new OB duplicates in real-world deployment, which were missed by OBI researchers for decades. The models, video illustration and demonstration of this work are available at: https://github.com/cszhangLMU/OBD-Finder/.  
  </ol>  
</details>  
**comments**: This is the long version of our OBD-Finder paper for AI-enabled
  Oracle Bone Duplicates Discovery (currently under review at the ECML PKDD
  2025 Demo Track). The models, video illustration and demonstration of this
  paper are available at: https://github.com/cszhangLMU/OBD-Finder/.
  Illustration video: https://www.youtube.com/watch?v=5QT4f0YIo0Q  
  
  



## Image Matching  

### [OBD-Finder: Explainable Coarse-to-Fine Text-Centric Oracle Bone Duplicates Discovery](http://arxiv.org/abs/2505.03836)  
[[code](https://github.com/cszhanglmu/obd-finder)]  
Chongsheng Zhang, Shuwen Wu, Yingqi Chen, Matthias Aßenmacher, Christian Heumann, Yi Men, Gaojuan Fan, João Gama  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Oracle Bone Inscription (OBI) is the earliest systematic writing system in China, while the identification of Oracle Bone (OB) duplicates is a fundamental issue in OBI research. In this work, we design a progressive OB duplicate discovery framework that combines unsupervised low-level keypoints matching with high-level text-centric content-based matching to refine and rank the candidate OB duplicates with semantic awareness and interpretability. We compare our approach with state-of-the-art content-based image retrieval and image matching methods, showing that our approach yields comparable recall performance and the highest simplified mean reciprocal rank scores for both Top-5 and Top-15 retrieval results, and with significantly accelerated computation efficiency. We have discovered over 60 pairs of new OB duplicates in real-world deployment, which were missed by OBI researchers for decades. The models, video illustration and demonstration of this work are available at: https://github.com/cszhangLMU/OBD-Finder/.  
  </ol>  
</details>  
**comments**: This is the long version of our OBD-Finder paper for AI-enabled
  Oracle Bone Duplicates Discovery (currently under review at the ECML PKDD
  2025 Demo Track). The models, video illustration and demonstration of this
  paper are available at: https://github.com/cszhangLMU/OBD-Finder/.
  Illustration video: https://www.youtube.com/watch?v=5QT4f0YIo0Q  
  
  



