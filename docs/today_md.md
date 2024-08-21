<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#TrackNeRF:-Bundle-Adjusting-NeRF-from-Sparse-and-Noisy-Views-via-Feature-Tracks>TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#BrewCLIP:-A-Bifurcated-Representation-Learning-Framework-for-Audio-Visual-Retrieval>BrewCLIP: A Bifurcated Representation Learning Framework for Audio-Visual Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#BrewCLIP:-A-Bifurcated-Representation-Learning-Framework-for-Audio-Visual-Retrieval>BrewCLIP: A Bifurcated Representation Learning Framework for Audio-Visual Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Learning-Part-aware-3D-Representations-by-Fusing-2D-Gaussians-and-Superquadrics>Learning Part-aware 3D Representations by Fusing 2D Gaussians and Superquadrics</a></li>
        <li><a href=#TrackNeRF:-Bundle-Adjusting-NeRF-from-Sparse-and-Noisy-Views-via-Feature-Tracks>TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks](http://arxiv.org/abs/2408.10739)  
Jinjie Mai, Wenxuan Zhu, Sara Rojas, Jesus Zarzar, Abdullah Hamdi, Guocheng Qian, Bing Li, Silvio Giancola, Bernard Ghanem  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural radiance fields (NeRFs) generally require many images with accurate poses for accurate novel view synthesis, which does not reflect realistic setups where views can be sparse and poses can be noisy. Previous solutions for learning NeRFs with sparse views and noisy poses only consider local geometry consistency with pairs of views. Closely following \textit{bundle adjustment} in Structure-from-Motion (SfM), we introduce TrackNeRF for more globally consistent geometry reconstruction and more accurate pose optimization. TrackNeRF introduces \textit{feature tracks}, \ie connected pixel trajectories across \textit{all} visible views that correspond to the \textit{same} 3D points. By enforcing reprojection consistency among feature tracks, TrackNeRF encourages holistic 3D consistency explicitly. Through extensive experiments, TrackNeRF sets a new benchmark in noisy and sparse view reconstruction. In particular, TrackNeRF shows significant improvements over the state-of-the-art BARF and SPARF by $\sim8$ and $\sim1$ in terms of PSNR on DTU under various sparse and noisy view setups. The code is available at \href{https://tracknerf.github.io/}.  
  </ol>  
</details>  
**comments**: ECCV 2024 (supplemental pages included)  
  
  



## Visual Localization  

### [BrewCLIP: A Bifurcated Representation Learning Framework for Audio-Visual Retrieval](http://arxiv.org/abs/2408.10383)  
Zhenyu Lu, Lakshay Sethi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Previous methods for audio-image matching generally fall into one of two categories: pipeline models or End-to-End models. Pipeline models first transcribe speech and then encode the resulting text; End-to-End models encode speech directly. Generally, pipeline models outperform end-to-end models, but the intermediate transcription necessarily discards some potentially useful non-textual information. In addition to textual information, speech can convey details such as accent, mood, and and emphasis, which should be effectively captured in the encoded representation. In this paper, we investigate whether non-textual information, which is overlooked by pipeline-based models, can be leveraged to improve speech-image matching performance. We thoroughly analyze and compare End-to-End models, pipeline models, and our proposed dual-channel model for robust audio-image retrieval on a variety of datasets. Our approach achieves a substantial performance gain over the previous state-of-the-art by leveraging strong pretrained models, a prompting mechanism and a bifurcated design.  
  </ol>  
</details>  
  
  



## Image Matching  

### [BrewCLIP: A Bifurcated Representation Learning Framework for Audio-Visual Retrieval](http://arxiv.org/abs/2408.10383)  
Zhenyu Lu, Lakshay Sethi  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Previous methods for audio-image matching generally fall into one of two categories: pipeline models or End-to-End models. Pipeline models first transcribe speech and then encode the resulting text; End-to-End models encode speech directly. Generally, pipeline models outperform end-to-end models, but the intermediate transcription necessarily discards some potentially useful non-textual information. In addition to textual information, speech can convey details such as accent, mood, and and emphasis, which should be effectively captured in the encoded representation. In this paper, we investigate whether non-textual information, which is overlooked by pipeline-based models, can be leveraged to improve speech-image matching performance. We thoroughly analyze and compare End-to-End models, pipeline models, and our proposed dual-channel model for robust audio-image retrieval on a variety of datasets. Our approach achieves a substantial performance gain over the previous state-of-the-art by leveraging strong pretrained models, a prompting mechanism and a bifurcated design.  
  </ol>  
</details>  
  
  



## NeRF  

### [Learning Part-aware 3D Representations by Fusing 2D Gaussians and Superquadrics](http://arxiv.org/abs/2408.10789)  
Zhirui Gao, Renjiao Yi, Yuhang Huang, Wei Chen, Chenyang Zhu, Kai Xu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Low-level 3D representations, such as point clouds, meshes, NeRFs, and 3D Gaussians, are commonly used to represent 3D objects or scenes. However, humans usually perceive 3D objects or scenes at a higher level as a composition of parts or structures rather than points or voxels. Representing 3D as semantic parts can benefit further understanding and applications. We aim to solve part-aware 3D reconstruction, which parses objects or scenes into semantic parts. In this paper, we introduce a hybrid representation of superquadrics and 2D Gaussians, trying to dig 3D structural clues from multi-view image inputs. Accurate structured geometry reconstruction and high-quality rendering are achieved at the same time. We incorporate parametric superquadrics in mesh forms into 2D Gaussians by attaching Gaussian centers to faces in meshes. During the training, superquadrics parameters are iteratively optimized, and Gaussians are deformed accordingly, resulting in an efficient hybrid representation. On the one hand, this hybrid representation inherits the advantage of superquadrics to represent different shape primitives, supporting flexible part decomposition of scenes. On the other hand, 2D Gaussians are incorporated to model the complex texture and geometry details, ensuring high-quality rendering and geometry reconstruction. The reconstruction is fully unsupervised. We conduct extensive experiments on data from DTU and ShapeNet datasets, in which the method decomposes scenes into reasonable parts, outperforming existing state-of-the-art approaches.  
  </ol>  
</details>  
  
### [TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks](http://arxiv.org/abs/2408.10739)  
Jinjie Mai, Wenxuan Zhu, Sara Rojas, Jesus Zarzar, Abdullah Hamdi, Guocheng Qian, Bing Li, Silvio Giancola, Bernard Ghanem  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural radiance fields (NeRFs) generally require many images with accurate poses for accurate novel view synthesis, which does not reflect realistic setups where views can be sparse and poses can be noisy. Previous solutions for learning NeRFs with sparse views and noisy poses only consider local geometry consistency with pairs of views. Closely following \textit{bundle adjustment} in Structure-from-Motion (SfM), we introduce TrackNeRF for more globally consistent geometry reconstruction and more accurate pose optimization. TrackNeRF introduces \textit{feature tracks}, \ie connected pixel trajectories across \textit{all} visible views that correspond to the \textit{same} 3D points. By enforcing reprojection consistency among feature tracks, TrackNeRF encourages holistic 3D consistency explicitly. Through extensive experiments, TrackNeRF sets a new benchmark in noisy and sparse view reconstruction. In particular, TrackNeRF shows significant improvements over the state-of-the-art BARF and SPARF by $\sim8$ and $\sim1$ in terms of PSNR on DTU under various sparse and noisy view setups. The code is available at \href{https://tracknerf.github.io/}.  
  </ol>  
</details>  
**comments**: ECCV 2024 (supplemental pages included)  
  
  



