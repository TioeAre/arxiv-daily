<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#ORB-SfMLearner:-ORB-Guided-Self-supervised-Visual-Odometry-with-Selective-Online-Adaptation>ORB-SfMLearner: ORB-Guided Self-supervised Visual Odometry with Selective Online Adaptation</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Towards-Global-Localization-using-Multi-Modal-Object-Instance-Re-Identification>Towards Global Localization using Multi-Modal Object-Instance Re-Identification</a></li>
        <li><a href=#Open-Set-Semantic-Uncertainty-Aware-Metric-Semantic-Graph-Matching>Open-Set Semantic Uncertainty Aware Metric-Semantic Graph Matching</a></li>
        <li><a href=#Obfuscation-Based-Privacy-Preserving-Representations-are-Recoverable-Using-Neighborhood-Information>Obfuscation Based Privacy Preserving Representations are Recoverable Using Neighborhood Information</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#JEAN:-Joint-Expression-and-Audio-guided-NeRF-based-Talking-Face-Generation>JEAN: Joint Expression and Audio-guided NeRF-based Talking Face Generation</a></li>
        <li><a href=#BRDF-NeRF:-Neural-Radiance-Fields-with-Optical-Satellite-Images-and-BRDF-Modelling>BRDF-NeRF: Neural Radiance Fields with Optical Satellite Images and BRDF Modelling</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [ORB-SfMLearner: ORB-Guided Self-supervised Visual Odometry with Selective Online Adaptation](http://arxiv.org/abs/2409.11692)  
Yanlin Jin, Rui-Yang Ju, Haojun Liu, Yuzhong Zhong  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Deep visual odometry, despite extensive research, still faces limitations in accuracy and generalizability that prevent its broader application. To address these challenges, we propose an Oriented FAST and Rotated BRIEF (ORB)-guided visual odometry with selective online adaptation named ORB-SfMLearner. We present a novel use of ORB features for learning-based ego-motion estimation, leading to more robust and accurate results. We also introduce the cross-attention mechanism to enhance the explainability of PoseNet and have revealed that driving direction of the vehicle can be explained through attention weights, marking a novel exploration in this area. To improve generalizability, our selective online adaptation allows the network to rapidly and selectively adjust to the optimal parameters across different domains. Experimental results on KITTI and vKITTI datasets show that our method outperforms previous state-of-the-art deep visual odometry methods in terms of ego-motion accuracy and generalizability.  
  </ol>  
</details>  
  
  



## Visual Localization  

### [Towards Global Localization using Multi-Modal Object-Instance Re-Identification](http://arxiv.org/abs/2409.12002)  
Aneesh Chavan, Vaibhav Agrawal, Vineeth Bhat, Sarthak Chittawar, Siddharth Srivastava, Chetan Arora, K Madhava Krishna  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Re-identification (ReID) is a critical challenge in computer vision, predominantly studied in the context of pedestrians and vehicles. However, robust object-instance ReID, which has significant implications for tasks such as autonomous exploration, long-term perception, and scene understanding, remains underexplored. In this work, we address this gap by proposing a novel dual-path object-instance re-identification transformer architecture that integrates multimodal RGB and depth information. By leveraging depth data, we demonstrate improvements in ReID across scenes that are cluttered or have varying illumination conditions. Additionally, we develop a ReID-based localization framework that enables accurate camera localization and pose identification across different viewpoints. We validate our methods using two custom-built RGB-D datasets, as well as multiple sequences from the open-source TUM RGB-D datasets. Our approach demonstrates significant improvements in both object instance ReID (mAP of 75.18) and localization accuracy (success rate of 83% on TUM-RGBD), highlighting the essential role of object ReID in advancing robotic perception. Our models, frameworks, and datasets have been made publicly available.  
  </ol>  
</details>  
**comments**: 8 pages, 5 figures, 3 tables. Submitted to ICRA 2025  
  
### [Open-Set Semantic Uncertainty Aware Metric-Semantic Graph Matching](http://arxiv.org/abs/2409.11555)  
Kurran Singh, John J. Leonard  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Underwater object-level mapping requires incorporating visual foundation models to handle the uncommon and often previously unseen object classes encountered in marine scenarios. In this work, a metric of semantic uncertainty for open-set object detections produced by visual foundation models is calculated and then incorporated into an object-level uncertainty tracking framework. Object-level uncertainties and geometric relationships between objects are used to enable robust object-level loop closure detection for unknown object classes. The above loop closure detection problem is formulated as a graph-matching problem. While graph matching, in general, is NP-Complete, a solver for an equivalent formulation of the proposed graph matching problem as a graph editing problem is tested on multiple challenging underwater scenes. Results for this solver as well as three other solvers demonstrate that the proposed methods are feasible for real-time use in marine environments for the robust, open-set, multi-object, semantic-uncertainty-aware loop closure detection. Further experimental results on the KITTI dataset demonstrate that the method generalizes to large-scale terrestrial scenes.  
  </ol>  
</details>  
  
### [Obfuscation Based Privacy Preserving Representations are Recoverable Using Neighborhood Information](http://arxiv.org/abs/2409.11536)  
Kunal Chelani, Assia Benbihi, Fredrik Kahl, Torsten Sattler, Zuzana Kukelova  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Rapid growth in the popularity of AR/VR/MR applications and cloud-based visual localization systems has given rise to an increased focus on the privacy of user content in the localization process.   This privacy concern has been further escalated by the ability of deep neural networks to recover detailed images of a scene from a sparse set of 3D or 2D points and their descriptors - the so-called inversion attacks.   Research on privacy-preserving localization has therefore focused on preventing these inversion attacks on both the query image keypoints and the 3D points of the scene map.   To this end, several geometry obfuscation techniques that lift points to higher-dimensional spaces, i.e., lines or planes, or that swap coordinates between points % have been proposed.   In this paper, we point to a common weakness of these obfuscations that allows to recover approximations of the original point positions under the assumption of known neighborhoods.   We further show that these neighborhoods can be computed by learning to identify descriptors that co-occur in neighborhoods.   Extensive experiments show that our approach for point recovery is practically applicable to all existing geometric obfuscation schemes.   Our results show that these schemes should not be considered privacy-preserving, even though they are claimed to be privacy-preserving.   Code will be available at \url{https://github.com/kunalchelani/RecoverPointsNeighborhood}.  
  </ol>  
</details>  
  
  



## NeRF  

### [JEAN: Joint Expression and Audio-guided NeRF-based Talking Face Generation](http://arxiv.org/abs/2409.12156)  
Sai Tanmay Reddy Chakkera, Aggelina Chatziagapi, Dimitris Samaras  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce a novel method for joint expression and audio-guided talking face generation. Recent approaches either struggle to preserve the speaker identity or fail to produce faithful facial expressions. To address these challenges, we propose a NeRF-based network. Since we train our network on monocular videos without any ground truth, it is essential to learn disentangled representations for audio and expression. We first learn audio features in a self-supervised manner, given utterances from multiple subjects. By incorporating a contrastive learning technique, we ensure that the learned audio features are aligned to the lip motion and disentangled from the muscle motion of the rest of the face. We then devise a transformer-based architecture that learns expression features, capturing long-range facial expressions and disentangling them from the speech-specific mouth movements. Through quantitative and qualitative evaluation, we demonstrate that our method can synthesize high-fidelity talking face videos, achieving state-of-the-art facial expression transfer along with lip synchronization to unseen audio.  
  </ol>  
</details>  
**comments**: Accepted by BMVC 2024. Project Page:
  https://starc52.github.io/publications/2024-07-19-JEAN  
  
### [BRDF-NeRF: Neural Radiance Fields with Optical Satellite Images and BRDF Modelling](http://arxiv.org/abs/2409.12014)  
Lulin Zhang, Ewelina Rupnik, Tri Dung Nguyen, Stéphane Jacquemoud, Yann Klinger  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Understanding the anisotropic reflectance of complex Earth surfaces from satellite imagery is crucial for numerous applications. Neural radiance fields (NeRF) have become popular as a machine learning technique capable of deducing the bidirectional reflectance distribution function (BRDF) of a scene from multiple images. However, prior research has largely concentrated on applying NeRF to close-range imagery, estimating basic Microfacet BRDF models, which fall short for many Earth surfaces. Moreover, high-quality NeRFs generally require several images captured simultaneously, a rare occurrence in satellite imaging. To address these limitations, we propose BRDF-NeRF, developed to explicitly estimate the Rahman-Pinty-Verstraete (RPV) model, a semi-empirical BRDF model commonly employed in remote sensing. We assess our approach using two datasets: (1) Djibouti, captured in a single epoch at varying viewing angles with a fixed Sun position, and (2) Lanzhou, captured over multiple epochs with different viewing angles and Sun positions. Our results, based on only three to four satellite images for training, demonstrate that BRDF-NeRF can effectively synthesize novel views from directions far removed from the training data and produce high-quality digital surface models (DSMs).  
  </ol>  
</details>  
  
  



