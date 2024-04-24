<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#slam>SLAM</a></li>
      <ul>
        <li><a href=#Multi-Session-SLAM-with-Differentiable-Wide-Baseline-Pose-Optimization>Multi-Session SLAM with Differentiable Wide-Baseline Pose Optimization</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Adaptive-Local-Binary-Pattern:-A-Novel-Feature-Descriptor-for-Enhanced-Analysis-of-Kidney-Abnormalities-in-CT-Scan-Images-using-ensemble-based-Machine-Learning-Approach>Adaptive Local Binary Pattern: A Novel Feature Descriptor for Enhanced Analysis of Kidney Abnormalities in CT Scan Images using ensemble based Machine Learning Approach</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#FINEMATCH:-Aspect-based-Fine-grained-Image-and-Text-Mismatch-Detection-and-Correction>FINEMATCH: Aspect-based Fine-grained Image and Text Mismatch Detection and Correction</a></li>
      </ul>
    </li>
  </ol>
</details>

## SLAM  

### [Multi-Session SLAM with Differentiable Wide-Baseline Pose Optimization](http://arxiv.org/abs/2404.15263)  
[[code](https://github.com/princeton-vl/multislam_diffpose)]  
Lahav Lipson, Jia Deng  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce a new system for Multi-Session SLAM, which tracks camera motion across multiple disjoint videos under a single global reference. Our approach couples the prediction of optical flow with solver layers to estimate camera pose. The backbone is trained end-to-end using a novel differentiable solver for wide-baseline two-view pose. The full system can connect disjoint sequences, perform visual odometry, and global optimization. Compared to existing approaches, our design is accurate and robust to catastrophic failures. Code is available at github.com/princeton-vl/MultiSlam_DiffPose  
  </ol>  
</details>  
**comments**: Accepted to CVPR 2024  
  
  



## Keypoint Detection  

### [Adaptive Local Binary Pattern: A Novel Feature Descriptor for Enhanced Analysis of Kidney Abnormalities in CT Scan Images using ensemble based Machine Learning Approach](http://arxiv.org/abs/2404.14560)  
Tahmim Hossain, Faisal Sayed, Solehin Islam  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    The shortage of nephrologists and the growing public health concern over renal failure have spurred the demand for AI systems capable of autonomously detecting kidney abnormalities. Renal failure, marked by a gradual decline in kidney function, can result from factors like cysts, stones, and tumors. Chronic kidney disease may go unnoticed initially, leading to untreated cases until they reach an advanced stage. The dataset, comprising 12,427 images from multiple hospitals in Dhaka, was categorized into four groups: cyst, tumor, stone, and normal. Our methodology aims to enhance CT scan image quality using Cropping, Resizing, and CALHE techniques, followed by feature extraction with our proposed Adaptive Local Binary Pattern (A-LBP) feature extraction method compared with the state-of-the-art local binary pattern (LBP) method. Our proposed features fed into classifiers such as Random Forest, Decision Tree, Naive Bayes, K-Nearest Neighbor, and SVM. We explored an ensemble model with soft voting to get a more robust model for our task. We got the highest of more than 99% in accuracy using our feature descriptor and ensembling five classifiers (Random Forest, Decision Tree, Naive Bayes, K-Nearest Neighbor, Support Vector Machine) with the soft voting method.  
  </ol>  
</details>  
**comments**: 17 pages, 5 tables, 4 figures  
  
  



## Image Matching  

### [FINEMATCH: Aspect-based Fine-grained Image and Text Mismatch Detection and Correction](http://arxiv.org/abs/2404.14715)  
Hang Hua, Jing Shi, Kushal Kafle, Simon Jenni, Daoan Zhang, John Collomosse, Scott Cohen, Jiebo Luo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Recent progress in large-scale pre-training has led to the development of advanced vision-language models (VLMs) with remarkable proficiency in comprehending and generating multimodal content. Despite the impressive ability to perform complex reasoning for VLMs, current models often struggle to effectively and precisely capture the compositional information on both the image and text sides. To address this, we propose FineMatch, a new aspect-based fine-grained text and image matching benchmark, focusing on text and image mismatch detection and correction. This benchmark introduces a novel task for boosting and evaluating the VLMs' compositionality for aspect-based fine-grained text and image matching. In this task, models are required to identify mismatched aspect phrases within a caption, determine the aspect's class, and propose corrections for an image-text pair that may contain between 0 and 3 mismatches. To evaluate the models' performance on this new task, we propose a new evaluation metric named ITM-IoU for which our experiments show a high correlation to human evaluation. In addition, we also provide a comprehensive experimental analysis of existing mainstream VLMs, including fully supervised learning and in-context learning settings. We have found that models trained on FineMatch demonstrate enhanced proficiency in detecting fine-grained text and image mismatches. Moreover, models (e.g., GPT-4V, Gemini Pro Vision) with strong abilities to perform multimodal in-context learning are not as skilled at fine-grained compositional image and text matching analysis. With FineMatch, we are able to build a system for text-to-image generation hallucination detection and correction.  
  </ol>  
</details>  
  
  



