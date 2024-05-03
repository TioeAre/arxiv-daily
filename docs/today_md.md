<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#NeRF-in-Robotics:-A-Survey>NeRF in Robotics: A Survey</a></li>
        <li><a href=#DiL-NeRF:-Delving-into-Lidar-for-Neural-Radiance-Field-on-Street-Scenes>DiL-NeRF: Delving into Lidar for Neural Radiance Field on Street Scenes</a></li>
      </ul>
    </li>
  </ol>
</details>

## NeRF  

### [NeRF in Robotics: A Survey](http://arxiv.org/abs/2405.01333)  
Guangming Wang, Lei Pan, Songyou Peng, Shaohui Liu, Chenfeng Xu, Yanzi Miao, Wei Zhan, Masayoshi Tomizuka, Marc Pollefeys, Hesheng Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Meticulous 3D environment representations have been a longstanding goal in computer vision and robotics fields. The recent emergence of neural implicit representations has introduced radical innovation to this field as implicit representations enable numerous capabilities. Among these, the Neural Radiance Field (NeRF) has sparked a trend because of the huge representational advantages, such as simplified mathematical models, compact environment storage, and continuous scene representations. Apart from computer vision, NeRF has also shown tremendous potential in the field of robotics. Thus, we create this survey to provide a comprehensive understanding of NeRF in the field of robotics. By exploring the advantages and limitations of NeRF, as well as its current applications and future potential, we hope to shed light on this promising area of research. Our survey is divided into two main sections: \textit{The Application of NeRF in Robotics} and \textit{The Advance of NeRF in Robotics}, from the perspective of how NeRF enters the field of robotics. In the first section, we introduce and analyze some works that have been or could be used in the field of robotics from the perception and interaction perspectives. In the second section, we show some works related to improving NeRF's own properties, which are essential for deploying NeRF in the field of robotics. In the discussion section of the review, we summarize the existing challenges and provide some valuable future research directions for reference.  
  </ol>  
</details>  
**comments**: 21 pages, 19 figures  
  
### [DiL-NeRF: Delving into Lidar for Neural Radiance Field on Street Scenes](http://arxiv.org/abs/2405.00900)  
Shanlin Sun, Bingbing Zhuang, Ziyu Jiang, Buyu Liu, Xiaohui Xie, Manmohan Chandraker  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Photorealistic simulation plays a crucial role in applications such as autonomous driving, where advances in neural radiance fields (NeRFs) may allow better scalability through the automatic creation of digital 3D assets. However, reconstruction quality suffers on street scenes due to largely collinear camera motions and sparser samplings at higher speeds. On the other hand, the application often demands rendering from camera views that deviate from the inputs to accurately simulate behaviors like lane changes. In this paper, we propose several insights that allow a better utilization of Lidar data to improve NeRF quality on street scenes. First, our framework learns a geometric scene representation from Lidar, which is fused with the implicit grid-based representation for radiance decoding, thereby supplying stronger geometric information offered by explicit point cloud. Second, we put forth a robust occlusion-aware depth supervision scheme, which allows utilizing densified Lidar points by accumulation. Third, we generate augmented training views from Lidar points for further improvement. Our insights translate to largely improved novel view synthesis under real driving scenes.  
  </ol>  
</details>  
**comments**: CVPR2024 Highlights  
  
  



