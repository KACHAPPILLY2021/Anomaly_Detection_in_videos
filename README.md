<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">


  <h1 align="center">Weakly Supervised Variational Auto-Encoder for Anomaly Detection </h1>


</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary><h3>Table of Contents</h3></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#visualization">Visualization</a></li>
      </ul>
      <ul>
        <li><a href="#dataset-info">Dataset Info</a></li>
      </ul>
    </li>
    <li>
      <a href="#methodology">Methodology</a>
      <ul>
        <li><a href="#baseline-approach">Baseline Approach</a></li>
      </ul>
      <ul>
        <li><a href="#multi-task-vae">Multi task VAE</a></li>
      </ul>
    </li>
    <li>
      <a href="#results">Results</a>
      <ul>
        <li><a href="#report-and-presentation">Report and Presentation</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#data-preparation">Data Preparation</a></li>
        <li><a href="#usage">Usage</a></li>
      </ul>
    </li>
    <li><a href="#contributors">Contributors</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project



The goal for Video anomaly detection (VAD) is to identify abnormal activities in a video sequence, more specifically, to make frame-level predictions indicating whether the frame is normal or not.  In this project, the focus was on the ```weakly-supervised``` setting, where only video-level annotations are provided. A ```Multi-Task Variational Auto-Encoder``` was designed to generate pseudo normal and abnormal video features which can be built on top of any existing frameworks.

Summary of tasks achieved:
* Proposed a generative approach to **generate pseudo video representations** using a Multi-Task Variational Auto-Encoder.
* Leveraged the shared encoder architecture to generate pseudo video features that came from different distributions while preserving the **temporal consistency** in the original videos.
* Showed that generated pseudo video features can **improve the performance** of a model, even if it's a simple network with only MLPs.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Visualization

The anomaly in this example is caused by a rickshaw passing through a pedestrian-only zone. The ground truth anomaly scores and the model's high predicted abnormal scores overlap significantly, indicating strong performance.


<div align="center">
  <h4 align="center"> Ground Truth (GT) labelled frames</h4>
</div>
<p align="center">
<img src="https://github.com/KACHAPPILLY2021/Anomaly_Detection_in_videos/blob/main/images/supp_1.PNG?raw=true" width=80% alt="frames">


<div align="center">
  <h4 align="center"> Ground truth vs Predicted Anomaly score (GT score either 0 or 1)</h4>
</div>
<p align="center">
<img src="https://github.com/KACHAPPILLY2021/Anomaly_Detection_in_videos/blob/main/images/supp_plot.PNG?raw=true" width=65% alt="sup_plot">


<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Dataset Info

* [ShanghaiTech](https://svip-lab.github.io/dataset/campus_dataset.html) : Medium-scale dataset containing 13 scenes with complex light conditions and camera angles.  The anomalies in this
dataset are caused by sudden motion, such as chasing and brawling.
* [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) : Large-scale dataset with over 1900 untrimmed videos. The background in UCF-Crime is not
static, unlike what is found in Shanghai dataset. The anomalies comprise 13 different classes like Abuse, Arrest,
Arson, Shooting, Stealing, Shoplifting.. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Methodology -->
## Methodology

* The input video is divided into snippet-level video sequences, each containing 16 consecutive frames. 
* The snippet-level features are then extracted from the video data using a pre-trained Inflated 3D Convolution (I3D) architecture, following the methodology commonly adopted in recent literature.


### Baseline Approach

1. Consists of 4 fully connected layers. 
2. Predict the abnormal score given each snippet feature.
3. For each video, obtain the top-3 snippet scores and average them. 
4. Supervised using the cross-entropy loss between the score obtained above and the video label. (0 for **normal** and 1 for **abnormal**).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Multi task VAE

*  First, a snippet feature is classified as normal or abnormal. (For normal videos we can ignore
this step). Then, reconstruct normal features using normal
decoder and abnormal features using abnormal decoder.
*  The objective is to minimize the **reconstruction loss**, and the **Kulback-Leibler divergence** between the predicted distribution and the standard deviation.


<div align="center">
  <h4 align="center"> Architecture of Multi-task VAE with shared encoder</h4>
</div>
<p align="center">
<img src="https://github.com/KACHAPPILLY2021/Anomaly_Detection_in_videos/blob/main/images/train.PNG?raw=true" width=60% alt="encode">

* To generate pseudo abnormal videos, T/2 snippet features are sampled
from a normal video and fed to the Multi-task VAE with the abnormal decoder to generate abnormal pseudo features.
*  To generate pseudo normal videos, the entire abnormal video features are fed to the Multi-task VAE with
the normal decoder to generate normal pseudo features.

Generate Pseudo Abnormal | Generate Pseudo Normal 
--- | --- 
<img src="https://github.com/KACHAPPILLY2021/Anomaly_Detection_in_videos/blob/main/images/normal.PNG?raw=true" width=100% alt="normal"> | <img src="https://github.com/KACHAPPILLY2021/Anomaly_Detection_in_videos/blob/main/images/abnormal.PNG?raw=true" width=100% alt="abnormal">

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Results and report -->
## Results

Our proposed method can be built on top of any existing frameworks. To show this, we tried to reproduce
the Robust Temporal Feature Magnitude (RTFM) framework by using their publicly available
[source code](https://github.com/tianyu0207/RTFM).

<div align="center">
  <h4 align="center"> Results on ShanghaiTech dataset</h4>
</div>

<div align="center">

Method | Feature | AUC (%)
--- | :---: | ---: | 
Baseline | I3D | 92.62 
Baseline + Multi-Task VAE | I3D | 94.21 (+1.59) 
RTFM | I3D | 95.86 
RTFM + Multi-Task VAE | I3D | 96.85 (+0.99) 

</div>

<div align="center">
  <h4 align="center"> ROC curves of baseline and RTFM, and its variants</h4>
</div>

<div align="center">
<img src="https://github.com/KACHAPPILLY2021/Anomaly_Detection_in_videos/blob/main/images/proper_roc.PNG?raw=true" width=50% alt="roc">
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Report and Presentation

* Detailed report for this project and additional results can be found [here](https://github.com/KACHAPPILLY2021/Anomaly_Detection_in_videos/blob/main/Anomaly_detection_report.pdf).
* Supplementary material can be accessed [here](https://github.com/KACHAPPILLY2021/Anomaly_Detection_in_videos/blob/main/Supplementary_Material.pdf).
* To check out presentation video. [![Youtube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtu.be/SjDJAVIiurs)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

These are the instructions to get started on the project.
To get a local copy up and running follow these simple steps.


### Data preparation
Please follow the data preparation guide in [link](https://github.com/louisYen/S3R/)

### Usage

1. Train baseline model

    ```sh
    python main.py
    ```
2. Generate pseudo snippet scores using the trained baseline model
    ```sh
    python test.py
    ```
3. Train the VAE using the pseudo snippet predictions
    ```sh
    python train_share_vae.py
    ```
4. Generate pseudo features using VAE
    ```sh
    python generate_pseudo.py
    ```
5. Train baseline/RTFM with the augmentation (pseudo features)
    ```sh
    python main.py
    ```

    Dont forget to specify the augmentation path in the file.
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTORS -->
## Contributors

- [Jeffin Johny](https://github.com/KACHAPPILLY2021)
- [Pin-Hao Huang](https://github.com/haogerhuang)
- [Po-Lun Chen]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Jeffin Johny K - [![MAIL](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:jeffinjk@umd.edu)
	
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://kachappilly2021.github.io/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](http://www.linkedin.com/in/jeffin-johny-kachappilly-0a8597136)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See [MIT](https://choosealicense.com/licenses/mit/) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com
