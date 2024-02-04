<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center"><img src="media/TUM.png" width="60">  Weakly Supervised Neural Scene Labeling</h1>
  <p align="center">
    <a href="https://github.com/doginono"><strong>Dogu Tamgac</strong></a>
    ·
    <a href="https://github.com/JuliusKoerner"><strong>Julius Koerner</strong></a>
  </p>
  <!-- 
<h3 align="center"><a href="https://arxiv.org/abs/2112.12130">Paper</a> | <a href="https://youtu.be/V5hYTz5os0M">Video</a> | <a href="https://pengsongyou.github.io/nice-slam">Project Page</a></h3>TABLE OF CONTENTS -->
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="./media/room0.gif" alt="Room0" width="80%">
  </a>
</p>
<p align="center">
Producing accurate dense geometric, color and instance segmentation on static indoor scenes.
</p>
<p align="center">
(Example output of Room 0 of the Replica Dataset)
</p>
<br>



<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#Precomputed WSNSL Results on Replica">Meshes</a>
    </li>
    <li>
      <a href="#Demo">Demo</a>
    </li>
    <li>
      <a href="#Replica">Replica</a>
    </li>
    <li>
      <a href="#Credits">Credits</a>
    </li>
  </ol>
</details>


## Installation

First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `wsnsl`. For linux, you need to install **libopenexr-dev** before creating the environment.
```bash
sudo apt-get install libopenexr-dev
    
conda env create -f environment.yaml
conda activate wsnsl
```

## Precomputed WSNSL Results on Replica
Some generated meshes from the Replica Dataset are provided in the repository ready to be examined. 
All of them can be found in the meshes folder.

```bash
#after cd to this repository
meshlab meshes/room0_final_mesh_seg.ply
meshlab meshes/room0_final_mesh_color.ply
```
If you are running on a remote sever, you can either set the Display variable or download it to your local machine before running the above command.
You can copy the directory for examplewith scp:
```bash
scp -r user@your.server.example.com:/path/to/repository/meshes/ /path/to/destination/
```

## Demo

You can run WSNSL on a short Replica Video of 60 frames yourself without having to download additional data-
 
```bash
python -W ignore run.py configs/Own/room1_small.yaml
```
The runtime visualizaions and meshes are stored into the output/Own/room1_small folder.

### Replica
You can download the Replica Dataset as below and then run WSNSL on it. 
Running it on a full video takes around 5-6 hours. 

```bash
bash scripts/download_replica.sh
```
Here is an example how to run it on Room 0. But you can also run it on all configs included in the configs/Own directory.
```bash
python -W ignore run.py configs/Own/room0.yaml
```
The runtime visualizations and the meshes are stored into the folder output/Own/room0

## Credits
We included code from the <a href="https://github.com/cvg/nice-slam">NICE-SLAM</a> repository, thank you for making it publically available.

