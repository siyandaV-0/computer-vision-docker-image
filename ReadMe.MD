<h1 style="Text-Align:center; background-color:DodgerBlue;">
Computer Vision Docker Image:
</h1>

---
<h4 style="Text-Align:left;color:black; background-color:Gold;">
    <ul>
    <li>This image builds a computer-vision environment in jupyter-lab <br>
    <li>The image builds opencv from source code to have cuda support. <br>
    <li>You can then run the cv algorithms using the GPU for faster processing.<br>
    <li>This will serve you well if you can deploy the container to a remote server.<br>
    </ul>
</h4>
<br>

<h2 style="Text-Align:left; background-color:MediumSeaGreen;">
Working Directory:
</h2>

<h4 style="Text-Align:left;color:black; background-color:Gold;">
Our working Directory is structured as follows:
</h4>

```
.
└── computer-vision-docker-image
    ├── build-runs
    │   ├── docker-build.sh
    │   └── docker-run.sh
    ├── dependecies
    │   ├── cv.requirements.txt
    │   ├── ubuntu-deps.sh
    │   └── opencv.sh
    ├── apps
    │   ├── example_notebook.ipynb
    │   ├── Lenna.png
    │   └── haarcascade_frontalface_default.xml    
    ├── Dockerfile
    ├── docker-compose.yml
    └── ReadME.md
```
<br>

<h2 style="Text-Align:left; background-color:MediumSeaGreen;">
To build the image run the docker-build bash script
</h2>

```shell
 bash build-runs/docker-build.sh
```
<h4 style="Text-Align:left;color:black; background-color:Gold;">
If you need to make updates to the content of the container, e.g. 
installing new packages with pip- you can specify these in the cv-requirements.txt file in the dependencies folder.
</h4>

<br> 

 ---

<h2 style="Text-Align:left; background-color:MediumSeaGreen;">
To launch the image run the docker-run bash script 
</h2>

```shell
 bash build-runs/docker-run.sh
```

<h4 style="Text-Align:left;color:black; background-color:Gold;">
The docker run bash scripts allow for ease of launching into the container's jupyter-lab. <br> 
This docker run script uses docker-compose commands to achieve this.<br> 
The docker-compose.yml file contains a set of commands that specify using the host's GUI via Xorg, <br>
which we require to have a display interface inside the container. 

</h4>
