
---
<h1 style="text-align:center;"> Computer Vision Image </h1>


- ### This repo consist of a docker image for a computer vision environment.
- ### The enviroment is linux-based - Ubuntu22.04
- ### An example notebook which shows the working environment is provided in the [./apps] folder.
- ### The exposed port to the jupyter-server is the default localhost port of 8888, and thus should always be mapped to this port.
    
### The following Computer Vision packages are installed:
```
- NVIDIA/Cuda 11.7
- Python 3.9
- Miniconda-latest
- OpenCV with CUDA support
- Tensoflow 2.9
- Pytorch+cuDNN 11.8
- Scikit-Learn
- Jupyter-lab Server

```
### Additional packages installed:
```
- Pandas
- Numpy
- Scipy
- Matplotlib
- Seaborn
- Plotly
- yellowbrick

```

### To run the image the following docker command can be used:
- This is an example command where the port mapping to the jupyter-server
  is 8000.

```
$ docker run --rm -it --user $(id -u):$(id -g) --group-add container_user --group-add sudo -v "${PWD}:/app" -p 8000:8888 -p 6006:6006 siyandav0/computer-vision-env

```