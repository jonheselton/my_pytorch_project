# my_pytorch_project

A project to learn about pytorch and building in containers.

Built in container `rocm/pytorch:latest` currently this is image ID `214ceb2d47a7`
[rocm/pytorch](https://hub.docker.com/r/rocm/pytorch/#!) on Docker Hub

## Docker Run command

`docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G -v $HOME/dockerx:/dockerx -w /dockerx rocm/pytorch:rocm6.0.2_ubuntu22.04_py3.10_pytorch_2.1.2 `

## GPU

Radeon 7900xtx ROCm 6.0.2 on Ubuntu 22.04

## Tutorial Used

The files in the tutorial directory are my work while working through the pytorch repositor [found here].(https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
