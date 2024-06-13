# syntax=docker/dockerfile:1
FROM archlinux:latest
RUN pacman -Syu --noconfirm rocm-hip-sdk
RUN pacman -Syu --noconfirm rocm-opencl-sdk
RUN pacman -Syu --noconfirm python-pytorch-rocm
RUN pacman -Syu --noconfirm python-torchvision

