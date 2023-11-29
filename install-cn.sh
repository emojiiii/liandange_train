#!/bin/bash

HF_HOME="huggingface"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1

python --version

if [ ! -d "venv" ]; then
    echo "正在创建虚拟环境..."
    python -m venv venv
    echo "创建虚拟环境失败，请检查是否安装了 python 并且 python 版本为64位版本，推荐使用 python 3.10，确保 python 目录已添加到 PATH 中"
fi

source ./venv/bin/activate


echo "安装依赖库 (如果在中国大陆网络环境下，可能无法使用源码安装方式，请使用 install.ps1 脚本)"
read -p "是否需要安装 Torch+xformers? 输入 y 代表是选择安装，输入 n 代表不安装 [y/n] (默认为 y): " install_torch
if [ "$install_torch" == "y" ] || [ "$install_torch" == "Y" ] || [ -z "$install_torch" ]; then
    pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://mirror.baidu.com/pypi/simple
    echo "torch 安装失败，请删除 venv 目录并重新运行脚本"

    pip install -U -I --no-deps xformers==0.0.19 -i https://mirror.baidu.com/pypi/simple
    echo "xformers 安装失败"
fi

pip install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple

pip install --upgrade lion-pytorch dadaptation -i https://mirror.baidu.com/pypi/simple

pip install --upgrade --pre lycoris-lora -i https://pypi.org/simple

pip install --upgrade fastapi uvicorn scipy -i https://mirror.baidu.com/pypi/simple

pip install --upgrade wandb -i https://mirror.baidu.com/pypi/simple

pip install --upgrade --no-deps pytorch-optimizer -i https://mirror.baidu.com/pypi/simple

pip install --upgrade prodigyopt -i https://pypi.org/simple

echo "安装 bitsandbytes..."
pip install bitsandbytes==0.41.1 --index-url https://jihulab.com/api/v4/projects/140618/packages/pypi/simple
# cp bitsandbytes_windows/*.dll ../venv/lib/python3.10/site-packages/bitsandbytes/
# cp bitsandbytes_windows/main.py ../venv/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py

echo "安装完成"
read -p "按任意键继续..." -n 1 -r
