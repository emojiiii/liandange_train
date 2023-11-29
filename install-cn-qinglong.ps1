$Env:HF_HOME = "huggingface"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
function InstallFail {
    Write-Output "安装失败。"
    Read-Host | Out-Null ;
    Exit
}

function Check {
    param (
        $ErrorInfo
    )
    if (!($?)) {
        Write-Output $ErrorInfo
        InstallFail
    }
}

./python310/python.exe --version

if (!(Test-Path -Path "venv")) {
    Write-Output "正在创建虚拟环境..."
     ./python310/python.exe -m venv venv
    Check "创建虚拟环境失败，请检查 python 是否安装完毕以及 python 版本是否为64位版本的python 3.10、或python的目录是否在环境变量PATH内。"
}

.\venv\Scripts\activate
Check "激活虚拟环境失败。"

Set-Location .\sd-scripts
Write-Output "安装程序所需依赖 (已进行国内加速，若在国外或无法使用加速源请换用 install.ps1 脚本)"
$install_torch = Read-Host "是否需要安装 Torch+xformers? 若您本次为首次安装请选择 y ，若本次为升级依赖安装则选择 n。[y/n] (默认为 y)"
if ($install_torch -eq "y" -or $install_torch -eq "Y" -or $install_torch -eq ""){
    pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://mirror.baidu.com/pypi/simple
    Check "torch 安装失败，请删除 venv 文件夹后重新运行。"
#    pip install --pre -U xformers -i https://mirror.baidu.com/pypi/simple
	pip install -U -I --no-deps xformers==0.0.19 -i https://mirror.baidu.com/pypi/simple
    Check "xformers 安装失败。"
}

pip install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple
Check "其他依赖安装失败。"
pip install --upgrade lion-pytorch dadaptation -i https://mirror.baidu.com/pypi/simple
Check "Lion、dadaptation 优化器安装失败。"
pip install --upgrade --pre lycoris-lora -i https://pypi.org/simple
Check "lycoris 安装失败。"
pip install --upgrade fastapi uvicorn scipy -i https://mirror.baidu.com/pypi/simple
Check "UI 所需依赖安装失败。"
pip install --upgrade wandb -i https://mirror.baidu.com/pypi/simple
Check "wandb 安装失败。"
pip install --upgrade --no-deps pytorch-optimizer -i https://mirror.baidu.com/pypi/simple
Check "pytorch-optimizer 安装失败。"
pip install --upgrade prodigyopt -i https://pypi.org/simple
Check "pytorch-optimizer 安装失败,此优化器暂无加速代理，请检查网络。"

Write-Output "安装 bitsandbytes..."
pip install bitsandbytes==0.41.1 --index-url https://jihulab.com/api/v4/projects/140618/packages/pypi/simple
#cp .\bitsandbytes_windows\*.dll ..\venv\Lib\site-packages\bitsandbytes\
#cp .\bitsandbytes_windows\main.py ..\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py

Write-Output "安装完毕"
Read-Host | Out-Null ;
