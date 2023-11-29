@echo off

set GIT_CONFIG_GLOBAL=%cd%\assets\gitconfig-cn
set GIT_TERMINAL_PROMPT=false
set PIP_FIND_LINKS=https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html
set PIP_INDEX_URL=https://mirror.baidu.com/pypi/simple
set PIP_NO_WARN_SCRIPT_LOCATION=0

where git > nul 2>&1
if %errorlevel% equ 0 (
    echo Git已安装
) else (
    if exist ".\git\cmd\git.exe" (
        set "PATH=%PATH%;%CD%\git\cmd"
        echo 已使用目录内的 Git
    ) else (
        echo 无法找到 Git，更新失败。
    )
)

echo 正在更新...
git reset --hard
git pull
echo 正在更新子模块...
git submodule init
git submodule update

python\python.exe -m pip install -r requirements.txt

cd sd-scripts
..\python\python.exe -m pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo 已全部更新成功
) else (
	echo 更新失败
)

pause