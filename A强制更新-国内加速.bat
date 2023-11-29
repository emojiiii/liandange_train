@echo off

set GIT_CONFIG_GLOBAL=%cd%\assets\gitconfig-cn
set GIT_TERMINAL_PROMPT=false
set PIP_FIND_LINKS=https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html
set PIP_INDEX_URL=https://mirror.baidu.com/pypi/simple
set PIP_NO_WARN_SCRIPT_LOCATION=0

where git > nul 2>&1
if %errorlevel% equ 0 (
    echo Git�Ѱ�װ
) else (
    if exist ".\git\cmd\git.exe" (
        set "PATH=%PATH%;%CD%\git\cmd"
        echo ��ʹ��Ŀ¼�ڵ� Git
    ) else (
        echo �޷��ҵ� Git������ʧ�ܡ�
    )
)

echo ���ڸ���...
git reset --hard
git pull
echo ���ڸ�����ģ��...
git submodule init
git submodule update

python\python.exe -m pip install -r requirements.txt

cd sd-scripts
..\python\python.exe -m pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo ��ȫ�����³ɹ�
) else (
	echo ����ʧ��
)

pause