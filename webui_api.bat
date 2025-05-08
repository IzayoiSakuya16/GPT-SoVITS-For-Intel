@echo off
:: 设置代码页为 UTF-8，支持中文显示
chcp 65001
echo.
:: 设置从weight.json文件中查找的版本,源码默认为v2，一般无需设置
:: set version=v3
:: 设置 Python 路径为相对路径
set PYTHON_PATH=runtime\python.exe

echo Current Python path（当前Python路径）: %PYTHON_PATH%
echo.
echo Current Python version（当前Python版本）:
"%PYTHON_PATH%" --version
echo.


echo Checking if pretrained models exist（检查预训练模型是否存在）：
"%PYTHON_PATH%" -c "import os; print('pretrained_models exists:', os.path.exists('GPT_SoVITS/pretrained_models'))"
echo.

echo Checking if weight.json file exists（检查当前目录下的weight.json文件是否存在，不存在会创建默认weight.json文件,默认使用v1版本）：
"%PYTHON_PATH%" -c "import os; print('weight.json exists:', os.path.exists('weight.json'))"
echo.

echo Reading the models in weight.json（读取当前目录下的weight.json文件中的模型）：
"%PYTHON_PATH%" -c "import json,os; print('SoVITS:'+str(data['SoVITS'])+'\nGPT:'+str(data['GPT'])) if os.path.exists('weight.json') and (data:=json.load(open('weight.json','r',encoding='utf-8'))) else print('weight.json not found（未找到weight.json文件，创建默认文件）')"
echo.

echo Starting API server（启动API服务器）
"%PYTHON_PATH%" webui_api.py
pause