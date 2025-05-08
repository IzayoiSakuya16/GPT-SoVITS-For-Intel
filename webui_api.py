from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from urllib.parse import urlparse
import numpy as np
import io
import scipy.io.wavfile as wav
import traceback
import logging
import os
import requests
import json


# 定义端口
PORT = 9876
# 定义主机
HOST = "0.0.0.0"
# 定义请求体模型
class TTSRequest(BaseModel):
    ref_wav_path: str
    prompt_text: str
    prompt_language: str = "中文"
    text: str
    text_language: str = "中文"
    how_to_cut: str = "不切"
    top_k: int = 15
    top_p: float = 0.6
    temperature: float = 0.6
    ref_free: bool = False
    speed: float = 1.0
    if_freeze: bool = False
    sample_steps: int = 64
    if_sr: bool = True
    pause_second: float = 0.3

# 添加新的请求体模型
class ChangeSovitsRequest(BaseModel):
    sovits_path: str
    prompt_language: str = "zh"
    text_language: str = "zh"

class ChangeGptRequest(BaseModel):
    gpt_path: str

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # 只显示消息内容，不显示其他信息
)
logger = logging.getLogger("uvicorn")



# 导入源码并执行初始化
import inference_webui_fast
from inference_webui_fast import inference
# 创建FastAPI应用
app = FastAPI()

# 添加CORS中间件，设置跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加全局异常处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error("请求参数验证错误:")
    logger.error(str(exc))
    return JSONResponse(
        status_code=422,
        content={"detail": f"参数验证错误: {str(exc)}"}
    )


@app.post("/tts")
async def tts_api(request: TTSRequest):
    try:
        # 检查是否是网络链接
        if request.ref_wav_path.startswith(('http://', 'https://')):
            # 下载音频文件
            response = requests.get(request.ref_wav_path)
            if response.status_code != 200:
                raise ValueError(f"无法下载音频文件: {request.ref_wav_path}")

            audio_data = response.content
            # 保存音频文件
            temp_audio_path = "temp_audio.wav"
            with open(temp_audio_path, "wb") as f:
                f.write(audio_data)
            request.ref_wav_path = temp_audio_path
       

        # 调用 inference 函数
        sr, audio_opt = next(inference(
            text=request.text,
            text_lang=request.text_language,
            ref_audio_path=request.ref_wav_path,
            aux_ref_audio_paths=[],  # 添加辅助参考音频路径参数
            prompt_text=request.prompt_text,
            prompt_lang=request.prompt_language,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            text_split_method=request.how_to_cut,
            batch_size=1,  # 添加默认值
            speed_factor=request.speed,
            ref_text_free=request.ref_free,
            split_bucket=False,  # 添加默认值
            fragment_interval=request.pause_second,
            seed=-1,  # 添加默认值
            keep_random=True,  # 添加默认值
            parallel_infer=True,  # 添加默认值
            repetition_penalty=1.35,  # 添加默认值
            sample_steps=request.sample_steps,
            super_sampling=request.if_sr
        ))
        
        # 添加日志记录
        logger.info(f"采样率: {sr}")
        logger.info(f"音频数据原始类型: {type(audio_opt)}")
        
        # 更新音频数据处理逻辑
        if isinstance(sr, tuple):
            sample_rate, audio_data = sr
            audio_opt = audio_data  # 使用元组中的音频数据
        else:
            sample_rate = sr
        
        # 将音频数据转换为WAV格式
        audio_bytes = io.BytesIO()
        
        # 检查并转换音频数据
        if isinstance(audio_opt, np.ndarray):
            # 已经是numpy数组，确保类型正确
            if audio_opt.dtype != np.int16:
                audio_opt = audio_opt.astype(np.int16)
        else:
            # 如果不是numpy数组，先转换为float32以避免溢出
            audio_opt = np.array(audio_opt, dtype=np.float32)
            # 归一化到[-32768, 32767]范围
            max_val = np.max(np.abs(audio_opt))
            if max_val > 0:
                audio_opt = (audio_opt / max_val * 32767).astype(np.int16)
            else:
                audio_opt = audio_opt.astype(np.int16)
        
        logger.info(f"处理后音频数据类型: {audio_opt.dtype}")
        logger.info(f"音频数据范围: [{np.min(audio_opt)}, {np.max(audio_opt)}]")
        
        # 写入WAV文件
        wav.write(audio_bytes, sample_rate, audio_opt)
        audio_bytes.seek(0)

        # 如果使用了临时文件，删除它
        if request.ref_wav_path == "temp_ref.wav":
            try:
                os.remove(request.ref_wav_path)
            except:
                pass
        
        return StreamingResponse(audio_bytes, media_type="audio/wav")
    except Exception as e:
        logger.error(f"错误: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=422, detail=str(e)) 

@app.post("/change_sovits_weights")
async def change_sovits_weights_api(request: ChangeSovitsRequest):
    try:
        # 记录切换前的配置
        logger.info("切换前的配置:")
        check_current_weights()
        
        # 执行切换
        result = inference_webui_fast.change_sovits_weights(
            sovits_path=request.sovits_path,
            prompt_language=request.prompt_language,
            text_language=request.text_language
        )
        
        # 记录切换后的配置
        logger.info("切换后的配置:")
        check_current_weights()
            
        return result
    except Exception as e:
        logger.error(f"错误: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/change_gpt_weights")
async def change_gpt_weights(data: dict):
    try:
        gpt_path = data["gpt_path"]
        
        # 记录切换前的配置
        logger.info("切换前的配置:")
        check_current_weights()
        
        # 执行切换
        inference_webui_fast.change_gpt_weights(gpt_path)
        
        # 记录切换后的配置
        logger.info("切换后的配置:")
        check_current_weights()
            
        return {"status": "success", "message": "GPT 模型切换成功"}
    except Exception as e:
        logger.error(f"错误: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/change_choices")
async def change_choices_api():
    try:
        # 记录当前配置
        logger.info("检查当前使用的json模型:")
        check_current_weights()
        
        # 调用 inference_webui 中的 change_choices 函数
        result = inference_webui_fast.change_choices()
        
        # 从结果中提取 choices
        sovits_choices = result[0]["choices"]
        gpt_choices = result[1]["choices"]
        
        # 打印可用的模型列表
        logger.info("可用的模型列表:")
        logger.info(f"SoVITS 模型: {sovits_choices}")
        logger.info(f"GPT 模型: {gpt_choices}")
        
        return sovits_choices, gpt_choices
        
    except Exception as e:
        logger.error(f"错误: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=422, detail=str(e))

def check_current_weights():
    """检查当前使用的模型"""
    with open("./weight.json", 'r', encoding="utf-8") as f:
        data = json.load(f)
        # 检查所有版本
        sovits_models = data["SoVITS"]
        gpt_models = data["GPT"]
        logger.info(f"SoVITS 模型: {sovits_models}")
        logger.info(f"GPT 模型: {gpt_models}")
        
        return data

# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)