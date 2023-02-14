import json
import re
import random
from tencentcloud.common import credential
from tencentcloud.asr.v20190614 import asr_client, models
asr_models = models
from tencentcloud.tts.v20190823 import tts_client, models
tts_models = models

engine_type = "16k_en"

def asr(SecretId, SecretKey, record_data):
    # 实例化一个认证对象，入参需要传入腾讯云账户 SecretId 和 SecretKey，此处还需注意密钥对的保密
    # 代码泄露可能会导致 SecretId 和 SecretKey 泄露，并威胁账号下所有资源的安全性。以下代码示例仅供参考，建议采用更安全的方式来使用密钥，请参见：https://cloud.tencent.com/document/product/1278/85305
    # 密钥可前往官网控制台 https://console.cloud.tencent.com/cam/capi 进行获取
    cred = credential.Credential(SecretId, SecretKey)

    # 实例化要请求产品的client对象,clientProfile是可选的
    client = asr_client.AsrClient(cred, "")

    # 实例化一个请求对象,每个接口都会对应一个request对象
    req = asr_models.CreateRecTaskRequest()
    params = {
      "Action": "CreateRecTask",
      "Version": "2019-06-14",
      "EngineModelType": engine_type,
      "ChannelNum": 1,
      "ResTextFormat": 0,
      "SourceType": 1,
      "Data": record_data,
    }
    req.from_json_string(json.dumps(params))
    resp = json.loads(client.CreateRecTask(req).to_json_string())
    task_id = resp["Data"]["TaskId"]
    
    client = asr_client.AsrClient(cred, "")
    # 实例化一个请求对象,每个接口都会对应一个request对象
    req = asr_models.DescribeTaskStatusRequest()
    params = {
      "Action": "DescribeTaskStatus",
      "Version": "2019-06-14",
      "TaskId": task_id,
    }

    while True:
        req.from_json_string(json.dumps(params))
        resp = json.loads(client.DescribeTaskStatus(req).to_json_string())

        if resp["Data"]["StatusStr"] == "failed":
            raise resp["Data"]["ErrorMsg"]
        elif resp["Data"]["StatusStr"] == "success":
            result = resp["Data"]["Result"]
            return re.sub("\[.*?\]", "", result)

def tts(SecretId, SecretKey, text):
    cred = credential.Credential(SecretId, SecretKey)
    client = tts_client.TtsClient(cred, "ap-beijing")

    # 实例化一个请求对象,每个接口都会对应一个request对象
    req = tts_models.TextToVoiceRequest()
    params = {
      "Action": "TextToVoice",
      "Version": "2019-08-23",
      "Text": text,
      "Region": "ap-beijing",
      "VoiceType": 1051,
      "SessionId": "".join([str(random.randint(0, 10)) for _ in range(10)]),
      "Codec": "wav",
    }
    req.from_json_string(json.dumps(params))

    # 返回的resp是一个TextToVoiceResponse的实例，与请求对象对应
    resp = json.loads(client.TextToVoice(req).to_json_string())
    return resp["Audio"]
