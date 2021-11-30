
import uvicorn
from fastapi import FastAPI
#from predict_01 import predict
import json
from pydantic import BaseModel
from predict import *
app = FastAPI()

class InputText(BaseModel):
    content:str

@app.post("/category_text")
async def root(
        query: InputText,
):
    # TODO 调用
    preds,p=predict(config,query.content)
    p=float(p[0])
    data=str({'预测类别':preds,'预测概率':p})
    return {"code": 200, "message": "成功", "data":data}

if __name__ == '__main__':
    
    uvicorn.run('main:app', port=8002, host='0.0.0.0',
                reload=True, proxy_headers=True, debug=True)
