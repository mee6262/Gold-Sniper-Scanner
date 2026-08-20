"""Optional AI gate for deterministic Gold Sniper candidates."""
from __future__ import annotations
import json, os, requests
from dotenv import load_dotenv
load_dotenv()


def review(candidate: dict) -> dict:
    key=os.getenv("AI_API_KEY")
    if not key:
        return {"enabled":False,"decision":"SKIP","quality":None,"risk":"UNKNOWN","reason":"AI_API_KEY not configured"}
    payload={
        "model":os.getenv("AI_MODEL","gpt-4o-mini"),
        "temperature":0,
        "response_format":{"type":"json_object"},
        "messages":[
            {"role":"system","content":"You are a strict XAUUSD setup reviewer. Review only the supplied deterministic candidate. Do not invent price data or change direction. Return JSON with decision PASS or REJECT, quality 0-100, risk LOW/MEDIUM/HIGH, flags array, reason string."},
            {"role":"user","content":json.dumps(candidate,default=str)}
        ]
    }
    try:
        r=requests.post(os.getenv("AI_API_URL","https://api.openai.com/v1/chat/completions"),headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},json=payload,timeout=20)
        r.raise_for_status()
        result=json.loads(r.json()["choices"][0]["message"]["content"])
        result["enabled"]=True
        return result
    except Exception as exc:
        return {"enabled":True,"decision":"ERROR","quality":None,"risk":"HIGH","flags":[str(exc)],"reason":str(exc)}
