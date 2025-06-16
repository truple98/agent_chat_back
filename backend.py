from pydantic import BaseModel # 데이터 타입 정의 모듈
from typing import List, Dict # 타입 힌트 모듈
from fastapi import FastAPI, HTTPException # 웹 서버 모듈, 예외 처리 모듈
from fastapi.middleware.cors import CORSMiddleware # 교차 출처 리소스 공유 설정 모듈
from dotenv import load_dotenv
from agent import process_query # 작성된 agent.py에서 process_query 함수 가져오기

load_dotenv()

# 타입 정의
class ChatMessage(BaseModel):
  role: str
  parts: List[Dict[str, str]]

class ChatRequest(BaseModel):
  contents: List[ChatMessage]

class ChatCandidate(BaseModel): # 챗봇 답변 타입
  content: ChatMessage

class ChatResponse(BaseModel):
  candidates: List[ChatCandidate]

app = FastAPI(title="법률 관련 채팅 서비스 API", description="법률 관련 질문에 답변해 드립니다.")

# CORS 미들웨어 설정
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"], # 모든 출처 허용 : 실제 서비스시 도메인으로 변경
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)

@app.get("/")
async def root():
  return {"message": "법률 관련 채팅 서비스"}

app.state.conversation_history = []

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
  """
  법률 질문에 답변 드립니다.
  """
  try:
    # 기존 대화 기록 가져오기
    conversation_history = app.state.conversation_history

    # 현재 사용자의 입력 메세지 가져오기
    current_user_message = request.contents[-1].parts[0].get("text", "") if request.contents else ""

    # AI 답변 생성
    response = await process_query(current_user_message, conversation_history)

    # 응답 형식 구성 및 반환
    return ChatResponse(
      candidates=[
        ChatCandidate(
          content=ChatMessage(
            role="model",
            parts=[{"text": response}]
          )
        )
      ]
    )
  
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"오류 발생 : {str(e)}")
  
# 프로그램 실행
if __name__ == "__main__":
  import uvicorn
  # uvicorn.run(app, host="0.0.0.0", port=8000)
  uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)