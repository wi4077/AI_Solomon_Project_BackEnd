import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cors import CORS
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import ChatUpstage
from langchain_openai import ChatOpenAI
from langchain_upstage import UpstageEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

app = FastAPI()

# 환경 변수 가져오기
load_dotenv()

# upstage
chat_upstage = ChatUpstage(
    model='solar-pro',
    temperature=0.1,
    top_p=0.8
)
chat_openai = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.1,
    top_p=1,
    max_tokens=256,
)
embedding_upstage = UpstageEmbeddings(model='solar-embedding-1-large')

pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=pinecone_api_key)
index_name = 'house-lease'

# create new index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

pinecone_vectorstore = PineconeVectorStore(index=pc.Index(index_name), embedding=embedding_upstage)
pinecone_retriever = pinecone_vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 15}
)

# ReRanker: CrossEncoder
model = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-base",
)

compressor = CrossEncoderReranker(model=model, top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=pinecone_retriever
)

origins = [
    "https://ai-solomon-project.vercel.app",  # Vercel 도메인
    "https://*.vercel.app",              # Vercel 서브도메인 허용
    "http://localhost:3000",             # 로컬 개발용
]


app.add_middleware(
    CORS,
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class MessageRequest(BaseModel):
    messages: str

class ChatMessage(BaseModel):
    role: str
    content: str


@app.get('/')
async def root():
    return {'reply': 'local host test'}

@app.post('/chat')
async def chat(req: MessageRequest):

    user_messages = req.messages
    print(f'User messages: {user_messages}')

    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            너는 계약 조항에 관한 법률 정보를 일반인이 이해하기 쉽게 설명해주는 법률 전문가야.
            항상 존댓말을 사용하고 쉽게 설명해 줘.
            
            사용자가 부동산 관련 질문을 하지 않는 경우(예: 'happy', '뭐 먹지' 등)
            부동산과 관련 있는 질문을 해달라고 해.

            반드시 아래 [context] 내용을 참고해서만 답변해야 해.
            먼저 [context] 안에 질문과 직접적으로 관련된 조항이나 사례가 있는지 반드시 확인을 해.
            만약 관련 조항(키워드, 판례 등)이 없으면 '관련 근거 없음'이라고 판단해.
            관련 근거가 없을 경우 해당 상황에 대해 명시적 규정이 보이지 않는다고 밝힌 뒤 일반적인 계약 사항 원칙을 설명해 줘.
            (예: 임차인의 중도 해지, 정당한 사유 기준 등)
            주의! [context]에 명시되지 않은 사실을 임의로 추정하거나 창작하지 마.
            사용자의 상황에 대해 유리하게 추정하지 말고 정확한 사실만 말해.

            특히 개인 사정에 따라 계약이 바뀌는 경우는 관련 조항, 판례가 명시되어 있지 않으면
            해당 상황에 대해 본 계약, 판례에서 확인되지 않는다고 먼저 밝힌 후 일반 원칙을 설명해 줘.
            [context]에 관련 조항이 있을 때만 그 조항을 근거로 판단해 줘.

            사용자가 계약 문구나 사례에 대해 질문하면 반드시 다음 형식으로만 대답해 줘:

            📘 [법적 해석]:
            🗣 [쉽게 말하면]:
            ⚠️ [주의할 점]:

            각 블록은 3~4줄, 250자 정도로 제한할게.
            전문 용어는 괄호로 간단히 해설을 달아 줘,
            [쉽게 말하면] 블록에 필요할 경우 '예: ...' 형태의 짧은 사례를 한 줄 제공해 줘.

            [context]
            {context}
            """,
        ),
        ("human", "임차인은 계약기간 중 임의 해지를 할 수 없으며, 이에 위반할 경우 잔여기간 월세를 위약금으로 청구할 수 있다."),
        ("ai", """
📘 [법적 해석]:
임차인은 계약 기간이 끝나기 전에는 중도에 계약을 해지할 수 없으며, 만약 해지할 경우 남은 기간 동안의 월세를 전부 내야 한다는 뜻입니다.

🗣 [쉽게 말하면]:
세입자가 계약 중간에 나가면, 그동안의 남은 월세를 전부 벌금처럼 내야 한다는 말이에요.

⚠️ [주의할 점]:
이 조항은 실제로는 ‘정당한 사유’가 있으면 중도 해지가 가능하므로, 무조건 위약금을 내야 하는 것은 아닙니다. 법률적으로는 효력이 제한적일 수 있어요.
        """),
        ("human", "{input}"),
    ])
    # Prompt chain
    chain = prompt | chat_openai | StrOutputParser()
    
    result_docs = compression_retriever.invoke(user_messages)
    context_text = "\n\n".join(d.page_content for d in result_docs)
    result = chain.invoke({'context': context_text, 'input': user_messages})

    print(f'Chain result: {result}')

    return {'reply': result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)