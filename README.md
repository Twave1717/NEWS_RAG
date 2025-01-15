![header](https://capsule-render.vercel.app/api?type=wave&color=auto&height=300&section=header&text=capsule%20render&fontSize=90)
# 🗞️NEWS_RAG

🚀 데모 페이지: [https://corag-news-rag.streamlit.app/](https://corag-news-rag.streamlit.app/)

---

**해당 프로젝트는 2024 성균관대학교 Co-DeepLearning Project에서 진행되었습니다.**

**NEWS RAG 챗봇**은 일주일 동안의 경제 신문 기사를 **RAG(Retrieval-Augmented Generation)** 방식으로 활용해 최신 경제 뉴스를 빠르게 제공합니다.

# **1. 주요 개념**

이 프로젝트는 **미국 증시 뉴스 챗봇**을 구축하기 위해 다양한 최신 기술과 개념을 통합하여 구현되었습니다.

## 1. LangChain🦜🔗

### 주요 역할

- **Retrieval-Augmented Generation (RAG)**: 검색과 생성 모델을 결합하여 더욱 정확하고 풍부한 응답을 생성합니다.
- **Vector Stores**: FAISS와 같은 벡터 데이터베이스를 사용하여 텍스트 데이터를 효율적으로 검색할 수 있도록 지원합니다.
- **Prompt Templates**: 사용자 질문에 맞춰 언어 모델에게 전달할 템플릿을 정의합니다.

### 코드

- 문서 분할
    
    ```python
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    ```
    
- 벡터 인덱스 생성
    
    ```python
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    
    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    ```
    
- Prompt 템플릿 정의
    
    ```python
    from langchain_core.prompts import PromptTemplate
    
    prompt = PromptTemplate.from_template(
        """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다.
        
        아래는 지금까지의 (최근) 대화 내용 일부입니다:
        {chat_history}
        
        당신의 임무는, 주어진 뉴스 문맥(context)을 사용하여 질문(question)에 답변하는 것입니다.
        만약 해당 문맥으로부터 답을 찾을 수 없다면
        "주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다" 라고 답하세요.
        
        # Question:
        {question}
        
        # Context:
        {context}
        
        # Answer:"""
    )
    
    ```
    

## 2. FAISS (Facebook AI Similarity Search)🕸️

**FAISS**는 Facebook AI Research에서 개발한 라이브러리로, 고차원 벡터의 유사성을 빠르게 검색할 수 있게 해줍니다.

### 주요역할

- **벡터 데이터베이스**: 텍스트 데이터를 벡터로 변환하여 저장하고, 유사한 벡터를 빠르게 검색할 수 있도록 지원합니다.
- **효율적인 검색**: 대량의 데이터에서도 빠른 검색 속도를 보장합니다.

### 코드

- **벡터 인덱스 생성**:
    
    ```python
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    
    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    ```
    
- **유사 문서 검색**:
    
    ```python
    context_docs = retriever.get_relevant_documents(user_input)
    ```
    

## 3. OpenAI API (GPT-4)֎

RAG을 통해 불러온 문서는 프롬프트 엔지니어링을 통해 GPT api에 전달되어 답변을 받습니다. 

### 역할

- **텍스트 생성**: 사용자 질문에 대한 응답을 생성합니다.
- **문맥 이해**: 제공된 뉴스 문맥을 바탕으로 정확한 답변을 도출합니다.

### 코드

- **LangChain과 통합**:
    
    ```python
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    ```
    
- **RAG 체인 실행**:
    
    ```python
    rag_chain = (
        {
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough(),
            "chat_history": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    ```
    

## 2. 요구사항

- **Python 3.11**
- **LLM API 키** (예: OpenAI)
- **인덱스 환경** (예: Elasticsearch, FAISS, Pinecone 등)

## **3. 설치 및 실행**

1. **프로젝트 클론 & 이동**
    
    ```bash
    git clone <https://github.com/Twave1717/NEWS_RAG.git>
    cd NEWS_RAG
    ```
    
2. **가상 환경 설정 (선택)**
    
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    ```
    
3. **패키지 설치**
    - 권장
    
    ```bash
    pip install poetry
    poetry shell
    poetry update
    ```
    
    - 차선책
    
    ```bash
    pip install -r requirements.txt
    ```
    
4. **서버 실행**
    
    ```bash
    streamlit run main.py
    ```
    

## **5. 스펙**

- **LLM**: OpenAI GPT-4o-mini 사용 예정
- **인덱스/검색**: FAISS 원하는 방식으로 대체 가능
- **데이터 전처리**: 기사 본문을 필요한 방식(문단 분할, 정규화 등)으로 처리

## 6. 라이선스

- **MIT License**
