# NEWS_RAG

**해당 프로젝트는 성균관대학교 Co-DeepLearning Project에서 진행되었습니다.**     

**NEWS RAG 챗봇**은 하루 경제 신문 기사를 **RAG(Retrieval-Augmented Generation)** 방식으로 활용해 최신 경제 뉴스를 빠르게 제공합니다.

## 1. 주요 개념
- **Retrieval**: DB나 인덱스에서 관련 문서를 찾아옴  
- **Augmented Generation**: LLM이 검색된 문서를 바탕으로 답변 생성  

## 2. 요구사항
- **Python 3.11**  
- **LLM API 키** (예: OpenAI)  
- **인덱스 환경** (예: Elasticsearch, FAISS, Pinecone 등)  

## 3. 설치 및 실행
1. **프로젝트 클론 & 이동**  
   ```bash
   git clone https://github.com/Twave1717/NEWS_RAG.git
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
5. **서버 실행**  
   ```bash
   streamlit run main.py
   ```

## 5. 스펙
- **LLM**: OpenAI GPT-4o-mini 사용 예정
- **인덱스/검색**: FAISS 원하는 방식으로 대체 가능  
- **데이터 전처리**: 기사 본문을 필요한 방식(문단 분할, 정규화 등)으로 처리  

## 6. 라이선스
- **MIT License**
