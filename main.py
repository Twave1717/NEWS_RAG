import streamlit as st
from streamlit_chat import message  # streamlit-chat 사용
import time
import os
from datetime import datetime, timedelta

import requests  # API 요청을 위해 추가

from dotenv import load_dotenv
load_dotenv()  # .env 파일 로드

# langchain-teddynote
from langchain_teddynote import logging
logging.langsmith("CH12-RAG")

from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain.schema import Document


##################################################
# (0) 매일 00:01(=12시1분)에 캐시를 갱신하기 위한 로직
##################################################
def refresh_cache_if_1201():
    """
    매일 00:01(=12시1분)이 되면 캐시를 초기화하여
    다음 호출 시 새로 뉴스를 크롤링(=API로 가져오기)하도록 합니다.
    """
    now = datetime.now()
    # 시/분이 각각 0,1 이면 cache clear
    if (now.hour == 0 and now.minute == 1):
        st.cache_resource.clear()
        st.experimental_rerun()


##################################################
# (1) DeepSearch API로 뉴스 불러오기 (Selenium 제거)
##################################################
def fetch_news_api():
    """
    전날부터 1주일치(즉, 어제 날짜 ~ 7일 전 날짜) 뉴스 기사를
    DeepSearch API로 모두 불러와 리스트로 반환합니다.

    반환 형식:
    [
      {
        "id": ...,
        "title": ...,
        "title_ko": ...,
        "summary": ...,
        "summary_ko": ...,
        "content_url": ...,
        "published_at": ...,
        ...
      },
      ...
    ]
    """
    # 날짜 설정 (어제 ~ 7일 전)
    end_date_dt = datetime.now() - timedelta(days=1)
    start_date_dt = datetime.now() - timedelta(days=7)
    date_from_str = start_date_dt.strftime("%Y-%m-%d")
    date_to_str = end_date_dt.strftime("%Y-%m-%d")

    # .env 파일에서 API_KEY 불러오기
    API_KEY = "5d1ca2322d974c129b89b2937f736bfa"
    if not API_KEY:
        st.error("DEEPSEARCH_API_KEY 환경 변수가 설정되지 않았습니다.")
        return []

    BASE_URL = "https://api-v2.deepsearch.com/v1/global-articles/economy"
    page_size = 100

    # articles 결과를 담을 리스트
    all_articles = []

    # 우선 1페이지를 호출하여 total_pages 확인
    page = 1
    url = (
        f"{BASE_URL}?api_key={API_KEY}"
        f"&date_from={date_from_str}&date_to={date_to_str}"
        f"&page={page}&page_size={page_size}"
    )
    resp = requests.get(url)
    if resp.status_code != 200:
        st.error(f"API 호출 실패: {resp.status_code}")
        return []

    data_json = resp.json()
    total_pages = data_json.get("total_pages", 1)
    items = data_json.get("data", [])
    all_articles.extend(items)

    # 2페이지 ~ total_pages까지 반복
    for page_num in range(2, total_pages + 1):
        url = (
            f"{BASE_URL}?api_key={API_KEY}"
            f"&date_from={date_from_str}&date_to={date_to_str}"
            f"&page={page_num}&page_size={page_size}"
        )
        resp = requests.get(url)
        if resp.status_code == 200:
            data_json = resp.json()
            items = data_json.get("data", [])
            all_articles.extend(items)
        else:
            st.warning(f"페이지 {page_num} 호출 실패: {resp.status_code}")
            break

    return all_articles


##################################################
# (2) RAG (인덱싱에 사용)
##################################################
def create_retriever_from_articles(articles):
    """
    가져온 articles를 바탕으로 2가지 문서를 만든 뒤,
    - (문서1) 모든 기사내용(영문/한글요약/본문url 등)을 합쳐서 하나의 큰 텍스트로
    - (문서2) "YYYY년 MM월 DD일 증시 요약" + 기사제목들만 모아 놓은 텍스트

    두 문서를 합쳐서 하나의 retriever를 생성합니다.
    """
    # (A) 모든 기사 내용을 합치는 문서
    all_text_list = []
    for art in articles:
        title = art.get("title", "")
        title_ko = art.get("title_ko", "")
        summary = art.get("summary", "")
        summary_ko = art.get("summary_ko", "")
        pub_at = art.get("published_at", "")
        content_url = art.get("content_url", "")
        reason = art.get("reason", "")

        one_text = (
            f"[기사ID: {art.get('id')}]\n\n"
            f"영문제목: {title}\n"
            # f"한글제목: {title_ko}\n"  # 한글 제목 제외
            f"발행일: {pub_at}\n"
            # f"내용URL: {content_url}\n"  # 내용 URL 제외
            f"영문요약: {summary}\n"
            # f"한글요약: {summary_ko}\n"  # 한글 요약 제외
            f"reason: {reason}\n\n"
            "----------------------------"
        )

        all_text_list.append(one_text)

    doc_text_1 = "\n".join(all_text_list)

    # (B) "YYYY년 MM월 DD일 증시 요약" 형태로 기사제목들만 모은 문서
    today_str = datetime.now().strftime("%Y년 %m월 %d일")
    header = f"{today_str} 증시 요약\n\n"
    titles_only_list = []
    for art in articles:
        t_ko = art.get("title_ko", "")
        t_en = art.get("title", "")
        if t_ko:
            titles_only_list.append(f"- {t_ko}")
        else:
            titles_only_list.append(f"- {t_en}")

    doc_text_2 = header + "\n".join(titles_only_list) + "\n"

    # (C) 두 문서를 하나의 리스트로
    docs = [
        Document(page_content=doc_text_1),
        Document(page_content=doc_text_2),
    ]

    # (D) 텍스트 분할 & 벡터 인덱스 생성
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    return retriever


prompt = PromptTemplate.from_template(
    """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다.

아래는 지금까지의 (최근) 대화 내용 일부입니다:
{chat_history}

당신의 임무는, 주어진 뉴스 문맥(context)을 사용하여 질문(question)에 답변하는 것입니다.

# Question:
{question}

# Context:
{context}

# Answer:"""
)

llm = ChatOpenAI(model_name="gpt-4", temperature=0)  # 모델 이름 수정
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


##################################################
# (2-1) 매번 새로 안 하고, "공유 데이터"를 캐시에 저장
##################################################
@st.cache_resource
def get_cached_articles_and_retriever():
    """
    - 서버가 실행된 후 최초 1회, 그리고 cache가 비워졌을 때만 실제 API에서 기사 불러오기(fetch_news_api).
    - 이후에는 동일 객체(articles, retriever) 반환.
    """
    articles = fetch_news_api()  # 1주일치 뉴스 기사 불러오기
    retriever = create_retriever_from_articles(articles)  # 문서화 + 벡터 DB 생성
    return articles, retriever


##################################################
# (3) Streamlit 메인
##################################################
def main():
    st.set_page_config(page_title="미국 증시 뉴스 챗봇", layout="wide")
    
    # 제목 설정
    st.title("미국 증시 뉴스 챗봇")
    
    # 시작일과 끝나는 날짜 계산
    end_date_dt = datetime.now() - timedelta(days=1)
    start_date_dt = datetime.now() - timedelta(days=7)
    start_date_str = start_date_dt.strftime("%Y년 %m월 %d일")
    end_date_str = end_date_dt.strftime("%Y년 %m월 %d일")
    
    # 서브타이틀 추가 (웃는 이모티콘 포함)
    st.markdown(f"😊 **{start_date_str} ~ {end_date_str}까지의 뉴스를 통해 증시를 알려드려요!**")
    
    # 매일 00:01(=12시1분)에 cache를 비우고, 새로 인덱싱하도록
    refresh_cache_if_1201()

    # (A) 캐시에 저장된(또는 새로 생성된) articles, retriever 가져오기
    if "articles" not in st.session_state or "retriever" not in st.session_state:
        # 처음(서버 구동 후 최초 접근)인 경우나, 세션에서 아직 안 불러온 경우
        articles, retriever = get_cached_articles_and_retriever()
        st.session_state["articles"] = articles
        st.session_state["retriever"] = retriever

    articles = st.session_state["articles"]
    retriever = st.session_state["retriever"]

    # ------------------- 카드뷰 (토글) -------------------
    with st.expander("불러온 뉴스 목록 (펼치기/접기)", expanded=False):
        st.write(f"불러온 기사 수: {len(articles)}개")

        scroll_css = """
        <style>
        .horizontal-scroll {
            display: flex;
            flex-direction: row;
            align-items: center;
            height: 50px;
            overflow-x: auto;
            white-space: nowrap;
            border: 1px solid #ccc;
            border-radius: 6px;
            background-color: #fafafa;
            padding: 5px;
        }
        .news-card {
            display: inline-flex;
            align-items: center;
            height: 40px;
            padding: 0 10px;
            margin-right: 10px;
            text-decoration: none;
            color: inherit;
            border-right: 1px solid #ddd;
        }
        .news-card:last-child {
            border-right: none;
            margin-right: 0;
        }
        .news-card:hover {
            background-color: #eaeaea;
        }
        .title {
            font-weight: bold;
            margin-right: 5px;
        }
        .date {
            color: #666;
            font-size: 0.9rem;
        }
        </style>
        """
        st.markdown(scroll_css, unsafe_allow_html=True)

        st.markdown('<div class="horizontal-scroll">', unsafe_allow_html=True)
        for art in articles:
            url = art.get("content_url", "#")
            title_en = art.get("title", "제목 없음")
            title_ko = art.get("title_ko", "")
            if title_ko.strip():
                display_title = title_ko
            else:
                display_title = title_en

            date_ = art.get("published_at", "날짜 없음")

            card_html = f"""
            <a class="news-card" href="{url}" target="_blank">
                <span class="title">{display_title}</span>
                <span class="date">({date_})</span>
            </a>
            """
            st.markdown(card_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------- streamlit-chat -------------------

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    user_input = st.text_input("질문을 입력하세요", "")

    # Send 버튼: 로딩중이면 disabled=True
    send_btn = st.button("Send", disabled=st.session_state.get("loading", False))

    if send_btn and user_input.strip():
        st.session_state["loading"] = True

        # (A) 사용자 메시지
        st.session_state["messages"].append({
            "role": "user",
            "content": user_input
        })

        # (B) 최근 대화 500자
        entire_chat_text = "".join(m["content"] for m in st.session_state["messages"])
        last_500_chars = entire_chat_text[-500:]

        # (C) 검색
        if retriever:
            context_docs = retriever.get_relevant_documents(user_input)
            context_text = "\n".join([doc.page_content for doc in context_docs])

            # (D) RAG
            chain_input = {
                "context": context_text,
                "question": user_input,
                "chat_history": last_500_chars,
            }
            answer = rag_chain.invoke(chain_input)
        else:
            answer = "인덱싱된 기사 내용이 없습니다."

        # (E) 봇 메시지
        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer
        })

        st.session_state["loading"] = False

    # 메시지 표시
    for i, msg in enumerate(st.session_state["messages"]):
        if msg["role"] == "user":
            message(msg["content"], is_user=True, key=f"user_{i}")
        else:
            message(msg["content"], is_user=False, key=f"assistant_{i}")


if __name__ == "__main__":
    main()
