import streamlit as st
import time
import os
from datetime import datetime, timedelta

# .env 로드 (LANGCHAIN_API_KEY 등)
from dotenv import load_dotenv
load_dotenv()

# LangChain / Teddynote 관련
from langchain_teddynote import logging
logging.langsmith("CH12-RAG")

import bs4
from langchain import hub
# langchain_text_splitters 대신 langchain.text_splitter를 써야 할 수도 있음
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- Selenium 관련 ---
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

# langchain 최신 버전 기준
from langchain.schema import Document

##################################################
# (1) 뉴스 크롤링 로직
##################################################
def get_total_pages(soup):
    navi_table = soup.find("table", class_="Nnavi")
    if not navi_table:
        return 1

    page_links = soup.find_all("a")
    max_page = 1
    for link_tag in page_links:
        href = link_tag.get("href", "")
        if "page=" in href:
            try:
                page_str = href.split("page=")[1].split("&")[0]
                page_num = int(page_str)
                if page_num > max_page:
                    max_page = page_num
            except:
                continue
    return max_page

def get_news_from_list_page(driver, page_url):
    driver.get(page_url)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "realtimeNewsList"))
        )
    except:
        return [], None

    soup = BeautifulSoup(driver.page_source, "html.parser")
    realtime_news_list = soup.find("ul", class_="realtimeNewsList")
    
    articles = []
    if realtime_news_list:
        li_tags = realtime_news_list.find_all("li", class_="newsList")
        for li in li_tags:
            dl_tags = li.find_all("dl")
            for dl in dl_tags:
                dd_subjects = dl.find_all("dd", class_="articleSubject")
                dd_summaries = dl.find_all("dd", class_="articleSummary")
                
                for subject_dd, summary_dd in zip(dd_subjects, dd_summaries):
                    a_tag = subject_dd.find("a")
                    title = a_tag.get_text(strip=True) if a_tag else "제목 없음"
                    link  = a_tag["href"] if a_tag else "링크 없음"
                    
                    date_span = summary_dd.find("span", class_="wdate")
                    date_ = date_span.get_text(strip=True) if date_span else "날짜 없음"
                    
                    articles.append({
                        "title": title,
                        "link": link,
                        "date": date_
                    })
    return articles, soup

def get_news_detail(driver, article, status_placeholder=None):
    # "크롤링중: 기사제목" 표시
    if status_placeholder:
        status_placeholder.write(f"뉴스를 가져오는 중: {article.get('title','(제목없음)')}")

    link = article.get("link", "")
    if not link.startswith("http"):
        link = "https://n.news.naver.com" + link
    article["URL"] = link

    driver.get(link)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "dic_area"))
        )
    except:
        article["content"] = "본문 로딩 실패"
        return article
    
    soup = BeautifulSoup(driver.page_source, "html.parser")
    content_tag = soup.find("article", id="dic_area")
    content = content_tag.get_text("\n", strip=True) if content_tag else "본문 없음"
    
    article["content"] = content
    return article

def crawl_news():
    """
    '어제' 날짜의 해외 증시 뉴스 크롤링
    """
    target_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    base_url = (
        "https://finance.naver.com/news/news_list.naver"
        "?mode=LSS3D&section_id=101&section_id2=258&section_id3=403"
        f"&date={target_date}"
    )

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

    status_placeholder = st.empty()

    all_articles = []
    try:
        # 1페이지
        first_page_url = base_url + "&page=1"
        articles, soup = get_news_from_list_page(driver, first_page_url)
        total_pages = get_total_pages(soup)
        
        # 2 ~ 마지막 페이지
        for page_num in range(2, total_pages + 1):
            page_url = f"{base_url}&page={page_num}"
            page_articles, _ = get_news_from_list_page(driver, page_url)
            articles.extend(page_articles)

        # 상세 본문 크롤링 (실시간 기사제목 표시)
        for art in articles:
            get_news_detail(driver, art, status_placeholder=status_placeholder)
            time.sleep(0.05)

    finally:
        driver.quit()
        status_placeholder.empty()

    return articles


##################################################
# (2) RAG (Retrieval QA) 세팅
##################################################
def create_retriever_from_articles(articles, status_placeholder=None):
    """
    크롤링된 기사들을 하나의 문서로 만들어 chunking하고,
    FAISS 기반의 retriever 생성
    """
    if status_placeholder:
        status_placeholder.write("인덱싱중...")

    # 기사 전체를 하나의 텍스트로 합침
    all_text = ""
    for art in articles:
        text_part = f"제목: {art['title']}\n날짜: {art['date']}\n{art.get('content','')}\n"
        all_text += text_part + "\n\n"

    docs = [Document(page_content=all_text)]

    # chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # FAISS 벡터스토어
    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    if status_placeholder:
        status_placeholder.empty()

    return retriever

# 프롬프트 템플릿
prompt = PromptTemplate.from_template(
    """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다.

아래는 지금까지의 (최근) 대화 내용 일부입니다:
{chat_history}

당신의 임무는, 주어진 뉴스 문맥(context) 을 사용하여 질문(question)에 답변하는 것입니다.
만약 해당 문맥으로부터 답을 찾을 수 없다면
"주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다" 라고 답하세요.

# Question:
{question}

# Context:
{context}

# Answer:"""
)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

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
# (3) Streamlit 메인
##################################################
def main():
    st.set_page_config(page_title="어제 미국 증시 뉴스 챗봇", layout="wide")
    st.title("어제 미국 증시 뉴스 챗봇")

    # ------------------------------------------------
    # 1) 세션에 articles가 없으면 크롤링 & 인덱싱 실행
    #    (같은 렌더링에서 UI가 이어서 표시되도록 st.stop() 사용 X)
    # ------------------------------------------------
    if "articles" not in st.session_state:
        with st.spinner("로딩중..."):
            # (A) 크롤링
            articles = crawl_news()
            st.session_state["articles"] = articles

            # (B) 인덱싱
            st.session_state["retriever"] = create_retriever_from_articles(
                articles, status_placeholder=st.empty()
            )

    # ------------------------------------------------
    # 2) 크롤링 완료 이후, 카드뷰 + 챗봇 UI 표시
    # ------------------------------------------------
    articles = st.session_state["articles"]
    retriever = st.session_state["retriever"]

    # (A) 뉴스 카드뷰 (토글)
    with st.expander("뉴스 기사 목록 (펼치기/접기)", expanded=False):
        st.write(f"크롤링된 기사 수: {len(articles)}개")

        # 카드뷰용 CSS
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
            url = art.get("URL", "#")
            title = art.get("title", "제목 없음")
            date_ = art.get("date", "날짜 없음")

            card_html = f"""
            <a class="news-card" href="{url}" target="_blank">
                <span class="title">{title}</span>
                <span class="date">({date_})</span>
            </a>
            """
            st.markdown(card_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # (B) 챗봇 UI (RAG QA)
    st.subheader("뉴스 기반 챗봇")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.text_input("질문을 입력하세요", "")
    if st.button("Send") and user_input.strip():
        # 1) 사용자 메시지 저장
        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        # 2) 최근 500자 추출
        entire_chat_text = "".join(msg["content"] for msg in st.session_state["chat_history"])
        last_500_chars = entire_chat_text[-500:]

        # 3) 문맥 검색
        context_docs = retriever.get_relevant_documents(user_input)
        context_text = "\n".join([doc.page_content for doc in context_docs])

        # 4) RAG 체인 실행
        chain_input = {
            "context": context_text,
            "question": user_input,
            "chat_history": last_500_chars,
        }
        answer = rag_chain.invoke(chain_input)

        # 5) 어시스턴트 메시지 추가
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})

    # (C) 대화 내역 표시
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.write(f"**User:** {msg['content']}")
        else:
            st.write(f"**Assistant:** {msg['content']}")

if __name__ == "__main__":
    main()
