import streamlit as st
from streamlit_chat import message  # streamlit-chat 사용
import time
import os
from datetime import datetime, timedelta
import html

from dotenv import load_dotenv
load_dotenv()

# langchain-teddynote
from langchain_teddynote import logging
logging.langsmith("CH12-RAG")

import bs4
from langchain import hub
# (버전에 따라) langchain.text_splitter 사용 필요성 있음
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Selenium 관련
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
# (0) 매일 00:01(=12시1분)에 캐시를 갱신하기 위한 로직
##################################################
def refresh_cache_if_1201():
    """
    매일 00:01(=12시1분)이 되면 캐시를 초기화하여
    다음 호출 시 새로 뉴스를 크롤링하도록 합니다.
    """
    now = datetime.now()
    # 시/분이 각각 0,1 이면 cache clear
    if (now.hour == 0 and now.minute == 1):
        st.cache_resource.clear()
        st.experimental_rerun()


##################################################
# (1) 뉴스 크롤링 로직 (기존 코드 유지)
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
    # (로딩 메시지 제거: status_placeholder 사용하지 않음)
    # 글씨들은 바꾸지 않으려고, 함수 인자는 그대로 두되 내부 활용은 생략
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


##################################################
# (1-2) 실제로 뉴스를 크롤링하는 함수
##################################################
def crawl_news():
    """
    크롤링을 수행해, 전날 자정 기준으로 뉴스 기사들을 가져옴.
    (로컬 Chrome 설치 없이도 가능한 환경이라면 그대로 사용)
    """

    # (1) URL 설정
    target_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    base_url = (
        "https://finance.naver.com/news/news_list.naver"
        "?mode=LSS3D&section_id=101&section_id2=258&section_id3=403"
        f"&date={target_date}"
    )

    # (2) ChromeOptions 설정
    chrome_options = Options()
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--headless")  
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("disable-infobars")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--remote-debugging-port=9222")

    # (3) WebDriver 실행
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
    except Exception as e:
        st.error(f"WebDriver 실행 실패: {e}")
        # 혹시 자동 설치 실패 시, 수동 경로를 지정
        driver = webdriver.Chrome(
            service=Service("/path/to/chromedriver"),
            options=chrome_options
        )

    # (4) 기사 수집
    all_articles = []
    try:
        first_page_url = f"{base_url}&page=1"
        articles, soup = get_news_from_list_page(driver, first_page_url)
        total_pages = get_total_pages(soup)

        for page_num in range(2, total_pages + 1):
            page_url = f"{base_url}&page={page_num}"
            page_articles, _ = get_news_from_list_page(driver, page_url)
            articles.extend(page_articles)

        for art in articles:
            get_news_detail(driver, art, status_placeholder=None)
            time.sleep(0.05)
    finally:
        driver.quit()

    return articles


##################################################
# (2) RAG (변경 없음 - 인덱싱)
##################################################
def create_retriever_from_articles(articles, status_placeholder=None):
    # (로딩 메시지 제거: status_placeholder 사용하지 않음)
    all_text = ""
    for art in articles:
        one_text = f"제목: {art['title']}\n날짜: {art['date']}\n{art.get('content','')}\n"
        all_text += one_text + "\n\n"

    docs = [Document(page_content=all_text)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    return retriever


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

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

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
# (2-1) 매번 새로 안 하고, "공유 데이터"를 캐시에 저장
##################################################
@st.cache_resource
def get_cached_articles_and_retriever():
    """
    - 서버가 실행된 후 최초 1회, 그리고 cache가 비워졌을 때만 실제 크롤링.
    - 이후에는 동일 객체(articles, retriever) 반환.
    """
    articles = crawl_news()
    retriever = create_retriever_from_articles(articles)
    return articles, retriever


##################################################
# (3) Streamlit 메인
##################################################
def main():
    st.set_page_config(page_title="미국 증시 챗봇 (Co-RAG)", layout="wide")
    st.title("어제 미국증시 뉴스 챗봇")

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

    # ------------------- streamlit-chat -------------------
    st.subheader("뉴스 기반 챗봇")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    user_input = st.text_input("질문을 입력하세요", "")

    # Send 버튼: 로딩중이면 disabled=True
    send_btn = st.button("Send", disabled=st.session_state["loading"] if "loading" in st.session_state else False)

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
