from langchain.chat_models import ChatOpenAI
import json 
from dotenv import load_dotenv
load_dotenv()
import requests
from newspaper import Article
from langchain.schema import HumanMessage
from langchain.callbacks import get_openai_callback

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}

article_url = "https://www.thehindu.com/news/international/israel-hamas-war-day-17-live-updates/article67451601.ece"

session = requests.Session()

try:
    response = session.get(article_url, headers=headers,timeout=10)

    if response.status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()
        print(article.title)
        print(article.text)

    else:
        print(f"Failed to fetch article at {article_url}")
except Exception as e:
    print(f"Error occurred while fetching article at {article_url}: {e}")

article_title = article.title
article_text = article.text

template = """You are a very good assistant that summarizes online articles in bulleted points.

Here's the article you want to summarize.

==================
Title: {article_title}

{article_text}
==================

Write a summary of the previous article.
"""

prompt = template.format(article_title=article_title, article_text=article_text)

messages = [HumanMessage(content=prompt)]

chat = ChatOpenAI(model_name="gpt-4", temperature=0,n=2)

#with get_openai_callback() as cb:
    #summary = chat(messages)
    #print(summary.content)
    #print(cb)