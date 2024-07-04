import requests
from newspaper import Article
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from typing import List
from pydantic import BaseModel, Field, validator
from langchain.prompts import PromptTemplate,FewShotPromptTemplate
load_dotenv()

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}

article_url = "https://www.thehindu.com/news/international/israel-hamas-war-live-updates-day-43-nov-18-2023/article67546700.ece"

session = requests.Session()

try:
    response = session.get(article_url, headers=headers,timeout=10)

    if response.status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()

    else:
        print(f"Failed to fetch article at {article_url}")
except Exception as e:
    print(f"Error occurred while fetching article at {article_url}: {e}")

article_title = article.title
article_text = article.text

examples = [
    {
        "article_title":"The Effects of Climate Change",
        "article_text":"""
        - Climate change is causing a rise in global temperatures.
        - This leads to melting ice caps and rising sea levels. 
        - Resulting in more frequent and severe weather conditions.
        """
    },
    {
        "article_title":"The Evolution of Artificial Intelligence",
        "article_text":"""
        - Artificial Intelligence (AI) has developed significantly over the past decade.
        - AI is now used in multiple fields such as healthcare, finance, and transportation.
        - The future of AI is promising but requires careful regulation.
        """
    },
]

prefix = """As an advanced AI, you've been tasked to summarize online articles into bulleted points. Here are a few examples of how you've done this in the past:
"""

suffix ="""
Now, here's the article you need to summarize:

==================
Title: {article_title}

{article_text}
==================

{format_instructions}
"""

example_template = """
Title: {article_title}
Summary: {article_text}
"""

example_prompt = PromptTemplate(
    template=example_template,
    input_variables=["article_title", "article_text"],
)

class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")
    summary: List[str] = Field(description="Summary of the article in a bulleted list format")

    @validator("summary")
    def validate_summary(cls, summary):
        if len(summary) < 3:
            raise ValueError("Summary must contain at least 3 bullet points")
        return summary
    
parser = PydanticOutputParser(pydantic_object=ArticleSummary)

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    prefix=prefix,
    suffix=suffix,
    example_prompt=example_prompt,
    input_variables=["article_title", "article_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    example_separator="\n\n"
)

model_input = few_shot_prompt_template.format_prompt(article_title=article_title, article_text=article_text)

chat = OpenAI(model_name="text-davinci-003", temperature=0)

summary = chat(model_input.to_string())
parsed_output = parser.parse(summary)
print(parsed_output.summary)
