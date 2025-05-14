from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
from dotenv import load_dotenv
import os

load_dotenv()

# Setup the model
groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model='Gemma-2-9b-it', groq_api_key=groq_api_key)

# Prompt
generic_template = "Translate to the following {language}"
prompt = ChatPromptTemplate.from_messages([
    ("system", generic_template),
    ("user", "{text}")
])

# Parser
parser = StrOutputParser()

# Compose the chain as a Runnable using `|`
chain = prompt | model | parser

# Setup FastAPI
app = FastAPI(
    title="Translator",
    version="0.1",
    description="Simple language translator API using LangChain Runnable interface"
)

# Add LangServe routes
add_routes(app, chain, path="/chain")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
