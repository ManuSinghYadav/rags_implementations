from dotenv import load_dotenv

from openai import OpenAI
import cohere


load_dotenv()

openai = OpenAI()

co = cohere.ClientV2()
