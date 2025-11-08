import pandas as pd
import openai
from dotenv import load_dotenv, dotenv_values


# load_dotenv()  # take environment variables
config = dotenv_values(".env")

a = 1
b = 2
c = 5
d = 2


print(a + b + c + d)

a = 'hello'

print(a)

print(config['OPENAI_API_KEY'])