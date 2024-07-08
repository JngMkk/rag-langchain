import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai.chat_models import ChatOpenAI

df = pd.read_csv("./implements/data/report.csv")

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
    df,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True,
)
print(agent.run("해당 csv 파일을 분석하고, 약 3줄에서 5줄로 요약해주세요."))
