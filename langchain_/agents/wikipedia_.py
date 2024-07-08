from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0, model="gpt-4")

# * llm-math는 나이 계산을 위해 사용
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    description="계산이 필요할 때 사용",
    verbose=True,
)
"""
- tools: 에이전트가 접근할 수 있는 툴로 여기서는 위키피디아와 llm-math를 사용
- llm: 에이전트로 사용할 언어 모델
- AgentType.ZERO_SHOT_REACT_DESCRIPTION: 툴의 용도와 사용 시기를 결정하는 에이전트. 따라서 이것을 사용하는 경우, 툴마다 설명을 제공해야 함.
    - REACT_DOCSTORE: 이 에이전트는 질문에 답하기 위해, 관련 정보를 조회할 수 있는 검색 도구가 필요함.
    - CONVERSATIONAL_REACT_DESCRIPTION: 메모리를 사용하여 과거에 시도했던 대화를 기억함.
"""

print(
    agent.invoke("에드 시런이 태어난 년도는 어떻게 되나요? 2024년도 현재 에드 시런은 몇 살인가요?")
)
"""
{'input': '에드 시런이 태어난 년도는 어떻게 되나요? 2024년도 현재 에드 시런은 몇 살인가요?', 'output': '에드 시런은 1991년에 태어났으므로, 2024년에는 33살이 됩니다.'}
"""
