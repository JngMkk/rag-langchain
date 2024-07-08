from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0, model_name="gpt-4", max_tokens=2048)
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template="7개의 팀을 보여줘 {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

query = "한국의 야구팀은?"
output = llm.invoke(prompt.format(subject=query))
print(output)
"""
content='두산 베어스, 롯데 자이언츠, 삼성 라이온즈, SK 와이번스, KIA 타이거즈, LG 트윈스, NC 다이노스'
response_metadata={
    'token_usage': {
        'completion_tokens': 62,
        'prompt_tokens': 57,
        'total_tokens': 119
    },
    'model_name': 'gpt-4-0613',
    'system_fingerprint': None,
    'finish_reason':'stop',
    'logprobs': None
}
id='run-e15dfa94-65dd-443e-b76c-e509ec7c16b3-0'
usage_metadata={
    'input_tokens': 57,
    'output_tokens': 62,
    'total_tokens': 119
}
"""

parsed_result = output_parser.parse(output.content)
print(parsed_result)
# ['두산 베어스', '롯데 자이언츠', '삼성 라이온즈', 'SK 와이번스', 'KIA 타이거즈', 'LG 트윈스', 'NC 다이노스']
