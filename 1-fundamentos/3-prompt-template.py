from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["name"],
    template="Hello, I'm {name}! Tell me a joke with my name!",
)

text = template.format(name="Alice")
print(text)