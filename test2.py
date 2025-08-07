from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from openai import OpenAI

# Method 1: Use OpenAI client directly to get reasoning content
# client = OpenAI(
#     api_key="EMPTY",
#     base_url="http://localhost:8000/v1"
# )

# # Get response with reasoning
# response = client.chat.completions.create(
#     model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
#     messages=[{"role": "user", "content": "9.11 and 9.8, which is greater?"}],
#     temperature=0
# )

# print("content:", response.choices[0].message.content)
# if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
#     print("reasoning_content:", response.choices[0].message.reasoning_content)

# Method 2: Custom LangChain wrapper to preserve reasoning
class ReasoningChatOpenAI(ChatOpenAI):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Convert LangChain messages to OpenAI format
        openai_messages = [{"role": "user", "content": msg.content} for msg in messages if hasattr(msg, 'content')]
        
        # Use OpenAI client directly
        client = OpenAI(api_key=self.openai_api_key, base_url=self.openai_api_base)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            temperature=self.temperature,
            **kwargs
        )
        
        # Extract reasoning content if available
        message = response.choices[0].message
        reasoning_content = getattr(message, 'reasoning_content', None)
        
        # Store reasoning in response metadata
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.messages import AIMessage
        
        ai_message = AIMessage(
            content=message.content,
            additional_kwargs={'reasoning_content': reasoning_content} if reasoning_content else {}
        )
        
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

# Use the custom wrapper
llm_with_reasoning = ReasoningChatOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    temperature=0
)

messages = [HumanMessage(content="9.11 and 9.8, which is greater?")]
response = llm_with_reasoning.invoke(messages)

print("\nUsing custom wrapper:")
print("content:", response.content)
if 'reasoning_content' in response.additional_kwargs and response.additional_kwargs['reasoning_content']:
    print("reasoning_content:", response.additional_kwargs['reasoning_content'])
