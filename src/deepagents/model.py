import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from openai import OpenAI

class ReasoningChatOpenAI(ChatOpenAI):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Convert LangChain messages to OpenAI format
        openai_messages = [{"role": "user", "content": msg.content} for msg in messages if hasattr(msg, 'content')]
        
        # Filter out tool-related kwargs that might not be supported
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['tools', 'tool_choice']}
        
        # Use OpenAI client directly
        client = OpenAI(api_key=self.openai_api_key, base_url=self.openai_api_base)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            temperature=self.temperature,
            **filtered_kwargs
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


def get_default_model():
    """Get the default model for the agent."""
    model_type = os.getenv("MODEL_TYPE", "gemini").lower()
    if model_type == "gemini":
        return get_gemini_model()
    elif model_type == "anthropic":
        return get_anthropic_model()
    elif model_type == "qwen":
        return get_qwen_model()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")    

def get_gemini_model():
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
        transport="rest",
        client_options={
            "api_endpoint": "https://generativelanguage.googleapis.com"
        },
        model_kwargs={
            "enable_thinking": True  # If you want to enable this feature,            
        }
    )
    """Get the Gemini model for the agent."""
    return llm

def get_anthropic_model():
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model_name="claude-sonnet-4-20250514", max_tokens=64000)

def get_qwen_model():
    llm_with_reasoning = ReasoningChatOpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        temperature=0
    )
    return llm_with_reasoning
