import os

def get_default_model():
    """Get the default model for the agent."""
    model_type = os.getenv("MODEL_TYPE", "gemini").lower()
    if model_type == "gemini":
        return get_gemini_model()
    elif model_type == "anthropic":
        return get_anthropic_model()
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
