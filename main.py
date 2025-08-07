import argparse
import os
import uuid
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler

load_dotenv()
os.environ["CURL_CA_BUNDLE"] = "/home/solmon/github/questmind/zscaler_root.crt"
os.environ["REQUESTS_CA_BUNDLE"] = "/home/solmon/github/questmind/zscaler_root.crt"
os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = "/home/solmon/github/questmind/zscaler_root.crt"


def main():

    parser = argparse.ArgumentParser(description="Run research agent")
    parser.add_argument("query", type=str, nargs="?", default="find best recipe of sourdough bread", help="Research query")
    parser.add_argument("--max_results", type=int, default=5, help="Maximum number of search results")
    parser.add_argument("--topic", type=str, choices=["general", "news", "finance"], default="general", help="Search topic")
    parser.add_argument("--include_raw_content", action="store_true", help="Include raw content in search results")
    parser.add_argument("--trace", action="store_true", help="Enable Langfuse tracing")
    args = parser.parse_args()

    from tavily import TavilyClient
    from src.deepagents import create_deep_agent

    def internet_search(
        query: str,
        max_results: int = 5,
        topic: str = "general",
        include_raw_content: bool = False,
    ):
        """
        Run a web search using Tavily.
        Args:
            query (str): The search query.
            max_results (int): Maximum number of results to return.
            topic (str): Search topic (general, news, finance).
            include_raw_content (bool): Whether to include raw content in results.
        Returns:
            dict: Search results from Tavily API.
        """
        tavily_async_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        return tavily_async_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )

    # research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

    # You have access to a few tools.

    # ## `internet_search`

    # Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
    # """

    research_instructions = """You are an expert researcher. Your job is to conduct thorough research on the internet for a given task, and then use the information gathered to recommend a shopping cart for purchase.

    You have access to a few tools.

    ## `internet_search`

    Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
    """



    agent = create_deep_agent(
        [internet_search],
        research_instructions,
    )

    invoke_kwargs = {
        "messages": [{"role": "user", "content": args.query}],
        "max_results": args.max_results,
        "topic": args.topic,
        "include_raw_content": args.include_raw_content,
    }

    if args.trace:
        langfuse_handler = CallbackHandler()
        config = {"configurable": {"thread_id": str(uuid.uuid4())},"callbacks": [langfuse_handler]}
        result = agent.invoke(invoke_kwargs, config=config)
    else:
        result = agent.invoke(invoke_kwargs)
    print(result)

if __name__ == "__main__":
    main()


