# from datasets import load_dataset
# from haystack.document_stores import InMemoryDocumentStore
# from haystack.nodes import Crawler
#
# dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
#
# document_store = InMemoryDocumentStore(use_bm25=True)
# document_store.write_documents(dataset)

from haystack import Pipeline
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import Crawler, PreProcessor

document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

crawl = True
crawler_output_dir = "crawled_files"

if crawl:
    document_store.delete_documents()

    crawler = Crawler(
        output_dir=crawler_output_dir,
        urls=["https://www.uni-osnabrueck.de/en"],  # This tells the Crawler which URLs to crawl
        filter_urls=["uni-osnabrueck"],  # Here, you can pass regular expressions that the crawled URLs must comply with
        crawler_depth=2  # This tells the Crawler to follow only the links that it finds on the initial URLs
    ) # This tells the Crawler where to store the crawled files
    # crawler.crawl(
    #     urls=["https://www.uni-osnabrueck.de"], # This tells the Crawler which URLs to crawl
    #     filter_urls=["uni-osnabrueck"], # Here, you can pass regular expressions that the crawled URLs must comply with
    #     crawler_depth=1 # This tells the Crawler to follow only the links that it finds on the initial URLs
    # )

    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=250,
        split_respect_sentence_boundary=True,
    )

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_node(component=crawler, name="crawler", inputs=['File'])
    indexing_pipeline.add_node(component=preprocessor, name="preprocessor", inputs=['crawler'])
    indexing_pipeline.add_node(component=document_store, name="document_store", inputs=['preprocessor'])

    indexing_pipeline.run()


from haystack.nodes import PromptNode, PromptTemplate, AnswerParser, BM25Retriever
from haystack.pipelines import Pipeline

retriever = BM25Retriever(document_store=document_store, top_k=1) # top_k: Count of returned documents

prompt_template = PromptTemplate(
    prompt="""Answer the question truthfully based solely on the given documents. If the documents do not contain the answer to the question, say that answering is not possible given the available information. Your answer should be no longer than 50 words.
    Documents:{join(documents)}
    Question:{query}
    Answer:
    """,
    output_parser=AnswerParser(),
)

prompt_node = PromptNode(
    model_name_or_path="google/flan-t5-large", default_prompt_template=prompt_template
)

generative_pipeline = Pipeline()
generative_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
generative_pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

from haystack.utils import print_answers

while True:
    question = input('question: ')
    response = generative_pipeline.run(question)
    #print('Prompt: ' + response["answers"][0].meta["prompt"])
    print("Answer:" + response["answers"][0].answer)
    print("Found in " + ', '.join([document.meta["url"] for document in response['documents']]))
    #print(response)
    #print_answers(response, details="minimum")

# response = generative_pipeline.run("What does Taylor Swift look like?")
# print_answers(response, details="minimum")

# TODO: Test on larger model due to 512 tokens limit
# from haystack.agents import Tool
#
# search_tool = Tool(
#     name="seven_wonders_search",
#     pipeline_or_node=generative_pipeline,
#     description="useful for when you need to answer questions about the seven wonders of the world",
#     output_variable="answers",
# )
#
# from haystack.nodes import PromptNode
#
# agent_prompt_node = PromptNode(
#     "google/flan-t5-large",
#     max_length=256,
#     stop_words=["Observation:"],
#     model_kwargs={"temperature": 0.5},
# )
#
# from haystack.agents.memory import ConversationSummaryMemory
# from haystack.nodes import PromptNode
#
# memory_prompt_node = PromptNode(
#     "philschmid/bart-large-cnn-samsum", max_length=256, model_kwargs={"task_name": "text2text-generation"}
# )
# memory = ConversationSummaryMemory(memory_prompt_node, prompt_template="{chat_transcript}")
#
# agent_prompt = """
# In the following conversation, a human user interacts with an AI Agent. The human user poses questions, and the AI Agent goes through several steps to provide well-informed answers.
# The AI Agent must use the available tools to find the up-to-date information. The final answer to the question should be truthfully based solely on the output of the tools. The AI Agent should ignore its knowledge when answering the questions.
# The AI Agent has access to these tools:
# {tool_names_with_descriptions}
#
# The following is the previous conversation between a human and The AI Agent:
# {memory}
#
# AI Agent responses must start with one of the following:
#
# Thought: [the AI Agent's reasoning process]
# Tool: [tool names] (on a new line) Tool Input: [input as a question for the selected tool WITHOUT quotation marks and on a new line] (These must always be provided together and on separate lines.)
# Observation: [tool's result]
# Final Answer: [final answer to the human user's question]
# When selecting a tool, the AI Agent must provide both the "Tool:" and "Tool Input:" pair in the same response, but on separate lines.
#
# The AI Agent should not ask the human user for additional information, clarification, or context.
# If the AI Agent cannot find a specific answer after exhausting available tools and approaches, it answers with Final Answer: inconclusive
#
# Question: {query}
# Thought:
# {transcript}
# """
#
# from haystack.agents import AgentStep, Agent
#
# def resolver_function(query, agent, agent_step):
#     return {
#         "query": query,
#         "tool_names_with_descriptions": agent.tm.get_tool_names_with_descriptions(),
#         "transcript": agent_step.transcript,
#         "memory": agent.memory.load(),
#     }
#
# from haystack.agents.base import Agent, ToolsManager
#
# conversational_agent = Agent(
#     agent_prompt_node,
#     prompt_template=agent_prompt,
#     prompt_parameters_resolver=resolver_function,
#     memory=memory,
#     tools_manager=ToolsManager([search_tool]),
# )
#
# result = conversational_agent.run("What did Rhodes Statue look like?")
# print(result)
#
# result = conversational_agent.run("When did it collapse?")
# print(result)
#
# conversational_agent.run("How tall was it?")
#
# conversational_agent.run("How long did it stand?")




# from haystack.document_stores import InMemoryDocumentStore
#
# document_store = InMemoryDocumentStore(use_bm25=True)
#
# from haystack.utils import fetch_archive_from_http
#
# doc_dir = "data/build_your_first_question_answering_system"
#
# fetch_archive_from_http(
#     url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt1.zip",
#     output_dir=doc_dir,
# )
#
# import os
# from haystack.pipelines.standard_pipelines import TextIndexingPipeline
#
# files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
# indexing_pipeline = TextIndexingPipeline(document_store)
# indexing_pipeline.run_batch(file_paths=files_to_index)
#
# from haystack.nodes import BM25Retriever
#
# retriever = BM25Retriever(document_store=document_store)
#
# from haystack.nodes import FARMReader
#
# reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
#
# from haystack.pipelines import ExtractiveQAPipeline
#
# pipe = ExtractiveQAPipeline(reader, retriever)
#
# prediction = pipe.run(
#     query="Who is the father of Arya Stark?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
# )
#
# from pprint import pprint
#
# pprint(prediction)
#
# from haystack.utils import print_answers
#
# print_answers(prediction, details="minimum")  ## Choose from `minimum`, `medium`, and `all`



# import os
# import time
# from getpass import getpass
#
# # model_api_key = os.getenv("HF_API_KEY", None) or getpass("Enter HF API key:")
#
# from haystack.nodes import PromptNode
#
# model_name = "google/flan-t5-large"
# prompt_node = PromptNode(model_name, max_length=256)
#
# from haystack.agents.memory import ConversationSummaryMemory
#
# summary_memory = ConversationSummaryMemory(prompt_node)
#
# from haystack.agents.conversational import ConversationalAgent
#
# conversational_agent = ConversationalAgent(prompt_node=prompt_node, memory=summary_memory)
#
# result = conversational_agent.run("Tell me three most interesting things about England")
# print(result)
#
# result = conversational_agent.run("Can you tell me more about the culture?")
# print(result)
#
# result = conversational_agent.run("Can you turn this info into a poem?")
# print(result)
