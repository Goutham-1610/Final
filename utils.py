from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SummaryIndex, PromptTemplate, SimpleDirectoryReader
from llama_index.readers.file import IPYNBReader, FlatReader, MarkdownReader, HTMLTagReader
from llama_index.core.node_parser import SentenceSplitter
import os
from dotenv import load_dotenv
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.objects import ObjectIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from subprocess import Popen, PIPE
import re, subprocess




class CodeReviewerBot:


    def __init__(self, file_path) -> None:
        self.file_path = file_path
        if not os.path.exists(file_path):
            print('File path not valid')
            exit()
        load_dotenv()


        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

    def files_in_codebase(self):
        fnames = []
        for path in os.listdir(self.file_path):
            if path != '.github' and path != '.git':
                temp = ''
                for i in path:
                    if i == '.':
                        break
                    temp += i
                path = temp
                fnames.append(path)
        return fnames

    def load_doc(self):
        notebook_parser = IPYNBReader()
        normal_parser = FlatReader()
        readme_parser = MarkdownReader()
        html_parser = HTMLTagReader()
        file_extractor = {".ipynb": notebook_parser, ".py": normal_parser, '.md': readme_parser, '.html': html_parser, ".css": normal_parser, ".js": normal_parser}
        node = []

        for path in os.listdir(self.file_path):
            pattern = r'\.(.*)$'
            match = re.search(pattern, path)
            if match != '.github' and match != '.git':
                file_path1 = os.path.join(self.file_path, path)
                splitter = SentenceSplitter(chunk_size=2000, chunk_overlap=200)
                node.append(splitter.get_nodes_from_documents(docs))
        return node

    def create_doc_agent(self, node):
        node = node
        file_name = self.files_in_codebase()

        query_engines = {}
        agents = {}
        all_nodes = []
        for fname, n in zip(file_name, node):
            all_nodes.extend(n)
            if not os.path.exists(f"./data/{fname}"):
                vector_index.storage_context.persist(
                    persist_dir=f"./data/{fname}"
                )
            else:
                vector_index = load_index_from_storage(
                    StorageContext.from_defaults(persist_dir=f"./data/{fname}"),
                    embed_model=Settings.embed_model
                )
            summary_index = SummaryIndex(n)
            summary_query_engine = summary_index.as_query_engine(llm=self.llm)

            # define tools
            query_engine_tools = [
                QueryEngineTool(
                    query_engine=vector_query_engine,
                    metadata=ToolMetadata(name="vector_tool",
                    description=(
                        "Useful for questions related to specific aspects of"
                        f" {fname}"
                        ),)
                ),
                QueryEngineTool(
                    query_engine=summary_query_engine,
                    description=(
                    ),)
                )
            ]

            # build agent
            agent = ReActAgent.from_tools(
                query_engine_tools,
                llm=self.llm,
            )
            agents[fname] = agent
            query_engines[fname] = vector_index.as_query_engine(
                similarity_top_k=3,
                llm=self.llm,
            )

        return agents

    def create_agents_tools(self):
        node = self.load_doc()
        agents = self.create_doc_agent(node)
        all_tools = []
        file_name = self.files_in_codebase()
        for fname in file_name:
            fname_summary = (
            )
            doc_tool = QueryEngineTool(
                query_engine=agents[fname],
                metadata=ToolMetadata(
                    name=f"tool_{fname}",
                    description=fname_summary,
                ),
            )
            all_tools.append(doc_tool)
        print(f"Tools available are :\n {all_tools}")
        return all_tools

    def main_agent(self, prompt=None):
        all_tools = self.create_agents_tools()

        obj_index = ObjectIndex.from_objects(
            all_tools,
            index_cls=VectorStoreIndex,
            embed_model=Settings.embed_model
        )

        file_name = self.files_in_codebase()
        file_str = " ,".join(f for f in file_name)
        context =  f"""
            Role: You are an Expert Code Reviewer
            System:
            Understand the code."""
        
        react_llm = self.llm
        agent1 = ReActAgent.from_tools(tool_retriever=obj_index.as_retriever(similarity_top_k=3),
                              llm=react_llm, context=context, verbose=True, max_iterations=100)
        if prompt is None:
            prompt = f"""""

        result = agent1.query(prompt)
        print('---------------------')
        print(result)
        return str(result)

    # Rest of the methods remain unchanged.
