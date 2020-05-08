from haystack import Finder
from haystack.indexing.cleaning import clean_wiki_text
from haystack.indexing.io import write_documents_to_db, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.elasticsearch import ElasticsearchRetriever
import ipdb

document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

doc_dir = "test/samples/docs"
# s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
# fetch_archive_from_http(url=s3_url, output_dir=doc_dir)
write_documents_to_db(document_store=document_store, document_dir=doc_dir, clean_func=clean_wiki_text, only_empty_db=False, split_paragraphs=True)
retriever = ElasticsearchRetriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
finder = Finder(reader, retriever)
ipdb.set_trace()