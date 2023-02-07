import functools
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from copy import deepcopy
from awsSchema.apigateway import Event, Response
from openai.embeddings_utils import get_embedding, cosine_similarity
from nicHelper.secrets import getSecret
import openai, time


class OpenaiSearch:
    def __init__(self, df, api_key):
        openai.api_key = api_key
        self.df = df
        start_time = time.time()
        self.embeddings = np.stack(df.embedding.values)
        print(f"Time taken to get embeddings: {time.time() - start_time} seconds")
        
    @functools.lru_cache(maxsize=1)
    def search(self, product_description, n=5):
        start_time = time.time()
        product_embedding = get_embedding(product_description, engine="text-embedding-ada-002")
        embedding_time = time.time() - start_time
        print(f"Time taken to get product embedding: {embedding_time} seconds")

        start_time = time.time()
        self.df['similarity'] = cosine_similarity(self.embeddings, product_embedding)
        similarity_time = time.time() - start_time
        print(f"Time taken to calculate cosine similarity: {similarity_time} seconds")

        start_time = time.time()
        results = self.df.sort_values("similarity", ascending=False).head(n)
        sorting_time = time.time() - start_time
        print(f"Time taken to sort results: {sorting_time} seconds")

        return results


def parse_event(event):
    eventCopy = deepcopy(event)
    body = Event.parseBody(eventCopy)
    try:
        search_term = body['search_term']
    except Exception as e:
        print(e)
        raise Exception('path is not in body')

    return search_term
    

def search(event, *args):
    try:
        start_time = time.time()
        api_key = getSecret('openai')['key']
        print(f"took {time.time() - start_time} seconds to get secret")
        start_time = time.time()
        df = feather.read_feather('/villa_database_with_float32_embeddings.feather')
        print(f"took {time.time() - start_time} seconds to read feather")
        openai_search = OpenaiSearch(df, api_key)
        start_time = time.time()
        search_term = parse_event(event)
        print(f"took {time.time() - start_time} seconds to parse event")
        results = openai_search.search(search_term)
        return_body = {"results": {"1":{"cprcode":str(results.iloc[0].cprcode), "pr_engname":results.iloc[0].pr_engname},
                                    "2":{"cprcode":str(results.iloc[1].cprcode), "pr_engname":results.iloc[1].pr_engname}, 
                                    "3":{"cprcode":str(results.iloc[2].cprcode), "pr_engname":results.iloc[2].pr_engname}, 
                                    "4":{"cprcode":str(results.iloc[3].cprcode), "pr_engname":results.iloc[3].pr_engname}, 
                                    "5":{"cprcode":str(results.iloc[4].cprcode), "pr_engname":results.iloc[4].pr_engname}}}
        return Response.returnSuccess(body=return_body)
    except Exception as e:
        print(e)
        return Response.returnError(str(e))