import functools
import numpy as np, json
import pandas as pd
import pyarrow.feather as feather
from awsSchema.apigateway import Response
from openai.embeddings_utils import cosine_similarity
from nicHelper.secrets import getSecret
import openai, time
from threading import Thread
from return_dataclass import InputClass, Output, Body
from lambdasdk.lambdasdk import Lambda
import openAiSearchCache as searchCache


class OpenaiSearch:
    initialized: bool = False
    df: pd.DataFrame = None
    embeddings: np.ndarray = None

    def __init__(self, api_key):
        openai.api_key = api_key
        
        self.loadDFThread = Thread(target=self.load_dataframe)
        self.loadDFThread.start()
    
    def load_dataframe(self):
        if not OpenaiSearch.initialized:
            OpenaiSearch.df = feather.read_feather('/villa_database_with_float32_embeddings.feather')
            OpenaiSearch.embeddings = np.stack(self.df.embedding.values)
            OpenaiSearch.initialized = True
            print("just initialized")
        else:
            print("already initialized")
        
    @functools.lru_cache(maxsize=1)
    def search(self, search_term, n=100):
        product_embedding = Lambda().invoke("openai-embedding-experiment",{'query':search_term})
        self.loadDFThread.join()
        OpenaiSearch.df['similarity'] = cosine_similarity(OpenaiSearch.embeddings, product_embedding)
        results = OpenaiSearch.df.sort_values("similarity", ascending=False).head(n)
        results["similarity_str"] = results["similarity"].astype(str)
        return {"search_results": results.to_dict(orient="records")}
    

def search(event, *args):
    try:
        api_key = getSecret('openai')['key']
        openai_search = OpenaiSearch(api_key)
        input_ = InputClass.from_dict(event)
        search_term = input_.queryStringParameters.search_term
        n = int(input_.queryStringParameters.num_items_to_return)
        if result := next(searchCache.CacheTable.query(f"{search_term}-{n}"), None):
            print("returned from cache")
            return json.loads(result.search_result)
        return_dict = openai_search.search(search_term=search_term, n=n)
        return_dict["search_input"] = input_.queryStringParameters.to_dict()
        return_body = Body.from_dict(return_dict)
        print("return_body", return_body)
        final_output = Output(body=return_body.to_json()).to_dict()
        table = searchCache.CacheTable(key=f"{search_term}-{n}", search_result=json.dumps(final_output))
        table.save()
        return final_output
    except Exception as e:
        print("error: ", e)
        return Response.returnError(str(e))