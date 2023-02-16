# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/return_dataclass.ipynb (unless otherwise specified).

__all__ = ['QueryInput', 'Item', 'InputClass', 'Field', 'Result', 'Hits', 'SearchResponse', 'Body', 'Output']

# Cell
from dataclasses_json import dataclass_json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional
import pandas as pd


# Cell
@dataclass_json
@dataclass
class QueryInput:
    search_term : str
    num_items_to_return: int = 100
    brands: Optional[List[str]] = None
    filters: Optional[List[str]] = None

@dataclass_json
@dataclass
class Item:
    cprcode: str
    pr_engname: str
    similarity_str: Optional[str] = None
    pr_filter: Optional[List[str]] = None

@dataclass_json
@dataclass
class InputClass:
    '''
    input class, search function should take this as input
    '''
    queryStringParameters: QueryInput

# Cell
@dataclass_json
@dataclass
class Field:
    cprcode:List[str]
    pr_engname:List[str]
    pr_filter_en:Optional[List[str]]=None

@dataclass_json
@dataclass
class Result:
    id:str
    fields:Field
@dataclass_json
@dataclass
class Hits:
    hit:List[Result]

@dataclass_json
@dataclass
class SearchResponse:
    hits: Hits
    def result(self)->List[Field]: return [result.fields for result in self.hits.hit]
    def resultDict(self)->List[dict]:
        return [{'cprcode':r.parsedCprcode, 'name':r.name} for r in self.result]
    def resultDf(self)->pd.DataFrame:
        return pd.DataFrame(self.resultDict)
    def nameSortedDf(self)->pd.DataFrame:
        return self.resultDf.sort_values(by='name')


# Cell
@dataclass_json
@dataclass
class Body:
    '''output from your custom search functions'''
    search_results: List[Item]
    search_input: QueryInput
    # suggestions: List[str] = field(default_factory=list)

@dataclass_json
@dataclass
class Output:
    '''
    output class, search function should return this as output
    body should be json payload of class Body
    '''
    body: str #json payload of class Body
    statusCode: int = 200
    headers: dict = field(default_factory=lambda:
        { 'Access-Control-Allow-Headers': '*',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': '*'})