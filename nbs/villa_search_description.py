import openai, sys
from openai.error import RateLimitError
import pandas as pd
from googletrans import Translator
from nicHelper.secrets import getSecret
from diskcache import Cache

current_key_dict = {"current_key": 0}
cache = Cache(directory='/tmp/')
#init data

@cache.memoize(tag='translateCache')
def translate(text):
    '''translate from thai to english'''
    if not text: return ''
    translator = Translator()
    return translator.translate(text, dest='en').text

def dropUnusedColumn(df)->pd.DataFrame:
    df1 = df.drop(['iprcode', 'oprcode', 'ordertype','pr_cgcode', 'pr_dpcode','pr_ggcode','pr_sa_method','pr_sucode1',
        'pr_suref3','prtype','pstype','depth','product_attribute_id', 'pr_country_th','warehouse','consign_inv','product_attribute_images',
        'related_products',	'enabled',	'preorder_delivery_type',	'preorder_fix_date',	'preorder_relative_day',	'priority_score','plu_no',	'sort_cat_sku',
        'avail_nationwide',	'portion_size',	'portion','weight',	'psqty',	'pr_use_original_img',	'max_qty_in_cart','height',	'width','dept',
        'sort_weight', 'salemode_unit', 'ba_nprice', 'sort_villa_sku','pr_abb','pr_name', 'pr_market','hema_sizedesc','pr_barcode', 'pr_barcode2',
        'pr_brand_th',
        ], axis=1)
    return df1

@cache.memoize(tag='getDescriptionCache')
def getDescription(inputText:str):
    try:
        r = openai.Completion.create(engine="text-davinci-003", prompt=inputText, max_tokens=500, temperature=.08)
        return r["choices"][0]["text"].replace("\n", "")
    except RateLimitError as e:
        CURRENT_KEY += 1
        openai.api_key = getSecret('openai')['keys'][CURRENT_KEY]
        return getDescription(inputText)
    except Exception as e:
        print(e)




current_key_dict = {"current_key": 0}

if __name__ == "__main__":

    current_key_dict['current_key'] = int(sys.argv[1])
    openai.api_key = getSecret('openai')['keys'][current_key_dict['current_key']]

    df = pd.read_csv('/home/sam/Repos/openAIInterface/nbs/dynamodb_export_full.csv')
    df1 = dropUnusedColumn(df.head(300))
    df1 = df1[df1["master_online"] == True]
    total_length = len(df1)
    print(total_length)
    individual_length = int(total_length/10)
    df1 = df1.head((current_key_dict['current_key']+1)*individual_length)
    df1 = df1.tail(individual_length)
    print("df1 length", len(df1))
    df1['translated_keyword'] = df1.pr_keyword_th.apply(translate)
    df1['translated_name1'] = df1.pr_name_th.apply(translate)
    df1['translated_name2'] = df1.hema_name_th.apply(translate)
    df1['translated_name3'] = df1.pr_online_name_th.apply(translate)
    df1['translated_content'] = df1.content_th.apply(translate)
    df1 = df1.drop(['pr_keyword_th', 'pr_name_th', 'hema_name_th', 'pr_online_name_th','content_th', 'hema_brand_th',], axis=1)
    df1.fillna('', inplace=True)
    df1['seoInputText'] = df1.apply(lambda x: f'please make a seo tag for this product, including 200 words description include the title and meta tag\n\
                           product name: {x.pr_engname}\n, alternative name:{x.hema_name_en}\n, alternative name2:{x.pr_online_name_en}\n, \
                            product country {x.pr_country_en}\n product keywords are {x.pr_keyword_en + x.translated_keyword + x.pr_filter_en + x.pr_filter_th}\n \
                            product categories are {",".join( (x.online_category_l1_en , x.online_category_l2_en , x.online_category_l3_en, x.villa_category_l1_en, x.villa_category_l2_en, x.villa_category_l3_en,x.villa_category_l4_en))}\n\
                            product contents are {x.content_en + x.translated_content + x.product_detail_description}\n\
                            product metas are {x.meta_title + x.meta_keywords + x.meta_description}\n\
                            product brand is {x.pr_brand_en}\n\
                             '
                            , axis=1)
    
    df1['seoOutputText'] = df1.seoInputText.apply(getDescription)

    df1.to_csv(f"/home/sam/Repos/openAIInterface/nbs/dynamodb_export_full_output_{current_key_dict['current_key']}.csv", index=False)
    

