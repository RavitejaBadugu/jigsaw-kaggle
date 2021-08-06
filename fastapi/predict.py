from transformers import XLMRobertaTokenizer
import numpy as np
import requests
import json
sent_length=210

x_tokenizer=XLMRobertaTokenizer.from_pretrained('tokenizer/')
URL='http://model_server:8501/v1/models/xlm_model:predict'

def get_predictions(text):
    test_inputs={}
    tokenized=x_tokenizer.tokenize(text)
    if len(tokenized)>sent_length-2:
        tokenized=['<s>']+tokenized[:(sent_length-2)]+['</s>']
    else:
        curr_len=len(tokenized)
        tokenized=['<s>']+tokenized+['</s>']+['<pad>']*(sent_length-curr_len-2)
    token_ids=x_tokenizer.convert_tokens_to_ids(tokenized)
    mask=np.char.not_equal('<pad>',tokenized).astype(np.int32)
    test_inputs['input_ids']=np.asarray(token_ids,dtype=np.int32).tolist()
    test_inputs['attention_mask']=np.asarray(mask,dtype=np.int32).tolist()
    data=json.dumps({"signature_name": "serving_default",
     "instances": [test_inputs]})
    headers={"content-type": "application/json"}
    response=requests.post(URL,data=data,headers=headers)
    response=json.loads(response.text)['predictions'][0][0]
    return response


