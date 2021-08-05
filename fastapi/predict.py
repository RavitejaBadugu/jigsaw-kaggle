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
    test_inputs['input_ids']=np.expand_dims(np.cast(token_ids,dtype=np.int32),axis=0)
    test_inputs['attention_mask']=np.expand_dims(np.cast(mask,dtype=np.int32),axis=0)
    #test_inputs['input_ids'][0,]=test_inputs['input_ids'][0,].
    #test_inputs['attention_mask'][0,]=test_inputs['attention_mask'][0,].tolist()
    print(test_inputs)
    data=json.dumps({"signature_name": "serving_default",
     "instances": [test_inputs]})
    headers={"content-type": "application/json"}
    response=json.loads(requests.post(URL,data=data,headers=headers))['predictions']
    return response


