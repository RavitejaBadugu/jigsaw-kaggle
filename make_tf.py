from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,BatchNormalization,ReLU,Dropout,Input
import tensorflow as tf
from transformers import TFXLMRobertaModel,XLMRobertaTokenizer



sent_length=210
xlmroberta_model=TFXLMRobertaModel.from_pretrained('jplu/tf-xlm-roberta-base')
ins1=Input((sent_length,),dtype=tf.int32,name='input_ids')
ins2=Input((sent_length,),dtype=tf.int32,name='attention_mask')
pre_model=xlmroberta_model({'input_ids':ins1,'attention_mask':ins2})
x=Dropout(0.1)(pre_model[1])
outs=Dense(1,activation='sigmoid')(x)
model=Model(inputs={'input_ids':ins1,'attention_mask':ins2},outputs=outs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),loss='binary_crossentropy')
model.load_weights('./xlm_roberta_jigsaw_weights.h5')

tf.saved_model.save(model,export_dir='tf_serving/models/1/')

x_tokenizer=XLMRobertaTokenizer.from_pretrained('jplu/tf-xlm-roberta-base')
x_tokenizer.save_pretrained('fastapi/tokenizer/')