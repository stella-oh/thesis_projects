
##############실행순서 ##############
1. data 폴더에 아래의 링크의 데이터를 넣으세요
2. preprocess.py
3. image_combined_query.ipynb
4. eval.py (대회제공 스크립트참조)


##########pretrained BERT dataset list##############
1. crawl-300d-2M.vec.zip(https://fasttext.cc/docs/en/english-vectors.html)
2. glove.840B.300d.zip(https://www.kaggle.com/takuok/glove840b300dtxt)
3. BERT-Base, Uncased(https://huggingface.co/transformers/pretrained_models.html)



##############대회제공 데이터셋 (약 30G)############## 
multimodal_train_sampleset.zip: (international) http://tianchi-public-us-east-download.oss-us-east-1.aliyuncs.com/231786/sample/multimodal_train_sampleset.zip
multimodal_labels.txt : (international) http://tianchi-public-us-east-download.oss-us-east-1.aliyuncs.com/231786/sample/multimodal_labels.txt
multimodal_train.zip : (international) http://tianchi-public-us-east-download.oss-us-east-1.aliyuncs.com/231786/multimodal_train.zip
multimodal_validpics.zip : (international) http://tianchi-public-us-east-download.oss-us-east-1.aliyuncs.com/231786/multimodal_validpics.zip
multimodal_valid.zip : (international) http://tianchi-public-us-east-download.oss-us-east-1.aliyuncs.com/231786/multimodal_valid.zip
multimodal_testA.zip : (international) http://tianchi-public-us-east-download.oss-us-east-1.aliyuncs.com/231786/multimodal_testA.zip
multimodal_submit_example_testA.zip : (international) http://tianchi-public-us-east-download.oss-us-east-1.aliyuncs.com/231786/multimodal_submit_example_testA.zip
multimodal_testB.zip(submit용 테스트셋) : (international) http://tianchi-public-us-east-download.oss-us-east-1.aliyuncs.com/231786/multimodal_testB.zip




##############파이썬 버전확인 ##############
python v.3.7.7

##############패키지 버전확인 ##############
tensorflow == 1.15.3
keras == 2.3.1
keras-bert == 0.84.0
cudatoolkit == 8.0
cudnn == 7.1.3
keras-embed-sim == 0.7.0
keras-pos-embd == 0.11.0
llvmlite == 0.32.1
swifter == 0.304
tqdm == 4.46.1


############## AWS cloud instance ##############
EC2 instance p2.xlarge
