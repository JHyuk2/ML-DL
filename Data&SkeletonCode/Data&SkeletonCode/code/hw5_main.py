# -*- coding: utf-8 -*-

from hw5_util import *

class CNN(torch.nn.Module):
    def __init__(self, vocab_size, output_dim):
        super(CNN, self).__init__()
        ##################################
        #           과제 입력            #
        ##################################

    def forward(self, inputs):
        ##################################
        #           과제 입력            #
        ##################################
        return output

class CNN_util(NLP_util):
    def create_w2i_l2i_i2l(self, data):
        """
            *** 입력된 train data를 사용하여 self.word2idx를 구축하고, self.labels를 사용하여 self.labels2idx, self.idx2labels을 구축하시면 됩니다. ***

            1) data는 train_data로, 아래와 같이 dict 자료형으로 구성되어 있습니다.
                {
                    "1":[
                            [
                                "user",
                                "아름아 그동안 잘 있었어?",
                                "opening"
                            ],
                            ...생략...
                        ]
                    
                    ...생략...
                }
            
            2) self.komoran을 사용하여 입력된 모든 발화를 형태소 분석한 후, word2idx를 구축하여, self.word2idx에 저장하시면 됩니다.

            3) self.labels는 아래와 같이 list 자료형이며, 화행 분석의 모든 레이블이 있습니다.
                self.labesl = ['ack', 'affirm', ... , 'yn-question']

            4) self.labels를 사용하여, labels2idx 및 idx2labels을 구축하여, 각각 self.labels2idx와 idx2labels에 저장하시면 됩니다.
        """
        return

    def convert_examples_to_features(self, data, max_sequence_length=50):
        input_ids = list()
        labels = list()
        """
            *** self.word2idx, self.labels2idx를 사용하여 입력된 data의 발화와 label을 index형태로 변환하시면 됩니다. ***

            1) data는 train_data 또는 test_data로, 아래와 같이 dict 자료형으로 구성되어 있습니다.
                {
                    "1":[
                            [
                                "user",
                                "아름아 그동안 잘 있었어?",
                                "opening"
                            ],
                            ...생략...
                        ]
                    
                    ...생략...
                }

            2) 구축된 self.word2idx, self.labels2idx를 사용하여 입력된 data의 발화와 label을 index 형태로 변환하시면 됩니다.

            3) input_ids는 2차원 list로, 입력된 data의 모든 발화를 index 형태로 변환하여 return 하시면 됩니다.

            4) labels는 1차원 list로, 입력된 data의 모든 발화의 label을 index 형태로 변환하여 return 하시면 됩니다.

            5) 입력된 문장의 sequence 길이를 맞춰주기 위해 본 과제에서는 max_sequence_length를 50 으로 설정하며,
               입력된 문장의 sequence 길이가 max_sequence_length 보다 짧은 경우, "<PAD>"을 사용하여 max_sequence_length을 맞춘다.
        """
        return input_ids, labels

    def eval(self, pred, label, dialogue_number):
        result = dict()
        """
            *** 입력된 값을 사용하여 모델을 성능을 평가하시면 됩니다. ***

            1) pred 변수는 모델이 예측한 label의 index로 구성된 1차원 리스트 입니다.
                [0, 2, 5, ..., 2]

            2) label 변수는 실제 정답 label로 구성된 1차원 리스트 입니다.
                [3, 2, 5, ..., 6]

            3) self.labels는 아래와 같이 화행 분석의 모든 레이블이 있습니다.
                self.labesl = ['ack', 'affirm', ... , 'yn-question']

            1, 2, 3)과 사전에 구축한 self.idx2labels을 사용하여 evaluation을 진행하시면 됩니다.

            4) self.data_dict는 test_data로, 아래와 같이 dict 자료형으로 구성되어 있습니다.
                {
                    "261":[
                                [
                                    "user",
                                    "아름아 그동안 잘 있었어?",
                                    "opening"
                                ],
                                ...생략...
                          ]
                    
                    ...생략...
                }

            5) argument로 입력받은 대화 번호(dialogue_number)를 사용하시면 됩니다.

            6) result 변수는 아래와 같이 만드시면 됩니다.
               ** 모델의 성능 부분을 백분율로 만든 후, result에 저장해야 됩니다. **

                result['micro precision'] = 모델의 micro averaging precision (float 자료형)
                result['macro precision'] = 모델의 macro averaging precision (float 자료형)

                result['micro recall'] = 모델의 micro averaging recall (float 자료형)
                result['macro recall'] = 모델의 macro averaging recall (float 자료형)

                result['micro f1'] = 모델의 micro averaging F1-score (float 자료형)
                result['macro f1'] = 모델의 macro averaging F1-score (float 자료형)

                result['dialogue number'] = 입력 받은 dialogue_number (int 자료형)

                result['matching'] = 입력 받은 dialogue_number의 모든 발화에 대한 매칭 결과가 tuple 자료형으로 되어있어야 합니다. (list 자료형)
                Ex) result['matching'] = [('아름아 잘 잤니?', 'opening', 'opening', 'True'),  
                                          ('아름아 일정 확인 좀 해줘', 'request', 'wh-question', 'False'),
                                          ...생략...
                                          ('해당 발화', '실제 레이블', '예측 레이블', '실제 레이블과 예측 레이블 비교 결과')
                                         ]

                result['matching']의 경우 해당 대화의 발화순으로 되어 있어야 합니다.

            모든 소수 계산과정에서 반올림, 올림, 버림은 하지 않습니다.
        """
        return result

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dialogue-number",
        type=int,
        required=True,
        dest="dialogue_number",
        choices=range(261, 301)
    )
    args = parser.parse_args()
    
    # 입력 JSON 파일 경로
    train_inpath = './SpeechAct_tr.json' 
    test_inpath = './SpeechAct_te.json'

    # ======= [여기를 수정하세요] =======
    std_name = '홍길동'
    std_ID = '2020123456'

    """
    * HyperParameter *

        loss_function_name은 string 자료형으로 다음 중 하나를 선택하시면 됩니다.
        ['MSELoss', 'CrossEntropyLoss', 'NLLLoss']

        optimizer_name string 자료형으로 다음 중 하나를 선택하시면 됩니다.
        ['Adam', 'AdamW', 'RMSprop', 'SGD']

        learning_rate는 float 자료형으로 입력하시면 됩니다.

        train_batch_size는 int 자료형으로 입력하시면 됩니다.

        num_train_epochs는 int 자료형으로 입력하시면 됩니다.
    """
    loss_function_name = str
    optimizer_name = str
    learning_rate = float
    train_batch_size = int
    num_train_epochs = int
    # ======= [여기까지 수정하세요] =======

    processing = CNN_util()
    processing.load_data(train_inpath)
    processing.load_data(test_inpath)
    model = CNN(len(processing.word2idx), len(processing.labels))
    processing.train(model, loss_function_name, optimizer_name, learning_rate, train_batch_size, num_train_epochs)
    pred, label = processing.predict(model)
    result = processing.eval(pred, label, args.dialogue_number)
    processing.save_result(result, std_name, std_ID)

if __name__ == "__main__":
    main()