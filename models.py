# models
from transformers import AutoTokenizer

def ret_tokenizer():
  # Load the cardiffnlp/twitter-roberta-base-sentiment tokenizer.
  print('cardiffnlp/twitter-roberta-base-sentiment tokenizer...')
  tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

  return tokenizer



'''
Original:  Our friends won't buy this analysis, let alone the next one we propose.
Tokenized:  ['our', 'friends', 'won', "'", 't', 'buy', 'this', 'analysis', ',', 'let', 'alone', 'the', 'next', 'one', 'we', 'propose', '.']
Token IDs:  [2256, 2814, 2180, 1005, 1056, 4965, 2023, 4106, 1010, 2292, 2894, 1996, 2279, 2028, 2057, 16599, 1012]
'''

# models
from transformers import AdamW, BertConfig
from transformers import AutoModelForSequenceClassification
def ret_model(args):
    test_model_name = args.test_model_name
    model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment",
        num_labels = 3,
        output_attentions = False, # 모델이 어탠션 가중치를 반환하는지 여부.
        output_hidden_states = False, # 모델이 all hidden-state를 반환하는지 여부.
    )

    return model

