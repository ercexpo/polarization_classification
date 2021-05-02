# polarization
=======
# BERT-based classifier for political polarization in English, Dutch, and Polish news articles:

In order to automatically classify online news articles as containing politically polarizing language or not, we developed a neural binary classifier built on top of a large transformer-based language model, namely BERT (Devlin et al.,2019).

BERT is a deep transformer model pre-trained on huge amounts of unlabeled text using a masked-word prediction training objective. By training on this objective with such a large amount of data, BERT builds a powerful language model which can then be fine-tuned to successfully complete specific language-related tasks, such as question answering and text classification. By pretraining the model on unlabeled text, the model is essentially taught to understand the target language. Then by fine-tuning, the model learns how to complete certain specific tasks with its knowledge of the language.

Given the BERT is trained on huge amounts of web data, including news articles, we did not feel that additional pre-training of the BERT model was necessary for the target domain. We utilize the Google's ‘bert-base-uncased’ pre-trained model for English, and the 'bert-base-multilingual-cased' model for our multilingual data. We implement our models using Huggingface’s Transformers package for Python. We encode our tokenized news articles using the pretrained BERT model. We then pass the BERT model’s final hidden layer to a linear softmax layer for binary-class prediction. 

Becuase BERT's input size is limited to 512 tokens, and the fact that some of our input articles exceed this limit, we are forced to truncate our articles prior to passing to our model. Rather than using the first 512 tokens when articles exceed the 512 token limit, we use the first and last 256 tokens in the article. This decision is based on our hypothesis that polarized statements tend to occur in the beginning or end of articles, rather than in the content-heavy middle. Our hypothesis was confirmed by improved model performance using this approach.

Our training dataset consists of 2278 articles coded by trained annotators for polarized language content - 782 from the United States, 738 from Poland, and 758 from the Netherlands. Additionally, we set aside 476 articles (approx. 20% of our dataset) for validation and testing.

Overall model results:

![ml_results](https://github.com/ercexpo/polarization_classification/blob/master/multilingual_results_polarization.png)
