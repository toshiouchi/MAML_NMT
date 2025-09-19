# We applied the meta-learning MAML model to machine translation.

## Motivation for learning machine translation with MAML.

As reported on these pages

https://github.com/toshiouchi/MAML_ImgClassification

https://github.com/toshiouchi/MAML_sentimental_analysis/tree/main

https://github.com/toshiouchi/MAML_sentimental_analysis_CLS/tree/main

, we have applied the MAML model to image classification and sentiment analysis. 
We believe that we have achieved a reasonable level of accuracy for tasks with small amounts of data.
My next target was machine translation using MAML. I googled "machine translation MAML" and found a paper titled "Meta-Learning for Low-Resource Neural Machine Translation."

https://aclanthology.org/D18-1398.pdf

It was very helpful.
As stated in this paper, the motivation is to verify whether learning works by training MAML for languages ​​with sufficient training data, and then fine-tuning for languages ​​with less training data.

## Learning language and number of data

The training data used was

http://www.statmt.org/europarl/

For this training, we used 1 million French-English fr-en sentence pairs, 1 million German-English de-en sentence pairs, and 500,000 Italian-English it-en sentence pairs. We believe there is sufficient training data for fr-en and de-en, but not enough for it-en. We registered 1 million fr-en and 1 million de-en sentences with 500,000 it-en sentences twice to bring the total to 1 million, and trained the MAML model. Using the results, we measured the level of accuracy achieved by fine-tuning with 500,000 it-en sentence pairs.

## Machine translation model.

The machine translation model used was a Transformer Encoder-Decoder model. Since the encoder and decoder each have 8 layers, the LayerNorm arrangement used was Pre-LN. The embedding_dim = 768, heads = 12, layers = 8 * 2, feed_forwad_dim = 768 * 4.

## Lexicons and language embeddings.

The dictionary and language embedding were done in the same way as for one source language, one target language machine translation. The source dictionary was created by separating fr, de, and it with spaces. There is no language distinction between the words fr, de, and it. The target dictionary is en. Words that occur less frequently are designated <unk>. Special words are <pad>, <sos>, <eos>, <unk>, <blank>, and <mask>. As a result, the number of words in the source dictionary was 49,528. The number of words in the target language was 20,556. Embeddings were done using source language embeddings and target language embeddings.

## Positional encoding

Positional encoding was performed using Positional Embedding.

## Learning fr+de+it→en machine translation with MAML.

The loss and WER curves are shown below. The WER is approximately 0.36 for both Train and Val after 200,000 steps (2 epochs). Test results using autoregressive inference show a WER of 0.595. However, it took about 12 days to train 200,000 steps using a single RTX-A6000 processor.

<img width="638" height="498" alt="MainLoss" src="https://github.com/user-attachments/assets/dee74cc5-9772-4aee-b487-5fb773594628" />

<img width="641" height="494" alt="MainWER" src="https://github.com/user-attachments/assets/284b18e8-2a4d-464d-9278-10bf67012ad6" />

## Fine-tuning it→en translation with initial model parameters which have been trained abobe MAML fr+de+it→en.

The loss and WER curves are shown below. The WER is about 0.34 for both Train and Val after 100,000 steps (approximately 3 epochs). The test results using autoregressive inference show a WER of 0.592.

<img width="636" height="492" alt="FTLoss" src="https://github.com/user-attachments/assets/efa5b508-1f24-408e-a5ba-97df444fcfda" />

<img width="671" height="534" alt="FTWER" src="https://github.com/user-attachments/assets/a00671dd-e1fd-40d9-a6e4-3608c082eec2" />

## For comparison, the model parameters were initialized and trained normally on 500,000 Italian-English sentence pairs.

The loss and WER curves are shown below. The WER is around 0.45 after 124,000 steps (5 epochs). Test results using autoregressive inference show a WER of around 0.648.

<img width="646" height="500" alt="OrdLoss" src="https://github.com/user-attachments/assets/0cc4f5ea-aceb-4016-8f06-4bb5b249c281" />

<img width="651" height="500" alt="OrdWER" src="https://github.com/user-attachments/assets/266b2f6a-3dde-4e15-ae31-679d37dfc4a6" />

## Consideration

From the above, it can be considered that for languages ​​with little data, training with MAML using languages ​​with a lot of data has some advantages compared to training only with languages ​​with little data. The reason is that the WER for MAML + Fine Tuning is 0.592, while the WER for regular training is 0.648. In addition, while overfitting occurs with regular training, overfitting is avoided with MAML + Fine Tuning.

