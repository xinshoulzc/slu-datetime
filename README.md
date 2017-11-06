Current training parameters:
```
--data_dir data/Qiyi_intent --train_dir model_intent --max_sequence_length 20 --task intent --bidirectional_rnn False --fasttext_model /work/ml/wordvectors/zh.bin --max_training_steps 6000 --word_embedding_size 300 --batch_size 50 --size 300
--data_dir data/Qiyi_tagging --train_dir model_tagging --max_sequence_length 20  --task tagging --bidirectional_rnn False --fasttext_model /work/ml/wordvectors/zh.bin --max_training_steps 6000 --word_embedding_size 300 --size 300 // learning rate 0.3
```