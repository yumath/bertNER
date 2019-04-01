# BERT fine-tune实践

1. #### 下载模型&代码

   BERT代码及模型下载地址：https://github.com/google-research/bert

2. #### 保存路径

3. #### 加载模型参数

4. #### 将输入Tokenize

   ```python
   def convert_single_example(char_line, tag_to_id, max_seq_length, tokenizer, label_line):
       """
       将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
       :param ex_index: index
       :param example: 一个样本
       :param label_list: 标签列表
       :param max_seq_length:
       :param tokenizer:
       :param mode:
       :return:
       """
       text_list = char_line.split(' ')
       label_list = label_line.split(' ')
   
       tokens = []
       labels = []
       for i, word in enumerate(text_list):
           # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），可以将所有的分字操作替换为list(input)
           token = tokenizer.tokenize(word)
           tokens.extend(token)
           label_1 = label_list[i]
           for m in range(len(token)):
               if m == 0:
                   labels.append(label_1)
               else:  # 一般不会出现else
                   labels.append("X")
       # 序列截断
       if len(tokens) >= max_seq_length - 1:
           tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
           labels = labels[0:(max_seq_length - 2)]
       ntokens = []
       segment_ids = []
       label_ids = []
       ntokens.append("[CLS]")  # 句子开始设置CLS 标志
       segment_ids.append(0)
       # append("O") or append("[CLS]") not sure!
       label_ids.append(tag_to_id["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
       for i, token in enumerate(tokens):
           ntokens.append(token)
           segment_ids.append(0)
           label_ids.append(tag_to_id[labels[i]])
       ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
       segment_ids.append(0)
       # append("O") or append("[SEP]") not sure!
       label_ids.append(tag_to_id["[SEP]"])
       input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
       input_mask = [1] * len(input_ids)
   
       # padding
       while len(input_ids) < max_seq_length:
           input_ids.append(0)
           input_mask.append(0)
           segment_ids.append(0)
           # we don't concerned about it!
           label_ids.append(0)
           ntokens.append("**NULL**")
   
       return input_ids, input_mask, segment_ids, label_ids
   ```

5. #### 获得embedding

   在上一过程中，将文本数据处理为以下函数的三个输入变量

   ```python
   # load bert embedding
   bert_config = modeling.BertConfig.from_json_file("chinese_L-12_H-768_A-12/bert_config.json")  # 配置文件地址。
   model = modeling.BertModel(
       config=bert_config,
       is_training=True,
       input_ids=self.input_ids,
       input_mask=self.input_mask,
       token_type_ids=self.segment_ids,
       use_one_hot_embeddings=False)
   embedding = model.get_sequence_output()
   ```

6. #### 冻结bert参数层

   ```python
   # bert模型参数初始化的地方
   init_checkpoint = "chinese_L-12_H-768_A-12/bert_model.ckpt"
   # 获取模型中所有的训练参数。
   tvars = tf.trainable_variables()
   # 加载BERT模型
   (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,init_checkpoint)
   tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
   print("**** Trainable Variables ****")
   # 打印加载模型的参数
   train_vars = []
   for var in tvars:
       init_string = ""
       if var.name in initialized_variable_names:
           init_string = ", *INIT_FROM_CKPT*"
       else:
           train_vars.append(var)
           print("  name = %s, shape = %s%s", var.name, var.shape,
                     init_string)
   grads = tf.gradients(self.loss, train_vars)
   (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
   
   self.train_op = self.opt.apply_gradients(
       zip(grads, train_vars), global_step=self.global_step)
   ```

7. #### 开始训练task层