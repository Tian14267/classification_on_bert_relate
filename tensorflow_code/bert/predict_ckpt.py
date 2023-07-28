#coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np
import tokenization
from tqdm import tqdm
from time import time
import modeling


class Bert_pred_ckpt(object):
	def __init__(self):
		self.max_seq_length = 128
		self.input_ids = tf.placeholder(tf.int32, [None, self.max_seq_length], name='input_ids')
		self.input_mask = tf.placeholder(tf.int32, [None, self.max_seq_length], name='input_mask')

		self.tokenizer = tokenization.FullTokenizer(vocab_file="./pretrain_models/chinese_L-12_H-768_A-12/vocab.txt",
													do_lower_case=True)

		bert_config = modeling.BertConfig.from_json_file("./pretrain_models/chinese_L-12_H-768_A-12/bert_config.json")
		num_labels = 97
		logits, probabilities = self._create_model(
			bert_config=bert_config, is_training=False,
			input_ids=self.input_ids, input_mask=self.input_mask,
            num_labels=num_labels, use_one_hot_embeddings=False)

		self.probabilities = tf.argmax(probabilities, 1)

		saver = tf.train.Saver()
		sess_config = tf.ConfigProto(allow_soft_placement=True)

		sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
		sess_config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=sess_config)
		model_path = 'outputs/selfdata_bert_base_zh_output_epoch12/model.ckpt-3000'
		saver.restore(sess=self.sess, save_path=model_path)
		print("################ load  model down! ##########################")

		if os.path.exists("./data/self_data/label.txt"):
			with open("./data/self_data/label.txt", "r", encoding="utf-8") as f:
				lines = f.readlines()
				lines_out = []
				for line in lines:
					line = line.replace("\n", "")
					lines_out.append(line)
			f.close()
			self.label_map = {}
			self.label_map_id = {}
			for i, one_label in enumerate(lines_out):
				self.label_map[one_label] = i
				self.label_map_id[i] = one_label
		###################################################################################################


	def _create_model(self,bert_config, is_training, input_ids, input_mask,
					 num_labels, use_one_hot_embeddings):
		"""Creates a classification model."""
		model = modeling.BertModel(
			config=bert_config,
			is_training=is_training,
			input_ids=input_ids,
			input_mask=input_mask,
			use_one_hot_embeddings=use_one_hot_embeddings)

		output_layer = model.get_pooled_output()

		hidden_size = output_layer.shape[-1].value

		output_weights = tf.get_variable(
			"output_weights", [num_labels, hidden_size],
			initializer=tf.truncated_normal_initializer(stddev=0.02))

		output_bias = tf.get_variable(
			"output_bias", [num_labels], initializer=tf.zeros_initializer())

		with tf.variable_scope("loss"):
			if is_training:
				# I.e., 0.1 dropout
				output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

			logits = tf.matmul(output_layer, output_weights, transpose_b=True)
			logits = tf.nn.bias_add(logits, output_bias)
			probabilities = tf.nn.softmax(logits, axis=-1)

			return logits, probabilities

	def _close(self):
		self.sess.close()

	def _covert_to_tensor(self,sentence):
		sentence_token = self.tokenizer.tokenize(sentence)
		if len(sentence_token) > self.max_seq_length - 2:
			sentence_token = sentence_token[0:(self.max_seq_length - 2)]

		tokens = []
		segment_ids = []
		tokens.append("[CLS]")
		segment_ids.append(0)
		for token in sentence_token:
			tokens.append(token)
			segment_ids.append(0)
		tokens.append("[SEP]")
		segment_ids.append(0)

		input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
		input_mask = [1] * len(input_ids)

		while len(input_ids) < self.max_seq_length:
			input_ids.append(0)
			input_mask.append(0)
			segment_ids.append(0)

		assert len(input_ids) == self.max_seq_length
		assert len(input_mask) == self.max_seq_length
		assert len(segment_ids) == self.max_seq_length

		def create_int_feature(values):
			f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
			return f

		features = {}

		features["input_ids"] = np.expand_dims(np.array(input_ids), axis=0)
		features["input_mask"] = np.expand_dims(np.array(input_mask), axis=0)


		return features

	def text(self,input):
		features = self._covert_to_tensor(input)

		pred = self.sess.run([self.probabilities], feed_dict={self.input_ids:features["input_ids"],
															  self.input_mask:features["input_mask"]})
		return pred

	def text_test(self):
		with open("./data/self_data/test_1k.txt", "r", encoding="utf-8") as f:
			lines = f.readlines()
			lines_out = []
			for line in lines:
				line = line.replace("\n", "")
				line_split = line.split("	")
				lines_out.append(line_split)
		f.close()
		strat = time()
		count = 0
		for one_line in tqdm(lines_out):
			features = self._covert_to_tensor(one_line[0])
			out = self.sess.run([self.probabilities], feed_dict={self.input_ids: features["input_ids"],
																  self.input_mask: features["input_mask"]})
			result = self.label_map_id[out[0][0]]

			if result == one_line[1]:
				count = count + 1
		end = time()
		acc = count/len(lines_out)
		print("Time cost: ",(end-strat))
		print("average time cost: ", (end - strat)/len(lines_out))
		return acc


if __name__=="__main__":
	bertpred = Bert_pred_ckpt()
	inputs = "没工作，没收入，"
	#pred = bertpred.text(inputs)[0]
	pred = bertpred.text_test()
	print("Result: ",pred)