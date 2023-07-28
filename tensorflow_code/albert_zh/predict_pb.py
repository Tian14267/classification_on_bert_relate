#coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np
import tokenization
from tqdm import tqdm
from time import time


class Bert_pred(object):
	def __init__(self,pb_path):
		self.tokenizer = tokenization.FullTokenizer(vocab_file="./pre_trained_model/albert_base_zh/vocab.txt", do_lower_case=True)
		self.max_seq_length =128
		with tf.Graph().as_default():
			self.output_graph_def = tf.GraphDef()
			with open(pb_path, "rb") as f:
				self.output_graph_def.ParseFromString(f.read())
				tf.import_graph_def(self.output_graph_def, name="")
			sess_config = tf.ConfigProto(allow_soft_placement=True)
			sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
			sess_config.gpu_options.allow_growth = True
			self.sess = tf.Session(config=sess_config)
			self.sess.run(tf.global_variables_initializer())
			# 定义输入的张量名称,对应网络结构的输入张量
			# input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
			self.input_ids = self.sess.graph.get_tensor_by_name("input_ids:0")
			self.input_mask = self.sess.graph.get_tensor_by_name("input_mask:0")
			# 定义输出的张量名称
			output_tensor_name = self.sess.graph.get_tensor_by_name("pred_prob:0")
			self.pred = tf.argmax(output_tensor_name, 1)

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

		print("################ load  model down! ##########################")


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


		features = {}

		features["input_ids"] = np.expand_dims(np.array(input_ids),axis=0)
		features["input_mask"] = np.expand_dims(np.array(input_mask),axis=0)
		features["segment_ids"] = np.expand_dims(np.array(segment_ids),axis=0)

		return features


	def _close(self):
		self.sess.close()

	def text_pre(self,input):
		features = self._covert_to_tensor(input)
		out=self.sess.run(self.pred, feed_dict={self.input_ids: features["input_ids"],self.input_mask: features["input_mask"]})
		result = self.label_map_id[out[0]]

		return result


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
			out=self.sess.run(self.pred, feed_dict={self.input_ids: features["input_ids"],self.input_mask: features["input_mask"]})
			result = self.label_map_id[out[0]]

			if result == one_line[1]:
				count = count + 1
		end = time()
		acc = count/len(lines_out)
		print("Time cost: ",(end-strat))
		print("average time cost: ", (end - strat)/len(lines_out))
		return acc



if __name__=="__main__":
	bertpred = Bert_pred(pb_path="./pb_model_dir/albert_base_zh.pb")
	#while 1:
		#inputs = ['龙岗区四联路30号路段多辆车辆违停,私设路障,严重影响车辆和行人通行。',
		#          '馨荔苑业主群，13：44分报警人发来短信：对不起，拨错号了，歉意。',
	    #        ]
	#print("开始输入：")
	#inputs = input()
	inputs = "没工作，没收入，"
	#pred = bertpred.text_pre(inputs)
	pred = bertpred.text_test()
	print("Result: ",pred)