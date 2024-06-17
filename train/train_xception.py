import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from network.models import model_selection
from network.mesonet import Meso4, MesoInception4
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset
# from dataloader import get_loader

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn import metrics

from dataset.dataloader import get_loader
from dataset.dataloader import get_loader_celebDF_fewimage, get_loader_celebDF_JYUNYI

def collate_fn(batch):
    imgs = [item['image'] for item in batch if item['image'] is not None]
    targets = [item['label'] for item in batch if item['image'] is not None]
    filenames = [item['filename'] for item in batch if item['image'] is not None]
    imgs = torch.stack(imgs)
    targets = torch.stack(targets)
    return {'image': imgs, 'label': targets, 'filename': filenames}

def main():
	args = parse.parse_args()
	name = args.name
	continue_train = args.continue_train
	# train_list = args.train_list
	# val_list = args.val_list
	epoches = args.epoches
	batch_size = args.batch_size
	model_name = args.model_name
	model_path = args.model_path
	output_path = os.path.join('./output', name)
	if not os.path.exists(output_path):
		os.mkdir(output_path)
	torch.backends.cudnn.benchmark=True
	# train_data, val_data, test_data, train_loader, val_loader, test_loader = get_loader(args)
	train_data, val_data, train_loader, val_loader = get_loader_celebDF_JYUNYI(args)
	train_dataset_size = len(train_data)
	val_dataset_size = len(val_data)
	model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
	
	# if continue_train:
	# 	model.load_state_dict(torch.load(model_path))

	state_dict = torch.load(model_path)

	# adjusted_state_dict = {}
	# for key, value in state_dict.items():
	# 	adjusted_key = key.replace('model.last_linear.1', 'model.last_linear')
	# 	adjusted_state_dict[adjusted_key] = value
	# model.load_state_dict(adjusted_state_dict)

	model.load_state_dict(state_dict)

	model = model.to(f'cuda:{device_ids[0]}')
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
	# optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
	model = nn.DataParallel(model)
	best_model_wts = model.state_dict()
	best_acc = 0.0
	iteration = 0

	train_acc_list = []
	accs_list = []
	train_auc_list = []
	roc_auc_list = []


	with open(args.file, "w") as file:
		for epoch in range(epoches):
			file.write('Epoch {}/{}\n'.format(epoch+1, epoches))
			file.write('-'*10 + '\n')
			print('Epoch {}/{}'.format(epoch+1, epoches))
			print('-'*10)
			model.train()
			train_loss = 0.0
			train_corrects = 0.0
			train_preds = []
			targets = []
			probability = []
		
			for item in train_loader:
				image, labels = item['image'], item['label']
				iter_loss = 0.0
				iter_corrects = 0.0
				image = image.to(f'cuda:{device_ids[0]}')
				labels = labels.to(f'cuda:{device_ids[0]}')
				optimizer.zero_grad()
				outputs = model(image)

				post_function = nn.Softmax(dim=1)
				p_score = post_function(outputs.data)
				fake_prob = p_score[:, 1] 

				_, preds = torch.max(outputs.data, 1)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
		
				iter_loss = loss.data.item()
				train_loss += iter_loss
				iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
				train_corrects += iter_corrects
				iteration += 1
				# if not (iteration % 20):
				# 	print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_corrects / batch_size))
					
				train_preds.extend(preds.detach().cpu().numpy()) #預測結果
				targets.extend(labels.data.detach().cpu().numpy()) #真實結果
				probability.extend(fake_prob.detach().cpu().numpy()) #預測為1的機率
		
			epoch_loss = train_loss / train_dataset_size
			epoch_acc = train_corrects / train_dataset_size
		
			train_preds = np.array(train_preds)
			train_targets = np.array(targets)
			train_acc = metrics.accuracy_score(train_preds,train_targets)
			f1 = f1_score(train_targets, train_preds, average='macro')
			re = recall_score(train_targets, train_preds, average='macro')
			pr = precision_score(train_targets, train_preds, average='macro')
			train_fpr, train_tpr, train_thresholds = metrics.roc_curve(train_targets, probability, pos_label=1)
			train_auc = metrics.auc(train_fpr, train_tpr)   

			# file.write('\ntrain_preds:\n{}\n'.format(train_preds))
			# file.write('\ntrain_targets:\n{}\n'.format(targets))

			# print('train_preds:',train_preds)
			# print('train_targets:',targets)

			file.write('\nepoch: {} , train loss: {:.4f} , train acc: {:.4f} , train auc: {:.4f}, train f1: {:.4f}, train re: {:.4f}, train pr: {:.4f}\n'.format(epoch+1, epoch_loss, epoch_acc, train_auc, f1, re, pr))
			print('epoch: {} , train loss: {:.4f} , train acc: {:.4f} , train auc: {:.4f}, train f1: {:.4f}, train re: {:.4f}, train pr: {:.4f}'.format(epoch+1, epoch_loss, epoch_acc, train_auc, f1, re, pr))


			val_loss = 0.0
			val_corrects = 0.0
		
			pred_labels = []
			target_labels = []
			probability = []
			model.eval()
			with torch.no_grad():
				for item in val_loader:
					image, labels = item['image'], item['label']
					image = image.to(f'cuda:{device_ids[0]}')
					labels = labels.to(f'cuda:{device_ids[0]}')
					outputs = model(image)
					post_function=nn.Softmax(dim=1)
					p_score = post_function(outputs.data)
					fake_prob = p_score[:, 1]

					_, preds = torch.max(outputs.data, 1)
					loss = criterion(outputs, labels)
					val_loss += loss.data.item()
					val_corrects += torch.sum(preds == labels.data).to(torch.float32)

					pred_labels.extend(preds.detach().cpu().numpy())
					target_labels.extend(labels.detach().cpu().numpy())
					probability.extend(fake_prob.detach().cpu().numpy())

				epoch_loss = val_loss / val_dataset_size
				epoch_acc = val_corrects / val_dataset_size
				
				if epoch_acc > best_acc:
					best_acc = epoch_acc
					best_model_wts = model.state_dict()

				preds_arr=np.array(pred_labels)
				labs_arr= np.array(target_labels)    
				accs = accuracy_score(labs_arr, preds_arr)
				f1 = f1_score(labs_arr, preds_arr, average='macro')
				re = recall_score(labs_arr, preds_arr, average='macro')
				pr = precision_score(labs_arr, preds_arr, average='macro')
				fpr, tpr, thresholds = metrics.roc_curve(labs_arr, probability, pos_label=1)
				roc_auc = metrics.auc(fpr, tpr)   

				# file.write('\nval_preds: {}\n'.format(pred_labels))
				# file.write('\nval_targets: {}\n'.format(target_labels))

				# print('val_preds:',pred_labels)
				# print('val_targets:',target_labels)

				file.write('\nepoch: {} , val loss: {:.4f} , val acc: {:.4f} , val auc: {:.4f} , val f1: {:.4f}, val re: {:.4f}, val pr: {:.4f}\n'.format(epoch, epoch_loss, epoch_acc, roc_auc, f1, re, pr))
				print('epoch: {} , val loss: {:.4f} , val acc: {:.4f} , val auc: {:.4f} , val f1: {:.4f}, val re: {:.4f}, val pr: {:.4f}'.format(epoch+1, epoch_loss, epoch_acc, roc_auc, f1, re, pr))

			scheduler.step()
			#if not (epoch % 40):
			torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))

			train_acc_list.append(train_acc)
			accs_list.append(accs)
			train_auc_list.append(train_auc)
			roc_auc_list.append(roc_auc)

		file.write('\nBest val Acc: {:.4f}\n'.format(best_acc))
		print('Best val Acc: {:.4f}'.format(best_acc))

		model.load_state_dict(best_model_wts)
		torch.save(model.module.state_dict(), os.path.join(output_path, "best.pkl"))

		# 繪製訓練和測試的準確率隨著epoch變動的曲線圖
		plt.plot(range(epoches), train_acc_list, label='Train Accuracy')
		plt.plot(range(epoches), accs_list, label='Test Accuracy')
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.title('Train and Test Accuracy over Epochs')
		plt.legend()
		plt.show()

		# 繪製訓練和測試的AUC隨著epoch變動的曲線圖
		plt.plot(range(epoches), train_auc_list, label='Train AUC')
		plt.plot(range(epoches), roc_auc_list, label='Test AUC')
		plt.xlabel('Epoch')
		plt.ylabel('AUC')
		plt.title('Train and Test AUC over Epochs')
		plt.legend()
		plt.show()

		# # 繪製訓練集的ROC曲線
		# plt.figure(figsize=(8, 6))
		# for i in range(epoches):
		# 	plt.plot(train_fpr[i], train_tpr[i], label=f'Epoch {i+1} (AUC = {train_auc[i]:.2f})')

		# plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')
		# plt.xlabel('False Positive Rate')
		# plt.ylabel('True Positive Rate')
		# plt.title('ROC Curve - Training Set')
		# plt.legend()
		# plt.grid()
		# plt.show()

		# # 繪製測試集的ROC曲線
		# plt.figure(figsize=(8, 6))
		# for i in range(epoches):
		# 	plt.plot(fpr[i], tpr[i], label=f'Epoch {i+1} (AUC = {roc_auc[i]:.2f})')

		# plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')
		# plt.xlabel('False Positive Rate')
		# plt.ylabel('True Positive Rate')
		# plt.title('ROC Curve - Test Set')
		# plt.legend()
		# plt.grid()
		# plt.show()






if __name__ == '__main__':
	parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parse.add_argument('--name', '-n', type=str, default='celebDF_JYUNYI_50%_Xception')
	# parse.add_argument('--train_list', '-tl' , type=str, default = './data_list/FaceSwap_c0_train.txt')
	# parse.add_argument('--val_list', '-vl' , type=str, default = './data_list/FaceSwap_c0_val.txt')
	parse.add_argument('--epoches', '-e', type=int, default='50')
	parse.add_argument('--model_name', '-mn', type=str, default='celebDF_JYUNYI_50%_Xception.pkl')
	parse.add_argument('--continue_train', type=bool, default=False)
	parse.add_argument('--model_path', '-mp', type=str, default='/ssd6/Roy/XceptionNet-Deepfake-master/output/first_conti_xception_c23_299/0_first_conti_xception_c23_299.pkl')
	parse.add_argument('--file', type=str, default='celebDF_JYUNYI_50%_Xception.txt')
	parse.add_argument('--image_size', type=int, default='299')
	parse.add_argument('--workers', type=int, default='8')

	parse.add_argument('--batch_size', '-bz', type=int, default=32)
	parse.add_argument('--test_img_batch', type=int, default=16)
	# parse.add_argument('--root_dir', type=str, default="/hdd1/DeepFakes_may/celeb-df_crop_face")
	# parse.add_argument('--train_file_path', type=str, default="/hdd1/DeepFakes_may/celeb-df/train.txt")
	# parse.add_argument('--val_file_path', type=str, default="/hdd1/DeepFakes_may/celeb-df/val.txt")
	# parse.add_argument('--test_file_path', type=str, default="/hdd1/DeepFakes_may/celeb-df/test.txt")
	parse.add_argument('--root_dir', type=str, default="/hdd1/JYUN-YI/celeb-df_crop_face")
	parse.add_argument('--train_file_path', type=str, default="/hdd1/DeepFakes_may/celeb-df_train_3084")
	parse.add_argument('--val_file_path', type=str, default="/hdd1/JYUN-YI/celeb-df/celeb-df_test_62")
	parse.add_argument('--test_file_path', type=str, default="/hdd1/JYUN-YI/celeb-df/celeb-df_test_62")

	os.environ['CUDA_VISIBLE_DEVICES']='0' 
	device_ids = [0]

	main()
