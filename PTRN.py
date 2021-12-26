import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# from model import *
from init_parameter import init_model
from data import *
from model_test import *
from util import *
import torch
from cuml import KMeans
import time
from transformers import AdamW, logging
import torch.nn as nn

class ModelManager:
    
    def __init__(self, args, data, pretrained_model=None):
        
        if pretrained_model is None:
            pretrained_model = BertForModel.from_pretrained(args.bert_model, cache_dir = "", num_labels = data.n_known_cls)
            if os.path.exists(args.pretrain_dir):
                pretrained_model = self.restore_model(args.pretrained_model)
        self.pretrained_model = pretrained_model
        self.labelMap = None
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id           
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(args.cluster_num_factor)
        if args.cluster_num_factor > 1:
            self.num_labels = self.predict_k(args, data) 
            print(self.num_labels)
        else:
            self.num_labels = data.num_labels       

        self.model_1 = BertForModel(args, self.num_labels)
        self.model_2 = BertForModel(args, self.num_labels)
        
        if args.pretrain:
            self.load_pretrained_model(args)

        if args.freeze_bert_parameters:
            self.freeze_parameters(self.model_1)
            self.freeze_parameters(self.model_2)
            
        self.model_1.to(self.device)
        self.model_2.to(self.device)

        self.optimizer = self.get_optimizer(args)
        num_train_examples = len(data.train_labeled_examples) + len(data.train_unlabeled_examples)
        self.num_training_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs
        self.num_warmup_steps= int(args.warmup_proportion * self.num_training_steps) 
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_training_steps) 
        

        self.best_eval_score = 0
        self.centroids = None

        self.test_results = None
        self.predictions = None
        self.true_labels = None
        self.score_list = []
        self.acc_list = []
        self.epoch_record = 0
        self.centroids_log = None

    def get_features_labels(self, dataloader, model, args, mode=False):
        
        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_pred = torch.empty(0,dtype=torch.long).to(self.device)
        # total_soft = torch.empty((0, self.num_labels)).to(self.device)
        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                feature, label, logits_softmax = model(input_ids, segment_ids, input_mask, feature_ext = True)
            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))
            total_pred = torch.cat((total_pred, label))
            # total_soft = torch.cat((total_soft, logits_softmax))
        if mode:
            return total_features, total_labels, total_labels
        return total_features, total_labels, total_pred
    
    def get_optimizer(self, args):
        param_optimizer = list(self.model_1.named_parameters())
        param_optimizer.extend(list(self.model_2.named_parameters()))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)
        return optimizer

    def predict_k(self, args, data):
           
        feats, _, _ = self.get_features_labels(data.train_semi_dataloader, self.pretrained_model, args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = data.num_labels, n_init=10).fit(feats)
        y_pred = km.labels_

        pred_label_list = np.unique(y_pred)
        drop_out = len(feats) / data.num_labels
        print('drop',drop_out)

        cnt = 0
        for label in pred_label_list:
            num = len(y_pred[y_pred == label]) 
            if num < drop_out:
                cnt += 1

        num_labels = len(pred_label_list) - cnt
        print('pred_num',num_labels)

        return num_labels
    
    def evaluation(self, args, data):
        feats, labels, pred_1 = self.get_features_labels(data.test_dataloader, self.model_1, args, mode=True)
        feats_2, _, pred_2 = self.get_features_labels(data.test_dataloader, self.model_2, args, mode=True)
        feats = feats.cpu().numpy()
        feats_2 = feats_2.cpu().numpy()
        km = KMeans(n_clusters = self.num_labels, n_init=10).fit((feats + feats_2)/2)
        y_true = labels.cpu().numpy()

        # align_labels = self.alignment(km, args)
        # onehot_labels = np.zeros((len(feats), self.num_labels), dtype=int)
        # for i in range(len(align_labels)):
        #     onehot_labels[i][align_labels[i]] = 1
        #     weight = F.kl_div(pred_2[i].log(), pred_1[i])
        #     weight = weight2.cpu().numpy()
        #     print(weight2)
        # onehot_labels = torch.tensor(onehot_labels).to(self.device)
        # y_pred = weight2*onehot_labels + 0.5*(1 - weight)*pred_1 + 0.5*(1 - weight)*pred_2
        # y_pred = torch.softmax(y_pred, 1)
        # _, y_pred = torch.max(y_pred, 1)
        # y_pred = y_pred.cpu().numpy()

        results = clustering_score(y_true, km.labels_)
        print(results)
        
    def alignment(self, km, args):
        if self.centroids is not None:

            old_centroids = self.centroids.cpu().numpy()
            new_centroids = km.cluster_centers_
            
            DistanceMatrix = np.linalg.norm(old_centroids[:,np.newaxis,:]-new_centroids[np.newaxis,:,:],axis=2) 
            row_ind, col_ind = linear_sum_assignment(DistanceMatrix)
            
            new_centroids = torch.tensor(new_centroids).to(self.device)
            self.centroids = torch.empty(self.num_labels ,args.feat_dim).to(self.device)
            
            alignment_labels = list(col_ind)
            for i in range(self.num_labels):
                label = alignment_labels[i]
                self.centroids[i] = new_centroids[label]
                
            pseudo2label = {label:i for i,label in enumerate(alignment_labels)}
            pseudo_labels = np.array([pseudo2label[label] for label in km.labels_])
            self.labelMap = pseudo2label
        else:
            self.centroids = torch.tensor(km.cluster_centers_).to(self.device)        
            pseudo_labels = km.labels_ 

        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(self.device)
        
        return pseudo_labels

    def update_pseudo_labels(self, pseudo_labels, args, data):
        train_data = TensorDataset(data.semi_input_ids, data.semi_input_mask, data.semi_segment_ids, pseudo_labels)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = args.train_batch_size)
        return train_dataloader

    def model_select(self, data, args):
        feats, _, _ = self.get_features_labels(data.train_semi_dataloader, self.model_1, args)
        feats_2, _, _ = self.get_features_labels(data.train_semi_dataloader, self.model_2, args)
        feats = feats.cpu().numpy()
        feats_2 = feats_2.cpu().numpy()
        km = KMeans(n_clusters = self.num_labels, n_init=10).fit((feats+feats_2)/2)
        score = metrics.silhouette_score(feats, km.labels_)
        return score

    def train(self, args, data): 
        print(self.num_labels)
        best_score = 0
        best_model = None
        best_model_2 = None
        wait = 0
        iter = 0
        

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  

            feats, _, _ = self.get_features_labels(data.train_semi_dataloader, self.model_1, args)
            feats_2, _, _ = self.get_features_labels(data.train_semi_dataloader, self.model_2, args)
            feats = feats.cpu().numpy()
            feats_2 = feats_2.cpu().numpy()
            print("feature extracting finished")
            km = KMeans(n_clusters = self.num_labels, n_init=10).fit(feats)
            print("k-means finished")
            score = metrics.silhouette_score(feats, km.labels_)
            print('score',score)
            self.score_list.append(score)

            if score > best_score:
                best_model = copy.deepcopy(self.model_1)
                best_model_2 = copy.deepcopy(self.model_2)
                wait = 0
                self.epoch_record = epoch
                self.centroids_log = self.centroids
                best_score = score
                print(best_score)
            else:
                wait += 1
                if wait >= args.wait_patient:
                    self.model_1 = best_model
                    self.model_2 = best_model_2
                    break
            pseudo_labels = self.alignment(km, args)
            train_dataloader = self.update_pseudo_labels(pseudo_labels, args, data)
            
            tr_loss = 0
            tr_loss_1_KNN = 0
            tr_loss_2_1 = 0
            tr_loss_1_2 = 0
            tr_loss_labeled = 0

            nb_tr_examples, nb_tr_steps = 0, 0
            nb_tr_steps_labeled = 0
            self.model_1.train()
            self.model_2.train()

            for batch in tqdm(train_dataloader, desc="Pseudo-Training"):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch


                loss_1_KNN, logits_1 = self.model_1(input_ids, segment_ids, input_mask, label_ids, mode='train')
                _, pred_1 = torch.max(logits_1.data, 1)
                loss_2_1, logits_2 = self.model_2(input_ids, segment_ids, input_mask, pred_1, mode='train')
                _, pred_2 = torch.max(logits_2.data, 1)
                loss_1_2, _ = self.model_1(input_ids, segment_ids, input_mask, pred_2, mode='train')
                loss = 0.3 * (1/3 * loss_1_2 +  1/3 * loss_2_1 +  1/3 * loss_1_KNN)
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                
                nb_tr_steps += 1
                iter += 1
                tr_loss_1_KNN += loss_1_KNN.item() 
                tr_loss_2_1 += loss_2_1.item()
                tr_loss_1_2 += loss_1_2.item()
                nn.utils.clip_grad_norm_(self.model_1.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.model_2.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
   
            for batch in tqdm(data.train_labeled_dataloader, desc="Labeled-Training"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    loss_1, _ = self.model_1(input_ids, segment_ids, input_mask, label_ids, mode="train")
                    loss_2, _ = self.model_2(input_ids, segment_ids, input_mask, label_ids, mode="train")
                    loss = 0.5 * loss_1 + 0.5 * loss_2
                    loss.backward()
                    tr_loss_labeled += loss.item()

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps_labeled += 1
            
            tr_loss = 0.5* (tr_loss / nb_tr_steps + tr_loss_labeled / nb_tr_steps_labeled)
            self.evaluation(args, data)
        
    def load_pretrained_model(self, args):
        pretrained_dict = self.pretrained_model.state_dict()
        classifier_params = ['classifier.weight','classifier.bias']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model_1.load_state_dict(pretrained_dict, strict=False)
        self.model_2.load_state_dict(pretrained_dict, strict=False)
        
    def restore_model(self, args, model):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model
    
    def freeze_parameters(self,model):
        for name, param in model.bert.named_parameters():  
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

if __name__ == '__main__':
    # torch.cuda.set_device(3)
    logging.set_verbosity_error()
    print('Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    data = Data(args)

    if args.pretrain:
        print('Pre-training begin...')
        manager_p = PretrainModelManager(args, data)
        manager_p.train()
        print('Pre-training finished!')
        manager = ModelManager(args, data, manager_p.model)
    else:
        manager_p = PretrainModelManager(args, data)
        manager = ModelManager(args, data, manager_p.model)
    
    print('Training begin...')
    start = time.time()
    manager.train(args,data)
    end = time.time()
    print(end - start)
    print('Training finished!')

    manager.centroids = manager.centroids_log
    print('Evaluation begin...')
    manager.evaluation(args, data)
    print(time.time() - end)
    print('Evaluation finished!')

