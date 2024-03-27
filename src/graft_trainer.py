import torch
import transformers
import torch.nn as nn
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import math 
from transformers import get_constant_schedule_with_warmup, get_constant_schedule, get_linear_schedule_with_warmup
from datasets import load_metric    

class graft_Trainer(nn.Module):
    def __init__(self, model_trainer):
        
        super(graft_Trainer, self).__init__()
        self.trainer = model_trainer
        self.model   = self.trainer.model
        self.args    = self.trainer.args
        
        self.trainer.select_trainable_parameters()
        self.params  = self.trainer.params
        
        
    
    ########################################################################################################################
    #We need to store pre-trained and fine-tuned model weights as well (Inefficient, need to think more on this)
    ########################################################################################################################
    def augment_models (self, pretrained_model, finetuned_model, model_args, device):   
        self.pretrained_model = pretrained_model
        self.finetuned_model  = finetuned_model
        self.device = device
        self.model_args = model_args
    
    ########################################################################################################################
    #The following function initializes the mask for grafting optimization
    ########################################################################################################################
    def create_binary_masks(self):
        
        self.trainable_name = []
        self.trainable_parameters = []
       
        for n in self.params: 
            self.trainable_name += [n]
            p = self.params[n]
            #self.trainable_parameters += [ torch.rand_like( p.data, device=self.device, requires_grad=False) ] 
            #!use all ones to begin for Gumbel-softmax
            self.trainable_parameters += [torch.ones_like(p.data, device=self.device, requires_grad=False)] 
        
        
        self.num_params = sum([p.numel() for p in self.trainable_parameters])  

        self.grad_directions = []
        for counter in range(len(self.trainable_name)):
            for pre_n, pre_p in self.pretrained_model.named_parameters():
                if pre_n == self.trainable_name[counter]: pretensor = pre_p


            for fine_n, fine_p in self.finetuned_model.named_parameters():
                if fine_n == self.trainable_name[counter]: finetensor = fine_p
                    
                    
            self.grad_directions += [ (finetensor - pretensor).detach() ]        
    ########################################################################################################################

    
    ########################################################################################################################
    #The following function resets the model to pretrained model weights
    ########################################################################################################################   
    def reset_model(self):
        sigmoid = torch.nn.Sigmoid()
        for counter in range(len(self.trainable_name)):
            for pre_n, pre_p in self.pretrained_model.named_parameters():
                if pre_n == self.trainable_name[counter]: pretensor = pre_p.to(self.device)



            with torch.no_grad():   
                for n, p in self.model.named_parameters():    
                    if n == self.trainable_name[counter]: 
                    #    frac = sigmoid(trainable_parameters[counter] - sigmoid_bias)
                        p += ( pretensor - p )
    ########################################################################################################################
    
    
    

    
    ########################################################################################################################
    #The following function gets the grafted model with a given mask (or the current trainable parameters)
    ########################################################################################################################
    def interpolate_model(self, round_=False, mask=None):  
        sigmoid = torch.nn.Sigmoid()
        sigmoid_bias = self.args.sigmoid_bias
        for counter in range(len(self.trainable_name)):
            for pre_n, pre_p in self.pretrained_model.named_parameters():
                if pre_n == self.trainable_name[counter]: pretensor = pre_p.to(self.device)


            for fine_n, fine_p in self.finetuned_model.named_parameters():
                if fine_n == self.trainable_name[counter]: finetensor = fine_p.to(self.device)

            with torch.no_grad():            
                for n, p in self.model.named_parameters():    
                    if n == self.trainable_name[counter]: 
                        if mask is not None:
                            frac = self.basepatch[counter] + (1. - 2. * self.basepatch[counter]) * mask[counter]
                        else:    
                            frac = self.basepatch[counter] + (1. - 2. * self.basepatch[counter]) * sigmoid(self.trainable_parameters[counter] - sigmoid_bias) 
                            if round_:
                                frac = torch.round(frac)
                        p += frac * ( finetensor - pretensor ) 
    ########################################################################################################################                   
    
    ########################################################################################################################
    # The following function gets the grafted model with a given mask (or the current trainable parameters) with Gumbel-softmax
    ########################################################################################################################
    def interpolate_model_Gumbel(self, mask=None, temp=1.0, round_=False):
        sigmoid = torch.nn.Sigmoid()
        for counter in range(len(self.trainable_name)):
            for pre_n, pre_p in self.pretrained_model.named_parameters():
                if pre_n == self.trainable_name[counter]: pretensor = pre_p.to(self.device)
                
            for fine_n, fine_p in self.finetuned_model.named_parameters():
                if fine_n == self.trainable_name[counter]: finetensor = fine_p.to(self.device)
            
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if n == self.trainable_name[counter]:
                        # Here we use Gumbel-softmax to sample the mask
                        if mask is not None:
                            frac = mask[counter]
                        else:
                            # frac = torch.zeros_like(self.trainable_parameters[counter], requires_grad=False)
                            # noise = torch.nn.functional.gumbel_softmax(self.trainable_parameters[counter], tau=tau, hard=False)  # TODO: How to generate Guembel distribution quickly
                            eps = 1e-20
                            uniform0 = torch.rand_like(self.trainable_parameters[counter], requires_grad=False)
                            uniform1 = torch.rand_like(self.trainable_parameters[counter], requires_grad=False)
                            noise = -torch.log(torch.log(uniform0 + eps)/torch.log(uniform1 + eps) + eps)
                            frac = sigmoid(((self.trainable_parameters[counter] + eps).log() - (1.0-self.trainable_parameters[counter] + eps).log() + noise)*temp)
                            self.basepatch[counter] = frac
                        if round_:
                            # round_ means we want to return the final model whose mask is sampled from bernoulli distribution with probability s
                            # s is just self.trainable_parameters[counter], i.e., the prob of bernoulli distribution
                            # frac = (torch.rand_like(frac) < self.trainable_parameters[counter]).float()
                            frac = torch.round(self.trainable_parameters[counter])
                        p += frac * (finetensor - pretensor)
    
    
    ##################################################################################################################
    # This function solve the v
    ##################################################################################################################
    def solve_v(self, sparsity_level=1.0):
        num_params = self.num_params
        k = num_params * sparsity_level # Here we need to transform k smoothly from dense to sparse
        a, b = 0, 0
        for counter in range(len(self.trainable_name)):
            b = max(b,self.trainable_parameters[counter].max()) # TODO: How to get b? wrong!
        def f(v):
            s = 0
            for counter in range(len(self.trainable_name)):
                s += (self.trainable_parameters[counter] - v).clamp(0, 1).sum() # TODO: How to get s? wrong!
            return s - k
        if f(0) > 0:
            return 0
        itr = 0
        while (1):
            itr += 1
            v = (a+b)/2
            obj = f(v)
            if abs(obj) < 1e-3 or itr > 20:
                break
            if obj > 0:
                a = v
            else:
                b = v
        return v
    
    ##################################################################################################################
    # This function constrain the score by the whole model
    ##################################################################################################################
    def constrainScoreByWhole(self, sparsity_level=1.0):
        total = 0
        for counter in range(len(self.trainable_name)):
            total += len(self.basepatch[counter])
        v = self.solve_v(sparsity_level=sparsity_level)
        for counter in range(len(self.trainable_name)):
            self.trainable_parameters[counter].sub_(v).clamp_(0, 1)
    
    
    ######
    #This function creates a BitDelta
    ######
    def BitDelta(self, round_=True, mask=None):
        sigmoid = torch.nn.Sigmoid()
        sigmoid_bias = self.args.sigmoid_bias
        for counter in range(len(self.trainable_name)):
            for pre_n, pre_p in self.pretrained_model.named_parameters():
                if pre_n == self.trainable_name[counter]: pretensor = pre_p.to(self.device)


            for fine_n, fine_p in self.finetuned_model.named_parameters():
                if fine_n == self.trainable_name[counter]: finetensor = fine_p.to(self.device)

            with torch.no_grad():            
                for n, p in self.model.named_parameters():    
                    if n == self.trainable_name[counter]: 
                        if mask is not None:
                            frac = self.basepatch[counter] + (1. - 2. * self.basepatch[counter]) * mask[counter]
                        else:
                            frac = self.basepatch[counter] + (1. - 2. * self.basepatch[counter]) * sigmoid(self.trainable_parameters[counter] - sigmoid_bias)
                            if round_:
                                frac = torch.round(frac)
                        diff = finetensor - pretensor
                        quantile = diff.float().abs().mean()
                        delta = torch.zeros_like(frac, requires_grad=False)
                        delta[diff<0] = -1
                        delta[frac>0] = 1
                        p += delta * quantile
    
    
    ########################################################################################################################
    #This function creates the basepatch used for initializing the mask for optimization!
    #If mask_path == "highest_movement", we simply pick the parameters that have moved the most during training
    ########################################################################################################################
    def create_basepatch(self):
        sigmoid = torch.nn.Sigmoid()
        sigmoid_bias = self.args.sigmoid_bias
        num_params = self.num_params
        mask_path = self.model_args.mask_path 
        sparsity_level =  self.model_args.sparsity_level
        
        
        #If mask is already stored somewhere, I simply load it!
        if mask_path != "highest_movement":
            basepatch = torch.load(mask_path, map_location=self.device)

            
            total = max([ torch.amax(p) for p in basepatch ])
            #if the max value is greater than 1., it means we have received masks without sigmoid
            if total > 1.:
                basepatch[mask_counter] = [ sigmoid( p - sigmoid_bias ) for p in basepatch ]
            
            basepatch = [ torch.round( torch.clip (p, 0., 1.) )  for p in basepatch ]
            print ('Total parameters in my graft: ', sum([ torch.sum(p*p) / (1. * num_params) for p in basepatch ]))
            
            
        elif mask_path == "highest_movement":

            threshold = int(sparsity_level * num_params)
            
            best_top = np.zeros(threshold)

            consider = self.grad_directions

            for p in consider:
                arr = np.absolute(np.ndarray.flatten(p.detach().cpu().numpy()))
                all_magnitude = np.concatenate( [np.absolute(arr), best_top] )
                best_top = -np.sort(-all_magnitude)[:threshold]  


            all_magnitude = np.asarray(best_top)  
            

            threshold = np.sort(all_magnitude)[ 0 ]

            basepatch = [torch.zeros_like(p, requires_grad=False) for p in self.trainable_parameters]


            for p, q in zip(consider, basepatch):
                q[torch.absolute(p) > threshold] = 1.

            print ('Total parameters in my stitch: ', sum([ torch.sum(p*p) / (1. * num_params) for p in basepatch ]))
        else:
            raise NotImplementedError("Not Implemented!")
            
        self.basepatch = basepatch
    ########################################################################################################################
    
    ########################################################################################################################
    # Here I want to write create_basepatch_Gumbel, which stores the frac, i.e., the sigmoid.
    ########################################################################################################################
    def create_basepatch_Gumbel(self):
        # the base batch should be all ones
        self.basepatch = [torch.ones_like(p, requires_grad=False) for p in self.trainable_parameters]
    
    ######################################################################################################################## 
    #For debugging, I re-defined evaluation here!
    ########################################################################################################################   
    def evaluate(self, dataloader, task_name, mode='dev'):
        if task_name.lower() not in [ 'qqp', 'mrpc' ]: 
            metric = load_metric("accuracy")
        else:
            metric = load_metric("f1")
            
        self.model.eval()
        hidden_states = []
        counter = 0 
        device = self.device
        for batch in dataloader:
            with torch.no_grad():
                if 'prompt' in self.model_args.few_shot_type :
                    loss, outputs = self.model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), mask_pos=batch["mask_pos"].to(device), labels=batch["labels"].to(device))
                elif ('finetune' in self.model_args.few_shot_type and  self.model_args.use_CLS_linearhead == 1) : 
                    loss, outputs = self.model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=batch["labels"].to(device))
                elif 'finetune' in self.model_args.few_shot_type :
                    outputs = self.model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device)).logits
                    
            predictions = torch.argmax(outputs, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            counter += 1
            if mode=='train' and counter >= self.args.gradient_accumulation_steps: break
            
        return metric
    ########################################################################################################################

    ########################################################################################################################
    #Use Hard-concrete distribution to get gradients of the prob
    ########################################################################################################################
    def concrete_stretched(self, alpha,l = 0, r=1):
        u = torch.zeros_like(alpha).uniform_().clamp_(0.0001,0.9999)
        s = (torch.sigmoid(u.log() - (1-u).log() + alpha)).detach()
        u = s*(r-l) + l
        t = u.clamp(0, 1000)
        z = t.clamp(-1000, 1)
        dz_dt = (t < 1).float().to(alpha.device).detach()
        dt_du = (u > 0).float().to(alpha.device).detach()
        du_ds = r - l
        ds_dalpha = (s*(1-s)).detach()
        dz_dalpha = dz_dt*dt_du*du_ds*ds_dalpha
        return dz_dalpha.detach()
    
    
    ########################################################################################################################
    #Main function that trains our graft!
    #We do not use an optimizer to train the graft, but compute the gradient w.r.t. the mask ourselves
    ########################################################################################################################    
    def train_graft (self, \
                     train_dataloader, \
                     valid_dataloader, \
                     eval_dataset, \
                     autoregressive, \
                     task_name, \
                    ):
        
        baseline = 0.  
        loss_fct = torch.nn.CrossEntropyLoss()
        first_batch = 0
        sigmoid = torch.nn.Sigmoid()
        checkpoint_location = self.model_args.checkpoint_location
        
        
        device = self.device
        lr = self.args.learning_rate
        sigmoid_bias = self.args.sigmoid_bias
        num_params = self.num_params
        
        for _ in tqdm( range(int(self.args.num_train_epochs)), 'Training the mask' ):
            total_grad = []
            
            first_batch = 0
            self.interpolate_model()

            for batch in train_dataloader:
                if 'prompt' in self.model_args.few_shot_type :
                    loss, outputs = self.model(input_ids=batch['input_ids'].to(device), \
                                               attention_mask=batch['attention_mask'].to(device), \
                                               mask_pos=batch["mask_pos"].to(device), \
                                               labels=batch["labels"].to(device), \
                                              )
                    
                elif ('finetune' in self.model_args.few_shot_type and  self.model_args.use_CLS_linearhead == 1) : 
                    loss, outputs = self.model(input_ids=batch['input_ids'].to(device), \
                                               attention_mask=batch['attention_mask'].to(device), \
                                               labels=batch["labels"].to(device), \
                                              )   
                    
                elif 'finetune' in self.model_args.few_shot_type :
                    loss = self.model(input_ids=batch['input_ids'].to(device), \
                                      attention_mask=batch['attention_mask'].to(device), \
                                      labels=batch['labels'].to(device), \
                                     ).loss
                    
                elif 'autoregressive' in self.model_args.few_shot_type :
                    input_ids=batch["input_ids"].to(device)
                    option_ids=batch["label_word_list"].to(device)

                    attention_mask=batch["attention_mask"].to(device)
                    token_type_ids=batch["token_type_ids"].to(device)
                    labels=batch["labels"].to(device)



                    #computing gradients for the slow weights!
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits  = outputs.logits.contiguous()



                    indices = torch.where(token_type_ids[..., 1:] == 1)
                    logits = logits[indices]
                    nlogits = []
                    for i in range(len(input_ids)):
                        nlogits += [ logits[i, option_ids[i]] ]
                    logits = torch.stack(nlogits, 0)


                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    loss = torch.mean(loss_fct(logits, labels.view(-1)))



                loss.backward()
                
                for n, p in self.model.named_parameters() :
                    if n in self.trainable_name :
                        if p.grad is None: print (n)

                grad = [p.grad.detach().clone() for n, p in self.model.named_parameters() if n in self.trainable_name]
                self.model.zero_grad()
                grad = [ g * p.to(device) for (g, p) in zip(grad, self.grad_directions) ]  # TODO :why g*p?


                if first_batch == 0:
                    total_grad = [lr * g for g in grad]
                else:
                    total_grad = [ p + lr * g for (p, g) in zip( total_grad, grad ) ]
                first_batch += 1 
                #restrict the number of loops
                if first_batch >= self.args.gradient_accumulation_steps: 
                    break

            total_grad = [ p / (1. * first_batch) for p in total_grad ]    
            self.reset_model()
    
            #Take the gradient step
            with torch.no_grad():
                for p, (g, s) in zip(self.trainable_parameters, zip(total_grad, self.basepatch)):
                    p -=  ( (1. - 2.*s) * g * sigmoid(p - sigmoid_bias) * (1. - sigmoid(p - sigmoid_bias)) )  # TODO :update the mask with closed form solution
        
            ######### Evaluation of current mask ###########
            self.interpolate_model(round_=True)
            if task_name.lower() not in [ 'qqp', 'mrpc' ]: key = "accuracy"
            else: key = "f1"
                
            if autoregressive:
                tr  = self.trainer.evaluate(train_dataset).compute()[key] 
                val = self.trainer.evaluate(eval_dataset).compute()[key] 
            else:            
                tr  = self.evaluate(train_dataloader, task_name, mode='train').compute()[key]
                val = self.evaluate(valid_dataloader, task_name).compute()[key]

            #store the mask with the best train + validation score
            bs_compare = val + tr

            if bs_compare > baseline:
                torch.save(self.trainable_parameters, checkpoint_location)
                baseline = bs_compare
               
            self.reset_model()   
            
    
    ########################################################################################################################
    #create a same size tensor to store the sample of frac
    ########################################################################################################################
    def create_frac_sample(self):
        self.frac_sample = [torch.zeros_like(p, requires_grad=False) for p in self.trainable_parameters]
        #Here we use zeros_like to create a tensor with the same size as the trainable_parameters 
        #so we could just add the sample to the frac_sample without initializing it
    ########################################################################################################################
    #reset the basepatch above to all ones
    ########################################################################################################################
    def reset_frac_sample(self):
        self.frac_sample = [torch.zeros_like(p, requires_grad=False) for p in self.trainable_parameters]
        #same as above, just change the name to remind myself
        #this function could be deleted later
            
    ########################################################################################################################
    # I want to write another train function which uses the Gumbel-softmax algorithm to optimize the mask
    ########################################################################################################################   
    def train_graft_Gumbel(self,\
                           train_dataloader,\
                            valid_dataloader,\
                            eval_dataset,\
                            autoregressive,\
                            task_name,\
                            ):
        # copy most of the code from train_graft
        baseline = 0.
        # loss_fct = torch.nn.CrossEntropyLoss() # we don't need this. This is for auto-regressive models
        first_batch = 0
        sigmoid = torch.nn.Sigmoid()
        checkpoint_location = self.model_args.checkpoint_location
        device = self.device
        lr = self.args.learning_rate
        sigmoid_bias = self.args.sigmoid_bias
        num_params = self.num_params
        T = int(self.args.num_train_epochs)
        t2 = int(0.8*T)
        t1 = int(0.2*T)
        I = 2 # sample I times to get the approximate frac. 2 for ResNet, in the paper.
        
        for t in tqdm( range(T), 'Training the mask' ):
            total_grad = []
            
            first_batch = 0
            # TODO: Get tau and temp from epoch
            tau = 1.0
            temp = 1.0
            tau = 0.97*(1.0-t/T)+0.03
            temp = 1.0/tau
            self.interpolate_model_Gumbel(temp=temp, round_=False)
            
            for batch in train_dataloader:
                if 'prompt' in self.model_args.few_shot_type :
                    loss, outputs = self.model(input_ids=batch['input_ids'].to(device), \
                                               attention_mask=batch['attention_mask'].to(device), \
                                               mask_pos=batch["mask_pos"].to(device), \
                                               labels=batch["labels"].to(device), \
                                              )
                elif ('finetune' in self.model_args.few_shot_type and  self.model_args.use_CLS_linearhead == 1) : 
                    loss, outputs = self.model(input_ids=batch['input_ids'].to(device), \
                                               attention_mask=batch['attention_mask'].to(device), \
                                               labels=batch["labels"].to(device), \
                                              )
                elif 'finetune' in self.model_args.few_shot_type :
                    loss = self.model(input_ids=batch['input_ids'].to(device), \
                                      attention_mask=batch['attention_mask'].to(device), \
                                      labels=batch['labels'].to(device), \
                                     ).loss
                # for now, we needn't consider autoregressive models
                
                loss.backward()
                
                for n, p in self.model.named_parameters() :
                    if n in self.trainable_name :
                        if p.grad is None: print (n)
                
                grad = [p.grad.detach().clone() for n, p in self.model.named_parameters() if n in self.trainable_name]
                self.model.zero_grad()
                # Here I need to think about the gradient accumulation
                
                if first_batch == 0:
                    total_grad = [lr * g for g in grad]
                else:
                    total_grad = [ p + lr * g for (p, g) in zip( total_grad, grad ) ]
                first_batch += 1
                #restrict the number of loops
                if first_batch >= self.args.gradient_accumulation_steps: 
                    break
            
            total_grad = [ p / (1. * first_batch) for p in total_grad ]
            self.reset_model()
            
            
            #Take the gradient step
            #There should sample Gumel distribution many times to get the approximate gradient
            #one way is to wirte a for loop to sample the frac many times. frac: sigmoid(Gumbel), i.e., the mask
            #Originally, the frac are stored in self.basepatch. But if we want to sample the frac many times, we should not use basepatch
            #my idea is to create a same size tensor as the trainable_parameters, and then sample the frac many times and take the average
            with torch.no_grad():
                self.create_frac_sample()  # create the frac_sample
                for i in range(I):
                    for counter in range(len(self.trainable_name)):
                        eps = 1e-20
                        uniform0 = torch.rand_like(self.trainable_parameters[counter], requires_grad=False)
                        uniform1 = torch.rand_like(self.trainable_parameters[counter], requires_grad=False)
                        noise = -torch.log(-torch.log(uniform0 + eps)/torch.log(uniform1 + eps) + eps)
                        frac = sigmoid(((self.trainable_parameters[counter] + eps).log() - (1.0-self.trainable_parameters[counter] + eps).log() + noise)*temp)
                        
                        self.frac_sample[counter] += frac / (I*1.0)  # accumulate the frac and take the average
                #for p, (g, s) in zip(self.trainable_parameters, zip(total_grad, self.basepatch)):
                for p, (g, s) in zip(self.trainable_parameters, zip(total_grad, self.frac_sample)):
                    # TODO:Here I need to think about the gradient accumulation
                    p -= g * s * (1. - s) * 1.0 / (p * (1. - p+eps)) * temp  #! 1.0/0.0
                self.reset_frac_sample()  # reset the frac_sample to all zeros
                # constrain the score by the whole model
                # TODO first we need to get sparsity_level, t1 and t2
                kf = self.model_args.sparsity_level
                k =  kf + (1.0-kf)*(1.0-(t-t1)/(t2-t1))**3  #TODO: 0~t1 and t2~T
                self.constrainScoreByWhole(sparsity_level=k)
            ######### Evaluation of current mask ###########
            self.interpolate_model_Gumbel(temp=temp, round_=True) # TODO: Here we need to think about the round_, whether we need to round the frac_sample?
            if task_name.lower() not in [ 'qqp', 'mrpc' ]: key = "accuracy"
            else: key = "f1"
            
            if autoregressive:
                tr  = self.trainer.evaluate(train_dataset).compute()[key] 
                val = self.trainer.evaluate(eval_dataset).compute()[key]
            else:
                tr  = self.evaluate(train_dataloader, task_name, mode='train').compute()[key]
                val = self.evaluate(valid_dataloader, task_name).compute()[key]
                
            #store the mask with the best train + validation score
            bs_compare = val + tr
            
            if bs_compare > baseline:
                torch.save(self.trainable_parameters, checkpoint_location)
                baseline = bs_compare
            
            self.reset_model()
            
        
    ########################################################################################################################
    
        
        
        
        
        