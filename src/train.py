#######################################################################################
#                                  Train/ evaluation 
#######################################################################################
import torch 
import torch.nn as nn 
from  src.data import get_batch

class Train:
    def __init__(self, model, train_data, val_data, n_embd, batch_size, block_size, num_epochs,optimizer,steps_per_epoch,val_every,device):

        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.nembd = n_embd
        self.batch_size = batch_size
        self.block_size = block_size
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.device = device
        self.steps_per_epoch = steps_per_epoch
        self.val_every = val_every

    def train_one_epoch(self): 
        self.model.train()
        total_loss = 0.0 

        for step in range(self.steps_per_epoch): 

            X,y = get_batch(
                self.train_data,
                self.batch_size,
                self.block_size,
                self.device
            )

            loss, _ = self.model(X,y)

            self.optimizer.zero_grad(set_to_none=True)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        total_avg_loss = total_loss / self.steps_per_epoch

        return total_avg_loss

    def evaluate(self): 
        self.model.eval()
        total_loss = 0.0 

        with torch.no_grad():
            for step in range(self.steps_per_epoch): 
                X,y = get_batch(
                        self.val_data,
                        self.batch_size,
                        self.block_size,
                        self.device
                    )
        
                loss, _ = self.model(X,y)
                
                total_loss += loss.item()
        
        total_avg_loss = total_loss / self.steps_per_epoch
        
        return total_avg_loss

    def fit(self): 

        History = {"Epochs": [],"train_losses": [], "val_losses": []}

        for epoch in range(self.num_epochs): 
            train_losses = self.train_one_epoch()

            if (epoch +1) % self.val_every == 0 or epoch == self.num_epochs -1 : 
                val_losses = self.evaluate()

                History["Epochs"].append(epoch +1)
                History["val_losses"].append(val_losses)
                History["train_losses"].append(train_losses)

                print(f"epochs : [{epoch +1}/{self.num_epochs}]"
                      f"train_loss {train_losses: .4f}"
                      f"val_loss {val_losses: .4f}")

        return History










