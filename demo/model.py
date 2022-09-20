import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_dim=1, output_dim=99*2, batchsize=32):
        super().__init__()
        self.batchsize = batchsize
        
        self.without_start_goal = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=3,stride=2),
            
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        
        self.project_start_and_goal = nn.Sequential(
            nn.Linear(4,8),
            nn.ReLU(inplace=True),
            nn.Linear(8,16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(16,32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(32,64),
            nn.ReLU(inplace=True)
        )
        
        self.conv_dim = 5408
        
        self.project_down = nn.Sequential(
            nn.Linear(self.conv_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        
        self.middle_dim = 128 + 64
        self.fc1 = nn.Linear(self.middle_dim, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, output_dim)
    
    def forward(self, map_info, start, goal, verbose=False):
        map_info = self.without_start_goal(map_info)
        map_info = (map_info.reshape(self.batchsize, 1, -1))
        if verbose:
            print("After first set of conv layers: ", map_info.size())
        map_info = self.project_down(map_info)
        
        start_and_goal = torch.cat((start, goal), 1).unsqueeze(1)
        start_and_goal = self.project_start_and_goal(start_and_goal)
        if verbose:
            print("Projected start and goal:", start_and_goal.size())
        
        output = torch.cat((map_info, start_and_goal), 2)
        if verbose:
            print("Concatenated: ", output.size())
                
        output = self.fc2(self.dropout(F.relu(self.fc1(output))))
        
        return output

def prepare_training_data(batch, batchsize=32, include_sdf=False, device="cuda:0"):
    map_tensor = batch["map_tensor"].unsqueeze(1).to(device)
    sdf_tensor = batch["sdf_data"].unsqueeze(1).to(device)
    expert_traj = batch["expert_trajectory"].to(device)
    label_traj = expert_traj[:, :2, 1:-1].reshape(batchsize, 1, -1)
    
    start = expert_traj[:, :2, 0].to(device)
    goal = expert_traj[:, :2, -1].to(device)
    
    if include_sdf:
        map_tensor = torch.cat((map_tensor, sdf_tensor), 1).double()
    else:
        map_tensor = map_tensor.double()
    
    return map_tensor, start, goal, label_traj

def loss_func(feat1, feat2):
    # maximize average magnitude of cosine similarity
    loss = nn.MSELoss()
    return loss(feat1, feat2)
