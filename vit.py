# Projection creator for each inout senor embedding
import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from pointnet.pointnet_cls import PointNetCls
from resnet.resnet import ImageFeatureExtractor, get_transform
from kitti_data_handler import PointCloudDataset, LazyDataLoader
import sys 
USE_POSITIONAL_EMBEDDING = True
IMAGE_FEATURES = 2048 #[bs, 2048]
LIDAR_FEATURES = 512 #[bs, 512]
NUM_FEATURES = IMAGE_FEATURES+LIDAR_FEATURES
NUM_MODALITIES = 2
test_pos_embedding = True

class ToTransformerEmbedding(nn.Module):
    '''
    Generates pre-embeddings to be fed into the ViT Transformer
    - Projects the different sensor dims into a common embedding dim
    - Appends the cls token to each sensor's projected dim
    - No position embedding as there is no sense of position among different 
      batches and since the embeddings are linear no need to add point-wise 
      position encoding 
    '''
    def __init__(self, input_dim:list, proj_dim:int, dropout:float):
        super().__init__()
        
        # converting the dimension of the input modality into a standard 
        # embedding size
        self.proj_lidar = nn.Linear(input_dim[0], proj_dim)
        self.proj_img = nn.Linear(input_dim[1], proj_dim)
        
        #Add cls token to each feature vector
        self.cls_token = nn.Parameter(torch.zeros(
            size=(NUM_MODALITIES , 1),requires_grad=True))
        if USE_POSITIONAL_EMBEDDING:
            self.positional_embedding = nn.Parameter(torch.randn(
                size=(1,NUM_MODALITIES,proj_dim+1), requires_grad=True))
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x:list):# x = [img_tensor=>(n, 2048), lidar_tensor=>(n, 512)]
        # x => [bs, C, H, W] or [bs, num_points, 3]
        x_0 = self.proj_lidar(x[0])
        x_1 = self.proj_img(x[1])
        
        # print("\n\n", x_0.size(), x_1.size())
        # print("\n\n",x_0.view(x_0.size(0), -1, x_0.size(1)).size(), 
            # x_1.view(x_1.size(0), -1, x_1.size(1)).size())
        x = torch.cat(
            [x_0.view(x_0.size(0), -1, x_0.size(1)), 
            x_1.view(x_1.size(0), -1, x_1.size(1))], 
            dim=1
            )
        cls_token = self.cls_token.expand(x.shape[0], -1,-1)
        x = torch.cat([cls_token, x], dim=-1)
        # print(x.size(), self.positional_embedding.size())
        if USE_POSITIONAL_EMBEDDING:
            x = x + self.positional_embedding
        x = self.dropout(x)
        return x


# ViT code from  https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
class FeedForward(nn.Module):
    def __init__(self, dim:int, hidden_dim:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

# Attention class that implements multi-head self attention mechanism
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        # Apply layer normalization
        x = self.norm(x)

        # Project input into Query, Key, Value matrices and split them
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # Separate Q,K,V and reshape to include number of heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # Calculate scaled dot-product attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Apply softmax to get attention weights
        attn = self.attend(dots)

        # Apply attention weights to values
        out = torch.matmul(attn, v)
        # Reshape output by combining head dimension with feature dimension
        out = rearrange(out, 'b h n d -> b n (h d)')
        # Final linear projection
        return self.to_out(out)


# to training the transformer as a final detection head 
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViT_multimodal_imgClassifier(nn.Module):
    '''
    ViT model for image classification
    '''
    def __init__(self, 
                 image_size:tuple, 
                 lidar_size:tuple,
                 l_c_proj_dim:int,# used to project the resnet and lidar to same dims = 768
                 num_classes:int, 
                 dropout:float,
                 hidden_dim:int,
                 depth:int, 
                 heads:int, 
                 dim_head:int,
                 device:str) -> None:
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        
        resent_output_dim = 2048
        pointnet_output_dim = 1024
        
        
        # pass image through the resnet model
        self.image_features_extractor = ImageFeatureExtractor()
        # pass lidar through the pointnet model
        self.lidar_features_extractor = PointNetCls(num_classes=num_classes)
        
        
        
        self.coplanar_projection = ToTransformerEmbedding(
            input_dim=[resent_output_dim, pointnet_output_dim], 
            proj_dim=l_c_proj_dim,
            dropout=dropout)
        self.transformer = Transformer(dim=l_c_proj_dim+1, depth=depth, 
                                       heads=heads, dim_head=dim_head, 
                                       mlp_dim=hidden_dim)
        self.linear_head = nn.Linear(l_c_proj_dim+1, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, x:list):
        # x = [img_tensor=>(n, 2048), lidar_tensor=>(n, 512)]
        img_features = self.image_features_extractor(x[0])
        _, _, _, gl_features = self.lidar_features_extractor(x[1])
        x = [img_features, gl_features.flatten(start_dim=1)]#[bs,2048],[bs, 4096]
        x = self.coplanar_projection(x)
        x = self.transformer(x)
        return self.logsoftmax(self.linear_head(x[:,0,:]))
    
    def _get_img_features(self, img_tensor):
        self.image_features_extractor.eval()
        transform = get_transform()
        img_tensor = transform(img_tensor).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.image_features_extractor(img_tensor.detatch)
        return features
    
    def _get_lidar_features(self, lidar_tensor):
        return self.lidar_features_extractor(lidar_tensor)
        
        
        
# if test_pos_embedding:
#     x =torch.randn(512, 512)
#     y = torch.rand(512,2048)
#     print(x.size(), y.size())
#     model = ToTransformerEmbedding([x.size()[-1], y.size()[-1]], 768, 0.1)

#     print(model(torch.nested.nested_tensor([x,y])).size())

def prepare_label(x, task_type:str='MLC'):
    def multi_label_classification(inp):
        # Extract first element from each tensor in the list and create flat tensor
        output = []
        for tensor in inp:
            # Get first element (1,15) -> (15,)
            first_elem = tensor[0][0]
            output.append(int(first_elem))
        
        # Stack into single tensor
        return output
    if task_type=="MLC":
        return multi_label_classification(x)
    
if __name__ == "__main__":
    # # Test Attention module with ToTransformerEmbedding input
    # ***************************
    # Data loading & Handling
    # ***************************
    data_dir = "/home/sm2678/csci_739_term_project/CSCI739/data"
    camera_dir = "left_images/{}/image_2"
    lidar_dir = "velodyne/{}/velodyne/"
    calib_dir = "calibration/{}/calib"
    label_dir = "labels/{}/label_2"
    batch_size = 4
    mode = "traing"
    
    dataset = PointCloudDataset(
        data_dir,lidar_dir, camera_dir, calib_dir, label_dir, 1024, "training",
        return_image=True, return_calib=True, return_labels=True
    )
    
    lazy_loader = LazyDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        label_data_loc=[0]
    )
    
    
    # ***************************
    # Model init
    # ***************************
    model = ViT_multimodal_imgClassifier(image_size=(512,512), lidar_size=(512,512), 
                                l_c_proj_dim=768, num_classes=9,
                                dropout=0.1, hidden_dim=3072, depth=6, 
                                heads=16, dim_head=64, device='cuda')

    # ***************************
    # Loss and optimizer
    # ***************************
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # ***************************
    # Training loop
    # ***************************
    epochs = 2
    ep=0
    total_loss = 0
    for epoch in range(epochs):
        ep+=1
        i=0
        total_epoch_loss = 0
        for batch in lazy_loader.dataloader:
            i+=1
            # Each batch will contain:
            point_clouds = batch['point_clouds']      # Shape: (batch_size, num_points, 3)
            images = batch['images']                  # Shape: (batch_size, C, H, W)
            labels = batch['labels']                  # List of label dictionaries
            calibrations = batch['calibrations']      # List of calibration dictionaries
            sample_ids = batch['sample_ids']          # List of sample IDs
            
            # # Print shapes and contents
            # print(f"Point clouds shape: {point_clouds.shape}")
            # print(f"Images shape: {images.shape}")
            # print(f"Number of labels: {len(labels)}")
            # print(f"Sample IDs: {sample_ids}")
            
            log_probs = model([images, point_clouds.permute(0,2,1)])
            print(labels)
            labels = prepare_label(labels)
            loss = criterion(log_probs, torch.tensor(labels, dtype=int))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss
        print(f"Epoch {ep}/{epochs}: Avg Epoch loss: {total_epoch_loss*batch_size/len(dataset)}")
        total_loss +=total_epoch_loss*batch_size/len(dataset)
    total_loss /= epochs
    print(f"Average training loss: {total_loss}")
        
    
    # model = ViT_multimodal_imgClassifier(image_size=(512,512), lidar_size=(512,512), 
    #                             l_c_proj_dim=768, num_classes=40, 
    #                             dropout=0.1, hidden_dim=3072, depth=6, 
    #                             heads=16, dim_head=64)
    # print(model(nested_input).size())

    
    

# create a model that takes in batch of data from ToTransformerEmbedding
# and then feeds it into a ViT model

# class ViTFusor(nn.Module):
#     def __init__(self, num_classes:int, dropout:float):
#         super().__init__()
#         self.transformer_embedding = ToTransformerEmbedding(
#             input_dim=[x.size()[-1], y.size()[-1]], 
#             proj_dim=768, 
#             dropout=0.1)  
        
#         self.transformer = nn.Transformer(
#             d_model = 768,
#             nhead = 16,
#             num_encoder_layers = 6,
#             dim_feedforward = 3072,
#         )
#         self.linear_head = nn.Linear(768, num_classes)
        
#     def forward(self, x:list):
#         x = self.transformer_embedding(x)
#         x = self.transformer(x)
#         return self.linear_head(x[:,0,:])

# def train_model(model, train_loader, val_loader, num_epochs:int, lr:float):
#     # data loader
#     from pointnet.data_loader import KittiDataset
    
#     train_loader = KittiDataset(
#         root_dir='../data/kitti/training',
#         split='train',
#         num_points=4096,
#         transform=None
#     )
    
    
    

        
        
            
        
        
        