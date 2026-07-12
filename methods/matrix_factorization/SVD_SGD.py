import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from surprise import SVD, Dataset, Reader

def svd_cf(train: np.ndarray):
    """
    使用 Surprise 套件的 SVD 模型進行矩陣分解預測。
    輸入: train (2D Numpy Array), 未評分項目需補 0
    輸出: pred (2D Numpy Array, shape 與 train 相同) 的預測評分矩陣
    """
    # 注意：這裡的 rating_scale 預設為 1~5 分，若對方資料範圍不同請提醒他修改
    reader = Reader(rating_scale=(1,5))
    
    # 將 numpy array 轉為 surprise 需要的格式
    train_data = [(u, i, train[u,i]) for u in range(train.shape[0]) for i in range(train.shape[1]) if train[u,i] > 0]
    df = pd.DataFrame(train_data, columns=["user", "item", "rating"])
    data = Dataset.load_from_df(df, reader)
    trainset = data.build_full_trainset()
    
    # 訓練模型
    model = SVD()
    model.fit(trainset)

    # 產生完整的預測矩陣
    pred = np.zeros_like(train, dtype=np.float32)
    for u in range(train.shape[0]):
        for i in range(train.shape[1]):
            pred[u,i] = model.predict(u, i).est

    return pred

def sgd_mf(train: np.ndarray, epochs=50, lr=0.01, dim=64):
    """
    使用 PyTorch 實作的 SGD Matrix Factorization (包含 Embedding 與 Bias)。
    輸入: train (2D Numpy Array), 未評分項目需補 0
    輸出: pred_mat (2D Numpy Array, shape 與 train 相同) 的預測評分矩陣
    """
    # 定義 PyTorch 模型
    class MF(nn.Module):
        def __init__(self, n_users, n_items, k=64):
            super(MF, self).__init__()
            self.user_emb = nn.Embedding(n_users, k)
            self.item_emb = nn.Embedding(n_items, k)
            self.user_bias = nn.Embedding(n_users, 1)
            self.item_bias = nn.Embedding(n_items, 1)
            self.global_bias = nn.Parameter(torch.tensor(0.0))
            
            # 初始化參數
            nn.init.normal_(self.user_emb.weight, std=0.01)
            nn.init.normal_(self.item_emb.weight, std=0.01)
            nn.init.constant_(self.user_bias.weight, 0.0)
            nn.init.constant_(self.item_bias.weight, 0.0)

        def forward(self, u, i):
            u_e = self.user_emb(u)
            i_e = self.item_emb(i)
            pred = (u_e * i_e).sum(dim=1, keepdim=True)
            pred += self.user_bias(u) + self.item_bias(i) + self.global_bias
            return pred.squeeze()

    # 準備硬體與資料
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_users, n_items = train.shape

    triplets = [(u, i, train[u, i]) for u in range(n_users) for i in range(n_items) if train[u, i] > 0]
    us, is_, rs = zip(*triplets)
    us_t = torch.tensor(us, dtype=torch.long).to(device)
    is_t = torch.tensor(is_, dtype=torch.long).to(device)
    rs_t = torch.tensor(rs, dtype=torch.float32).to(device)

    # 初始化模型與優化器
    model = MF(n_users, n_items, k=dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) 
    loss_fn = nn.MSELoss()

    # 訓練迴圈
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(us_t, is_t)
        loss = loss_fn(output, rs_t)
        loss.backward()
        optimizer.step()

    # 預測階段：利用矩陣相乘一次算出所有預測值
    model.eval()
    with torch.no_grad():
        P = model.user_emb.weight  # (n_users, dim)
        Q = model.item_emb.weight  # (n_items, dim)
        b_u = model.user_bias.weight # (n_users, 1)
        b_i = model.item_bias.weight.T # (1, n_items)
        mu = model.global_bias
        
        preds = torch.matmul(P, Q.T) + b_u + b_i + mu
        pred_mat = preds.cpu().numpy()

    return pred_mat