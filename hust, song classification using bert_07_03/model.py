import torch
import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask): #3 
        outputs = self.bert( #4 Đưa dữ liệu vào mô hình BERT 
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.pooler_output # 5 Lấy ra vector đặc trưng từ BERT 
        x = self.dropout(pooled) # 6 Áp dụng dropout để tránh overfitting
        return self.fc(x) # 7 Dự đoán genre