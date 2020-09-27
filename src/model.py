import config
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(config.ROBERTA_PATH, config=conf)
        self.drop_out = nn.Dropout(0.1)
        self.Cov1S = nn.Conv1d(768 * 2, 128 , kernel_size = 2 ,stride = 1 )
        self.Cov1E = nn.Conv1d(768 * 2, 128, kernel_size = 2 ,stride = 1 )
        self.Cov2S = nn.Conv1d(128 , 64 , kernel_size = 2 ,stride = 1)
        self.Cov2E = nn.Conv1d(128 , 64 , kernel_size = 2 ,stride = 1)
        self.lS = nn.Linear(64 , 1)
        self.lE = nn.Linear(64 , 1)
        torch.nn.init.normal_(self.lS.weight, std=0.02)
        torch.nn.init.normal_(self.lE.weight, std=0.02)

        
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)
        out = out.permute(0,2,1)
        
        same_pad1 = torch.zeros(out.shape[0] , 768*2 , 1).cuda()
        same_pad2 = torch.zeros(out.shape[0] , 128 , 1).cuda()

        out1 = torch.cat((same_pad1 , out), dim = 2)
        out1 = self.Cov1S(out1)
        out1 = torch.cat((same_pad2 , out1), dim = 2)
        out1 = self.Cov2S(out1)
        out1 = F.leaky_relu(out1)
        out1 = out1.permute(0,2,1)
        start_logits = self.lS(out1).squeeze(-1)
        #print(start_logits.shape)

        out2 = torch.cat((same_pad1 , out), dim = 2)
        out2 = self.Cov1E(out2)
        out2 = torch.cat((same_pad2 , out2), dim = 2)
        out2 = self.Cov2E(out2)
        out2 = F.leaky_relu(out2)
        out2 = out2.permute(0,2,1)
        end_logits = self.lE(out2).squeeze(-1)
        #print(end_logits.shape)


        return start_logits, end_logits