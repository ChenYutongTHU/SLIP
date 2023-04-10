from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class Mengzi_Encoder(torch.nn.Module):
    def __init__(self, model_path, output_dim):
        super().__init__()
        self.mengzi = T5ForConditionalGeneration.from_pretrained(model_path)
        del self.mengzi.decoder
        del self.mengzi.lm_head
        self.text_projection = torch.nn.Linear(self.mengzi.config.d_model, output_dim)
    
    @property
    def dtype(self):
        return self.text_projection.weight.dtype
    

    def encode_text(self, input_ids, attn_layer=-1):
        attention_mask = (input_ids!=0).long() #B,L
        encoder_outputs = self.mengzi.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        eos_pos = torch.sum(attention_mask,dim=1)-1 #B,
        last_hidden_states = encoder_outputs[0] #B,L,D
        eos_embed = last_hidden_states[torch.arange(0, input_ids.shape[0]),eos_pos] #B,D
        text_embed = self.text_projection(eos_embed) #B,D
        return text_embed, None, None, None

class Mengzi_Tokenizer(object):
    def __init__(self, model_path, max_length=32):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.max_length=max_length

    def __call__(self,text):
        o = self.tokenizer(text, max_length=self.max_length, truncation=True, padding='max_length')
        return torch.tensor(o['input_ids'])


def load(model_path, device, context_length, output_dim):
    model = Mengzi_Encoder(model_path=model_path, output_dim=output_dim)
    tokenizer = Mengzi_Tokenizer(model_path=model_path, max_length=context_length)
    return model.to(device), tokenizer