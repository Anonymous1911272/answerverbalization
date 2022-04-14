import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from modeling_outputs import ModelOutputs
from constants import QUESTION_GENERATION_TASK
from utils import load_huggingface_pretrained_object
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


class PretrainedModel(nn.Module):
    def __init__(self, model_name, config, vocab_size, tasks):
        super(PretrainedModel, self).__init__()
        self.name = model_name
        self.config = config
        self.tasks = tasks 
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        self.use_question_autoencoder = True if QUESTION_GENERATION_TASK in self.tasks else False 

        self.model = load_huggingface_pretrained_object(self.name, self.config)
        self.model.resize_token_embeddings(vocab_size)

        self.question_encoder = self.model.get_encoder()
        self.answer_decoder = self.model.get_decoder()

        if self.use_question_autoencoder:
            self.question_decoder = copy.deepcopy(self.model.get_decoder())
            # assert hex(id(self.question_decoder)) != hex(id(self.model.decoder)) 
            # assert hex(id(self.question_decoder.block)) != hex(id(self.model.decoder.block))
        

    def forward(self, input_questions, labels):

        encoder_outputs = self.question_encoder(**input_questions)
        encoder_hidden_states = encoder_outputs.last_hidden_state
             
        question_seq2seq_lm_outputs = None
        question_loss = torch.tensor(0.0, requires_grad=self.training)
        
        if self.training and self.use_question_autoencoder:
            question_decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(input_questions['input_ids'])
            question_decoder_outputs = self.question_decoder(input_ids=question_decoder_input_ids, encoder_hidden_states=encoder_hidden_states, output_hidden_states=True)
            question_sequence_outputs = question_decoder_outputs[0]
            question_lm_logits = self.model.lm_head(question_sequence_outputs)
            question_loss = self.loss_fct(question_lm_logits.view(-1, question_lm_logits.size(-1)), input_questions['input_ids'].view(-1))

            question_seq2seq_lm_outputs = Seq2SeqLMOutput(
                    loss=question_loss,
                    logits=question_lm_logits, 
                    past_key_values=question_decoder_outputs.past_key_values, 
                    decoder_hidden_states=question_decoder_outputs.hidden_states, 
                    decoder_attentions=question_decoder_outputs.attentions, 
                    cross_attentions=question_decoder_outputs.cross_attentions, 
                    encoder_last_hidden_state=encoder_outputs.last_hidden_state, 
                    encoder_hidden_states=encoder_outputs.hidden_states, 
                    encoder_attentions=encoder_outputs.attentions)
            

        if labels is not None:
            # shifting lm labels to the right
            answer_decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels['input_ids']) 
            # answer_decoder_outputs = self.model(input_questions['input_ids'], labels['input_ids'])
                      
        answer_decoder_outputs = self.answer_decoder(input_ids=answer_decoder_input_ids, encoder_hidden_states=encoder_hidden_states)
        answer_sequence_outputs = answer_decoder_outputs.last_hidden_state
        answer_lm_logits = self.model.lm_head(answer_sequence_outputs)

        # answer_loss = torch.tensor(0.0, requires_grad=self.training)
        
        if labels is not None:
            answer_loss = self.loss_fct(answer_lm_logits.view(-1, answer_lm_logits.size(-1)), labels['input_ids'].view(-1)) 
            
            if self.use_question_autoencoder:
                
                total_loss = torch.mean(torch.stack([ (1-0.5) *question_loss,  0.5 *answer_loss]))
        
            else:

                total_loss = answer_loss

        answer_seq2seq_lm_outputs = Seq2SeqLMOutput(
                loss=answer_loss,
                logits=answer_lm_logits,
                past_key_values=answer_decoder_outputs.past_key_values,
                decoder_hidden_states=answer_decoder_outputs.hidden_states,
                decoder_attentions=answer_decoder_outputs.attentions,
                cross_attentions=answer_decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions)
         
        
        return ModelOutputs({'loss':total_loss, 'answer_seq2seq_lm_outputs': answer_seq2seq_lm_outputs, 'question_seq2seq_lm_outputs': question_seq2seq_lm_outputs})

    def generate(self, input_questions, num_beams=1):
        return self.model.generate(**input_questions, num_beams=num_beams)


    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def get_num_trainable_parameters(self):
        return len(self.get_trainable_parameters())

'''
class Encoder(nn.Module):
    def __init__(self, vocabulary, device, embed_dim=args.emb_dim, layers=args.layers,
                 heads=args.heads, pf_dim=args.pf_dim, dropout=args.dropout, max_positions=args.max_positions):
        super().__init__()
        input_dim = len(vocabulary)
        self.padding_idx = vocabulary.stoi[PAD_TOKEN]
        self.dropout = dropout
        self.device = device

        input_dim, embed_dim = vocabulary.vectors.size()
        self.scale = math.sqrt(embed_dim)
        self.embed_tokens = nn.Embedding(input_dim, embed_dim)
        self.embed_tokens.weight.data.copy_(vocabulary.vectors)
        self.embed_positions = PositionalEmbedding(embed_dim, dropout, max_positions)

        self.layers = nn.ModuleList([EncoderLayer(embed_dim, heads, pf_dim, dropout, device) for _ in range(layers)])

    def forward(self, src_tokens):
        src_mask = (src_tokens != self.padding_idx).unsqueeze(1).unsqueeze(2)

        x = self.embed_tokens(src_tokens) * self.scale
        x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, heads, pf_dim, dropout, device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadedAttention(embed_dim, heads, dropout, device)
        self.pos_ff = PositionwiseFeedforward(embed_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_tokens, src_mask):
        x = self.layer_norm(src_tokens + self.dropout(self.self_attn(src_tokens, src_tokens, src_tokens, src_mask)))
        x = self.layer_norm(x + self.dropout(self.pos_ff(x)))

        return x


class Decoder(nn.Module):
    def __init__(self, vocabulary, device, embed_dim=args.emb_dim, layers=args.layers,
                 heads=args.heads, pf_dim=args.pf_dim, dropout=args.dropout, max_positions=args.max_positions):
        super().__init__()

        output_dim = len(vocabulary)
        self.pad_id = vocabulary.stoi[PAD_TOKEN]
        self.pf_dim = pf_dim
        self.dropout = dropout
        self.device = device
        self.max_positions = max_positions

        self.scale = math.sqrt(embed_dim)
        self.embed_tokens = nn.Embedding(output_dim, embed_dim)
        self.embed_positions = PositionalEmbedding(embed_dim, dropout, max_positions)

        self.layers = nn.ModuleList([DecoderLayer(embed_dim, heads, pf_dim, dropout, device) for _ in range(layers)])

        self.linear_out = nn.Linear(embed_dim, output_dim)

    def make_masks(self, src_tokens, trg_tokens):
        src_mask = (src_tokens != self.pad_id).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = (trg_tokens != self.pad_id).unsqueeze(1).unsqueeze(3)
        trg_len = trg_tokens.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return src_mask, trg_mask

    def forward(self, src_tokens, trg_tokens, encoder_out):
        src_mask, trg_mask = self.make_masks(src_tokens, trg_tokens)

        x = self.embed_tokens(trg_tokens) * self.scale
        x += self.embed_positions(trg_tokens)
        h = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layers:
            h = layer(h, encoder_out, trg_mask, src_mask)

        x = h.contiguous().view(-1, h.shape[-1])
        x = self.linear_out(x)

        return x, h


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, heads, pf_dim, dropout, device):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadedAttention(embed_dim, heads, dropout, device)
        self.src_attn = MultiHeadedAttention(embed_dim, heads, dropout, device)
        self.pos_ff = PositionwiseFeedforward(embed_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embed_trg, embed_src, trg_mask, src_mask):
        x = self.layer_norm(embed_trg + self.dropout(self.self_attn(embed_trg, embed_trg, embed_trg, trg_mask)))
        x = self.layer_norm(x + self.dropout(self.src_attn(x, embed_src, embed_src, src_mask)))
        x = self.layer_norm(x + self.dropout(self.pos_ff(x)))

        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, heads, dropout, device):
        super().__init__()
        assert embed_dim % heads == 0
        self.attn_dim = embed_dim // heads
        self.heads = heads
        self.dropout = dropout

        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)

        self.scale = torch.sqrt(torch.FloatTensor([self.attn_dim])).to(device)

        self.linear_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.linear_q(query)
        K = self.linear_k(key)
        V = self.linear_v(value)

        Q = Q.view(batch_size, -1, self.heads, self.attn_dim).permute(0, 2, 1, 3) # (batch, heads, sent_len, attn_dim)
        K = K.view(batch_size, -1, self.heads, self.attn_dim).permute(0, 2, 1, 3) # (batch, heads, sent_len, attn_dim)
        V = V.view(batch_size, -1, self.heads, self.attn_dim).permute(0, 2, 1, 3) # (batch, heads, sent_len, attn_dim)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale # (batch, heads, sent_len, sent_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = F.softmax(energy, dim=-1) # (batch, heads, sent_len, sent_len)
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        x = torch.matmul(attention, V) # (batch, heads, sent_len, attn_dim)
        x = x.permute(0, 2, 1, 3).contiguous() # (batch, sent_len, heads, attn_dim)
        x = x.view(batch_size, -1, self.heads * (self.attn_dim)) # (batch, sent_len, embed_dim)
        x = self.linear_out(x)

        return x


class PositionwiseFeedforward(nn.Module):
    def __init__(self, embed_dim, pf_dim, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, pf_dim)
        self.linear_2 = nn.Linear(pf_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x):
        x = torch.relu(self.linear_1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.linear_2(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        pos_embed = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        pos_embed = pos_embed.unsqueeze(0)
        self.register_buffer('pos_embed', pos_embed)

    def forward(self, x):
        return Variable(self.pos_embed[:, :x.size(1)], requires_grad=False)

'''

'''
class Encoder(nn.Module):
    def __init__(self, pretrained_encoder, question_decoder=None, gated_mechanism=None):
        self.pretrained_encoder = pretrained_encoder
        self.question_decoder = None
        self.gated_mechanism = None

            
    def forward(inputs):
        pretrained_encoder_outputs = self.pretrained_encoder(inputs['input_ids'])
        hidden_states = pretrained_encoder_outputs[0]
            
        question_seq2seq_lm_outputs = None
        question_loss = torch.tensor(0.0, requires_grad=self.training)
        if self.use_question_autoencoder:
            question_decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(inputs['input_ids'])
            question_decoder_outputs = self.question_decoder(input_ids=question_decoder_input_ids, encoder_hidden_states=hidden_states, output_hidden_states=True)
            
            if self.training:
                question_sequence_outputs = question_decoder_outputs[0]
                question_lm_logits = self.model.lm_head(question_sequence_outputs)
                question_loss = self.loss_fct(question_lm_logits.view(-1, question_lm_logits.size(-1)), inputs['input_ids'].view(-1))

                question_seq2seq_lm_outputs = Seq2SeqLMOutput(
                        loss=question_loss,
                        logits=question_lm_logits, 
                        past_key_values=question_decoder_outputs.past_key_values, 
                        decoder_hidden_states=question_decoder_outputs.hidden_states, 
                        decoder_attentions=question_decoder_outputs.attentions, 
                        cross_attentions=question_decoder_outputs.cross_attentions, 
                        encoder_last_hidden_state=encoder_outputs.last_hidden_state, 
                        encoder_hidden_states=encoder_outputs.hidden_states, 
                        encoder_attentions=encoder_outputs.attentions)
            
            
        if self.use_gated_mechanism:
            hidden_states = self.gated_mechanism(hidden_states, question_decoder_outputs[-1])
        
        return BaseModelOutput(
                last_hidden_state=hidden_states, 
                hidden_states=None, 
                attentions=None
                ), question_seq2seq_lm_outputs

class PretrainedModel(nn.Module, GenerationMixin):
    def __init__(self, model_name, config, vocab_size, use_question_autoencoder, use_gated_mechanism):
        super(PretrainedModel, self).__init__()

        self.name = model_name
        self.config = config
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        self.use_question_autoencoder = use_question_autoencoder
        self.use_gated_mechanism = use_gated_mechanism

        self.model = load_huggingface_pretrained_object(model_name, self.config)
        self.model.resize_token_embeddings(vocab_size) 
        
        self.question_encoder = self.model.get_encoder()

        if self.use_question_autoencoder:
            self.question_decoder = copy.deepcopy(self.model.get_decoder())

        if self.use_gated_mechanism:
            self.gated_mechanism = GatedCNN(a,b)

        self.answer_decoder = self.model.get_decoder()

    def forward(
            self, 
            input_ids=None, 
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True):

        encoder_outputs = self.question_encoder(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                inputs_embeds=inputs_embeds, 
                head_mask=head_mask, 
                output_attentions=output_attentions, 
                output_hidden_states=output_hidden_states, 
                return_dict=return_dict)

        hidden_states = encoder_outputs.last_hidden_state
        
        question_seq2seq_lm_outputs = None
        question_loss = torch.tensor(0.0, requires_grad=self.training)
        
        if self.use_question_autoencoder:
            question_decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(input_ids)
            question_decoder_outputs = self.question_decoder(input_ids=question_decoder_input_ids, encoder_hidden_states=hidden_states, output_hidden_states=True)
            
            if self.training:
                question_sequence_outputs = question_decoder_outputs[0]
                question_lm_logits = self.model.lm_head(question_sequence_outputs)
                question_loss = self.loss_fct(question_lm_logits.view(-1, question_lm_logits.size(-1)), input_ids.view(-1))

                question_seq2seq_lm_outputs = Seq2SeqLMOutput(
                        loss=question_loss,
                        logits=question_lm_logits, 
                        past_key_values=question_decoder_outputs.past_key_values, 
                        decoder_hidden_states=question_decoder_outputs.hidden_states, 
                        decoder_attentions=question_decoder_outputs.attentions, 
                        cross_attentions=question_decoder_outputs.cross_attentions, 
                        encoder_last_hidden_state=encoder_outputs.last_hidden_state, 
                        encoder_hidden_states=encoder_outputs.hidden_states, 
                        encoder_attentions=encoder_outputs.attentions)
            

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # shifting lm labels to the right
            answer_decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels['input_ids']) 
            
            
        else:
            if "Bart" in self.name:
                # Bart is concatenating input --> different than other models!
                answer_decoder_input_ids = self.prepare_decoder_input_ids_from_labels(input_ids)
            else:
                answer_decoder_input_ids = self.model.prepare_inputs_for_generation(input_ids)


        if self.use_gated_mechanism:
            hidden_states = self.gated_mechanism(hidden_states, question_decoder_outputs[-1])
        
        # answer_decoder_outputs = self.answer_decoder(**answer_decoder_input_ids)
        
        
        if self.training:    
            answer_decoder_outputs = self.answer_decoder(
                    input_ids=answer_decoder_input_ids, 
                    attention_mask=decoder_attention_mask,
                    inputs_embeds=decoder_inputs_embeds,
                    past_key_values=past_key_values,
                    encoder_hidden_states=hidden_states,
                    encoder_attention_mask=attention_mask,
                    head_mask=decoder_head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict)

        else:
            
            answer_decoder_outputs = self.answer_decoder(
                    input_ids=answer_decoder_input_ids['decoder_input_ids'], 
                    attention_mask=decoder_attention_mask,
                    inputs_embeds=decoder_inputs_embeds,
                    past_key_values=past_key_values,
                    encoder_hidden_states=hidden_states,
                    encoder_attention_mask=attention_mask,
                    head_mask=decoder_head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict)

        
        answer_sequence_outputs = answer_decoder_outputs[0]
        answer_lm_logits = self.model.lm_head(answer_sequence_outputs)

        answer_loss = torch.tensor(0.0, requires_grad=True)
        if labels is not None:
            answer_loss = self.loss_fct(answer_lm_logits.view(-1, answer_lm_logits.size(-1)), labels['input_ids'].view(-1))


        answer_seq2seq_lm_outputs = Seq2SeqLMOutput(
                loss=answer_loss,
                logits=answer_lm_logits,
                past_key_values=answer_decoder_outputs.past_key_values,
                decoder_hidden_states=answer_decoder_outputs.hidden_states,
                decoder_attentions=answer_decoder_outputs.attentions,
                cross_attentions=answer_decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions)
         
       
        total_loss = torch.mean(torch.stack([question_loss, answer_loss]))

        return ModelOutputs({'loss':total_loss, 'answer_seq2seq_lm_outputs': answer_seq2seq_lm_outputs, 'question_seq2seq_lm_outputs': question_seq2seq_lm_outputs})
        
    
    def _get_encoder_outputs(self, input_ids):
        encoder_outputs = self.question_encoder(input_ids)
        hidden_states = encoder_outputs[0]
        
        if self.use_question_autoencoder:
            question_decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(input_ids)
            question_decoder_outputs = self.question_decoder(input_ids=question_decoder_input_ids, encoder_hidden_states=hidden_states, output_hidden_states=True)
            
        if self.use_gated_mechanism:
            hidden_states = self.gated_mechanism(hidden_states, question_decoder_outputs[-1])
            
        return BaseModelOutput(
                last_hidden_state=hidden_states, 
                hidden_states=None,
                attentions=None)
    
    def generate_verbalized_answer(input_questions, decoder_input_ids=None):
        # depending on the model, input will not be provided in same way:
        # BART:
        # One solution: 1. get_decoder_input_ids for each model
        #               2. give the generated logits to a BeamSearch class
        # 
        #encoder_outputs = self.encoder(input_questions['input_ids'])
        #hidden_states = encoder_outputs[0]
        outputs = self.generate(input_questions['input_ids'])

    def get_config(self):
        return self.model.config

    def get_decoder_start_token_id(self):
        decoder_start_token_id = self.get_config().decoder_start_token_id
        
        if decoder_start_token_id is not None:
            return decoder_start_token_id

        else:
            return self.get_config().bos_token_id

    def prepare_inputs_for_generation(self, input_ids):
        return self.model.prepare_inputs_for_generation(input_ids)
'''

    
