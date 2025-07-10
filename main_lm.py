import os
import gc
import hydra
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
from speechbrain.pretrained import EncoderDecoderASR
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.submodules.rnnt_beam_decoding import BeamRNNTInfer
from pyctcdecode import BeamSearchDecoderCTC
from pyctcdecode.alphabet import Alphabet
from pyctcdecode.language_model import LanguageModel
from jiwer import wer

from data import *
from forward import *
from utils import *

from transformers import AutoTokenizer, AutoModelForCausalLM
import ipdb

llm_model  = 'openchat3.5'
if llm_model == 'llama':
    SYSTEM = 'You are a sentence correction assistant'
    INSTRUCTION = "Please generate correction of the given <<Sentence>> considering its pronunciation and meaning based on the suitable explanation. Generate the output in form of '\n<<Explanation>>: {explanation of correction}\n\n<<Correction>>: {corrected sentence}\n'"
    TEMPLATE = "[INST] «SYS»\n{}\n«/SYS»\n\n{}\n\n<<Sentence>>: {}[/INST]"
elif llm_model == 'openchat3.5':
    TEMPLATE = {'role': 'user', 'content': "Given <<Sentence>> is the transcription of utterance. Please generate correction of the <<Sentence>> considering its pronunciation and meaning based on the generated explanation. Generate the output in form of '\n<<Explanation>>: {explanation of correction}\n\n<<Correction>>: {corrected sentence}'\n\n<<Sentence>>: "}

def forward_and_adapt(args, model, processor, optimizer, scheduler, wavs, lens, step):
    global original_model, lm_tokenizer, correction_lm

    optimizer.zero_grad()
    blank_index = get_blank_index(args, model, processor)

    for i, wav in enumerate(wavs):
        wav = wav[:lens[i]].unsqueeze(0)
        outputs, pseudo_labels = get_logits_and_pseudo_labels(args, model, processor, wav, torch.FloatTensor([lens[i]]).to(wav.device))
        if "original" in args.method or "em_uncertainty" in args.method or "em_sparse" in args.method:
            predicted_ids = torch.argmax(outputs, dim=-1)
            non_blank = torch.where(predicted_ids != blank_index, 1, 0).bool()

            if args.em_coef > 0:
                if "original" in args.method:
                    if args.not_blank:
                        e_loss = softmax_entropy(outputs / args.temp)[non_blank].mean(0).mean()
                    else: 
                        e_loss = softmax_entropy(outputs / args.temp).mean(0).mean() 
                (args.em_coef * e_loss / (len(wavs))).backward(retain_graph=True)
            if 1 - args.em_coef > 0:
                c_loss = mcc_loss(outputs / args.temp, class_num=outputs.shape[-1], reweight=True)
                ((1 - args.em_coef) * c_loss / (len(wavs))).backward(retain_graph=True)
        if 'beam_search_max' in args.method or 'beam_search_all' in args.method or 'beam_search_negative_sampling' in args.method:
            criterion = nn.CrossEntropyLoss(ignore_index=blank_index) if args.not_blank else nn.CrossEntropyLoss()
            if 'beam_search_max' in args.method:
                char_history = pseudo_labels[0].to(args.device)
                if args.certain_only:
                    selected_frame = set()
                    top_idx, top_prob = -1, 0
                    for frame_idx, (output, char_idx) in enumerate(zip(outputs.squeeze(0), char_history)):
                        probs = torch.softmax(output, dim=-1)
                        if probs[char_idx] > args.prob_threshold:
                            selected_frame.add(frame_idx)
                        if char_idx != blank_index and probs[char_idx].item() > top_prob:
                            top_idx = frame_idx
                            top_prob = probs[char_idx].item()
                    selected_frame.add(top_idx)
                    selected_frame = sorted(selected_frame)
                    selected_outputs, selected_char_history = outputs.squeeze(0)[selected_frame], char_history[selected_frame]
                    loss = criterion(selected_outputs / args.temp, selected_char_history)
                else:
                    loss = criterion(outputs / args.temp, char_history)
                (loss / len(wavs)).backward(retain_graph=True)
            elif 'beam_search_all' in args.method:
                loss = 0
                for char_history in pseudo_labels[:args.num_positives]:
                    char_history = char_history.to(args.device)
                    if args.certain_only:
                        selected_frame = set()
                        top_idx, top_prob = -1, 0
                        for frame_idx, (output, char_idx) in enumerate(zip(outputs.squeeze(0), char_history)):
                            probs = torch.softmax(output, dim=-1)
                            if probs[char_idx] > args.prob_threshold:
                                selected_frame.add(frame_idx)
                            if char_idx != blank_index and probs[char_idx].item() > top_prob:
                                top_idx = frame_idx
                                top_prob = probs[char_idx].item()
                        selected_frame.add(top_idx)
                        selected_frame = sorted(selected_frame)
                        selected_outputs, selected_char_history = outputs.squeeze(0)[selected_frame], char_history[selected_frame]
                        loss += criterion(selected_outputs / args.temp, selected_char_history) / len(pseudo_labels)
                    else:
                        loss += criterion(outputs / args.temp, char_history) / len(pseudo_labels)
                (loss / len(wavs)).backward(retain_graph=True)
            if 'beam_search_negative_sampling' in args.method:
                negative_outputs = outputs.clone()
                negative_loss = 0
                char_history = pseudo_labels[0].to(args.device)
                if args.negative_sampling_method == "random":
                    for _ in range(args.num_negatives):
                        negative_char_history = torch.randint_like(input=char_history, high=negative_outputs.shape[-1]).to(args.device)
                        negative_mask = (negative_char_history != char_history) & (char_history != 0)

                        selected_frame = []
                        for frame_idx, mask in enumerate(negative_mask):
                            if mask:
                                selected_frame.append(frame_idx)
                        selected_negative_outputs = negative_outputs.squeeze(0)[selected_frame]
                        selected_negative_char_history = negative_char_history[selected_frame]
                        if len(selected_negative_outputs) > 0:
                            negative_loss += -criterion(selected_negative_outputs / args.temp, selected_negative_char_history) / args.num_negatives
                elif args.negative_sampling_method == "beam_candidate":
                    for out_idx in range(len(pseudo_labels))[-args.num_negatives:]:
                        negative_char_history = pseudo_labels[out_idx].to(args.device)
                        negative_mask = (negative_char_history != char_history) & (char_history != 0)

                        selected_frame = []
                        for frame_idx, mask in enumerate(negative_mask):
                            if mask:
                                selected_frame.append(frame_idx)
                        selected_negative_outputs = negative_outputs.squeeze(0)[selected_frame]
                        selected_negative_char_history = negative_char_history[selected_frame]
                        if len(selected_negative_outputs) > 0:
                            negative_loss += -criterion(selected_negative_outputs / args.temp, selected_negative_char_history) / args.num_negatives
                elif args.negative_sampling_method == 'ns3l':
                    negative_mask = torch.where(torch.softmax(negative_outputs, dim=-1) < args.ns_threshold * (10 / negative_outputs.shape[-1]), 1, 0)
                    negative_loss += torch.mean(-torch.log(1 - torch.sum(negative_mask * torch.softmax(negative_outputs / args.temp, dim=-1), dim=-1)))
                if torch.is_tensor(negative_loss):
                    (args.ns_coef * negative_loss / len(wavs)).backward(retain_graph=True)
        if 'renyi_em' in args.method:
            predicted_ids = torch.argmax(outputs, dim=-1)
            non_blank = torch.where(predicted_ids != blank_index, 1, 0).bool()
            # added by esyoon 2024-02-25-22:31:53
            if isinstance(model, Wav2Vec2ForCTC):
                pred_text = processor.decode(
                        outputs.squeeze(0).detach().cpu().numpy(),
                        beam_width=args.beam_width,
                    ).text
            elif isinstance(model, EncoderDecoderASR):
                pred_text = model.transcribe_batch(wav, wav_lens=torch.ones(1).to(args.device))[0]

            elif isinstance(model, nemo_asr.models.EncDecCTCModelBPE):
                greedy_predictions = outputs.argmax(dim=-1, keepdim=False)
                pred_text = model.wer.decode(greedy_predictions, predictions_lengths=torch.tensor([outputs.shape[1]]))[0][0].upper()
            
            elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
                # vocabulary = model.tokenizer.tokenizer.get_vocab()
                # labels_map = {v: k for k, v in vocabulary.items()}
                # token_list = [labels_map[c] for c in pseudo_labels[0].tolist() if c < model.decoding.blank_id - model.decoding.num_extra_outputs]
                # pred_text = model.tokenizer.tokenizer.detokenize(token_list).upper()
                pred_text = transcribe_batch(args, model, processor, wavs, lens)[0]

                # pred_text = model.wer.decode(greedy_predictions, predictions_lengths=encoder_length)[0][0].upper()
            if args.llm_model == 'llama':
                lm_input = TEMPLATE.format(SYSTEM, INSTRUCTION, pred_text)
                lm_input_tensor = lm_tokenizer(lm_input, return_tensors='pt')
                lm_input_tensor = to_device(lm_input_tensor, args.device)
                do_generation = True
                max_length = 512
                lm_generated_text = None
                num_loops = 0
                ctc_loss = 0
                while do_generation:
                    num_loops += 1
                    with torch.no_grad():
                        lm_output = correction_lm.generate(**lm_input_tensor, max_length = max_length)
                    lm_output = lm_tokenizer.decode(lm_output[0], skip_special_tokens=True)
                    if "<<Correction>>" in lm_output.split('[/INST]')[1]:
                        try:
                            lm_generated_text = lm_output.split('[/INST]')[1].split('<<Correction>>: ')[1].split('\n')[0].replace('.', '')
                        except:
                            do_generation=True
                            max_length += 128
                    elif "Corrected sentence" in lm_output.split('[/INST]')[1]:
                        try:
                            lm_generated_text = lm_output.split('[/INST]')[1].split("Corrected sentence: ")[1].split('\n')[0].replace('.', '')
                        except:
                            do_generation=True
                            max_length += 128
                    if lm_generated_text:
                        if '"' in lm_generated_text:
                            lm_generated_text = lm_generated_text.replace('"', '')
                        do_generation=False
                        break
                    else:
                        do_generation = True
                    if num_loops > 5:
                        print("skip this one")
                        
                        do_generation = False
                        break
                if lm_generated_text:
                    ctc_loss_reduction = "mean"
                    ctc_zero_infinity = True

                    log_probs = nn.functional.log_softmax(outputs, dim=-1, dtype=torch.float32).transpose(0,1)

                    if isinstance(processor, Wav2Vec2Processor): 
                        processed_lm_output = processor(text=lm_generated_text, return_tensors='pt')
                        processed_lm_output = to_device(processed_lm_output, args.device)
                        ctc_loss = nn.functional.ctc_loss(
                            log_probs,
                            processed_lm_output['input_ids'],
                            torch.LongTensor([outputs.shape[1]]),
                            torch.LongTensor([processed_lm_output['input_ids'].shape[1]]),
                            blank=blank_index,
                            reduction=ctc_loss_reduction,
                            zero_infinity=ctc_zero_infinity,
                        )
                    elif isinstance(model, nemo_asr.models.EncDecCTCModelBPE):
                        processed_lm_output = torch.tensor([model.tokenizer.text_to_ids(pred_text)]).to(args.device)
                        ctc_loss = nn.functional.ctc_loss(
                            log_probs,
                            processed_lm_output,
                            torch.LongTensor([outputs.shape[1]]),
                            torch.LongTensor([processed_lm_output.shape[1]]),
                            blank=blank_index,
                            reduction=ctc_loss_reduction,
                            zero_infinity=ctc_zero_infinity,
                        )    
                    elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
                        processed_lm_output = torch.tensor([model.tokenizer.text_to_ids(pred_text)]).to(args.device)
                        ctc_loss = nn.functional.ctc_loss(
                            log_probs,
                            processed_lm_output,
                            torch.LongTensor([outputs.shape[1]]),
                            torch.LongTensor([processed_lm_output.shape[1]]),
                            blank=blank_index,
                            reduction=ctc_loss_reduction,
                            zero_infinity=ctc_zero_infinity,
                        ) 

                    # ctc_loss_reduction = "mean"
                    # ctc_zero_infinity = True
                    # ctc_loss = nn.functional.ctc_loss(
                    #     log_probs,
                    #     processed_lm_output['input_ids'],
                    #     torch.LongTensor([outputs.shape[1]]),
                    #     torch.LongTensor([processed_lm_output['input_ids'].shape[1]]),
                    #     blank=blank_index,
                    #     reduction=ctc_loss_reduction,
                    #     zero_infinity=ctc_zero_infinity,
                    # )
            elif args.llm_model == 'openchat3.5':
                NEW_TEMPLATE = {}
                NEW_TEMPLATE['role'] = 'user'
                NEW_TEMPLATE['content'] = TEMPLATE['content'] + pred_text

                lm_input_with_template= lm_tokenizer.apply_chat_template([NEW_TEMPLATE], add_generation_prompt=True, tokenize=True, return_tensors='pt')
                lm_input_tensor = lm_input_with_template.to(args.device)
                max_length = 512
                lm_generated_text = None
                ctc_loss = 0
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        lm_output = correction_lm.generate(lm_input_tensor, max_new_tokens = max_length, pad_token_id=lm_tokenizer.eos_token_id)
                lm_output = lm_tokenizer.decode(lm_output[0], skip_special_tokens=True)
                try:
                    lm_generated_text = lm_output.split('<<Correction>>: ')[-1].replace('.', '').upper()
                except:
                    lm_generated_text = None
                if lm_generated_text:
                    ctc_loss_reduction = "mean"
                    ctc_zero_infinity = True

                    log_probs = nn.functional.log_softmax(outputs, dim=-1, dtype=torch.float32).transpose(0,1)

                    if isinstance(processor, Wav2Vec2Processor): 
                        processed_lm_output = processor(text=lm_generated_text, return_tensors='pt')
                        processed_lm_output = to_device(processed_lm_output, args.device)

                        ctc_loss = nn.functional.ctc_loss(
                            log_probs,
                            processed_lm_output['input_ids'],
                            torch.LongTensor([outputs.shape[1]]),
                            torch.LongTensor([processed_lm_output['input_ids'].shape[1]]),
                            blank=blank_index,
                            reduction=ctc_loss_reduction,
                            zero_infinity=ctc_zero_infinity,
                        )
                    elif isinstance(model, nemo_asr.models.EncDecCTCModelBPE):
                        processed_lm_output = torch.tensor([model.tokenizer.text_to_ids(pred_text)]).to(args.device)
                        ctc_loss = nn.functional.ctc_loss(
                            log_probs,
                            processed_lm_output,
                            torch.LongTensor([outputs.shape[1]]),
                            torch.LongTensor([processed_lm_output.shape[1]]),
                            blank=blank_index,
                            reduction=ctc_loss_reduction,
                            zero_infinity=ctc_zero_infinity,
                        )
                    elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
                        processed_lm_output = torch.tensor([model.tokenizer.text_to_ids(pred_text)]).to(args.device)
                        ctc_loss = nn.functional.ctc_loss(
                            log_probs,
                            processed_lm_output,
                            torch.LongTensor([outputs.shape[1]]),
                            torch.LongTensor([processed_lm_output.shape[1]]),
                            blank=blank_index,
                            reduction=ctc_loss_reduction,
                            zero_infinity=ctc_zero_infinity,
                        )
                    # processed_lm_output = processor(text=lm_generated_text, return_tensors='pt')
                    # processed_lm_output = to_device(processed_lm_output, args.device)
                    # log_probs = nn.functional.log_softmax(outputs, dim=-1, dtype=torch.float32).transpose(0,1)

                    # ctc_loss_reduction = "mean"
                    # ctc_zero_infinity = True
                    # ctc_loss = nn.functional.ctc_loss(
                    #     log_probs,
                    #     processed_lm_output['input_ids'],
                    #     torch.LongTensor([outputs.shape[1]]),
                    #     torch.LongTensor([processed_lm_output['input_ids'].shape[1]]),
                    #     blank=blank_index,
                    #     reduction=ctc_loss_reduction,
                    #     zero_infinity=ctc_zero_infinity,
                    # )
            # =============================================================================================================
            if args.not_blank:
                e_loss = renyi_entropy((outputs / args.temp)[non_blank], alpha=args.renyi_entropy_alpha)
                if non_blank.sum().item() == 0:
                    e_loss = renyi_entropy(outputs / args.temp, alpha=args.renyi_entropy_alpha)
            else:
                e_loss = renyi_entropy(outputs / args.temp, alpha=args.renyi_entropy_alpha)
                
            if args.coef_adapting:
                ((args.tta_coef*e_loss + (e_loss/(e_loss + ctc_loss))*args.llm_coef*ctc_loss) / (len(wavs))).backward(retain_graph=True)
                # ((args.tta_coef*e_loss + ((args.steps-(step-1))/args.steps)*args.llm_coef*ctc_loss) / (len(wavs))).backward(retain_graph=True)
                # ((args.tta_coef*e_loss + ((args.steps-(step-1))/args.steps)*args.llm_coef*ctc_loss) / (len(wavs))).backward(retain_graph=True)
            else:
                ((args.tta_coef*e_loss + args.llm_coef*ctc_loss) / (len(wavs))).backward(retain_graph=True)
        if 'kld_ori' in args.method:
            assert 0 <= args.kld_weight <= 1
            # TODO: implement bias parameter

            # naive pseudo-labeling
            predicted_ids = torch.argmax(outputs, dim=-1)
            non_blank = torch.where(predicted_ids != blank_index, 1, 0).bool()
            # e_loss = renyi_entropy(outputs, alpha='inf')
            e_loss = renyi_entropy((outputs / args.temp)[non_blank], alpha='inf')
            ((1 - args.kld_weight) * e_loss / (len(wavs))).backward(retain_graph=True)

            # kld loss
            original_outputs, _ = get_logits_and_pseudo_labels(args, original_model, processor, wav, torch.FloatTensor([lens[i]]).to(wav.device))
            probs = torch.softmax(outputs, dim=-1)
            original_probs = torch.softmax(original_outputs, dim=-1)
            kl_div_loss = F.kl_div(torch.log(probs), original_probs.detach(), reduction="batchmean")
            (args.kld_weight * kl_div_loss / (len(wavs))).backward(retain_graph=True)
        if 'kld_comb' in args.method:
            # Renyi em
            predicted_ids = torch.argmax(outputs, dim=-1)
            non_blank = torch.where(predicted_ids != blank_index, 1, 0).bool()
            if args.not_blank:
                e_loss = renyi_entropy((outputs / args.temp)[non_blank], alpha=args.renyi_entropy_alpha)
            else:
                e_loss = renyi_entropy(outputs / args.temp, alpha=args.renyi_entropy_alpha)
            ((1 - args.kld_weight) * e_loss / (len(wavs))).backward(retain_graph=True)

            # negative sampling
            criterion = nn.CrossEntropyLoss(ignore_index=blank_index) if args.not_blank else nn.CrossEntropyLoss()
            negative_outputs = outputs.clone()
            negative_loss = 0
            char_history = pseudo_labels[0].to(args.device)
            negative_mask = torch.where(torch.softmax(negative_outputs, dim=-1) < args.ns_threshold * (10 / negative_outputs.shape[-1]), 1, 0)
            negative_loss += torch.mean(-torch.log(1 - torch.sum(negative_mask * torch.softmax(negative_outputs / args.temp, dim=-1), dim=-1)))
            if torch.is_tensor(negative_loss):
                ((1 - args.kld_weight) * args.ns_coef * negative_loss / len(wavs)).backward(retain_graph=True)

            # kld loss
            original_outputs, _ = get_logits_and_pseudo_labels(args, original_model, processor, wav, torch.FloatTensor([lens[i]]).to(wav.device))
            probs = torch.softmax(outputs, dim=-1)
            original_probs = torch.softmax(original_outputs, dim=-1)
            kl_div_loss = F.kl_div(torch.log(probs), original_probs.detach(), reduction="batchmean")
            (args.kld_weight * kl_div_loss / (len(wavs))).backward(retain_graph=True)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args):
    if args.seed:
        set_seed(args.seed)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    global logger
    logger = get_logger(args)
    logger.info(OmegaConf.to_yaml(args))

    dataset = load_dataset(args.dataset_name, args.dataset_dir, args.batch_size, args.extra_noise, args.noise_type, args.noise_snr)
    gt_texts, ori_transcriptions, transcriptions_1, transcriptions_3, transcriptions_5, transcriptions_10, transcriptions_20, transcriptions_40 = [], [], [], [], [], [], [], []

    global original_model, lm_tokenizer, correction_lm

    model = get_model(args)
    original_model = get_model(args)
    params, _ = collect_params(model, args.train_params)
    optimizer, scheduler = get_optimizer(args, params, opt_name=args.optimizer, lr=args.lr, scheduler=args.scheduler)
    processor = Wav2Vec2Processor.from_pretrained(args.asr, sampling_rate=args.sample_rate, return_attention_mask=True) if isinstance(model, Wav2Vec2ForCTC) else None

    lm_tokenizer = AutoTokenizer.from_pretrained(args.lm_model, padding_side="right", use_fast=False,)
    correction_lm = AutoModelForCausalLM.from_pretrained(args.lm_model, torch_dtype=torch.float16, use_flash_attention_2=True).to(args.device)
    correction_lm.eval()

    

    if isinstance(model, Wav2Vec2ForCTC):
        decoder_processor = Wav2Vec2ProcessorWithLM.from_pretrained(args.processor)
    elif isinstance(model, EncoderDecoderASR):
        decoder_processor = None
    elif isinstance(model, nemo_asr.models.EncDecCTCModelBPE):
        decoder_processor = BeamSearchDecoderCTC(
            alphabet=Alphabet(labels=model.decoder.vocabulary+[""], is_bpe=True),
            language_model=LanguageModel.load_from_dir(args.processor),
        )
    elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
        decoder_processor = BeamRNNTInfer(
            model.decoding.decoding.decoder.to(args.device),
            model.decoding.decoding.joint.to(args.device),
            beam_size=args.beam_width,
            return_best_hypothesis=False,
        )

    episodic = args.episodic
    steps = args.steps

    if episodic:
        original_model_state, original_optimizer_state, original_scheduler_state = copy_model_and_optimizer(model, optimizer, scheduler)

    for batch_idx, batch in enumerate(dataset):
        if args.dataset_name == "commonvoice" and batch_idx >= 1000:
            break

        lens, wavs, texts, _ = batch
        if isinstance(model, Wav2Vec2ForCTC):
            wavs = processor(wavs, sampling_rate=args.sample_rate, return_tensors="pt", padding="longest").input_values.to(args.device)
        else:
            wavs = pad_sequence([torch.from_numpy(wav) for wav in wavs], batch_first=True).to(args.device)
        lens = lens.to(args.device)

        gt_texts.extend(texts)
        ori_transcription = transcribe_batch(args, original_model, processor, wavs, lens)
        ori_transcriptions.extend(ori_transcription)
        ori_wer = wer(list(texts), list(ori_transcription))

        logger.info(f"{batch_idx}/{len(dataset)}")
        logger.info(f"gt text: {' '.join(list(texts))}")
        logger.info(f"original WER: {ori_wer}")
        logger.info(f"original text: {' '.join(list(ori_transcription))}")

        if episodic:
            model, optimizer, scheduler = load_model_and_optimizer(model, optimizer, scheduler, original_model_state, original_optimizer_state, original_scheduler_state)

        for step_idx in range(1, steps + 1):
            model = set_rnn_to_train(model)
            forward_and_adapt(args, model, decoder_processor, optimizer, scheduler, wavs, lens, step_idx)
            transcription = transcribe_batch(args, model, processor, wavs, lens)

            if step_idx in [1, 3, 5, 10, 20, 40]:
                transcription_list = eval(f"transcriptions_{step_idx}")
                transcription_list.extend(transcription)
                ada_wer = wer(list(texts), list(transcription))
                logger.info(f"adapt-{step_idx} WER: {ada_wer}")
                logger.info(f"adapt-{step_idx} text: {' '.join(list(transcription))}")

        gc.collect()
        torch.cuda.empty_cache()
        logger.info("\n")

    logger.info(OmegaConf.to_yaml(args))
    logger.info(f"number of data : {len(dataset)}")
    logger.info(f"original WER: {wer(gt_texts, ori_transcriptions)}")
    for step_idx in [1, 3, 5, 10, 20, 40]:
        if step_idx > steps:
            break
        transcription_list = eval(f"transcriptions_{step_idx}")
        logger.info(f"TTA-{step_idx}: {wer(gt_texts, transcription_list)}")



if __name__ == '__main__':
    main()