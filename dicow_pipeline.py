import os
import re
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
from librosa import load as libr_load
from soundfile import write as sf_write


class DiCoW_Pipeline(AutomaticSpeechRecognitionPipeline):

    ############################################
    # STNO MASK
    ############################################
    @staticmethod
    def get_stno_mask(diar_mask, s_index):

        non_target_mask = torch.ones((diar_mask.shape[0],), dtype=torch.bool)
        non_target_mask[s_index] = False

        sil_frames = (1 - diar_mask).prod(axis=0)
        anyone_else = (1 - diar_mask[non_target_mask]).prod(axis=0)

        target_spk = diar_mask[s_index] * anyone_else
        non_target_spk = (1 - diar_mask[s_index]) * (1 - anyone_else)

        overlapping_speech = diar_mask[s_index] - target_spk

        stno_mask = torch.stack(
            [sil_frames, target_spk, non_target_spk, overlapping_speech],
            axis=0
        )

        return stno_mask

    ############################################
    # INIT
    ############################################
    def __init__(self, *args, speaker_embedding_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.speaker_embedding_model = speaker_embedding_model
        self.type = "seq2seq_whisper"
        self.diarization_mask = None

    ############################################
    # PREPROCESS
    ############################################
    def preprocess(self, inputs, chunk_length_s=0, stride_length_s=None):

        # Accept either {"array": np.ndarray, "sampling_rate": int} (preferred — pre-resampled)
        # or {"audio_filepath": path} (resamples on the fly, slower)
        if "array" in inputs:
            audio_input = {"array": inputs["array"], "sampling_rate": inputs["sampling_rate"]}
        else:
            mixed_audio_path = inputs["audio_filepath"]
            audio_arr, sr = libr_load(mixed_audio_path, sr=16000, mono=True)
            audio_input = {"array": audio_arr, "sampling_rate": sr}

        ####################################################
        # Load diarization mask
        ####################################################
        if self.diarization_mask is not None:
            diarization_mask = self.diarization_mask
        else:
            audio_name = os.path.basename(inputs.get("audio_filepath", "audio")).replace(".wav", "")
            diar_mask_path = os.path.join("diarisation_masks", f"{audio_name}_mask.pt")
            diarization_mask = torch.load(diar_mask_path)

        ####################################################
        # Run base Whisper preprocessing
        ####################################################
        generator = super().preprocess(
            audio_input,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s
        )

        samples = next(generator)

        ####################################################
        # Create STNO masks
        ####################################################

        # Align diarization mask length to mel features (same as original DiCoW pipeline).
        # Original builds the mask at exactly input_features.shape[-1] // 2 by zero-initialising
        # and filling speech regions, so we match that: pad with zeros (silence) or trim.
    
        target_len = samples["input_features"].shape[-1] // 2
        current_len = diarization_mask.shape[-1]
        if current_len < target_len:
            pad = torch.zeros(diarization_mask.shape[0], target_len - current_len,
                              dtype=diarization_mask.dtype, device=diarization_mask.device)
            diarization_mask = torch.cat([diarization_mask, pad], dim=-1)
            print(f"Zero-padded diarization_mask from {current_len} to {target_len} frames")
        elif current_len > target_len:
            diarization_mask = diarization_mask[:, :target_len]
            print(f"Trimmed diarization_mask from {current_len} to {target_len} frames")
        
        
        num_speakers = diarization_mask.shape[0]

        stno_masks = []
        for i in range(num_speakers):
            stno_mask = self.get_stno_mask(diarization_mask, i)
            stno_masks.append(stno_mask)
        stno_masks = torch.stack(stno_masks, axis=0)
        
        
        print("stno_masks shape:", stno_masks.shape)
        print()

        ####################################################
        # 5️⃣ Expand features for target speaker only
        ####################################################

        samples["stno_mask"] = stno_masks.to(
            samples["input_features"].device,
            dtype=samples["input_features"].dtype
        )

        samples["input_features"] = samples["input_features"].repeat(
            num_speakers,
            1,
            1
        )

        samples["attention_mask"] = torch.ones(
            samples["input_features"].shape[0],
            samples["input_features"].shape[2],
            dtype=torch.bool,
            device=samples["input_features"].device
        )

        ####################################################
        # Optional cleanup
        ####################################################

        if "num_frames" in samples:
            del samples["num_frames"]

        yield samples

    ############################################
    # FORWARD
    ############################################
    def _forward(self, model_inputs, return_timestamps=False, **generate_kwargs):
        attention_mask = model_inputs.pop("attention_mask", None)
        stride = model_inputs.pop("stride", None)
        segment_size = model_inputs.pop("segment_size", None)
        is_last = model_inputs.pop("is_last")

        if stride is not None and segment_size is not None:
            raise ValueError("segment_size must be used only when stride is None")

        # Consume values so we can let extra information flow freely through
        # the pipeline (important for `partial` in microphone)
        if "input_features" in model_inputs:
            inputs = model_inputs.pop("input_features")
        elif "input_values" in model_inputs:
            inputs = model_inputs.pop("input_values")
        else:
            raise ValueError(
                "Seq2Seq speech recognition model requires either a "
                f"`input_features` or `input_values` key, but only has {model_inputs.keys()}"
            )

        # custom processing for Whisper timestamps and word-level timestamps
        if return_timestamps and self.type == "seq2seq_whisper":
            generate_kwargs["return_timestamps"] = return_timestamps
            if return_timestamps == "word":
                generate_kwargs["return_token_timestamps"] = True
                generate_kwargs["return_segments"] = True
            generate_kwargs["input_features"] = inputs

        tokens = self.model.generate(
            attention_mask=attention_mask,
            **generate_kwargs,
            **model_inputs,
        )
        # whisper longform generation stores timestamps in "segments"
        if return_timestamps == "word" and self.type == "seq2seq_whisper":
            if "segments" not in tokens:
                out = {"tokens": tokens["sequences"], "token_timestamps": tokens["token_timestamps"]}
            else:
                token_timestamps = [
                    torch.cat([segment["token_timestamps"] for segment in segment_list])
                    for segment_list in tokens["segments"]
                ]
                out = {"tokens": tokens["sequences"], "token_timestamps": token_timestamps}
        else:
            out = {"tokens": tokens}
        if self.type == "seq2seq_whisper":
            if stride is not None:
                out["stride"] = stride

        # Leftover
        extra = model_inputs
        return {"is_last": is_last, **out, **extra}

    @staticmethod
    def postprocess_text(input_string):
        pattern = r"<\|([\d.]+)\|>"
        matches = re.finditer(pattern, input_string)
        timestamps = [(float(match.group(1)), match.start(), match.end()) for match in matches]
        if not timestamps or len(timestamps) <= 2:
            return input_string

        # The whole algorithm boils down to either removing the entire chain of timestamps - the case where all of them are the same (i.e. ...<a><a><a>... -> ......)
        # or removing all but the corner ones (i.e. <a><b><c><c><d> -> <a><d>) - the case where we have end and start timestamps and some rubbish in-between.

        processed_timestamps = []
        i = 0
        while i < len(timestamps):
            ts, st, et = timestamps[i]

            if i < len(timestamps) - 1 or processed_timestamps[-1][-1] != st:
                processed_timestamps.append((ts, st, et))

            if i == len(timestamps) - 1:
                break

            j = i + 1
            nts, nst, net = timestamps[j]
            all_equal_ts = nts == ts
            prev_et = et
            while nst - prev_et == 0:
                # Skip all but the last timestamp. If the last in the chain has the same TS as the processed_timestamps tail, pop processed_timestamps.
                # If not, append it while skipping all the previous ones.
                # In other words, keep appending (-2, X, X) as long as the next one is in the chain and then decide what to do with the last one if the next one is not in the chain.

                if j == len(timestamps) - 1:
                    if net == len(input_string) and prev_et != nst:
                        processed_timestamps.append((nts, nst, net))
                        j += 1
                    break
                else:
                    if timestamps[j + 1][1] - net == 0:
                        processed_timestamps.append((-2, nst, net))
                    else:
                        if all_equal_ts:
                            # If there's a chain of eq timestamps at the beginning, we need to keep at least one.
                            if i != 0:
                                processed_timestamps[i] = (-1, st, et)
                            processed_timestamps.append((-2, nst, net))
                        else:
                            # If there's a chain of tags at the beginning with all ts not being equal, we need to keep the last one.
                            if i == 0:
                                processed_timestamps[i] = (-2, st, et)
                            processed_timestamps.append((nts, nst, net))
                        j += 1
                        break

                j += 1
                prev_et = net
                nts, nst, net = timestamps[j]
                all_equal_ts = all_equal_ts and nts == ts

            i = j

        result = []
        prev_end = 0
        for i, (ts, st, et) in enumerate(processed_timestamps):
            result.append(f'{input_string[prev_end:st]}')
            if ts == -1:
                result.append(' ')
            elif ts == -2:
                # Empty string, so no need to append anything
                pass
            else:
                result.append(f'<|{ts:.2f}|>')
            prev_end = et

        return "".join(result)

    def postprocess(
            self, model_outputs, decoder_kwargs: Optional[Dict] = None, return_timestamps=None, return_language=None
    ):
        per_spk_outputs = self.tokenizer.batch_decode(
            model_outputs[0]['tokens'], decode_with_timestamps=True, skip_special_tokens=True
        )
        formatted_lines = []
        for spk, text in enumerate(per_spk_outputs):
            processed_text = self.postprocess_text(text)

            # Split on each timestamp pair
            # This regex finds "<|start|>...<|end|>" pairs with everything inside
            segments = re.findall(r"(<\|\d+\.\d+\|>.*?<\|\d+\.\d+\|>)", processed_text)
            # Build the output for this speaker
            speaker_header = f"🗣️ Speaker {spk}:\n"
            speaker_body = "\n".join(segments)
            formatted_lines.append(f"{speaker_header}{speaker_body}")

        full_text = "\n\n".join(formatted_lines)
        return {"text": full_text, "per_spk_outputs": per_spk_outputs}
