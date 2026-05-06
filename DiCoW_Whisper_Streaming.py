#!/usr/bin/env python3
import sys
import numpy as np
import librosa
from functools import lru_cache
import logging

import torch


logger = logging.getLogger(__name__)

@lru_cache(10**6)
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a

def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s]


class DiCoWASR:

    sep = " "
    SAMPLING_RATE = 16000

    def __init__(self, pipeline, lan="en", target_speaker_idx=0, logfile=sys.stderr):
        self.model = pipeline
        self.original_language = None if lan == "auto" else lan
        self.target_speaker_idx = target_speaker_idx
        self.logfile = logfile

    def transcribe(self, audio, init_prompt="", diar_mask=None):
        self.model.diarization_mask = diar_mask
        result = self.model(
            {"array": audio, "sampling_rate": self.SAMPLING_RATE},
            return_timestamps=True,
        )
        self.model.diarization_mask = None
        return result

    def ts_words(self, result):
        spk_output = result["per_spk_outputs"][self.target_speaker_idx]
        if not spk_output.strip():
            return []
        segments = self._extract_segments_with_timing(spk_output)
        if not segments:
            logger.warning(f"DiCoWASR: output has text but no timestamps, skipping: {spk_output!r}")
        return segments

    def segments_end_ts(self, result):
        spk_output = result["per_spk_outputs"][self.target_speaker_idx]
        segments = self._extract_segments_with_timing(spk_output)
        return [end for _, end, _ in segments]

    @staticmethod
    def _extract_segments_with_timing(text):
        import re
        pattern = r'<\|([\d.]+)\|>'
        matches = list(re.finditer(pattern, text))
        segments = []
        for i in range(0, len(matches) - 1, 2):
            start = float(matches[i].group(1))
            end = float(matches[i + 1].group(1))
            seg_text = text[matches[i].end():matches[i + 1].start()].strip()
            if seg_text:
                segments.append((start, end, seg_text))
        return segments


class HypothesisBuffer:

    def __init__(self, logfile=sys.stderr):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []

        self.last_commited_time = 0
        self.last_commited_word = None

        self.logfile = logfile

    def insert(self, new, offset):
        # compare self.commited_in_buffer and new. 
        # Inserts only the words in new that extend the commited_in_buffer, it means they are roughly behind last_commited_time and new in content
        # the new tail is added to self.new
        
        # 1. Fix the timestamp (relative -> absolute time)
        # Whisper output (relative): [(0.0, 0.5, "good"), (0.5, 1.0, "morning")]
        # offset = 10.0  ← audio buffer started at t=10s in the file
        # After: [(10.0, 10.5, "good"), (10.5, 11.0, "morning")]
        new = [(a+offset,b+offset,t) for a,b,t in new]

        # 2. Keep only words AFTER the committed boundary
        # last_commited_time = 10.5  (we already committed "good" which ended at 10.5)
        # [(10.0, 10.5, "good"),    ← starts at 10.0, which is < 10.4 → DROPPED
        #  (10.5, 11.0, "morning")] ← starts at 10.5, which is > 10.4 → KEPT
        # self.new = [(10.5, 11.0, "morning")]
        self.new = [(a,b,t) for a,b,t in new if a > self.last_commited_time-0.1]

        # 3. 5-gram dedup:
        # Motivation: remove any duplicated words slipped through the filter.
        # Check if the first N words of self.new match the last N words of commited_in_buffer. 
        # If they match → drop them from self.new.
        if len(self.new) >= 1:
            a,b,t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # it's going to search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in commited and new. If they are, they're dropped.
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1,min(min(cn,nn),5)+1):  # 5 is the maximum 
                        c = " ".join([self.commited_in_buffer[-j][2] for j in range(1,i+1)][::-1])
                        tail = " ".join(self.new[j-1][2] for j in range(1,i+1))
                        if c == tail:
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            words_msg = " ".join(words)
                            logger.debug(f"removing last {i} words: {words_msg}")
                            break

    def flush(self):
        # returns commited chunk = the longest common prefix of 2 last inserts.
        # updates self.buffer, self.new, self.commited_in_buffer
        '''
        buffer: [(285.12, 287.02, 'i dunno i mean')]
        new:    [(285.12, 288.0,  'i dunno i mean it is')]

        → committed:  (285.12, 287.02, 'i dunno i mean')
        → new[0] becomes: (287.02, 288.0, 'it is')  ← waits in buffer for next iteration
        '''
        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            ba, bb, bt = self.buffer[0]

            # 1. Exact match: unchanged — commits new's entry with its own timestamps
            if nt == bt:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            
            # 2. Prefix match: commit buffer entry if new is a strict extension of it
            else:
                bt_words = bt.split()
                nt_words = nt.split()
                if len(nt_words) > len(bt_words) and nt_words[:len(bt_words)] == bt_words:
                    # new extends buffer — commit buffer's stable entry and timestamps
                    commit.append((ba, bb, bt))
                    self.last_commited_word = bt
                    self.last_commited_time = bb
                    self.buffer.pop(0)
                    # trim new[0] to just the new words, using bb as the split point
                    remainder = ' '.join(nt_words[len(bt_words):])
                    self.new[0] = (bb, nb, remainder)
                else:
                    break
        
        # update buffer
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer

class OnlineASRProcessor:

    SAMPLING_RATE = 16000
    DIAR_FPS = 50  # diarization mask frame rate (frames per second)

    def __init__(self, asr, logfile=sys.stderr):
        self.asr = asr
        self.logfile = logfile
        self.init()

    def init(self, offset=None):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([],dtype=np.float32)
        self.diarisation_buffer = None  # [num_speakers, num_frames] torch.Tensor, or None if not using diarization
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.buffer_time_offset = 0
        if offset is not None:
            self.buffer_time_offset = offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited = []

    def insert_audio_chunk(self, audio, mask_chunk=None):
        self.audio_buffer = np.append(self.audio_buffer, audio)
        if mask_chunk is not None:
            if self.diarisation_buffer is None:
                self.diarisation_buffer = mask_chunk
            else:
                self.diarisation_buffer = torch.cat([self.diarisation_buffer, mask_chunk], dim=-1)

    def prompt(self):
        # Extract the prompt and context from the committed words
        """Returns a tuple: (prompt, context), where "prompt" is a 200-character suffix of commited text that is inside of the scrolled away part of audio buffer. 
        "context" is the commited text that is inside the audio buffer. It is transcribed again and skipped. It is returned only for debugging and logging reasons.
        """
        # 1. Find where buffer starts in the transcript
        # self.commited[:k] → text before current audio buffer
        # self.commited[k:] → text inside current buffer
        k = max(0,len(self.commited)-1)
        while k > 0 and self.commited[k-1][1] > self.buffer_time_offset:
            k -= 1

        # 2. Extract prompt and limit to 200 characters
        p = self.commited[:k]
        p = [t for _,_,t in p]
        prompt = []
        l = 0
        while p and l < 200:  # 200 characters prompt size
            x = p.pop(-1)
            l += len(x)+1
            prompt.append(x)

        # 3. Get context/"non-prompt" (current buffer text)
        non_prompt = self.commited[k:]

        # return (prompt, context)
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(t for _,_,t in non_prompt)

    def process_iter(self):
        """Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, "").
        The non-emty text is confirmed (committed) partial transcript.
        """
        buf_start = self.buffer_time_offset
        buf_end = self.buffer_time_offset + len(self.audio_buffer) / self.SAMPLING_RATE
        buf_dur = len(self.audio_buffer) / self.SAMPLING_RATE
        logger.debug("=" * 60)
        logger.debug(f"[OnlineASRProcessor] audio buffer: start={buf_start:.3f}s  end={buf_end:.3f}s  duration={buf_dur:.3f}s")
        logger.debug(f"[OnlineASRProcessor] buffer_time_offset: {self.buffer_time_offset:.3f}s")
        logger.debug(f"[OnlineASRProcessor] committed words so far: {self.commited}")

        prompt, non_prompt = self.prompt()
        logger.debug(f"PROMPT: {prompt}")
        logger.debug(f"CONTEXT: {non_prompt}")
        logger.debug(f"transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}")
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt, diar_mask=self.diarisation_buffer)
        # transform to [(beg,end,"word1"), ...]
        tsw = self.asr.ts_words(res)
        logger.debug("[Whisper Output]:")
        logger.debug(tsw)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        tb = self.transcript_buffer
        logger.debug("[HypothesisBuffer] after insert:")
        logger.debug(f"  transcription buffer : {tb.buffer}")
        logger.debug(f"  new                  : {tb.new}")
        logger.debug(f"  commited_in_buffer   : {tb.commited_in_buffer}")
        logger.debug(f"  last_commited_time   : {tb.last_commited_time:.3f}s")

        o = self.transcript_buffer.flush()
        logger.debug("[HypothesisBuffer] after flush:")
        logger.debug(f"  transcription buffer : {tb.buffer}")
        logger.debug(f"  new                  : {tb.new}")
        logger.debug(f"  commited_in_buffer   : {tb.commited_in_buffer}")
        logger.debug(f"  last_commited_time   : {tb.last_commited_time:.3f}s")
        logger.debug(f"  flushed (committed)  : {o}")

        self.commited.extend(o)
        completed = self.to_flush(o)
        logger.debug(f">>>>COMPLETE NOW: {completed}")
        the_rest = self.to_flush(self.transcript_buffer.complete())
        logger.debug(f"INCOMPLETE: {the_rest}")

        self.trim_buffer()

        logger.debug(f"len of buffer now: {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f}")
        return self.to_flush(o)

    def trim_buffer(self):
        """Trim the audio buffer after each inference call.

        On commit: trim to last_commited_time so the next inference starts from
        the committed boundary. This prevents re-outputting already-committed
        segments whose filtered-out start time would clear the remainder in the
        HypothesisBuffer.

        Safety fallback: if nothing committed but buffer exceeds 30s (Whisper's
        hard window limit), trim to the last committed word to avoid silent
        truncation of the tail.
        """
        t = self.transcript_buffer.last_commited_time
        if t > self.buffer_time_offset:
            # New commitment — trim to the committed boundary
            logger.debug(f"--- trimming buffer to last committed time {t:.2f}s")
            self.chunk_at(t)
        elif len(self.audio_buffer) / self.SAMPLING_RATE > 30:
            # Safety: buffer approaching Whisper's 30s hard limit with no new commits
            if self.commited:
                t = self.commited[-1][1]
                logger.debug(f"--- safety trim to last committed word at {t:.2f}s")
                self.chunk_at(t)

    def chunk_at(self, time):
        """trims the hypothesis, audio buffer, and diarisation buffer at "time"
        """
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds*self.SAMPLING_RATE):]
        if self.diarisation_buffer is not None:
            cut_frames = round(cut_seconds * self.DIAR_FPS)
            self.diarisation_buffer = self.diarisation_buffer[:, cut_frames:]
        self.buffer_time_offset = time

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        logger.debug(f"last, noncommited: {f}")
        self.buffer_time_offset += len(self.audio_buffer)/16000
        return f


    def to_flush(self, sents, sep=None, offset=0, ):
        # [(word-level timestamps)] -> (segment-level output)
        
        # concatenates the timestamped words or sentences into one sequence that is flushed in one line
        # sents: [(beg1, end1, "sentence1"), ...] or [] if empty
        # return: (beg1,end-of-last-sentence,"concatenation of sentences") or (None, None, "") if empty
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sents)
        if len(sents) == 0:
            b = None
            e = None
        else:
            b = offset + sents[0][0]
            e = offset + sents[-1][1]
        return (b,e,t)

class VACOnlineASRProcessor(OnlineASRProcessor):
    '''Wraps OnlineASRProcessor with VAC (Voice Activity Controller). 

    It works the same way as OnlineASRProcessor: it receives chunks of audio (e.g. 0.04 seconds), 
    it runs VAD and continuously detects whether there is speech or not. 
    When it detects end of speech (non-voice for 500ms), it makes OnlineASRProcessor to end the utterance immediately.
    '''

    def __init__(self, online_chunk_size, *a, **kw):
        self.online_chunk_size = online_chunk_size

        self.online = OnlineASRProcessor(*a, **kw)

        # VAC:
        import torch
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        from silero_vad_iterator import FixedVADIterator
        self.vac = FixedVADIterator(model)  # we use the default options there: 500ms silence, 100ms padding, etc.  

        self.logfile = self.online.logfile
        self.init()

    def init(self):
        self.online.init()
        self.vac.reset_states()
        self.current_online_chunk_buffer_size = 0

        self.is_currently_final = False

        self.status = None  # or "voice" or "nonvoice"
        self.audio_buffer = np.array([],dtype=np.float32)
        self.buffer_offset = 0  # in frames

    def clear_buffer(self):
        self.buffer_offset += len(self.audio_buffer)
        self.audio_buffer = np.array([],dtype=np.float32)


    def insert_audio_chunk(self, audio):
        res = self.vac(audio)
        self.audio_buffer = np.append(self.audio_buffer, audio)

        if res is not None:
            frame = list(res.values())[0]-self.buffer_offset
            if 'start' in res and 'end' not in res:
                self.status = 'voice'
                send_audio = self.audio_buffer[frame:]
                self.online.init(offset=(frame+self.buffer_offset)/self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.clear_buffer()
            elif 'end' in res and 'start' not in res:
                self.status = 'nonvoice'
                send_audio = self.audio_buffer[:frame]
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
            else:
                beg = res["start"]-self.buffer_offset
                end = res["end"]-self.buffer_offset
                self.status = 'nonvoice'
                send_audio = self.audio_buffer[beg:end]
                self.online.init(offset=(beg+self.buffer_offset)/self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
        else:
            if self.status == 'voice':
                self.online.insert_audio_chunk(self.audio_buffer)
                self.current_online_chunk_buffer_size += len(self.audio_buffer)
                self.clear_buffer()
            else:
                # We keep 1 second because VAD may later find start of voice in it.
                # But we trim it to prevent OOM. 
                self.buffer_offset += max(0,len(self.audio_buffer)-self.SAMPLING_RATE)
                self.audio_buffer = self.audio_buffer[-self.SAMPLING_RATE:]


    def process_iter(self):
        if self.is_currently_final:
            return self.finish()
        elif self.current_online_chunk_buffer_size > self.SAMPLING_RATE*self.online_chunk_size:
            self.current_online_chunk_buffer_size = 0
            ret = self.online.process_iter()
            return ret
        else:
            print("no online update, only VAD", self.status, file=self.logfile)
            return (None, None, "")

    def finish(self):
        ret = self.online.finish()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        return ret


class DiarisationOnlineASRProcessor(OnlineASRProcessor):
    """
    Wraps OnlineASRProcessor with diarization-based speaker activity control.
    Analogous to VACOnlineASRProcessor but uses the diarization mask for the
    target speaker as the activity signal instead of the Silero VAD model.

    Detects 0→1 (speech start) and 1→0 (speech end) transitions in the target
    speaker's diarization row and mirrors VACOnlineASRProcessor's behaviour:
      - speech start: init self.online at the correct absolute offset, push audio from first active frame
      - speech end  : push audio up to last active frame, set is_currently_final=True
      - continuing voice  : push full buffered audio to self.online, clear buffer
      - continuing silence: keep at most 1s of audio in case speech starts mid-buffer
    """

    def __init__(self, online_chunk_size, asr, target_speaker_idx=0, logfile=sys.stderr):
        self.online_chunk_size = online_chunk_size
        self.target_speaker_idx = target_speaker_idx
        self.online = OnlineASRProcessor(asr, logfile=logfile)
        self.logfile = self.online.logfile
        self.init()

    def init(self):
        self.online.init()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        self._inference_has_run = False  # True once process_iter() has run inference for the current segment
        self.status = None  # None, "voice", or "nonvoice"
        self.audio_buffer = np.array([], dtype=np.float32)
        self.diar_buffer = None  # [num_speakers, num_frames] torch.Tensor or None
        self.buffer_offset = 0  # in audio samples

    def clear_buffer(self):
        self.buffer_offset += len(self.audio_buffer)
        self.audio_buffer = np.array([], dtype=np.float32)
        self.diar_buffer = None

    def insert_audio_chunk(self, audio, mask_chunk=None):
        # Stage into outer pre-buffer; self.online only receives trimmed slices at transitions
        self.audio_buffer = np.append(self.audio_buffer, audio)
        if mask_chunk is not None:
            self.diar_buffer = mask_chunk if self.diar_buffer is None \
                               else torch.cat([self.diar_buffer, mask_chunk], dim=-1)

        if mask_chunk is None:
            return

        # True if target speaker is active anywhere in this 1s chunk
        chunk_active = bool(mask_chunk[self.target_speaker_idx].any().item())

        # ── Case 1: 0→1  speech START ─────────────────────────────────
        # Trim leading silence: find first active frame in accumulated diar_buffer,
        # init self.online at that absolute offset, push only the speech portion.
        # [0 0 0 0 | 1 1 1] → first_frame=4 → push audio[start_sample:]
        if chunk_active and self.status != 'voice':
            combined_row = self.diar_buffer[self.target_speaker_idx]
            active_frames = combined_row.nonzero(as_tuple=True)[0]
            first_frame = int(active_frames[0].item()) if len(active_frames) else 0
            start_sample = round(first_frame / self.DIAR_FPS * self.SAMPLING_RATE)

            send_audio = self.audio_buffer[start_sample:]
            send_mask  = self.diar_buffer[:, first_frame:]
            offset_sec = (self.buffer_offset + start_sample) / self.SAMPLING_RATE

            logger.debug(f"[DiarisationOnlineASRProcessor] SPEECH START at {offset_sec:.3f}s")
            self.online.init(offset=offset_sec)
            self.online.insert_audio_chunk(send_audio, send_mask)
            self.current_online_chunk_buffer_size += len(send_audio)
            self.status = 'voice'
            self.clear_buffer()

        # ── Case 2: 1→0  speech END ───────────────────────────────────
        # Trim trailing silence: find last active frame in accumulated diar_buffer,
        # push only up to that point, then signal finish() via is_currently_final.
        # [1 1 1 | 0 0 0 0] → last_frame=3 → push audio[:end_sample]
        elif not chunk_active and self.status == 'voice':
            combined_row = self.diar_buffer[self.target_speaker_idx]
            active_frames = combined_row.nonzero(as_tuple=True)[0]
            last_frame  = int(active_frames[-1].item()) + 1 if len(active_frames) else 0
            end_sample  = round(last_frame / self.DIAR_FPS * self.SAMPLING_RATE)

            send_audio = self.audio_buffer[:end_sample]
            send_mask  = self.diar_buffer[:, :last_frame]

            end_sec = (self.buffer_offset + end_sample) / self.SAMPLING_RATE
            logger.debug(f"[DiarisationOnlineASRProcessor] SPEECH END at {end_sec:.3f}s")
            self.online.insert_audio_chunk(send_audio, send_mask)
            self.current_online_chunk_buffer_size += len(send_audio)
            self.is_currently_final = True  # process_iter() will call finish() next
            self.status = 'nonvoice'
            self.clear_buffer()

        else:
            # ── Case 3: continuing VOICE — no transition, push full chunk to self.online
            if self.status == 'voice':
                self.online.insert_audio_chunk(self.audio_buffer, self.diar_buffer)
                self.current_online_chunk_buffer_size += len(self.audio_buffer)
                self.clear_buffer()

            # ── Case 4: continuing SILENCE — push nothing to self.online.
            # Keep at most 1s as a lookback window: if speech starts mid-buffer,
            # Case 1 can search diar_buffer across this window for the exact first_frame.
            else:
                max_samples = self.SAMPLING_RATE
                trim = max(0, len(self.audio_buffer) - max_samples)
                if trim > 0:
                    self.buffer_offset += trim  # advance offset to keep absolute timestamps correct
                    self.audio_buffer = self.audio_buffer[-max_samples:]
                    if self.diar_buffer is not None:
                        trim_frames = round(trim / self.SAMPLING_RATE * self.DIAR_FPS)
                        self.diar_buffer = self.diar_buffer[:, trim_frames:]

    def process_iter(self):
        if self.is_currently_final:
            return self.finish()
        elif self.current_online_chunk_buffer_size > self.SAMPLING_RATE * self.online_chunk_size:
            self.current_online_chunk_buffer_size = 0
            self._inference_has_run = True
            return self.online.process_iter()
        else:
            logger.debug(f"no online update, diarization status={self.status}")
            return (None, None, "")

    def finish(self):
        # Force inference only if no inference has run yet for this segment
        # (i.e. short segment < online_chunk_size). Avoids re-running on segments
        # that already had inference with a small leftover, which degrades output.
        if not self._inference_has_run and self.current_online_chunk_buffer_size > 0:
            self.online.process_iter()
        ret = self.online.finish()
        self.current_online_chunk_buffer_size = 0
        self._inference_has_run = False
        self.is_currently_final = False
        return ret

