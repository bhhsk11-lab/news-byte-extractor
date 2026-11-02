# Free voice assistant on Cloud Run (CPU only)

A small server that gives your **AI Chat Firefox extension** a speaking and listening voice, running on your own
Cloud Run service (2 vCPU, 1 GiB target). No GPU. Nothing is billed while you are not talking.

| Job | Model | Why |
|---|---|---|
| Speech → text | **Moondream Parakeet Redux** (Parakeet 0.6B v3 with a ternary-quantised encoder, weights 178 MB, 25 languages, via Moondream's Photon engine). The only speech-to-text model in the default build | Small memory, fast on CPU, built-in voice detection and chunking |
| Text → speech | **Kokoro-82M** (fp32, ONNX) | Natural English voice, runs on CPU |

The extension already speaks the OpenAI speech API, so **no extension changes are needed**. You only paste a URL and a key.

---

## 1. Deploy (about 10 minutes, once)

1. Open **Google Cloud Shell** (the `>_` icon in the Cloud Console). It already has `gcloud` logged in.
2. Upload this folder (Cloud Shell menu `⋮` → *Upload*) and open it:
   ```bash
   unzip news-byte-extractor.zip -d news-byte-extractor && cd news-byte-extractor
   ```
3. Deploy:
   ```bash
   ./deploy.sh YOUR_PROJECT_ID
   ```
   The first build downloads the models into the image and takes 5 to 10 minutes. When it finishes it prints your
   **URL** and a random **API key**. The key is also saved in `.api_key`.

The script uses region **asia-south1 (Mumbai)**. Per Google's location list it is a Tier 1 region, so you get the full free
tier, and it is the closest region to Jaipur. Change it with `REGION=... ./deploy.sh`.

## 2. Test it (before touching the extension)

```bash
URL=https://news-byte-extractor-xxxxx-el.a.run.app      # printed by deploy.sh
KEY=$(cat .api_key)

curl $URL/health                                  # first call wakes the server: 15-30 s, then instant

curl -X POST $URL/v1/audio/speech -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
     -d '{"input":"Hello, this is a test. नमस्ते, कैसे हैं आप?","voice":"marin"}' -o test.mp3

curl -X POST $URL/v1/audio/transcriptions -H "Authorization: Bearer $KEY" -F file=@test.mp3 -F language=en
```
Play `test.mp3`, then check that the last command returns the same text as JSON.

## 3. Connect the extension

Extension popup → **Speech engines & images**:

| Field | Value |
|---|---|
| Speech & image API base URL | `https://news-byte-extractor-xxxxx-el.a.run.app/v1` (keep the `/v1`) |
| Speech & image API key | the key from `.api_key` |
| Speech input (your microphone) | **API** |
| Transcription model | anything. The server ignores it (always Parakeet Redux) |
| Speech model | anything. The server ignores it |
| Language I speak | *English*. Redux picks the language itself and the server ignores this field (the model is multilingual; English is not enforced) |

Firefox will ask permission to contact the new server: click **Allow**. Then pick a voice: any female name
(Marin, Coral, Nova ...) gives the female Kokoro voice, any male name (Cedar, Ash ...) gives the male one.
Hindi text automatically switches to a Hindi voice of the same gender.

**Tip:** open `https://YOUR-URL/health` in a tab when you sit down to talk. It wakes the server, so your first
sentence does not wait for a cold start.

---

## What it costs (free tier)

From Google's Cloud Run pricing page, per month, per billing account, Tier 1 regions:
**180,000 vCPU-seconds, 360,000 GiB-seconds, 2 million requests.**

With 2 vCPU the CPU allowance is the limit: 180,000 / 2 = **90,000 seconds, about 25 hours of busy time per month**
(memory only uses 180,000 of its 360,000). Cloud Run bills only while a request is running, so only the seconds spent
transcribing or speaking count. A typical spoken turn is roughly 5 to 10 busy seconds, which is on the order of
**10,000+ turns a month** (an estimate, not a measurement).

Things that are **not** free, and what to do:
* **`--min-instances 1`** keeps the server warm but bills idle time all month (roughly $25, by the idle rates on the pricing page). Leave it at 0.
* The container image (about 2 GB) sits in Artifact Registry. Storage above its free amount costs cents per month.
* Outbound audio is small (MP3 at 48 kbps, about 21 MB per hour of speech), so egress is pennies.
* Because your account is upgraded (billing enabled), anything past the free tier is charged. `deploy.sh` allows
  `--max-instances 3` by default; use `MAX_INSTANCES=1 ./deploy.sh` so it can never scale into a big bill. Also set a **budget alert** of $1 to $5 in
  *Billing → Budgets & alerts*.

## Tuning

Change these with `gcloud run services update news-byte-extractor --region asia-south1 --update-env-vars NAME=value`:

| Setting | Default | When to change |
|---|---|---|
| `STT_BACKEND` | `redux` | Set in the Dockerfile (`ARG STT_BACKEND`), because the build downloads that model. `sherpa` = the old English-only Parakeet TDT 0.6B v2 int8 (+ Silero VAD), needs about 1.2-1.65 GB RAM for STT alone |
| `STT_WORKERS` | `1` | Recordings transcribed at the same moment. Redux always uses 1 |
| `STT_SEPARATE_COPIES` | `0` | `0` = the workers share ONE loaded Parakeet (about 1.2 GB). `1` = one Parakeet per worker (+1.2 GB each, needs `MEMORY=5Gi`) |
| `STT_THREADS` | `2` | CPU threads for speech-to-text (`deploy.sh` sets it from `CPU`) |
| `VAD_MODE` | `neural` | `neural` = Silero VAD network, `energy` = simple loudness rule (also the automatic fallback) |
| `VAD_THRESHOLD` (see below) | `0.4` | lowered from 0.5 so soft speech is kept |
| `LEVELER` | `1` | Evens out loudness before the voice detector (see "Level control"). `0` = old peak-normalise |
| `TARGET_DBFS` | `-20` | Level every speech piece is set to, in dB below full scale |
| `MAX_GAIN_DB` | `40` | Most a quiet passage may be lifted |
| `PIECE_MAX_S` | `12` | Longest piece sent to the model; longer speech is cut at pauses |
| `RETRY_DROPS` | `1` | Decode a piece a second time when it looks like words were lost |
| `VAD_THRESHOLD` | `0.5` | Speech probability needed to count as speech. `0.3` if quiet speech is cut, `0.7` if noise gets through |
| `STT_QUEUE_MAX` | `8` | If more recordings than this are waiting, the server answers HTTP 429 instead of hanging |
| `MAX_AUDIO_SECONDS` | `120` | Longest recording accepted |
| `TTS_WORKERS` | `1` | Sentences spoken at the same moment. Each Kokoro copy needs about 0.55 GB RAM |
| `TRIM_TTS` | `1` | Cuts silence at the start/end of speech |
| `TTS_CACHE_MAX` | `64` | Recent spoken sentences replayed instantly. `0` turns it off |
| `MP3_KBPS` | `48` | `64` = better sound, more data |
| `VOICE_EN_FEMALE` / `VOICE_EN_MALE` | `af_heart` / `am_michael` | any Kokoro English id |

## Speech-to-text: Moondream Parakeet Redux (build `voice-v14-moondream-redux-leveler`)

* Default model: **moondream/parakeet-redux**, run on CPU with `moondream.photon(..., device="cpu")`. It is NVIDIA's Parakeet 0.6B **v3** with the encoder weights reduced to -1/0/+1; weights shrink from 1.2 GB to 178 MB (per Moondream's release post, Sep 2026).
* **Not English only.** It supports 25 languages and detects the language automatically. If someone speaks another language the server will transcribe that language. The old English-only v2 model is still available with `STT_BACKEND=sherpa`.
* Accuracy, from Moondream's own tests (not run here): English average word error 6.55% (original v3: 6.26%); with background noise 9.04% (original: 6.72%). Redux is weaker in noise, so keep the microphone close and the room quiet. The README of the previous build quoted about 6.0% for v2 on a different benchmark, so those numbers cannot be compared directly.
* Long recordings: the server cuts speech at pauses (Silero VAD) into pieces of at most 12 s and decodes each piece alone (see "Level control and lost sentences").
* **Memory is not measured yet.** The logs print `MEMORY after STT load`, `after TTS load` and `after warm-up`: read them on the first deploy. Kokoro fp32 alone is about 0.55 GB, so 1 GiB total is tight. If Cloud Run reports `exceeded memory limit`, deploy with `MEMORY=2Gi`.
* **Not tested here:** this build was written without network access, so the Photon calls (`md.photon`, `transcribe(audio=ndarray, sample_rate=16000, timestamps="none")`) follow Moondream's documentation but have not been run. Photon's own dependencies (for example whether it pulls in PyTorch, which would make the image much bigger and use more RAM) are also unchecked: look at the build log and the image size.
* The model is loaded once at build time into the image (`fetch_stt.py`). If that download fails the build continues with a warning and the server downloads the weights at first start (slow cold start).
* One recording is transcribed at a time (Photon's thread-safety is not documented); the others wait in the queue (`STT_QUEUE_MAX`).
* **License:** CC-BY-4.0, same as NVIDIA's Parakeet. Keep the attribution: "Moondream Parakeet Redux, based on NVIDIA Parakeet TDT 0.6B v3". Check the Photon engine license on moondream.ai/photon before commercial use.
* `/health` shows `"build": "voice-v14-moondream-redux-leveler"` and `"stt": "moondream-parakeet-redux"`.

### Going back to the old model

Set `ARG STT_BACKEND=sherpa` in the Dockerfile and deploy with `MEMORY=3Gi ./deploy.sh`. The sections below about Silero VAD, 1.2-1.65 GB memory and two workers describe that backend.

## Level control and lost sentences (build `voice-v14`)

What the internet check (Nov 2026) found, and what was done:

* **Why quiet sentences vanish.** The speech model itself is fairly level-independent (it normalises its features per recording). What skips quiet sentences is everything in front of it: the silence gate, the voice detector and "loud enough" rules. The old energy rule dropped anything about 22 dB below the loudest moment. Fix: a slow automatic gain control (`audio_clean.level`) lifts quiet passages up to +40 dB (and lowers very loud ones by up to 12 dB) BEFORE the voice detector, then each speech piece is set to the same level (`normalize_piece`). Synthetic test: sentences at -10, -30 and -52 dBFS all reached the model at -20 dBFS. Background-only audio is not boosted on its own. A research paper on gain control also warns that badly chosen gain raises word errors, so the gain is smooth (about 0.2 s), has a ceiling, and a soft limiter instead of hard clipping.
* **"50 dB".** A recording has no absolute sound pressure: dB SPL depends on the microphone and the room. The equivalent here is a fixed digital level. Default `TARGET_DBFS=-20` (a normal, healthy speech level). Change it with `TARGET_DBFS=-26 ./deploy.sh`-style env updates if you want it lower.
* **Why whole sentences are lost (a second cause, found in several open-source projects, Oct-Nov 2026).** Parakeet TDT decoders sometimes return only blanks for several seconds of clear speech, especially after a pause of 1.5 s or more, in windows of 20-30 s, or when a window starts/ends right on the voice. Reports say Parakeet v3 (the base of Redux) loses speech at every window length tested (5 of 58 windows at 20 s), while v2 lost none up to 30 s. Fixes applied: speech is cut at pauses into pieces of at most 12 s (`PIECE_MAX_S`), long silences are never sent to the model, and a piece whose result looks too short (no words, or under 0.6 words per second in a piece of 4 s or more) is decoded again with 0.5 s of faint noise on both edges; the result with more words wins. `/health` shows `stt_recovered`, how many pieces this saved.
* **Neural networks added:** Silero VAD (voice detector) now runs for both backends, with threshold 0.4. **Not added, on purpose:** a noise-removal network. A 2026 study found speech enhancement lowers Parakeet and Whisper accuracy in every tested case (the models were trained on noisy audio and do not know enhancement artefacts). A high-pass filter below about 70 Hz is applied; that is not a neural network and is safe.
* **Accuracy honesty.** For English on a CPU, the old v2 backend (`STT_BACKEND=sherpa`) is more accurate and more robust than Redux (Moondream's own numbers: Redux 6.55% vs 6.26% for v3 on English, 9.04% vs 6.72% in noise; v2 about 6.0% on the Open ASR Leaderboard) but needs about 1.2-1.65 GB of RAM. Redux is chosen here for the 1 GB limit.
* **Untested options for later.** A third-party package `voxrt` (Rust runtime for Redux, Apache-2.0, Linux x86-64 wheels, no PyTorch) and community ONNX exports of Redux exist; they could use less RAM than Photon, but were not tested here.

## The audio neural networks: what is used, and what is not (internet check, Nov 2026; written for the sherpa backend)

**How YouTube-style captions are made.** Google does not publish its pipeline. Secondary sources describe it as one large neural speech model trained on a huge amount of audio (RNN-T or Conformer type), then language-model rescoring and context biasing, then punctuation and casing, then confidence filtering. The accuracy comes from the size and training data of that one model, not from a separate noise-removal network in front of it. Parakeet TDT 0.6B v2 is the same kind of model (a FastConformer transducer with punctuation built in), so this server already has that part.

**Added: a neural voice detector (Silero VAD).** A ~2 MB network decides where people really talk, and only those pieces go to Parakeet. Pauses under 0.5 s stay inside one piece, 0.2 s of real audio is kept around each piece so no first or last word is clipped, and speech longer than 20 s is cut at a pause chosen by the network (the old rule cut at the quietest moment). It is a voice detector, not a speech-to-text model, so Parakeet is still the only speech-to-text model. If the network file cannot load, the server falls back to the energy rule and logs `neural VAD unavailable`.

**Deliberately NOT added: a noise-removal network (DeepFilterNet, MetricGAN, RNNoise, GTCRN ...).** Published tests say it makes accuracy worse for modern ASR:
* A 2025 study with 500 recordings in nine noise conditions found that MetricGAN+ denoising raised the word error rate in all 30 tested setups. For Parakeet the increase was +1.4 to +5.9 points with background noise, up to +11.9 with white noise. Whisper was hurt more.
* A 2026 study of a large audio-separation model with Whisper found better-sounding audio but higher word errors.
* These tests used one enhancer (MetricGAN+) and one separation model (SAM-Audio), not every denoiser, and the authors say enhancement is not harmful in every setting. But I found no test showing a gain for Parakeet, so I did not ship one.

**Deliberately NOT added: hotwords (context biasing) for Parakeet.** It is the closest thing to YouTube's context biasing. sherpa-onnx supports it for NeMo transducers, but needs `modified_beam_search`, and a fix in sherpa-onnx pull request 4084 describes that mode dropping speech for NeMo TDT models. I could not confirm that the fix is in version 2.0.1, so a silent loss of words is possible. Test it yourself before turning it on.

**What really helps accuracy:** a good microphone close to your mouth, input volume that is not clipped (the log shows `clip=` and `snr=`), and no other people talking nearby.

## Parallel work and heavy load

* **Inside one server:** `STT_WORKERS=2` lets two recordings be transcribed at the same moment. By default both use the same loaded model: the sherpa-onnx maintainer states that one recognizer can be used by several threads, with a separate stream per recording (which this server does). I could not test real speed here, and I could not confirm that the Python layer lets both run truly in parallel. If `/health` shows `stt_busy` 2 but the log times look serial, set `STT_SEPARATE_COPIES=1` with `MEMORY=5Gi`.
* **Across servers:** `deploy.sh` allows `MAX_INSTANCES=3` with `--concurrency 4`. Cloud Run starts extra servers by itself when load grows. Use `MAX_INSTANCES=1 ./deploy.sh` to forbid scaling out.
* **Waiting line:** past `STT_QUEUE_MAX` waiting recordings the server answers 429.
* **Memory:** default `MEMORY=1Gi` (Redux). Use `MEMORY=3Gi` with the sherpa backend.
* Text-to-speech stays at one sentence at a time per server (`TTS_WORKERS=1`); speech-to-text and text-to-speech do run at the same time as each other.
* `/health` shows `stt_workers`, `stt_busy` and `stt_waiting`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Deploy fails with "failed to start and listen on PORT" | `gcloud run services logs read news-byte-extractor --region asia-south1 --limit 60`, look for `MODEL LOADING FAILED` |
| First sentence takes 15 to 30 s | Cold start. Open `/health` first |
| `401 Invalid API key` | The key in the extension differs from `.api_key` |
| `Container ... exceeded memory limit` | Redeploy with `MEMORY=2Gi` (Redux) or `MEMORY=4Gi` (sherpa), and read the `MEMORY after ...` log lines |
| Wrong words, or "heard NOTHING" in logs | Check the `peak= voice= snr= clip=` part of the log line: too quiet, clipped (microphone too loud) or noisy. Clipped audio turns words into other words: lower the microphone volume |
| Silence from the speaker | Look for `bad audio ... skipped` in the logs |

**Important:** do not swap in `kokoro-v1.0.int8.onnx`: it returned corrupt audio (NaN) in testing.

## Security

Every speech request needs your API key (`Authorization: Bearer ...`). To rotate it: delete `.api_key` and run `./deploy.sh` again.
