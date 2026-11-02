# Free voice assistant on Cloud Run (CPU only)

A small server that gives your **AI Chat Firefox extension** a speaking and listening voice, running on your own
Cloud Run service (2 vCPU / 2 GiB). No GPU. Nothing is billed while you are not talking.

| Job | Model | Why |
|---|---|---|
| Speech → text | **NVIDIA Parakeet TDT 0.6B v2** (int8) by default. Moonshine and faster-whisper are options | Runs on CPU, English only. See *Speech-to-text engines* below |
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
| Transcription model | anything (for example `whisper-small`). The server ignores it |
| Speech model | anything. The server ignores it |
| Language I speak | *English* (or Automatic). This build is English only |

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
* Because your account is upgraded (billing enabled), anything past the free tier is charged. `deploy.sh` sets
  `--max-instances 1` so it can never scale into a big bill. Also set a **budget alert** of $1 to $5 in
  *Billing → Budgets & alerts*.

## Tuning

Change these with `gcloud run services update news-byte-extractor --region asia-south1 --update-env-vars NAME=value`:

| Setting | Default | When to change |
|---|---|---|
| `WHISPER_BEAM` | `2` | `1` = fastest, `3`-`5` = most accurate, slower |
| `ALLOWED_LANGS` | `en` | Automatic language detection only chooses from these. Add more with `;` between them (gcloud treats commas specially), e.g. `en;hi;pa` |
| `WHISPER_HOTWORDS` | empty | Names / terms you say often, separated by `;`, e.g. `Jaipur;Claude;Kokoro`. Helps Whisper spell them right |
| `DENOISE` | `auto` | `auto` = clean only noisy recordings and double-check unsure ones, `on` = always try the cleaned version first, `off` = never |
| `SNR_CLEAN_DB` | `18` | Recordings cleaner than this are never noise-gated |
| `RETRY_LOGPROB` | `-0.50` | If Whisper's confidence is below this, the other version of the recording is also tried |
| `MAX_AUDIO_SECONDS` | `120` | Longest recording accepted (protects the 2 GiB memory limit) |
| `TRIM_TTS` | `1` | Cuts silence at the start/end of speech, so replies start sooner and sound tighter |
| `TTS_CACHE_MAX` | `64` | Number of recent spoken sentences kept in memory and replayed instantly. `0` turns it off |
| `MP3_KBPS` | `48` | `64` = better sound, more data |
| `VOICE_EN_FEMALE` / `VOICE_EN_MALE` | `af_heart` / `am_michael` | any Kokoro English id (`af_bella`, `bf_emma`, `am_adam` ...) |
| `VOICE_HI_FEMALE` / `VOICE_HI_MALE` | `hf_alpha` / `hm_omega` | `hf_beta` / `hm_psi` |
| `WHISPER_VAD` | `1` | `0` if it cuts off quiet speech |

Changing the **speech-to-text engine** needs a rebuild: edit `ARG STT_ENGINE=parakeet` in the `Dockerfile` (see the engine table below), then deploy again. Do not change `STT_ENGINE` only in the Cloud Run settings: the model files are baked into the image at build time.

The Whisper-only settings below (`WHISPER_BEAM`, `WHISPER_HOTWORDS`, `ALLOWED_LANGS`, `RETRY_LOGPROB`, `DENOISE`, `SNR_CLEAN_DB`, `WHISPER_VAD`) apply only when `STT_ENGINE=whisper`.

For the Whisper engine only:
* `base.en`: about twice as fast and lighter, a little less accurate.
* `small.en`: the default.
* `medium.en`: more accurate, but too slow for live talking on 2 vCPU, and it may not fit in 2 GiB next to the voice model.

Look at live timings in *Cloud Run → news-byte-extractor → Logs*. Every call logs a line like
`STT 4.1s audio in 1.9s (x2.2 realtime)`. A figure above 1.0 means faster than real time.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Deploy fails with "failed to start and listen on PORT" | Read the logs: `gcloud run services logs read news-byte-extractor --region asia-south1 --limit 60`. Look for `MODEL LOADING FAILED`. `/health` also shows the error text. |
| First sentence takes 15 to 30 s | Cold start. Open `/health` first (see the tip above). |
| `401 Invalid API key` | The key in the extension differs from `.api_key`. |
| `Container ... exceeded memory limit` in logs | Use Whisper `base`, or raise `--memory 3Gi` (more GiB-seconds used, still inside the free memory allowance if you use about 25 busy hours). |
| Hindi words transcribed wrongly | Set *Language I speak* to **Hindi** or **Hinglish** instead of Automatic. Automatic now only chooses between English and Hindi, so it no longer mistakes Hindi for Urdu. |
| Names / rare words spelled wrong | Set `WHISPER_HOTWORDS=Name1;Name2`, or send them in the extension's prompt field. |
| Hindi reply sounds odd inside an English sentence | Hindi words in Roman letters ("aap kaise ho") are read with an English accent. Devanagari is read correctly. |
| Silence from the speaker | Check the logs for `bad audio ... skipped`. The server never sends broken audio, and the extension falls back to a browser voice on an error. |

**Important:** do not swap in `kokoro-v1.0.int8.onnx` to save space. In testing it returned corrupt audio (NaN values) for
ordinary English sentences and was slower than fp32 on a modern Xeon.

## Security

The service is public, but every speech request needs your API key (`Authorization: Bearer ...`).
To rotate the key: delete `.api_key` and run `./deploy.sh` again. Put the new key in the extension.

## Does 2 vCPU / 2 GiB work on Cloud Run? (checked Nov 2026)

* **Allowed:** yes. Cloud Run services accept 2 vCPU with 128 MiB to 8 GiB of memory, so `--cpu 2 --memory 2Gi` is a valid pair. Startup CPU boost gives a 2-vCPU service 4 vCPU while starting.
* **Free tier:** request-based billing gives 180,000 vCPU-seconds, 360,000 GiB-seconds and 2 million requests per month. Mumbai (`asia-south1`) is a Tier 1 region, so it gets the full free tier (Delhi `asia-south2` is Tier 2, so avoid it). At 2 vCPU that is 90,000 busy seconds (25 hours) a month; at 2 GiB the memory used for that is 180,000 GiB-seconds, inside the allowance.
* **Memory is the tight part.** Measured here: Kokoro takes about 540 MB. The faster-whisper README lists `small` int8 at about 1.0 to 1.5 GB for a 13-minute file, and short voice clips use less. With Python itself, expect roughly 1.3 to 1.7 GiB in use. That fits in 2 GiB but with little spare room, which is why recordings are capped at 120 seconds. Watch *Cloud Run, Metrics, Memory utilization* after your first day. If it stays above about 85%, change `--memory 2Gi` to `3Gi` in `deploy.sh`. At 3 GiB a full 90,000 busy seconds is 270,000 GiB-seconds, still inside the free allowance.
* **Not measured:** Whisper's own memory and speed on Cloud Run's CPU. I could not download Whisper from Hugging Face in my test environment, so those numbers come from the faster-whisper documentation, not from a run.

## Noise: what the server does, and why it does not do more

Heavy AI noise removers (RNNoise, DeepFilterNet) sound clean to people but usually make Whisper **less** accurate, because of the artefacts they leave. Published tests found this, so this server uses a gentle method instead (`audio_clean.py`, numpy only):

1. Removes rumble below 70 Hz and lifts quiet microphones to a healthy level.
2. Learns the steady background (fan, AC, hiss, mains hum) from the quietest part of the recording and turns it down by at most 10 dB. It never mutes it fully, so there is no "underwater" sound.
3. Measures how noisy the recording is. Clean recordings are sent as they are. Noisy ones try the cleaned version first.
4. If Whisper is unsure (low confidence) it listens again to the other version and keeps the more confident result. Clear speech never pays for this second pass.

Test I ran (Whisper small.en int8, 4 synthetic English clips of 55 s in total, noise at 5 dB, word error rate, lower is better):

| Noise | Untouched | High-pass only | Gate (used here) |
|---|---|---|---|
| none | 1.9% | 1.9% | 1.9% |
| fan | 1.9% | 1.9% | 1.9% |
| mains hum | 1.9% | 1.9% | 1.9% |
| hiss | 2.5% | 2.6% | 1.9% |
| people talking | 13.5% | 16.1% | 12.3% |

Meaning: Whisper already copes well with steady noise, the gate never made things worse here and helped a little with hiss. The sample is small and the speech is synthetic, so treat it as a sanity check, not proof. **Other people talking nearby is the real problem** and no simple filter fixes it: use a headset or sit closer to the microphone.

## What changed in build `voice-v4`

* **Language detection** only chooses between English and Hindi (no more Hindi heard as Urdu).
* **Quiet microphones** are normalised, silent recordings are skipped, and phantom phrases ("Thank you.") are filtered.
* **Beam search 2**, retry at a higher temperature for garbled windows, padded voice detection so first/last words are not cut.
* **Warm-up** now really runs Whisper, language detection and both voices at startup (the earlier warm-up used silence, which the new silence check would skip).
* **Speech output:** silence trimmed, even loudness, emoji / `[links](url)` / `&` / `%` cleaned, repeated sentences cached.
* `.api_key` is no longer uploaded with your source when deploying.

**More accuracy, more cost (untested here):** in the `Dockerfile` set `ARG WHISPER_MODEL=large-v3-turbo`. It is a larger model, so expect more memory (raise `--memory`) and slower answers on 2 vCPU. Try it and read the `STT ... (x realtime)` log lines before keeping it.


---

## English-only build (v5)

This version only understands English. Hindi and Hinglish speech will come out as wrongly spelled English words.
To change speed, run `CPU=4 MEMORY=2Gi ./deploy.sh` (4 vCPU, about twice the price per active hour).
`/health` shows `"build": "voice-v5-en"` when this version is running.

## Speech-to-text engines (build `voice-v6-fast`)

faster-whisper is already the fast way to run Whisper. The slowness comes from Whisper itself: it always processes a 30-second window, even for a 2-second command, so most of the work is wasted on padding. The other engines have no such window.

I ran every engine on the same test machine (1 CPU thread, int8 ONNX, same runtime), with 10 short spoken commands (about 2.5 s each) and 4 long clips:

| Engine | RAM | Time per short command | Word errors, clean | Word errors, noisy room (8 dB) | Languages |
|---|---|---|---|---|---|
| Moonshine tiny | 0.3 GB | 0.09 s | 4.1% | 7.4% | English |
| Moonshine base | 0.5 GB | 0.19 s | 2.9% | 12.1% | English |
| **Parakeet 0.6B v2 (default)** | 1.0 GB | 0.46 s | 2.9% | 2.9% | English |
| Whisper base.en | 0.5 GB | 0.65 s | 4.0% | 6.4% | English |
| Distil-Whisper small.en | 0.8 GB | 1.05 s | 2.9% | 9.5% | English |
| Whisper small.en | 1.1 GB | 1.75 s | 2.9% | 4.1% | English |

Read this carefully:
* The test is small (about 68 words per row, so one word is about 1.5 points) and the speech is computer-generated, which is easier than real voices. Use it for the **ranking**, not for exact numbers.
* The Whisper rows ran through the ONNX runtime, **not** faster-whisper (the faster-whisper files could not be downloaded in my test environment). faster-whisper is probably faster than shown, so the real gap to Whisper is likely smaller than the table.
* Moonshine is the fastest and smallest, but in noise it made more mistakes in my test. Parakeet stayed accurate in noise.
* Parakeet writes numbers as digits ("25 minutes") and adds capital letters and punctuation.
* The noise gate (`audio_clean.py`) is **not** used for Parakeet and Moonshine. I tested it: it made them slightly worse (for example Parakeet with mains hum went from 0.9% to 4.7% errors). It stays in the Whisper path, where it was neutral or a little helpful.
* Parakeet and Moonshine are **English only**. For Hindi you need `STT_ENGINE=whisper` with the multilingual `small` model (the earlier `voice-v4` build).
* Parakeet is licensed CC-BY-4.0 on its model card (attribution required). Moonshine English models are MIT.
* One artificial test, the same three sentences repeated seven times with no pause, made Parakeet skip some repeats. Normal speech did not (55 s of different sentences: 149 of 150 words right).
* Recordings are cut at the quietest moment into pieces of at most 20 s, which keeps memory low.

**Memory on Cloud Run (2 GiB):** measured in one process with the voice model loaded: Parakeet 1.2 GB after start-up, about 1.5 to 1.65 GB during a long recording. Moonshine tiny: 0.65 GB after start-up, 1.0 GB peak. Parakeet leaves less spare room than Moonshine, so watch the memory graph. If it stays above about 85%, use `moonshine-base` or raise `--memory` to `3Gi`.

---

## v6 preload + audio stats

* **Models load before the port opens** (`PRELOAD=1`, default). On Cloud Run an idle container's CPU is slowed to almost nothing once the startup check has passed, so loading in the background took 6-14 minutes and the first recording waited ~16 s. Now the revision becomes ready only when the models are loaded (about 20-40 s of full CPU). Set `PRELOAD=0` to get the old behaviour back.
* **Every transcription line shows how the audio looked** when it reached the server: `peak=... voice=... snr=... clip=...%` with a plain warning when it is too quiet, clipped (microphone too loud) or noisy. Clipped audio (more than ~0.5 % of samples at the limit) turns words into other words; lower the microphone input volume in the computer's sound settings.
