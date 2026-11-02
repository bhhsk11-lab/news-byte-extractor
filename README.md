# Free voice assistant on Cloud Run (CPU only)

A small server that gives your **AI Chat Firefox extension** a speaking and listening voice, running on your own
Cloud Run service (2 vCPU / 2 GiB). No GPU. Nothing is billed while you are not talking.

| Job | Model | Why |
|---|---|---|
| Speech → text | **faster-whisper** (Whisper `small.en`, int8, English). The only speech-to-text engine | Whisper accuracy, runs on CPU, MIT license |
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
| Transcription model | anything. The server ignores it (always the Whisper model baked into the image) |
| Speech model | anything. The server ignores it |
| Language I speak | *English*. This build is English only |

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
| `STT_WORKERS` | `2` | Recordings transcribed at the same moment. They share one Whisper model (extra RAM is only working space) and split the CPU threads: `WHISPER_THREADS / STT_WORKERS` each |
| `STT_QUEUE_MAX` | `8` | If more recordings than this are waiting, the server answers HTTP 429 instead of hanging |
| `WHISPER_BEAM` | `2` | `1` = fastest, `3`-`5` = most accurate, slower |
| `WHISPER_HOTWORDS` | empty | Names / terms you say often, separated by `;`, e.g. `Jaipur;Claude;Kokoro` |
| `DENOISE` | `auto` | `auto` = clean only noisy recordings and double-check unsure ones, `on` = always try the cleaned version first, `off` = never |
| `SNR_CLEAN_DB` | `18` | Recordings cleaner than this are never noise-gated |
| `RETRY_LOGPROB` | `-0.50` | If Whisper's confidence is below this, the other version of the recording is also tried |
| `WHISPER_VAD` | `1` | `0` if it cuts off quiet speech |
| `MAX_AUDIO_SECONDS` | `120` | Longest recording accepted |
| `TTS_WORKERS` | `1` | Sentences spoken at the same moment. Each Kokoro copy needs about 0.55 GB RAM |
| `TRIM_TTS` | `1` | Cuts silence at the start/end of speech |
| `TTS_CACHE_MAX` | `64` | Recent spoken sentences replayed instantly. `0` turns it off |
| `MP3_KBPS` | `48` | `64` = better sound, more data |
| `VOICE_EN_FEMALE` / `VOICE_EN_MALE` | `af_heart` / `am_michael` | any Kokoro English id |

## Speech-to-text: faster-whisper only (build `voice-v10-faster-whisper`)

* The only speech-to-text engine is **faster-whisper** (Whisper through CTranslate2), int8, CPU, English. Moonshine, Parakeet and sherpa-onnx are removed.
* The model is downloaded at **build time** (`fetch_stt.py`) and baked into the image. Choose it with `--build-arg WHISPER_MODEL=...` in the `Dockerfile` (`ARG WHISPER_MODEL=small.en`), then deploy again. Changing it only in Cloud Run settings does nothing.

| Model | Notes |
|---|---|
| `base.en` | about twice as fast as small, lighter, a little less accurate |
| `small.en` (default) | good accuracy, about 1.0 to 1.5 GB RAM |
| `medium.en` | more accurate but too slow for live talking on 2 vCPU, and may not fit |

From the Open ASR Leaderboard (English average word error, lower is better): Whisper large-v3 7.44, small.en 8.59, base.en 10.32, tiny.en 12.81.

* What the server does for accuracy: removes rumble, lifts quiet microphones, drops silence with the built-in Silero voice detector, filters phantom phrases such as "Thank you.", retries at a higher temperature for garbled windows, and listens to a second, cleaned version of the recording when it is unsure.
* `/health` shows `"build": "voice-v10-faster-whisper"` and `"stt": "faster-whisper:small.en"`.

## Parallel work and heavy load

* **Inside one server:** `STT_WORKERS=2` (default) lets two recordings be transcribed at the same moment. This uses faster-whisper's `num_workers` (documented as true parallelism when `transcribe()` is called from several Python threads, at the cost of more memory). Others wait in line; past `STT_QUEUE_MAX` the server answers 429.
* **Across servers:** `deploy.sh` allows `MAX_INSTANCES=3` with `--concurrency 4`. Cloud Run starts extra servers by itself when load grows. Use `MAX_INSTANCES=1 ./deploy.sh` to forbid scaling out.
* **Memory:** default `MEMORY=3Gi`. With `STT_WORKERS=1` you can use `2Gi`. Look in the logs for `MEMORY after STT load: X GB resident` to see the real number.
* **CPU is shared:** with 2 vCPU and 2 workers each recording gets 1 thread, so one recording alone is slower than with `STT_WORKERS=1` (2 threads). Pick `1` if you are the only user, `2` for several users.
* Text-to-speech stays at one sentence at a time per server (`TTS_WORKERS=1`). Speech-to-text and text-to-speech do run at the same time as each other.
* `/health` shows `stt_workers`, `stt_busy` and `stt_waiting`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Deploy fails with "failed to start and listen on PORT" | `gcloud run services logs read news-byte-extractor --region asia-south1 --limit 60`, look for `MODEL LOADING FAILED` |
| First sentence takes 15 to 30 s | Cold start. Open `/health` first |
| `401 Invalid API key` | The key in the extension differs from `.api_key` |
| `Container ... exceeded memory limit` | Use `STT_WORKERS=1`, `WHISPER_MODEL=base.en`, or `MEMORY=4Gi` |
| Names / rare words spelled wrong | Set `WHISPER_HOTWORDS=Name1;Name2`, or send them in the extension's prompt field |
| Wrong words, or "heard NOTHING" in logs | Check the `peak= voice= snr= clip=` part of the log line: too quiet, clipped (microphone too loud) or noisy |
| Silence from the speaker | Look for `bad audio ... skipped` in the logs |

**Important:** do not swap in `kokoro-v1.0.int8.onnx`: it returned corrupt audio (NaN) in testing.

## Security

Every speech request needs your API key (`Authorization: Bearer ...`). To rotate it: delete `.api_key` and run `./deploy.sh` again.
