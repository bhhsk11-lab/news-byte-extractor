# Free voice assistant on Cloud Run (CPU only)

A small server that gives your **AI Chat Firefox extension** a speaking and listening voice, running on your own
Cloud Run service (2 vCPU / 2 GiB). No GPU. Nothing is billed while you are not talking.

| Job | Model | Why |
|---|---|---|
| Speech → text | **faster-whisper `small`** (int8) | Runs on CPU, handles English and Hindi |
| Text → speech | **Kokoro-82M** (fp32, ONNX) | Natural voice, English + **Hindi**, runs on CPU |

The extension already speaks the OpenAI speech API, so **no extension changes are needed**. You only paste a URL and a key.

---

## 1. Deploy (about 10 minutes, once)

1. Open **Google Cloud Shell** (the `>_` icon in the Cloud Console). It already has `gcloud` logged in.
2. Upload this folder (Cloud Shell menu `⋮` → *Upload*) and open it:
   ```bash
   unzip voice-server.zip -d voice-server && cd voice-server
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
URL=https://voice-server-xxxxx-el.a.run.app      # printed by deploy.sh
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
| Speech & image API base URL | `https://voice-server-xxxxx-el.a.run.app/v1` (keep the `/v1`) |
| Speech & image API key | the key from `.api_key` |
| Speech input (your microphone) | **API** |
| Transcription model | anything (for example `whisper-small`). The server ignores it |
| Speech model | anything. The server ignores it |
| Language I speak | *Automatic*, *Hindi*, or *Hinglish* |

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

Change these with `gcloud run services update voice-server --region asia-south1 --update-env-vars NAME=value`:

| Setting | Default | When to change |
|---|---|---|
| `WHISPER_BEAM` | `1` | `2` or `3` = slightly more accurate, slower |
| `MP3_KBPS` | `48` | `64` = better sound, more data |
| `VOICE_EN_FEMALE` / `VOICE_EN_MALE` | `af_heart` / `am_michael` | any Kokoro English id (`af_bella`, `bf_emma`, `am_adam` ...) |
| `VOICE_HI_FEMALE` / `VOICE_HI_MALE` | `hf_alpha` / `hm_omega` | `hf_beta` / `hm_psi` |
| `WHISPER_VAD` | `1` | `0` if it cuts off quiet speech |

Changing the **Whisper model** needs a rebuild: edit `ARG WHISPER_MODEL=small` in the `Dockerfile`, then run `./deploy.sh` again.
* `base`: faster and lighter, but weaker Hindi.
* `small`: the default.
* `medium`: clearly better Hindi, but too slow for live talking on 2 vCPU, and it may not fit in 2 GiB next to the voice model.

Look at live timings in *Cloud Run → voice-server → Logs*. Every call logs a line like
`STT 4.1s audio in 1.9s (x2.2 realtime)`. A figure above 1.0 means faster than real time.

## Troubleshooting

| Symptom | Fix |
|---|---|
| First sentence takes 15 to 30 s | Cold start. Open `/health` first (see the tip above). |
| `401 Invalid API key` | The key in the extension differs from `.api_key`. |
| `Container ... exceeded memory limit` in logs | Use Whisper `base`, or raise `--memory 3Gi` (more GiB-seconds used, still inside the free memory allowance if you use about 25 busy hours). |
| Hindi words transcribed wrongly | Set *Language I speak* to **Hindi** or **Hinglish** instead of Automatic. |
| Hindi reply sounds odd inside an English sentence | Hindi words in Roman letters ("aap kaise ho") are read with an English accent. Devanagari is read correctly. |
| Silence from the speaker | Check the logs for `bad audio ... skipped`. The server never sends broken audio, and the extension falls back to a browser voice on an error. |

**Important:** do not swap in `kokoro-v1.0.int8.onnx` to save space. In testing it returned corrupt audio (NaN values) for
ordinary English sentences and was slower than fp32 on a modern Xeon.

## Security

The service is public, but every speech request needs your API key (`Authorization: Bearer ...`).
To rotate the key: delete `.api_key` and run `./deploy.sh` again. Put the new key in the extension.
