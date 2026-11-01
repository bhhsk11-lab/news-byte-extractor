#!/usr/bin/env bash
# One-command deploy to Cloud Run.   Usage:  ./deploy.sh  [PROJECT_ID]
# Run it inside this folder, in Google Cloud Shell or any terminal where `gcloud` is logged in.
set -euo pipefail

PROJECT="${1:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-asia-south1}"          # Mumbai: closest to India, and full free tier (Tier 1 pricing)
SERVICE="${SERVICE:-voice-server}"
KEY_FILE=".api_key"

[ -n "$PROJECT" ] || { echo "Pass your project id:  ./deploy.sh my-project-id"; exit 1; }

# the key the extension will send as "Bearer <key>". Generated once, kept in .api_key
if [ ! -s "$KEY_FILE" ]; then
  (command -v openssl >/dev/null && openssl rand -hex 24 || head -c 24 /dev/urandom | xxd -p -c 48) > "$KEY_FILE"
fi
API_KEY="$(cat "$KEY_FILE")"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project "$PROJECT"

# --cpu/--memory      : the 2 vCPU / 2 GiB you have
# --max-instances 1   : hard stop on scaling -> you can never be billed for a stampede of instances
# --concurrency 4     : Whisper and Kokoro each run one job at a time inside; extra requests just queue
# --cpu-boost         : extra CPU while the container starts (faster cold start)
# --min-instances 0   : scale to zero = nothing billed while you are not talking
gcloud run deploy "$SERVICE" \
  --project "$PROJECT" --region "$REGION" --source . \
  --cpu 2 --memory 2Gi --cpu-boost \
  --min-instances 0 --max-instances 1 --concurrency 4 \
  --timeout 120 --allow-unauthenticated \
  --set-env-vars "API_KEY=${API_KEY},WHISPER_THREADS=2,KOKORO_THREADS=2"

URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)')"
cat <<MSG

==========================  DONE  ==========================
Put these in the extension:  Settings > Voice
  Speech-to-text  base URL :  ${URL}/v1
  Text-to-speech  base URL :  ${URL}/v1
  API key (both)           :  ${API_KEY}

Quick test (wakes the server, takes ~10-20 s the first time):
  curl ${URL}/health
============================================================
MSG
