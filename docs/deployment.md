# Deployment and configuration

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `AWS_S3_BUCKET` | Empty | Your bucket for model downloads and UI image uploads; optional for local models |
| `AWS_DEFAULT_REGION` | Boto3 configuration locally; `eu-north-1` in Compose | AWS region |
| `AWS_S3_MODEL_PREFIX` | `ml-models` | S3 prefix containing the three model directories |
| `MODEL_DIR` | `ml-model-dir` | Local model parent directory; fixed to `/app/ml-model-dir` in Compose |
| `API_URL` | `http://localhost:8000/api/v1/` | API base URL for Streamlit; Compose sets `http://app:8000/api/v1/` |
| `BIND_ADDRESS` | `127.0.0.1` in Compose | Host interface for Nginx and Streamlit |

Docker Compose reads `.env` for variable substitution. Running Python directly does **not** automatically load `.env`; export the desired variables in your shell. Do not place AWS keys in committed files or container images.

## Models in S3

Use the following key structure:

```text
s3://your-model-bucket/
├── ml-models/
│   ├── tinybert-sentiment-analysis/   # config, weights, tokenizer
│   ├── tinybert-disaster-tweets/      # config, weights, tokenizer
│   └── vit-human-pose-classification/# config, weights, image processor
└── ml-images/                        # Optional UI uploads
```

The notebooks show how to export and upload the full model directories. Their original bucket names are exercise values; replace them with your own before running those cells.

For serving, the AWS identity needs `s3:ListBucket` on the model prefix and `s3:GetObject` on model objects. Streamlit uploads additionally need `s3:PutObject` and `s3:GetObject` on `ml-images/*`. Additional permissions may be needed for your encryption configuration. Use a separate provisioning identity for the EC2 administration notebooks.

Downloads go into a temporary directory and become visible as a cached directory only after every file succeeds. Empty prefixes and failed transfers do not leave a usable-looking cache. A completed transfer does not validate the model itself; the loader will report `503` if artifacts are incompatible or incomplete.

## Local AWS credentials with containers

The optional override restores the credential mount from the course without making it mandatory for local-only inference:

```bash
cp .env.example .env
# Edit .env: set AWS_S3_BUCKET and AWS_DEFAULT_REGION.
# Point this to an existing credentials directory; do not commit it.
export AWS_CREDENTIALS_DIR="$HOME/.aws"
docker compose -f docker-compose.yml -f docker-compose.aws-local.yml up --build
```

`AWS_CREDENTIALS_DIR` defaults to `./.aws`. The directory is mounted read-only into the API and Streamlit containers. Set `AWS_PROFILE` if you use a named profile. Temporary credentials also require a valid session token. Profiles depending on SSO or external credential processes may require additional tooling beyond this image.

Boto3 uses its [credential provider chain](https://docs.aws.amazon.com/boto3/latest/guide/credentials.html). Compose combines the base and optional credential files using [Docker's merge rules](https://docs.docker.com/compose/how-tos/multiple-compose-files/merge/).

## EC2 workflow

1. Review the AWS notebook cells and replace exercise instance, bucket, region, AMI, key-pair, and security-group values. These notebooks can create billable resources and contain stop/delete/terminate examples; run individual cells deliberately.
2. Create an EC2 instance with enough RAM and disk for Torch, container images, and the models you will load. Memory use grows as more models are requested. The notebooks are exercises rather than an idempotent infrastructure deployment.
3. Attach an instance role with the S3 permissions above. The original full-access role note is retained for learning; serving needs only the relevant bucket/prefix access.
4. Install Docker Engine and the Compose plugin, clone this repository, copy `.env.example` to `.env`, and set your bucket and region.
5. For a deliberately accessible demo, set `BIND_ADDRESS=0.0.0.0`. Restrict security-group ingress to intended users for port 80 (API) and 8501 (Streamlit), and restrict SSH to your own address. The API's port 8000 is only on the Compose network.
6. Run `docker compose up --build -d`. The base file lets Boto3 use the instance role; the local-credentials override is unnecessary on EC2. Check `docker compose ps` and `docker compose logs app`.
7. Open `/docs`, then make a prediction to validate model loading and S3 permissions. `/health` alone does not prove inference is ready.
8. Stop containers with `docker compose down`. This leaves your model directory intact and does not stop EC2 or delete S3 data. Use your AWS controls to stop or remove resources when finished.

This setup uses HTTP and has no authentication. Keep it restricted to trusted demo users. Production exposure would also require HTTPS, authentication, controls on image URL fetching, resource limits, and operational monitoring.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| API returns `503` | Check API logs, full model exports, bucket/prefix configuration, credentials, and S3 permissions |
| Local model fails to load | Include tokenizer/image processor files and compatible weights; replace incomplete cached folders |
| UI cannot reach API | Locally use `localhost:8000`; in Compose Streamlit uses service name `app:8000` |
| `422` response | Check email format, nonempty text, batch limits, HTTP(S) URLs, and image readability |
| First request times out | Model download/load can exceed the UI or Nginx 300-second read timeout; inspect logs and retry once loading finishes |
| Image upload fails | Configure a bucket and upload permissions; direct image URLs avoid the S3 upload step |
| Container runs out of memory | Load fewer models or increase host resources; each API worker loads its own copies |
