# AuditGym — reproducible offline image (SPEC §15). Run with: make docker
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_FROZEN=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv==0.11.26 \
 && useradd -m -u 1001 agent \
 && useradd -m -u 1002 grader

WORKDIR /app
COPY . .

# Dev deps are included: `auditgym all` runs pytest inside the container.
RUN uv sync --frozen

# Pre-generate the public splits at build time, so that the run needs no network and
# private/ can be locked down: owned by `grader`, files 0400, directories 0500.
# The generator publishes via TemporaryDirectory (mode 0700), so open/ is made world-readable first.
RUN uv run --no-sync auditgym generate --split train --n 10 --config medium --out data \
 && uv run --no-sync auditgym generate --split dev --n 10 --config medium --out data \
 && find data -type d -exec chmod 0755 {} + \
 && find data -type f -exec chmod 0444 {} + \
 && find data -type d -name private -exec chown -R grader:grader {} + \
 && find data -type d -name private -exec chmod 0500 {} + \
 && find data -path '*/private/*' -type f -exec chmod 0400 {} + \
 && mkdir -p runs reports logs \
 && chown -R agent:agent runs reports \
 && chown grader:grader logs && chmod 0700 logs \
 && install -o grader -g grader -m 0600 /dev/null logs/grader.jsonl

# The runner starts as root only to split UIDs (SPEC §9.5): the grader worker switches to
# `grader` before reading private/, and `auditgym run` switches to `agent` before any policy step.
ENV AUDITGYM_GRADER_USER=grader AUDITGYM_AGENT_USER=agent
ENTRYPOINT ["uv", "run", "--no-sync", "auditgym"]
CMD ["all"]
