# Running TrustGate with a local Ollama judge

TrustGate's Judge Lab can use a real local model as the LLM-as-judge instead of the
simulated judge. The Ollama backend is built in and uses only the standard library, so there
is nothing extra to `pip install`.

## 1. Install Ollama and pull a model

Install from https://ollama.com/download, then pull a model:

```bash
ollama pull llama3.2        # ~3B, light: runs on ~4-6 GB of free RAM
# or, if you have plenty of RAM/VRAM:
ollama pull llama3.1        # 8B, needs roughly 8-16 GB free
```

Confirm the server is up and the model is present:

```bash
ollama list
curl http://localhost:11434/api/tags
```

### Choose a model your machine can load

An 8B model can need well over 8 GB of RAM to load. If you see
`out-of-memory during startup: failed to allocate buffer of size ...`, the model is too big
for the free memory on your machine. Use a smaller one (`llama3.2`, `llama3.2:1b`,
`qwen2.5:3b`, or `phi3:mini`) and pass that name below.

## 2. Run the Judge Lab against the real model

Swap the simulated judge for Ollama with one flag:

```bash
trustgate judge-lab --judge ollama --ollama-model llama3.2
```

This runs the same bias probes (length, position), self-consistency, judge-versus-human
agreement, and threshold calibration, but scored by the local model. Everything else about
the pipeline is unchanged.

## 3. Notes

- **First call is slow.** The model loads into memory on the first request (tens of seconds).
  The adapter waits up to 5 minutes, so let it finish; later calls are faster.
- **Context window is capped for you.** Ollama otherwise tries to allocate the model's full
  128k-token context, whose KV cache needs ~14 GB and fails to load on a normal laptop with
  `out-of-memory ... failed to allocate buffer for kv cache`. The judge sets `num_ctx=4096`,
  which is ample for these prompts. Override with `OLLAMA_NUM_CTX` if you want more or less.
- **Custom host or port:** construct `OllamaJudge(host="http://your-host:11434")` in code, or
  run Ollama on the default `localhost:11434`.
- **Small models may score noisily.** The judge is asked to return a bare number (or "A"/"B"
  for comparisons). Larger instruct-tuned models follow this format better; a tiny model may
  add prose, which the parser tolerates but which adds noise. Prefer `llama3.1` if it fits.
