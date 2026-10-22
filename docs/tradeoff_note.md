# Trade-off note: ONNX Runtime instead of PyTorch for text encoding

**Decision.** The service encodes search queries, and the build encodes products, with ONNX Runtime running
each pinned model's own ONNX export, instead of PyTorch through sentence-transformers. The first real search
request had crashed the service: the FAISS and PyTorch wheels each bundle an OpenMP runtime, both loaded into
one process, and PyTorch inference crashed once FAISS was loaded. Taking PyTorch out of the process removes the
conflict (the tested ONNX Runtime builds add no second OpenMP runtime), whereas the alternatives only avoided
the trigger (one OpenMP thread, a lucky import order, `KMP_DUPLICATE_LIB_OK`) or gave up approximate search.

**The number (measured): a 64% smaller dependency footprint**, 1,436 MB down to 522 MB unpacked (401 to 164 MB
compressed), comparing dependency-only images on the same Linux arm64 base. The complete application image,
810 MB unpacked, also holds data and artifacts that this decision does not change. The switch was acceptable
only because behaviour stayed equivalent: embeddings of all 30,000 products agree to a cosine of at least
0.9999997, and 54 of 60 development top-10 lists are identical. Each of the other six swaps two listings with
identical text, whose scores differ by at most 1.2e-7. Full verification: `docs/decisions.md`, D17.

**What it costs.** Encoding the catalogue offline takes about 20 s longer (36 s to 57 s for bge-small). We also
now own what sentence-transformers did for us (tokenizer settings, truncation, pooling, normalisation), so a
model or library upgrade could change vectors silently. Pooling is read from each model's own config,
configurations we do not reproduce are rejected, and a parity script compares every vector and top-10 list
against the PyTorch reference.

**With two weeks instead of six hours:**
1. Run the parity script in CI on every model, revision or runtime upgrade, failing on a cosine below 0.99999
   or on any top-10 change that is not a near-tie (score gap above 1e-6) on the labelled development queries.
2. Test INT8 quantisation (MiniLM's repository ships INT8 exports; bge-small would be quantised with ONNX
   Runtime's tools) and adopt it only if nDCG@10 on the labelled development set falls by less than 0.01, with
   a bootstrap confidence interval, recording the size and latency gained.
3. Load-test the container under a one-CPU Kubernetes limit at 1, 8 and 32 concurrent clients, and choose
   `ENCODER_THREADS` and the worker count from p95 latency and throughput; the single-thread default comes from
   single-request timings only.
4. Repeat the macOS-versus-Linux comparison on amd64, where FAISS and ONNX Runtime ship different builds, and
   pin the base image by digest.
