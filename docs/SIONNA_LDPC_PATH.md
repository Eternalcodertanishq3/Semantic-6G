# Sionna / LDPC Upgrade Path

The Windows path uses the local convolutional-code baseline so Phase 1.1 remains runnable without changing the installed PyTorch 2.6 environment.

Use this Linux container path for Sionna PHY/LDPC experiments:

```powershell
docker build -f docker/Dockerfile.sionna -t semantic-6g-sionna .
docker run --rm -it -v "${PWD}:/workspace" semantic-6g-sionna
```

Inside the container:

```bash
python scripts/sanity_check.py
python evaluate.py --config config.yaml --max-batches 2 --snr-count 3
```

Next LDPC integration step:

1. Add an LDPC-backed FEC adapter that imports `sionna.fec.ldpc` only inside the adapter.
2. Keep the adapter selected by config, for example `classical.fec: sionna_ldpc`.
3. Preserve the same assertions already used by `evaluate.py`: equal complex symbols per image and equal average transmit power.
4. Run the same SNR sweep for both `conv_viterbi` and `sionna_ldpc`, then compare both baselines in the write-up.

This path is intentionally isolated. A failed Sionna import in Linux should never break the Windows convolutional-code baseline.

### Phase 2B: CDL Channels via Sionna
Phase 2B will replace the CDL approximation in `channel_models.py` with Sionna's exact 3GPP CDL-B/CDL-C implementation for publication-quality results.
