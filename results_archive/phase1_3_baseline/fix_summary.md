# Text Semantic Codec Fix Summary (Phase 1.3)

* **Root Cause Diagnosis**: The `TextSemanticDecoder` was missing both teacher forcing during training and autoregressive loop structures during evaluation.
* **Symptom Explained**: By feeding a static sequence to the GRU at every time step, the model bypassed true sequential modeling. It learned to generate a single flat prior, causing the token accuracy to remain flat and completely insensitive to channel SNR.
* **Fix Applied**: Rewrote `TextSemanticDecoder` to accept a `targets` tensor. Enabled standard seq2seq teacher forcing during training and autoregressive decoding during evaluation.
* **Retraining & Verification**: The text model was retrained on Tiny Shakespeare for 40 epochs. The token accuracy curve now demonstrates SNR sensitivity, and the qualitative text check generates cohesive (though still LM-biased) language rather than static priors.
