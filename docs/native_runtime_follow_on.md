# TritPack Native Runtime Follow-On

## Goal

Replace `llama.cpp` with a native `TritPackNativeBackend` without changing the desktop app's model registry, artifact schema, chat/session schema, or UI flow.

## Scope

- Start with one GGUF-backed causal LM family.
- CPU-only first for correctness and observability.
- Reuse the existing `InferenceBackend` contract and `prepare_model` lifecycle.
- Maintain the same end-user flows:
  - import/download GGUF
  - convert to TritPack
  - prepare model
  - chat and stream tokens

## Backend milestones

1. Implement `TritPackNativeBackend::prepare_model` so the app can switch between prepared TritPack-native and reconstructed-GGUF paths without UI changes.
2. Implement model loading and tensor materialization against the current TritPack artifact layout.
3. Add token generation, KV cache management, and sampling parity with the existing `RuntimeProfile`.
4. Add backend capability reporting and runtime diagnostics that mirror the current `llama.cpp` backend.
5. Add CPU parity tests against a known-good baseline on one reference model.

## Constraints

- Do not leak backend-specific flags into persistence or UI state.
- Keep `ModelRecord`, `ArtifactRecord`, `ChatSession`, and `RuntimeProfile` stable.
- Preserve the reconstructed GGUF cache path as a fallback while the native backend reaches feature parity.
