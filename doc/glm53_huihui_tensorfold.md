# GLM-5.3-Flash: huihui GGUF -> TensorFold EXL3

Do not run `convert.py` on the GGUF. This file is the spec for that conversion.
Upstream converter: https://github.com/turboderp-org/exllamav3 (`convert.py`).
This branch is documentation only. The quant itself runs on the machine that has the weights and the GPU.

## What "done" looks like

A Hugging Face directory TensorFold can serve the way Mia AI Lab serves
[Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold):

- Routed-expert `gate` / `up` / `down` (MoE layers, Mia uses layers 3-44) are EXL3, 4 bits, codebook `mcg`, output scales on.
- Per expert matrix: `trellis` int16 `[K/16, N/16, 16*bits]`, `suh` fp16 `[K]`, `svh` fp16 `[N]`, `mcg` marker (multiplier `0xCBAC1FED`).
- Everything else stays BF16: attention, KDA, shared experts, dense MLPs, embeddings, `lm_head`, vision, and by default the whole MTP block.
- `quantization_config.json`: `quant_method=exl3`, `codebook=mcg`, `bits=4`, `scope=glm53_routed_experts_only`.
- The huihui ablation is present in the BF16 tensors it actually changed, and nowhere else.

TensorFold reads this. Her older vLLM overlay rejects a non-`mcg` codebook. A full-model EXL3 (default `mul1`, 6-bit head, quantized attention) will not match this ABI.

Decode identity (do not "simplify" it):

```text
y = ((((x * suh) @ H_K) @ W_q) @ H_N) * svh + bias
```

`H` is the 128x128 Hadamard divided by sqrt(128). `out_scales` means per-column scales were folded into `svh`.

## The huihui artifact

Repo: https://huggingface.co/huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF

Published facts, from that card:

- GGUFs are abliterated copies of `unsloth/GLM-5.3-Flash-GGUF`, not a BF16 export.
- Only layers 15-35 inclusive (0-based) were ablated.
- All expert modules were left unablated.
- Quants in the repo: `UD-IQ1_S`, `UD-IQ4_XS`, `UD-Q4_K_XL`, plus `mmproj-model-bf16.gguf`.
- There is no BF16 / safetensors copy of the abliterated weights.

`convert.py` wants an unquantized HF directory (`config.json`, `tokenizer.json`, BF16/FP16 safetensors). Pointing it at a `.gguf` fails. Dequantizing the GGUF and then quantizing that to EXL3 is a double quant. Do not do it. `UD-IQ1_S` cannot carry an ablation worth transplanting.

## Why the experts can stay a normal EXL3

huihui did not edit experts. Mia's checkpoint (and Brandon M. Music's TR3, which hers replaces) is already "routed experts EXL3 4bpw mcg, everything else original BF16". The ablation only has to be added onto the BF16 tensors inside layers 15-35.

Same pattern as https://huggingface.co/lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated : they edited BF16 `o_proj` and left every TR3 expert tensor byte-identical. Do not assume huihui only touched `o_proj`. Measure the diff. Whatever differs inside layers 15-35, and is not an expert, gets the delta. Nothing else does.

Prefer this over re-running the quantizer:

1. Start from `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` (Apache-2.0 quant; base model MIT). Experts are already the TensorFold ABI. Her card says the non-expert tensors are unchanged BF16 from `zai-org/GLM-5.3-Flash-BF16`.
2. Transplant the ablation onto those BF16 tensors.
3. Do not retouch `trellis` / `suh` / `svh` / `mcg`.

Only quantize experts yourself if you cannot use that checkpoint. That is a separate, multi-hundred-GB GPU job. It is not required to preserve this ablation.

## Delta transplant

Use the highest-bit huihui quant the user actually has. Of the three published, that is `UD-Q4_K_XL`. Download the **same** quant from `unsloth/GLM-5.3-Flash-GGUF`. Same quant name, same shard layout. Do not mix IQ4 with Q4.

For every tensor:

1. Dequantize both GGUFs to fp32.
2. `delta = huihui - unsloth`.
3. Classify.

Required checks, before any write. Stop on the first failure and print the tensor names.

- Expert tensors: max abs delta is numerical noise. huihui said they were not ablated. If experts differ, stop.
- Layers outside 15-35: max abs delta is numerical noise. If not, stop. Do not "include them anyway".
- Layers 15-35, non-experts: some tensors move. Record which ones, the max abs delta, and delta RMS / tensor RMS.
- If a 15-35 delta is on the order of the Q4 rounding noise (delta RMS much smaller than the dequant error of unsloth vs a BF16 reference, or the diff is scattered like quant noise rather than a structured edit), stop. Q4 ate the ablation. Do not add noise onto BF16.
- GGUF name -> HF safetensor name must come from the llama.cpp / Unsloth GLM-5.3-Flash converter map, not from a guessed rename. An unmapped tensor that has a real delta is a hard stop.

Then, shard by shard, without loading the whole model:

```text
new_bf16 = checkpoint_bf16 + delta
```

`checkpoint_bf16` is the tensor already in the EXL3 checkpoint (Mia: original BF16). Cast back to bf16 only at the store. Do not change dtype, shape, or key of any other tensor. Write a new directory. Do not modify the downloaded checkpoint in place.

Vision: the repo ships `mmproj-model-bf16.gguf`. Compare it to the vision tensors in the checkpoint (or to Unsloth's mmproj) before replacing anything. Default is leave vision alone. The card only claims layers 15-35 of the language model.

## If you must quantize experts yourself

Source is `zai-org/GLM-5.3-Flash-BF16` (or FP8 dequantized to BF16 once). Never the GGUF.

`doc/optimize.md` recipe: every budgeted tensor key, integer or half-integer rate. In `exllamav3/conversion/allocation.py`, `create_q_strategy_from_recipe` stores the rate and only adds tensors with `bpw <= 8` to the bit budget. `16` means unquantized (same convention as `--head_bits 16`). Confirm the quantize loop copies the original weight when the target is 16 before launching a full job. One missing or extra recipe key aborts.

```sh
python convert.py \
  -i /models/GLM-5.3-Flash-BF16 \
  -o /models/GLM-5.3-Flash-EXL3-4bpw-experts \
  -w /scratch/exl3-work \
  -rcp /models/experts-only-4.yaml \
  -cb mcg \
  --out_scales always \
  -hb 16 \
  -mb 16 \
  -vb 16 \
  -eb 16 \
  -d 0
```

Recipe contents:

- Routed-expert linears (gate, up, down): `4`
- Every other budgeted linear: `16`
- Do not pass `-hq`. It spends extra bits on attention and shared experts. That is the opposite of this ABI.
- Do not pass `-b` together with `-rcp` and expect `-b` to win. `--bits` is ignored when a recipe is set.
- `-mb 16` leaves the MTP block BF16. Mia quantized MTP experts at 4 bits. Leaving MTP in BF16 still loads and avoids quantizing the MTP attention by accident. Do not flip this unless you have enumerated MTP module keys and can mark only the expert linears as 4.
- `--out_scales always` because Mia's card says output scales are on. Default `auto` can drop scales on MoE gate/up.
- Codebook `mcg`, not the default `mul1`.
- Resume is explicit: `python convert.py -w /scratch/exl3-work -r`.
- Work dir must fit another copy of the output. Disk for a from-scratch quant is roughly BF16 (~642 GB) + work + output (~176 GB).

Apply the GGUF delta **after** compile, onto BF16 tensors only, with the same checks. Quantizing experts on the stock BF16 and then splicing is the right order: huihui never changed experts, so their Hessians should match the base model.

## Verify

- `python -m tensorfold.cuda.exl3.inspect <out_dir>` if TensorFold is installed. Expect 4-bit `mcg` on routed experts only.
- Expert tensor checksums identical to the donor EXL3 checkpoint.
- BF16 checksums identical except the measured layer 15-35 set.
- One expert tile: `trellis` + `suh` + `svh` + `mcg` decodes through exllamav3's reconstruct and matches that linear's BF16 source within EXL3 error, on the donor, before the splice. The splice must not change that decode.
- Spot-check a refusal prompt vs the stock EXL3 and vs llama.cpp on the original huihui GGUF. The EXL3 will not match llama.cpp token-for-token (different runtime, Q4 vs BF16+EXL3). It should move in the same direction. If it still refuses exactly like the stock EXL3, the delta did not land.

## Do not

- Do not convert the GGUF.
- Do not quantize attention, shared experts, embeddings, the head, or the vision tower.
- Do not use `mul1` for this GLM checkpoint.
- Do not apply the delta to expert tensors.
- Do not apply the delta outside layers 15-35 unless the diff check failed and a human looked at it.
- Do not invent GGUF-to-HF names.
