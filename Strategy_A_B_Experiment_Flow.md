# Strategy A/B 실험 플로우 (Uncertainty-risk / Subgroup sensitivity)

이 문서는 **concat baseline**(FiLM 없음) 기준으로, 현재 파이프라인에서

- **Strategy A**: `σ_log`(logTTC 표준편차) 기반 uncertainty-risk 산출
- **Strategy B**: 난이도 proxy(occlusion/density 등)로 subgroup을 나눠 `σ_log`가 커지는지 검증

을 **재현 가능하게** 수행하고, 산출되는 결과물(이미지/CSV/JSON)을 **무엇을 봐야 하고 어떻게 비교할지**를 정리합니다.

---

## 0) 이번 패치에서 반영한 핵심 수정

### 0-1) PIT 계산 버그 수정
- Flow에서 `y -> u_latent`를 얻으면, 캘리브레이션이 맞을 때 `u_latent ~ N(0,1)` 입니다.
- PIT는 **Uniform(0,1)** 이어야 하며 정의는 아래입니다.
  - `PIT = Φ(u_latent)` (표준정규 CDF)
- 이전에 `u_latent`를 [0,1]로 **클리핑**했다면 PIT가 필연적으로 왜곡됩니다(중앙 몰림/한쪽 몰림 등).
- 이번 패치에서 `diagnose_tau_gate_flow.py`는 반드시 `Φ(u_latent)`를 사용합니다.

### 0-2) Gate target recall threshold 자동 적용
- `--gate_target_recall 0.90`을 주면, 각 tau별로 **positive에서 target recall을 만족하는 threshold를 자동** 계산해 gated subset을 구성합니다.
- 기존처럼 `--gate_thresholds ...` 그리드가 있더라도, 내부적으로는 우선 **auto threshold**를 계산하고,
  재현성을 위해 필요하면 **그리드에서 가장 가까운(<=) 값으로 스냅**할 수 있습니다.

### 0-3) Strategy A: σ_log(logTTC) 샘플링 기반 계산
- `σ_log`는 Flow에서 샘플링한 `logTTC` 샘플들의 표준편차입니다.
  - `y_std ~ Flow(cond)`를 S개 샘플
  - `logTTC = y_std * target_sigma + target_mu`
  - `σ_log = std(logTTC_samples)`
- **전체 val** 기준, 그리고 **gate 통과(gated)** 기준을 각각 산출할 수 있습니다.

### 0-4) Strategy B: subgroup split 옵션
- `diagnose_tau_gate_flow.py`에서 subgroup split 방식을 선택할 수 있습니다.
  - `--subgroup_split median` : median 기준 2-bin (low/high)
  - `--subgroup_split quantile3` : 3-bin (low/mid/high)
- 분석 대상 feature는 `--subgroup_features`에 콤마로 넣습니다.
  - 기본값: `x__density,x__occlusion_low_points_ratio_dyn_30m`

---

## 1) 실험 준비

### 1-1) 입력 CSV
- minimal_v2 / minimal_v3 모두 가능
- 누수 방지를 위해 다음은 학습 입력에서 제외되어야 합니다.
  - `segment_id`, `frame_label` (지문)
  - `label`, `x__best_ttci`, `y_soft` (정답/정답 proxy)
- export 단계에서 이미 제거/무시하도록 맞춘 상태를 가정합니다.

### 1-2) 환경
- Python 3.8
- GPU 권장(Flow 샘플링 S=512를 전체 val에서 하려면 CPU는 매우 느릴 수 있음)

---

## 2) 학습(Concat baseline)

아래는 예시 커맨드입니다(경로/파일명은 본인 환경에 맞게 수정).

```bash
python "$SCRIPTS"/odd_risk_pipeline_test_antigravity/odd_risk_pipeline/train_gate_expert_flow.py \
  --csv "$OUT"/gssm_inputs_train_minimal_v3_noleak.csv \
  --out "$OUT"/runs/run_concat_minimalv3 \
  --ttc_floor 0.05 --ttc_cap 10.0 \
  --context_mode odd_only \
  --flow_cond_mode concat \
  --flow_feature_split auto \
  --expert_ctx_block_drop_prob 0.0
```

산출물(런 디렉토리):
- `gate.pt`, `expert_flow.pt`
- `preprocess_state.json`, `feature_schema.json` 등

---

## 3) Eval: Strategy A (uncertainty-risk) 산출

`eval_risk_models.py`는:
- 기본 metric(PR-AUC, ROC-AUC, Brier, ECE)
- expert NLL
- `p_gate_mean`, `p_event_mean`
- `σ_log`(옵션) + 저장

을 산출합니다.

### 3-1) 전체 val + gated 모두 σ_log 계산

```bash
python "$SCRIPTS"/odd_risk_pipeline_test_antigravity/odd_risk_pipeline/eval_risk_models.py \
  --csv "$OUT"/gssm_inputs_train_minimal_v3_noleak.csv \
  --run "$OUT"/runs/run_concat_minimalv3 \
  --split val \
  --ttc_floor 0.05 --ttc_cap 10.0 \
  --label_mode ttc_sstar --sstar_mode closing_speed \
  --tau 0.5 --amax 6.0 \
  --uncertainty_samples 512 \
  --uncertainty_scope both \
  --gate_target_recall 0.90 \
  --save_dir "$OUT"/eval_concat_minimalv3 \
  --save_preds
```

Eval 결과에서 Strategy A로 보는 핵심:
- `σ_log_mean_all` vs `σ_log_mean_gated` (gated가 더 커지는지)
- `σ_log`와 `p_event`의 상관(단조성), 그리고 subgroup에서 `σ_log`가 더 민감하게 반응하는지

---

## 4) Diagnose: PIT / ttc_pdf + Strategy B(subgroup)

`diagnose_tau_gate_flow.py`는:
- tau sweep에 따른 PR-AUC/Brier/ECE 민감도
- gate threshold 자동 선택(Recall 타겟)
- PIT histogram/KS
- TTC PDF(낮은 risk vs 높은 risk 조건)
- Strategy A 요약(σ_log all/gated)
- Strategy B subgroup summary (median 또는 quantile3)

을 한 번에 뽑습니다.

### 4-1) median split (2-bin)

```bash
python "$SCRIPTS"/odd_risk_pipeline_test_antigravity/odd_risk_pipeline/diagnose_tau_gate_flow.py \
  --csv  "$OUT"/gssm_inputs_train_minimal_v3_noleak.csv \
  --run  "$OUT"/runs/run_concat_minimalv3 \
  --split val \
  --ttc_floor 0.05 --ttc_cap 10.0 \
  --label_mode ttc_sstar --sstar_mode closing_speed \
  --amax 6.0 \
  --tau_sweep 0.1,0.2,0.3,0.5,0.7,1.0 \
  --gate_target_recall 0.90 \
  --out_dir "$OUT"/diag_concat_minimalv3 \
  --flow_diag_dir "$OUT"/diag_flow_concat_minimalv3 \
  --uncertainty_samples 512 \
  --uncertainty_scope both \
  --subgroup_features x__density,x__occlusion_low_points_ratio_dyn_30m \
  --subgroup_split median \
  --subgroup_on both \
  --calib_bins 15 \
  --save_preds
```

### 4-2) quantile3 split (3-bin)

```bash
... (위 커맨드 동일) ... \
  --subgroup_split quantile3
```

진단 결과에서 Strategy B로 보는 핵심:
- 각 tau 폴더(`tau_0.5` 등) 아래의 `subgroup_summary.csv`
- 예를 들어 `x__occlusion_low_points_ratio_dyn_30m`에서:
  - `low/mid/high`로 갈수록 `sigma_log_mean_*`이 증가하는지
  - gated에서 이 효과가 더 강해지는지

---

## 5) 결과 비교 방법(최소 체크리스트)

### 5-1) baseline(minimalv2) vs patch(minimalv3) 비교
**Metric만** 보지 말고 아래를 같이 봐야 합니다.

1) **PIT**
- `pit_hist.png` + `pit_ks_cdf.png`
- KS가 줄고, histogram이 더 uniform에 가까워지면 **분포 학습 정합성**이 개선된 겁니다.

2) **TTC PDF**
- `ttc_pdf_low_vs_high.png`
- high-risk 조건에서 tail이 제대로 살아나는지(= collapse 완화)

3) **σ_log 동작**
- `sigma_log_mean_all` vs `sigma_log_mean_gated`
- gated subset에서 σ_log가 증가하는게 “관심 상황에서 불확실성 증가”라는 주장에 유리합니다.

4) **subgroup sensitivity (Strategy B)**
- `subgroup_summary.csv`에서 proxy 난이도(occlusion/density)가 높아질수록 `σ_log`가 증가하는지

### 5-2) Strategy A vs Strategy B를 “무엇으로 비교하나?”

- **Strategy A(Uncertainty-risk)**는 “점수 정의”입니다.
  - `risk = p_event`(예: P(TTC < s*...))와 별도로
  - `uncertainty = σ_log`를 risk의 또 다른 축으로 제시
  - 논문에서는 `p_event` + `σ_log`를 같이 보고 “보수적 정책의 근거”로 씁니다.

- **Strategy B(ODD sensitivity)**는 “검증 프레임”입니다.
  - ODD가 직접 바뀌지 않아도, 난이도 proxy(occlusion/density 등)로 subgroup을 나눠
  - 어려운 subgroup에서 `σ_log`가 체계적으로 증가하는지 보여줍니다.

즉:
- A는 **정의/지표(what)**
- B는 **민감도 검증(how to validate without paired ODD)**

---

## 6) Diagnose의 subgroup split 옵션 정리

- `--subgroup_split median`
  - bin: `low`, `high`
  - 기준: feature의 median

- `--subgroup_split quantile3`
  - bin: `low`, `mid`, `high`
  - 기준: 33% / 66% quantile

- `--subgroup_on`
  - `all`: 전체 val 기준으로만 subgroup 통계
  - `gated`: gate 통과 샘플 기준으로만 subgroup 통계
  - `both`: 둘 다

---

## 7) 산출물 요약

### diagnose output (예)
`$OUT/diag_concat_minimalv3/` 아래:
- `run_..._diag_val_tau0.5.json` : tau별 요약 json
- `run_..._summary.csv` : tau sweep summary
- `run_.../pit_hist.png` : 전체 PIT 히스토그램
- `run_.../pit_stats.json` : PIT 통계
- `run_.../ttc_pdf_low_vs_high.png` : TTC PDF 비교
- `run_.../sigma_log_all_hist.png` : σ_log 분포
- `run_.../tau_0.5/subgroup_summary.csv` : subgroup 통계(핵심)

---

## 8) 권장 설정

- `--uncertainty_samples 512`
- `--subgroup_split quantile3` (논문 그림/표에서 추세를 보여주기 좋음)
- `--subgroup_features x__density,x__occlusion_low_points_ratio_dyn_30m` + 필요시 추가

