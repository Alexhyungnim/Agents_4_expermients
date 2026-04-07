# NEXT STEPS: experiment design training scaffold integration

이 문서는 `Agents_4_expermients` 레포에 새로 추가한 `exp_design_factory_next/` 스캐폴드를 기준으로,
앞으로 해야 할 일을 정리한 작업 계획 문서이다.

현재 레포에는 기존 실험 코드와 산출물이 이미 존재한다.
특히 `RAG_system.ipynb`, `analyze_smoke_run.py`, `rubricsetup.py`, `smoke_run/`, `smoke_run_analysis/`, `smoke_run_analysis_by_case/` 등이 있으며, 새 스캐폴드는 기존 루트를 바로 덮어쓰지 않고 `exp_design_factory_next/` 하위에 안전하게 추가된 상태다. fileciteturn6file0

---

## 0. 현재 목표

단기 목표는 다음과 같다.

1. 기존 notebook / smoke analysis 흐름을 새 스캐폴드 구조로 점진적으로 옮긴다.
2. strong LLM 기반 candidate generation + rubric judging 자동화 파이프라인을 연결한다.
3. `accepted_silver`, `rejected`, `weak_rejected` 데이터셋을 자동 생성한다.
4. SFT가 실제로 돌아가는 최소 end-to-end 파이프라인을 완성한다.
5. 그 다음 DPO와 validator 학습으로 확장한다.

---

## 1. 지금 바로 해야 할 일

### 1-1. scaffold를 git에 올리기
- [ ] `exp_design_factory.zip`은 커밋하지 말고 삭제하거나 무시한다.
- [ ] `exp_design_factory_next/` 폴더만 git add 한다.
- [ ] 현재 브랜치 `260407_chani`에 commit / push 한다.

권장 명령어:

```bash
rm -f exp_design_factory.zip
git add exp_design_factory_next
git commit -m "Add experiment design training scaffold"
git push
```

### 1-2. README 없이 바로 작업하지 말고 이 문서를 먼저 레포에 추가
- [ ] 이 파일을 예를 들어 `TODO_exp_design_factory.md` 또는 `NEXT_STEPS_exp_design_factory.md` 이름으로 레포 루트에 둔다.
- [ ] 이후 작업은 이 파일의 체크박스를 기준으로 진행한다.

---

## 2. 1차 통합 우선순위

### Priority 1. chunk extraction 정리
대상 파일:
- `RAG_system.ipynb`
- `exp_design_factory_next/scripts/03_extract_relevant_chunks.py`

할 일:
- [ ] 기존 notebook의 chunk filtering 로직과 새 스크립트의 차이를 비교한다.
- [ ] 키워드 그룹, chunk scoring, bad section 제거 규칙을 하나로 통일한다.
- [ ] JSONL 출력 포맷을 확정한다.

완료 기준:
- [ ] 논문 입력 → relevant chunks JSONL 저장이 notebook 없이 script로 가능하다.

### Priority 2. task builder 구체화
대상 파일:
- `exp_design_factory_next/scripts/04_build_tasks.py`

할 일:
- [ ] 논문 1편당 최소 2개 이상의 task가 나오도록 설계한다.
- [ ] task 유형을 명시적으로 나눈다.

추천 task 유형:
- [ ] full experiment design
- [ ] measurement plan only
- [ ] control / variable extraction
- [ ] feasibility check
- [ ] flawed proposal repair

완료 기준:
- [ ] `tasks.jsonl`이 의미 있는 task 다양성을 가진다.

### Priority 3. strong generator API 연결
대상 파일:
- `exp_design_factory_next/scripts/05_generate_candidates.py`
- `exp_design_factory_next/prompts/generate_candidate.md`

할 일:
- [ ] 실제 provider API를 연결한다.
- [ ] 후보 생성 개수 `n_candidates_per_task`를 설정한다.
- [ ] 출력이 항상 동일 JSON schema를 따르도록 강제한다.

완료 기준:
- [ ] task 1개당 candidate proposal 여러 개가 자동 생성된다.

### Priority 4. rubric judge API 연결
대상 파일:
- `exp_design_factory_next/scripts/07_judge_candidates.py`
- `exp_design_factory_next/prompts/judge_candidate.md`
- `rubricsetup.py`
- `analyze_smoke_run.py`

할 일:
- [ ] 기존 rubric 항목을 새 judge schema에 반영한다.
- [ ] hard fail 조건을 명시한다.
- [ ] rule-based validator를 추가한다.

rule check 예시:
- [ ] evidence에 없는 equipment 사용 여부
- [ ] control 누락 여부
- [ ] replicate 비어 있음 여부
- [ ] measurement / analysis mismatch 여부

완료 기준:
- [ ] candidate별 `total_score`, `hard_fail`, `overall_verdict`가 자동 생성된다.

### Priority 5. SFT dataset 생성
대상 파일:
- `exp_design_factory_next/scripts/08_build_sft_dataset.py`
- `exp_design_factory_next/scripts/11_train_sft.py`

할 일:
- [ ] `accept_silver`만 골라 SFT용 JSONL을 생성한다.
- [ ] tokenizer / LoRA / 4-bit 설정이 로컬 환경에서 도는지 확인한다.

완료 기준:
- [ ] 최소 1회 SFT trial이 끝까지 돈다.

---

## 3. 기존 repo와 새 scaffold의 연결 관계

### 기존 파일 → 새 구조 매핑

#### `RAG_system.ipynb`
역할:
- 논문 검색
- full text relevance 확인
- chunk filtering
- paper card / proposal 생성 prototype

이전 대상:
- `exp_design_factory_next/scripts/00_collect_metadata.py`
- `exp_design_factory_next/scripts/03_extract_relevant_chunks.py`
- `exp_design_factory_next/scripts/04_build_tasks.py`
- `exp_design_factory_next/scripts/05_generate_candidates.py`

#### `rubricsetup.py`
역할:
- rubric 설계 / 평가 항목 정의

이전 대상:
- `exp_design_factory_next/configs/rubric/experiment_rubric_v1.yaml`
- `exp_design_factory_next/prompts/judge_candidate.md`
- `exp_design_factory_next/scripts/07_judge_candidates.py`

#### `analyze_smoke_run.py`
역할:
- JSON 결과 파싱
- rubric score 집계
- case별 시각화

이전 대상:
- `exp_design_factory_next/scripts/10_build_validator_dataset.py`
- `exp_design_factory_next/scripts/13_eval_generator.py`
- `exp_design_factory_next/scripts/14_eval_validator.py`

---

## 4. 데이터 생성 전략

### accepted / rejected / weak_rejected 분리

- `accept_silver`: strong generator + judge 통과
- `reject`: strong generator 결과지만 judge fail 또는 hard fail
- `weak_rejected`: weak RAG / loose prompt 기반 low-quality proposal

할 일:
- [ ] 각 candidate에 대해 failure reason까지 저장한다.
- [ ] failure type taxonomy를 정한다.

추천 failure taxonomy:
- [ ] resource hallucination
- [ ] missing controls
- [ ] weak measurement plan
- [ ] poor analysis plan
- [ ] evidence mismatch
- [ ] unsupported novelty claim

완료 기준:
- [ ] DPO용 rejected와 validator용 negative를 동시에 만들 수 있다.

---

## 5. 데이터셋 목표 수량

### 파일럿 목표
- [ ] papers: 50 ~ 100
- [ ] tasks per paper: 2 ~ 4
- [ ] strong candidates per task: 3 ~ 4
- [ ] weak candidates per task: 1 ~ 2
- [ ] accept_silver: 최소 200+
- [ ] rejected: 최소 500+

### 1차 논문/프로젝트 수준 목표
- [ ] papers: 150 ~ 300
- [ ] accept_silver: 500 ~ 1500
- [ ] rejected: 2000+

주의:
- [ ] 논문 수보다 task 수와 candidate 수를 늘리는 것이 더 중요하다.
- [ ] 같은 논문에서 여러 task를 뽑아 example을 증식시킨다.

---

## 6. 실험 도메인 계획

### Phase A. additive manufacturing / NiTi
- [ ] 현재 파이프라인을 먼저 이 도메인에서 안정화한다.

### Phase B. welding / materials experiments
- [ ] prompt와 rubric이 domain-specific wording에 과적합되지 않는지 확인한다.

### Phase C. battery
- [ ] held-out domain generalization용 테스트셋을 만든다.
- [ ] 3D printing / welding에서 학습한 설계 구조가 battery에서도 통하는지 본다.

완료 기준:
- [ ] domain transfer 실험이 가능해진다.

---

## 7. training 로드맵

### Stage 1. SFT only
- [ ] `accept_silver`만으로 Qwen 7B 또는 유사 instruct model에 LoRA SFT
- [ ] sequence length / batch / grad accumulation 조정

완료 기준:
- [ ] base instruct 대비 proposal schema 안정성 개선

### Stage 2. DPO
- [ ] `accept_silver` vs `rejected` pair로 DPO dataset 생성
- [ ] chosen / rejected pair 품질 확인

완료 기준:
- [ ] rejected 유형을 더 잘 피하는 방향으로 개선

### Stage 3. validator / reranker
- [ ] validator dataset 생성
- [ ] inference에서 best-of-N + judge reranking 실험

완료 기준:
- [ ] single-shot generation보다 품질 향상 확인

---

## 8. 검증 및 평가

### 자동 평가
- [ ] rubric 평균 점수
- [ ] hard fail 비율
- [ ] accepted 비율
- [ ] evidence mismatch 비율

### 정성 평가
- [ ] strong generator 원본 vs fine-tuned model 출력 비교
- [ ] weak RAG output vs DPO 이후 output 비교

### 일반화 평가
- [ ] seen domain test
- [ ] held-out domain test

---

## 9. 운영 원칙

- [ ] 기존 루트 파일은 즉시 덮어쓰지 않는다.
- [ ] 새 파이프라인은 당분간 `exp_design_factory_next/` 안에서만 개발한다.
- [ ] notebook에서 검증된 로직만 script로 옮긴다.
- [ ] hardcoded API key는 금지한다.
- [ ] 결과 JSON schema는 초기에 고정하고 자주 흔들지 않는다.
- [ ] 논문 단위 split을 유지하여 leakage를 막는다.

---

## 10. 이번 주 현실적인 체크리스트

### Must
- [ ] scaffold 폴더 git commit / push
- [ ] TODO md 파일 git commit / push
- [ ] chunk extraction script 정리
- [ ] task builder 정리
- [ ] strong generator API 연결

### Should
- [ ] judge API 연결
- [ ] accept / reject 분류 저장
- [ ] SFT dataset 생성 테스트

### Nice to have
- [ ] weak RAG rejected 자동 생성
- [ ] battery held-out split 초안
- [ ] validator dataset builder 정리

---

## 11. 나중에 정리할 것

- [ ] `exp_design_factory_next/` 이름을 최종 이름으로 바꿀지 결정
- [ ] README에 새 파이프라인 설명 추가
- [ ] notebook 의존성을 줄이고 script-first 구조로 전환
- [ ] Slurm / cluster 실행 경로 정리

---

## 12. 메모

현재 레포 README도 notebook logic를 `scripts/search_and_filter.py`, `scripts/build_paper_cards.py`, `scripts/propose_experiment.py` 같은 script-first 구조로 옮기는 방향을 권장하고 있다. 따라서 새 scaffold는 기존 레포 철학과도 충돌하지 않고, 오히려 그 방향을 더 체계화하는 역할을 한다. fileciteturn6file0

