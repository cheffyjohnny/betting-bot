# Crypto Bot — Apex Mode

업비트 자동매매 봇. 페이퍼 트레이딩 중.

**GitHub:** https://github.com/cheffyjohnny/betting-bot
**대상:** BTC / ETH / XRP / SOL (업비트 KRW)
**초기 자본:** 1,000,000 KRW

## 실행 구조

| 시간 | 파일 | 역할 |
|------|------|------|
| KST 09:00 (매일) | `apex_bot.py` | 기어 판단 + 진입/로테이션/피라미딩 |
| KST 11,15,19,23,03,07시 | `stop_checker.py` | 스탑 체크 + HWM 갱신 (4시간마다) |

## Apex Mode 전략 V5 (2026-04-19 업데이트)

**핵심 철학:** 줄 때 최대한 가져가고, 빼앗길 때 절대 내주지 마라

### 4기어 레짐 (BTC 일봉 기준)

**BEAST** — 6개 조건 중 3개 이상 충족 시
- 조건: BTC>MA50, BTC>MA200, RSI<70, 거래량>MA20×1.2, 7일모멘텀>+5%, 볼린저밴드폭 확장(20일 평균×1.1)
- 행동: 90% 투입 + top1 코인 + 리버모어 피라미딩 3단계 (40/30/30)

**CRUISE** — 기본값
- 행동: 70% 투입 + top2 코인 분산

**CAUTION** — BTC>MA200 (bull macro) + BTC<MA50 시
- 행동: 40% 투입 + top1 코인 (방어적 포지션)

**BUNKER** — 즉시 트리거
- bull macro (BTC>MA200): RSI>80 또는 7일낙폭<-12% 시만 발동
- bear macro (BTC<MA200): BTC<MA50 / RSI>80 / 7일낙폭<-12% 중 1개라도 해당 시
- 행동: 전량 즉시 청산 → 현금 100%

### 리스크 관리

- **하드스탑:** 평균 진입가 -8% (즉시 청산)
- **트레일링 스탑:** HWM - ATR(14)×2 (bear macro) / ATR(14)×3 (bull macro)
- **부분 익절:** +15% 도달 시 50% 청산
- **피라미딩 트리거:** 전 단계 진입가 +3% 돌파 시 추가 매수 (BEAST 전용, 연속 2일+ 조건 필수)
- **로테이션 임계값:** bull macro 시 새 코인 스코어가 20% 이상 높을 때만 교체
- **현금 버퍼:** 총 자산의 5% 이상 항상 현금 유지 (매수 시 available_krw 기준)
- **Bear macro 비중 축소:** CRUISE/CAUTION에서 MA200 하향 코인은 배분의 50%만 투입

### 모멘텀 스코어

```
스코어 = (7일 수익률 × 0.4) + (30일 수익률 × 0.3) + (거래량 증가율 × 0.3)
MA50 아래 코인은 후보 제외
MA200 하향 코인은 후보 포함되지만 비중 50% 축소
```

### 볼린저 밴드 (V5.2 추가)

```
기간: 20일, 표준편차 ×2

상단 밴드 = MA20 + 표준편차×2
중간 밴드 = MA20
하단 밴드 = MA20 - 표준편차×2

밴드 폭(Bandwidth) = (상단 - 하단) / MA20
%B = (현재가 - 하단) / (상단 - 하단)   → 0=하단, 1=상단
```

**활용 방식:**
- **BEAST 조건** — 밴드 폭 > 20일 평균 밴드 폭 × 1.1 → 변동성 확장 중인 장세에서만 공격 진입
- **진입 타이밍** — 개별 코인 %B >= 0.8 이면 신규 진입 보류 (상단밴드 근처 고점 매수 방지)

## 파일 구조

### 메인 (현재 운용 중)
- `apex_bot.py` — Apex Mode 메인 봇
- `stop_checker.py` — 4시간 스탑 체커

### 레거시 (보존, 미사용)
- `momentum_bot.py` — 이전 MA50+모멘텀 전략 ← 건드리지 말 것
- `strategy.py` / `indicators.py` / `paper_trader.py` — 스윙봇 전략A 컴포넌트

### 백테스트
- `momentum_rotation.py` — 200MA 필터 포함 백테스트
- `momentum_exhaustive.py` — 파라미터 전수 조사 (~7,000가지)
- `ultimate_backtest.py` — 전략 4종 비교
- `apex_backtest.py` — V1/V4/V5 전략 상승장/하락장 비교 백테스트

### 데이터
- `data/apex_bot_state.json` — 현재 포지션 상태 (Apex)
- `data/apex_bot_log.jsonl` — Apex 일별 실행 로그
- `data/stop_checker_log.jsonl` — 스탑 체커 로그
- `data/momentum_bot_state.json` — 이전 봇 상태 (레거시)
- `data/bot_log.jsonl` — 이전 봇 로그 (레거시)

### GitHub Actions
- `.github/workflows/daily_bot.yml` — apex_bot 매일 KST 09:00
- `.github/workflows/stop_checker.yml` — stop_checker 4시간마다

## 이전 전략 백테스트 결과 (momentum_bot 기준)

| 목표 | 조합 | 6년 수익 | 최악년도 |
|------|------|---------|---------|
| 수익 최고 | 룩백15일/보유1일/50MA | +186,658% | -18% (2022) |
| 매년 플러스 | 룩백13일/보유8일/50MA | +64,750% | +6% (2022) |
| 매년 10%+ | 룩백31일/보유5일/50MA | +6,703% | +10% (2022) |

## 다음 단계

1. 페이퍼 트레이딩 검증 (1개월)
2. 결과 좋으면 실거래 전환 (업비트 API 키 필요)
3. 바이낸스 선물 계좌 → 하락장 숏 전략 추가
4. MVRV 온체인 지표 추가 검토

## V5 백테스트 결과 (apex_backtest.py 기준)

| 기간 | 전략 | 수익률 | BTC 홀딩 | MDD |
|------|------|--------|---------|-----|
| 2024 (상승장) | V5 | +33.34% | +136.67% | -53.87% |
| 2025~26 (하락장) | V5 | +24.88% | -8.74% | -18% |

- 상승장에서 BTC 홀딩을 이기기 어려운 구조적 이유: BUNKER 기간 수익 0%
- 하락장 방어가 이 전략의 핵심 강점

## 변경 이력

### V5.2 — 2026-05-28 (`4943172`)

**배경:** 볼린저 밴드 추가로 BEAST 품질 강화 및 고점 진입 방지

**수정:**
- `calc_bollinger()` 추가 — 상단/하단/중간선, 밴드 폭, %B 계산 (20일, ×2σ)
- BEAST 조건 5개 → 6개: 볼린저 밴드 폭 확장 조건 추가 (`BB_WIDTH_MULT = 1.1`)
- 신규 진입 시 %B >= 0.8 이면 진입 보류 (`BB_ENTRY_MAX = 0.8`)
- 대시보드/로그에 `bb_pct_b`, `bb_expanding` 필드 추가

### V5.1 — 2026-05-08 (`faf84c0` / `f3fb86c`)

**배경:** 3주 페이퍼 트레이딩 결과 분석 후 구조적 문제 3가지 발견

**문제:**
1. ETH가 bear macro(MA200 하향)임에도 CRUISE가 계속 top2로 선정 → 3번 진입해서 순손익 ~0
2. BEAST 1일 만에 종료되는 상황에서 피라미딩 집행 (5/6 고점 근처 Lot2 진입)
3. CRUISE 70% 배분 구조상 현금이 1.85%(18,737원)까지 소진됨

**수정:**
- `MIN_CASH_RATIO = 0.05` — 총 자산 5% 현금 버퍼 (available_krw 기준 매수)
- `BEAST_STREAK_MIN = 2` — BEAST 연속 2일 이상일 때만 피라미딩 허용 (`state['beast_streak']` 추적)
- `above_ma200` — 전 코인 220봉 수집 후 MA200 체크, CRUISE/CAUTION에서 하향 코인 비중 50% 축소
- fetch limit 80 → 220 통일 (BTC 별도 재수집 제거)
- 로그에 `beast_streak`, `above_ma200` 필드 추가

## 이슈 해결 이력

- numpy bool JSON 오류: `above_ma`가 numpy.bool_ → `bool()` 변환 (`cdb240a`)
- Actions push 충돌: `git add` → `git commit` → `git pull --rebase` → `git push` 순서로 해결
- BTC MA200 nan: limit=80으로 부족 → limit=220으로 재수집 처리
- pandas_ta Python 3.11 설치 실패: requirements.txt에서 제거 (레거시 파일에서만 사용)
- stop_checker.py V5 동기화: bull_macro 포지션에 ATR×3 적용 (`82901e2`)
