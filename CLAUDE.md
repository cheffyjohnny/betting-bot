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

## Apex Mode 전략 (2026-04-18 도입)

**핵심 철학:** 줄 때 최대한 가져가고, 빼앗길 때 절대 내주지 마라

### 3기어 레짐 (BTC 일봉 기준)

**BEAST** — 5개 조건 중 3개 이상 충족 시
- 조건: BTC>MA50, BTC>MA200, RSI<70, 거래량>MA20×1.2, 7일모멘텀>+5%
- 행동: 90% 투입 + top1 코인 + 리버모어 피라미딩 3단계 (40/30/30)

**CRUISE** — 기본값
- 행동: 70% 투입 + top2 코인 분산

**BUNKER** — 즉시 트리거 (1개라도 해당 시)
- 트리거: BTC<MA50 / BTC RSI>80 / 7일낙폭<-12%
- 행동: 전량 즉시 청산 → 현금 100%

### 리스크 관리

- **하드스탑:** 평균 진입가 -8% (즉시 청산)
- **트레일링 스탑:** HWM - ATR(14)×2 (4시간마다 갱신)
- **부분 익절:** +15% 도달 시 50% 청산
- **피라미딩 트리거:** 전 단계 진입가 +3% 돌파 시 추가 매수 (BEAST 전용)

### 모멘텀 스코어

```
스코어 = (7일 수익률 × 0.4) + (30일 수익률 × 0.3) + (거래량 증가율 × 0.3)
MA50 아래 코인은 후보 제외
```

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

## 이슈 해결 이력

- numpy bool JSON 오류: `above_ma`가 numpy.bool_ → `bool()` 변환 (`cdb240a`)
- Actions push 충돌: `git pull --rebase` 추가로 해결
- BTC MA200 nan: limit=80으로 부족 → limit=220으로 재수집 처리
