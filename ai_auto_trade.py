import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 텐서플로우 경고 제거

import openai
from dotenv import load_dotenv
import os

import numpy as np
import pandas as pd
import ta
import time
import ccxt
import time
import datetime

# ✅ 누적 수익 추적 변수
total_profit = 0.0
last_report_time = time.time()

# 📊 자동 성능 리포트용 전역 변수
total_long_entries = 0
total_short_entries = 0
total_wins = 0
total_losses = 0
all_profits = []


from openai import OpenAI
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from telegram_alert import send_telegram_message
# ✅ Bybit API 키 설정
api_key = "RSoesIOuV0GKsnvOGs"
api_secret = "eNHgrD75mmkX0DY61yA3GA2JhHezDac9q0GA"

# ✅ Bybit 연결
exchange = ccxt.bybit({
    'apiKey': api_key,
    'secret': api_secret,
    'options': {'defaultType': 'swap'},  # USDT 선물
})
exchange.load_markets()

symbol = "BTCUSDT"
amount = 0.002
leverage = 5
MIN_AMOUNT = 0.002

import pandas as pd
import talib

def calculate_indicators(df):
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    df["RSI"] = talib.RSI(close, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df["MACD"] = macd
    df["MACD_SIGNAL"] = macd_signal
    df["MACD_HIST"] = macd_hist

    upper, middle, lower = talib.BBANDS(close, timeperiod=20)
    df["UPPER_BAND"] = upper
    df["MIDDLE_BAND"] = middle
    df["LOWER_BAND"] = lower

    # 🔥 새로운 지표 추가
    df["ADX"] = talib.ADX(high, low, close, timeperiod=14)
    df["BB_WIDTH"] = (upper - lower) / middle  # 밴드폭 비율
    df["OBV"] = talib.OBV(close, volume)

    return df

def calculate_amount_based_on_confidence(current_price, predicted_price, usdt_balance):
    confidence = abs(predicted_price - current_price) / current_price

    # 예측 차이에 따라 퍼센트 조정
    if confidence > 0.006:       # 0.6% 이상 차이
        percent = 0.4            # 40% 강하게 진입
    elif confidence > 0.004:
        percent = 0.25
    elif confidence > 0.002:
        percent = 0.15
    else:
        percent = 0.05           # 아주 약한 예측 → 작게 진입

    amount = (usdt_balance * percent) / current_price
    amount = round(amount, 4)

    # 최소 거래량 유지
    MIN_TRADE_AMOUNT = 0.002  # Bybit 최소 주문량 (BTC 기준)
    if amount < MIN_TRADE_AMOUNT:
        print(f"⚠️ 수량이 너무 작아 {MIN_TRADE_AMOUNT} BTC로 고정합니다. (계산 수량: {amount})")
        amount = MIN_TRADE_AMOUNT
    return amount


def generate_trade_reason(signal_type, current_price, predicted_price, rsi=None, macd=None):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    prompt = (
        f"트레이딩 봇이 {signal_type} 포지션을 진입했습니다.\n"
        f"현재 가격: {current_price:.2f} USDT\n"
        f"예측 가격: {predicted_price:.2f} USDT\n"
    )
    if rsi is not None:
        prompt += f"RSI: {rsi:.2f}\n"
    if macd is not None:
        prompt += f"MACD: {macd:.2f}\n"

    prompt += "왜 이 시점에서 진입했는지 간단하게 설명해줘."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❗ GPT 분석 실패: {e}")
        return "GPT 전략 설명 불가"

# ✅ 캔들 데이터 가져오기
def get_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=100):
    retries = 3
    for i in range(retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            # 🔥 기술적 지표 계산 추가
            df = calculate_indicators(df)

            # ✅ 디버깅용 컬럼 확인 로그
            print("✅ OHLCV + 지표 컬럼 목록:", df.columns.tolist())

            return df
        except Exception as e:
            print(f"⚠️ OHLCV 데이터 가져오기 실패 (시도 {i+1}/{retries}): {e}")
            time.sleep(3)  # 3초 대기 후 재시도

    # 3번 실패하면 예외 발생
    raise Exception("❌ OHLCV 데이터 요청 실패. 인터넷 또는 Bybit API 상태 확인 필요.")

scaler = MinMaxScaler(feature_range=(0, 1))

def prepare_data(df):
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    df = df.dropna()  # ✅ 결측치 제거
    df["sentiment"] = 0.0  # ✅ 학습 시 기본 감성값

    feature_cols = ["close", "RSI", "MACD", "ADX", "BB_WIDTH", "OBV", "volume", "sentiment"]

    # ✅ 값 확인 (디버깅용)
    print("📊 학습용 통계:\n", df[feature_cols].describe())

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])

    sequence_length = 20
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # 종가 예측

    return np.array(X), np.array(y), scaler

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

sequence_length = 60

# ✅ LSTM 모델 정의
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 8)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")
# ✅ 데이터 준비
df = get_ohlcv()
X, y, scaler = prepare_data(df)


# ✅ 모델 학습
model.fit(X, y, epochs=20, batch_size=32)

# ✅ LSTM 모델 생성
def create_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 3)),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ✅ 자동매매 실행
def open_long(current_price, predicted_price):
    try:
        exchange.set_leverage(leverage, symbol)
    except Exception as e:
        print(f"⚠️ 레버리지 설정 스킵됨: {e}")

    # 현재 잔고 조회
    balance = exchange.fetch_balance()
    usdt = balance['total'].get('USDT', 0)

    # 뉴스 감성 점수 가져오기
    news_list = get_google_news()
    news_sentiment_score = analyze_news_sentiment(news_list)

    # 최신 RSI & MACD 가져오기
    df = get_ohlcv()
    latest_rsi = df["RSI"].iloc[-1]
    latest_macd = df["MACD"].iloc[-1]

    # 매매 이유 출력
    print(f"📊 매매 결정을 위한 요소:")
    print(f"- AI 예측가: {predicted_price:.2f} USDT (현재가: {current_price:.2f} USDT)")
    print(f"- 뉴스 감성 분석 점수: {news_sentiment_score:.2f}")
    print(f"- RSI: {latest_rsi:.2f}, MACD: {latest_macd:.2f}")

    # 수량 및 비용 계산
    amount = calculate_amount_based_on_confidence(current_price, predicted_price, usdt)
    estimated_cost = current_price * amount / leverage  # ✅ 먼저 계산
    print(f"🧮 주문 수량: {amount}, 레버리지: {leverage}x")
    print(f"💰 현재가: {current_price}, 주문 비용: {estimated_cost:.2f} USDT, 잔고: {usdt:.2f} USDT")

    if amount < MIN_AMOUNT:
        print(f"❗ 주문 수량이 너무 작습니다. 계산된 수량: {amount} BTC (최소: {MIN_AMOUNT})")
        return None

    if estimated_cost < 10.0:
        print(f"❗ 주문 금액이 너무 작습니다. 계산된 금액: {estimated_cost:.2f} USDT (최소: 5 USDT)")
        return None

    if estimated_cost > usdt:
        print(f"🚫 잔고 부족: 주문 비용 {estimated_cost:.2f} USDT > 잔고 {usdt:.2f} USDT")
        return None


    # 진입
    order = exchange.create_market_buy_order(symbol, amount)
    msg = f"📈 [롱 진입] {symbol}\n진입가: {current_price:.2f} USDT\n수량: {amount}\n레버리지: {leverage}x"

    reason = generate_trade_reason("롱", current_price, predicted_price, latest_rsi, latest_macd)
    msg += f"\n\n🤖 GPT 전략 분석:\n{reason}"
    send_telegram_message(msg)
    print(msg)

    print(f"🎯 예상 가격: {predicted_price:.2f} USDT | 실제 진입 가격: {current_price:.2f} USDT")
    print(f"💰 현재 잔고: {usdt:.2f} USDT | 진입 수량: {amount} BTC")

    # 트레일링 스탑 감시
    entry_price = current_price
    peak_price = entry_price
    elapsed = 0
    max_wait = 180
    interval = 5
    trail_start = 0.002
    trail_distance = 0.001
    stop_loss = -0.005
    global total_profit

    while elapsed < max_wait:
        time.sleep(interval)
        exit_price = exchange.fetch_ticker(symbol)['last']
        pnl_rate = (exit_price - entry_price) / entry_price

        print(f"⏱️ 경과: {elapsed}s | 현재가: {exit_price:.2f} | 수익률: {pnl_rate*100:.2f}%")

        if pnl_rate >= trail_start:
            if exit_price > peak_price:
                peak_price = exit_price
            elif (peak_price - exit_price) / peak_price >= trail_distance:
                print("🔔 트레일링 스탑 발동! 익절합니다.")
                break

        if pnl_rate <= stop_loss:
            print("🛑 손절 조건 도달! 즉시 청산.")
            break

        elapsed += interval
    global total_profit, total_long_entries, total_wins, total_losses, all_profits

    # 수익 계산 및 통계 업데이트
    profit = (exit_price - entry_price) * amount
    total_profit += profit
    all_profits.append(profit)
    total_long_entries += 1

    if profit > 0:
        total_wins += 1
    else:
        total_losses += 1

    # 수익 계산
    profit = (exit_price - entry_price) * amount
    total_profit += profit
    pnl_msg = f"💰 롱 포지션 실현 손익: {profit:.2f} USDT (누적 수익: {total_profit:.2f} USDT)"
    send_telegram_message(pnl_msg)
    print(pnl_msg)
   
    return order

def open_short(current_price, predicted_price):
    try:
        exchange.set_leverage(leverage, symbol)
    except Exception as e:
        print(f"⚠️ 레버리지 설정 스킵됨: {e}")

    # 현재 잔고 조회
    balance = exchange.fetch_balance()
    usdt = balance['total'].get('USDT', 0)

    # 뉴스 감성 점수 가져오기
    news_list = get_google_news()
    news_sentiment_score = analyze_news_sentiment(news_list)

    # 최신 RSI & MACD 가져오기
    df = get_ohlcv()
    latest_rsi = df["RSI"].iloc[-1]
    latest_macd = df["MACD"].iloc[-1]

    # 매매 이유 출력
    print(f"📊 매매 결정을 위한 요소:")
    print(f"- AI 예측가: {predicted_price:.2f} USDT (현재가: {current_price:.2f} USDT)")
    print(f"- 뉴스 감성 분석 점수: {news_sentiment_score:.2f}")
    print(f"- RSI: {latest_rsi:.2f}, MACD: {latest_macd:.2f}")

    # 수량 및 비용 계산
    amount = calculate_amount_based_on_confidence(current_price, predicted_price, usdt)
    estimated_cost = current_price * amount / leverage  # ✅ 여기서 먼저 계산
    print(f"🧮 주문 수량: {amount}, 레버리지: {leverage}x")
    print(f"💰 현재가: {current_price}, 주문 비용: {estimated_cost:.2f} USDT, 잔고: {usdt:.2f} USDT")

    if amount < MIN_AMOUNT:
        print(f"❗ 주문 수량이 너무 작습니다. 계산된 수량: {amount} BTC (최소: {MIN_AMOUNT})")
        return None

    if estimated_cost < 10.0:
        print(f"❗ 주문 금액이 너무 작습니다. 계산된 금액: {estimated_cost:.2f} USDT (최소: 5 USDT)")
        return None

    if estimated_cost > usdt:
        print(f"🚫 잔고 부족: 주문 비용 {estimated_cost:.2f} USDT > 잔고 {usdt:.2f} USDT")
        return None

    # 진입
    order = exchange.create_market_sell_order(symbol, amount)
    msg = f"📉 [숏 진입] {symbol}\n진입가: {current_price:.2f} USDT\n수량: {amount}\n레버리지: {leverage}x"

    reason = generate_trade_reason("숏", current_price, predicted_price, latest_rsi, latest_macd)
    msg += f"\n\n🤖 GPT 전략 분석:\n{reason}"
    send_telegram_message(msg)
    print(msg)

    print(f"🎯 예상 가격: {predicted_price:.2f} USDT | 실제 진입 가격: {current_price:.2f} USDT")
    print(f"💰 현재 잔고: {usdt:.2f} USDT | 진입 수량: {amount} BTC")

    # 트레일링 스탑 감시
    entry_price = current_price
    trough_price = entry_price
    elapsed = 0
    max_wait = 180
    interval = 5
    trail_start = 0.002
    trail_distance = 0.001
    stop_loss = -0.005
    global total_profit

    while elapsed < max_wait:
        time.sleep(interval)
        exit_price = exchange.fetch_ticker(symbol)['last']
        pnl_rate = (entry_price - exit_price) / entry_price

        print(f"⏱️ 경과: {elapsed}s | 현재가: {exit_price:.2f} | 수익률: {pnl_rate*100:.2f}%")

        if pnl_rate >= trail_start:
            if exit_price < trough_price:
                trough_price = exit_price
            elif (exit_price - trough_price) / trough_price >= trail_distance:
                print("🔔 트레일링 스탑 발동! 익절합니다.")
                break

        if pnl_rate <= stop_loss:
            print("🛑 손절 조건 도달! 즉시 청산.")
            break

        elapsed += interval
    global total_profit, total_short_entries, total_wins, total_losses, all_profits

    profit = (entry_price - exit_price) * amount
    total_profit += profit
    all_profits.append(profit)
    total_short_entries += 1

    if profit > 0:
        total_wins += 1
    else:
        total_losses += 1

    return order

    import datetime
    # 수익 계산
    profit = (entry_price - exit_price) * amount
    total_profit += profit
    pnl_msg = f"💰 숏 포지션 실현 손익: {profit:.2f} USDT (누적 수익: {total_profit:.2f} USDT)"
    send_telegram_message(pnl_msg)
    print(pnl_msg)
   

def send_summary_report(current_price, predicted_price):
    # 잔고 조회
    balance = exchange.fetch_balance()
    usdt_balance = balance['total'].get('USDT', 0)

    # 포지션 상태 확인 (현재 보유 중인지)   
    try:
        positions = exchange.private_get_v5_position_list({
            "category": "linear",
            "symbol": symbol
        })

        pos_data = positions['result']['list'][0] if positions['result']['list'] else None

        if pos_data and float(pos_data.get('size', 0)) > 0:
            pos_side = pos_data.get('side', 'UNKNOWN')
            entry_price = float(pos_data.get('entryPrice', 0))
            size = float(pos_data.get('size', 0))

            current_value = current_price * size
            entry_value = entry_price * size
            pnl = current_value - entry_value if pos_side == 'Buy' else entry_value - current_value
            pnl_percent = (pnl / entry_value) * 100 if entry_value > 0 else 0

            pos_info = (
                f"{pos_side} 보유 중 ({size} BTC @ {entry_price:.2f})\n"
                f"💰 평가손익: {pnl:.2f} USDT ({pnl_percent:.2f}%)"
            )
        else:
            pos_info = "포지션 없음"
    except Exception as e:
        pos_info = f"❗ 포지션 조회 실패: {e}"

    now = datetime.datetime.now().strftime('%H:%M')
    message = (
        f"🧠 AI 트레이딩 요약 리포트 ({now})\n"
        f"현재가: {current_price:.2f} USDT\n"
        f"예측가: {predicted_price:.2f} USDT\n"
        f"잔고: {usdt_balance:.2f} USDT\n"
        f"포지션: {pos_info}"
    )
    send_telegram_message(message)

def send_performance_report():
    total_entries = total_long_entries + total_short_entries
    if total_entries == 0:
        return  # 매매 없으면 스킵

    win_rate = (total_wins / total_entries) * 100 if total_entries > 0 else 0
    avg_profit = sum(all_profits) / len(all_profits) if all_profits else 0
    max_loss = min(all_profits) if all_profits else 0

    report = (
        "📊 [BTC 자동매매 리포트]\n"
        f"총 진입: {total_entries}회 (롱 {total_long_entries} / 숏 {total_short_entries})\n"
        f"승리: {total_wins} | 패배: {total_losses}\n"
        f"승률: {win_rate:.1f}%\n"
        f"누적 수익: {total_profit:.2f} USDT\n"
        f"평균 수익률: {avg_profit:.2f} USDT\n"
        f"최대 손실: {max_loss:.2f} USDT"
    )
    send_telegram_message(report)
    print(report)

def adjust_trade_decision(current_price, predicted_price, df, sentiment_score):
    """
    시그널 투표 기반으로 더 자주 매매하는 공격 전략
    """

    # ① AI 예측 시그널
    ai_signal = "LONG" if predicted_price > current_price else "SHORT"

    # ② 뉴스 감성 시그널
    if sentiment_score > 0.2:
        sentiment_signal = "LONG"
    elif sentiment_score < -0.2:
        sentiment_signal = "SHORT"
    else:
        sentiment_signal = "NEUTRAL"

    # ③ RSI + MACD 시그널
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]

    if rsi < 50 and macd > 0:
        indicator_signal = "LONG"
    elif rsi > 50 and macd < 0:
        indicator_signal = "SHORT"
    else:
        indicator_signal = "NEUTRAL"


    # 시그널 투표
    signals = [ai_signal, sentiment_signal, indicator_signal]
    long_votes = signals.count("LONG")
    short_votes = signals.count("SHORT")

    # ✅ 새로운 판단 로직: 하나라도 더 많으면 진입!
    if long_votes > short_votes:
        decision = "long"
    elif short_votes > long_votes:
        decision = "short"
    else:
        decision = "hold"

    print(f"\n📊 시그널 종합: AI={ai_signal}, 뉴스={sentiment_signal}, 지표={indicator_signal}")
    print(f"🧠 시그널 투표결과: LONG={long_votes}, SHORT={short_votes}")
    print(f"✅ 최종 매매 판단: {decision}")

    return decision


# ✅ 실시간 예측용 코드
def predict_price(df):
    recent_data = df[["close", "RSI", "MACD", "ADX", "BB_WIDTH", "OBV"]].dropna().values[-sequence_length:]
    recent_scaled = scaler.transform(recent_data)
    X = np.array([recent_scaled])
    return model.predict(X)[0][0]

def run_ai_bot():
    loop_count = 0
    sequence_length = 20  # ✅ 시퀀스 길이 정의

    while True:
        df = get_ohlcv()
     # ✅ 스케일링할 컬럼 6개
        feature_cols = ["close", "RSI", "MACD", "ADX", "BB_WIDTH", "OBV", "volume", "sentiment"]
        df["sentiment"] = analyze_news_sentiment(get_google_news())  # 🔥 실시간 감성 점수 넣기
        df = df.dropna()  # 결측값 제거 (지표 누락 방지)

        # ✅ 스케일러 학습
        scaler = MinMaxScaler()
        scaler.fit(df[feature_cols])

        # ✅ 최근 20개 데이터로 예측 준비
        recent_data = df[feature_cols].iloc[-sequence_length:]
        recent_scaled = scaler.transform(recent_data)
        X_input = np.array([recent_scaled])
        predicted_price_scaled = model.predict(X_input)[0][0]
# 예측된 값만 되살리기 (close만 되살리면 됨)
        predicted_price = scaler.inverse_transform([[predicted_price_scaled] + [0]*(len(feature_cols)-1)])[0][0]

        # ✅ 현재 상태 추출
        current_price = df["close"].iloc[-1]
        latest_rsi = df["RSI"].iloc[-1]
        latest_macd = df["MACD"].iloc[-1]
        news_sentiment_score = analyze_news_sentiment(get_google_news())

        # ✅ 로그 출력
        print(f"\n✅ [트레이딩 로그]")
        print(f"📈 현재가: {current_price:.2f}")
        print(f"🤖 예측가: {predicted_price:.2f}")
        print(f"📊 RSI: {latest_rsi:.2f}, MACD: {latest_macd:.2f}")
        print(f"📰 감성 점수: {news_sentiment_score:.2f}")

        # ✅ 매매 판단
        decision = adjust_trade_decision(current_price, predicted_price, df, news_sentiment_score)
        print(f"🧠 최종 매매 판단: {decision}")

        if decision == "long":
            open_long(current_price, predicted_price)
        elif decision == "short":
            open_short(current_price, predicted_price)
        else:
            print("⏸️ 조건 미충족. 관망 중...")

        # ✅ 5회차마다 텔레그램 보고
        if loop_count % 5 == 0:
            send_summary_report(current_price, predicted_price)

        loop_count += 1
        time.sleep(60)  # 🔁 1분 간격

import threading

def schedule_report():
    while True:
        time.sleep(3600)  # 1시간마다
        send_performance_report()

# 백그라운드로 실행
report_thread = threading.Thread(target=schedule_report)
report_thread.daemon = True
report_thread.start()

import requests
from bs4 import BeautifulSoup

def get_google_news():
    url = "https://news.google.com/search?q=bitcoin&hl=en&gl=US&ceid=US:en"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    articles = []
    for item in soup.select("article h3 a"):
        title = item.text
        link = "https://news.google.com" + item["href"][1:]
        articles.append({"title": title, "link": link})
    
    return articles[:5]  # 최신 뉴스 5개만 반환

# 테스트 실행
news_list = get_google_news()
for news in news_list:
    print(f"📰 {news['title']} ({news['link']})")

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def analyze_news_sentiment(news_list):
    sentiment_scores = []
    
    for news in news_list:
        score = sia.polarity_scores(news["title"])["compound"]
        sentiment_scores.append(score)
    
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return avg_sentiment

# 실행 테스트
sentiment_score = analyze_news_sentiment(news_list)
print(f"📊 뉴스 감성 점수: {sentiment_score:.2f}")

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def summarize_news_with_gpt(news_list):
    prompt = "다음은 최신 비트코인 관련 뉴스 제목입니다:\n\n"
    for news in news_list:
        prompt += f"- {news['title']}\n"
    
    prompt += "\n이 뉴스들이 시장에 어떤 영향을 미칠지 요약해서 설명해줘."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❗ GPT 분석 실패: {e}")
        return "GPT 뉴스 요약 불가"

# 실행 테스트
news_summary = summarize_news_with_gpt(news_list)
print(f"🤖 GPT 뉴스 해석:\n{news_summary}")

def train_lstm_model():
    import numpy as np
    import pandas as pd
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler

    # 1️⃣ 과거 데이터 불러오기
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="5m", limit=1000)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # 2️⃣ 기술 지표 계산
    df = calculate_indicators(df)  # 이건 이미 너 코드에 있어!
    df["sentiment"] = 0.0
    # 3️⃣ 입력 / 출력 준비
    feature_cols = ["close", "RSI", "MACD", "ADX", "BB_WIDTH", "OBV", "volume", "sentiment"]
    df = df.dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])

    sequence_length = 20
    X, y, scaler = prepare_data(df)
    X, y = np.array(X), np.array(y)

    # 4️⃣ LSTM 모델 구성
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    # 5️⃣ 모델 저장
    model.save("lstm_model.h5")
    print("✅ LSTM 모델 학습 완료 및 저장됨!")

train_lstm_model()

# ✅ 시작
run_ai_bot()