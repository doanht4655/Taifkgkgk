import requests
import time
from datetime import datetime, timedelta, timezone
from rich.console import Console

TELEGRAM_TOKEN = "8368121215:AAFIWchNFFpIa8ui3F4RhysD5nLWPuI4pFU"
CHAT_IDS = ["-1002639168970"]
ADMIN_ID = "7509896689"

strategies = ["streak", "repeat", "pattern_2x"]
strategy_stats = {s: {"correct": 0, "total": 0} for s in strategies}

last_sent_session_id = None
correct = 0
wrong = 0
last_prediction = None
last_prediction_session_id = None
is_bot_enabled = True
last_update_id = 0

console = Console()


def get_result(sum_dice):
    return "Tài" if sum_dice >= 11 else "Xỉu"


def get_vn_time():
    VN_TZ = timezone(timedelta(hours=7))
    return datetime.now(VN_TZ).strftime("%H:%M:%S - %d/%m/%Y")


def predict_by_strategy(data, strategy):
    def res(i):
        d = data[i]
        return get_result(d["FirstDice"] + d["SecondDice"] + d["ThirdDice"])

    if strategy == "streak":
        streak_val = res(0)
        streak_len = 1
        for i in range(1, len(data)):
            if res(i) == streak_val:
                streak_len += 1
            else:
                break
        return "Xỉu" if streak_len >= 3 and streak_val == "Tài" else "Tài" if streak_len >= 3 else streak_val

    elif strategy == "repeat":
        return res(1)

    elif strategy == "pattern_2x":
        if len(data) < 6:
            return res(0)
        pattern = [res(i) for i in range(len(data))]
        if pattern[0] == pattern[1]:
            if pattern[2] == pattern[0]:
                return "Xỉu" if pattern[0] == "Tài" else "Tài"
            return pattern[0]
        return pattern[0]

    return res(0)


def get_final_prediction(data):
    predictions = {s: predict_by_strategy(data, s) for s in strategies}
    values = list(predictions.values())  # FIX LỖI count()
    final = max(set(values), key=values.count)
    return final, predictions


def fetch_data():
    try:
        res = requests.get("https://taixiu1.gsum01.com/api/luckydice1/GetSoiCau")
        return res.json()[:10]
    except Exception as e:
        console.print(f"[red]Lỗi fetch: {e}[/]")
        return []


def send_telegram_message(text):
    for chat_id in CHAT_IDS:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": f"```{text}```",
                    "parse_mode": "Markdown"
                },
            )
        except Exception as e:
            console.print(f"[red]Gửi lỗi: {e}[/]")


def check_telegram_command():
    global is_bot_enabled, last_update_id
    try:
        res = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates?offset={last_update_id + 1}"
        )
        updates = res.json().get("result", [])
        for update in updates:
            last_update_id = update["update_id"]
            msg = update.get("message", {})
            if str(msg.get("from", {}).get("id")) != ADMIN_ID:
                continue
            text = msg.get("text", "").lower()
            if text == "/on":
                is_bot_enabled = True
                console.print("[green]✅ Bot BẬT[/]")
            elif text == "/off":
                is_bot_enabled = False
                console.print("[red]⛔ Bot TẮT[/]")
    except Exception as e:
        console.print(f"[red]Lỗi lệnh: {e}[/]")


def predict_and_send(data):
    global last_prediction, last_prediction_session_id
    global correct, wrong, last_sent_session_id

    current = data[0]
    session_id = current["SessionId"]
    total = sum([current["FirstDice"], current["SecondDice"], current["ThirdDice"]])
    current_result = get_result(total)

    if last_sent_session_id == session_id:
        return
    if last_sent_session_id is None:
        last_sent_session_id = session_id - 1

    last_sent_session_id = session_id

    if last_prediction and last_prediction_session_id == session_id - 1:
        is_correct = last_prediction == current_result
        correct += int(is_correct)
        wrong += int(not is_correct)

        final_prediction, individuals = get_final_prediction(data)

        for s, pred in individuals.items():
            strategy_stats[s]["total"] += 1
            strategy_stats[s]["correct"] += int(pred == current_result)

        accuracy = correct * 100 / (correct + wrong)
        history = ""
        for i in range(5, 0, -1):
            if i < len(data):
                d = data[i]
                h_res = get_result(d["FirstDice"] + d["SecondDice"] + d["ThirdDice"])
                history += f"│ ➤ Phiên {d['SessionId']}: {h_res} ({d['FirstDice']}-{d['SecondDice']}-{d['ThirdDice']})\n"

        reasons = {
            "streak": "🧠 Chiến thuật 1: Cầu bệt ≥ 3 → đảo chiều",
            "repeat": "🧠 Chiến thuật 2: Theo kết quả trước",
            "pattern_2x": "🧠 Chiến thuật 3: Phân tích chuỗi 2 phiên"
        }

        name_map = {
            "streak": "Chiến thuật 1",
            "repeat": "Chiến thuật 2",
            "pattern_2x": "Chiến thuật 3"
        }

        msg = "┌──────────── 🎯 *TOOL BACCARAT* 🎯 ────────────┐\n"
        msg += f"│ 🆔 *Phiên hiện tại:* {session_id:<29}│\n"
        msg += f"│ 🎲 *Kết quả:* {current_result} ({current['FirstDice']}-{current['SecondDice']}-{current['ThirdDice']}){' ' * (25 - len(current_result))}│\n"
        msg += f"│ 📌 *Dự đoán trước:* {last_prediction} → {'✅ ĐÚNG' if is_correct else '❌ SAI':<12}│\n"
        msg += "├──────────── 🔍 *Lý do dự đoán* ──────────────┤\n"
        for s in strategies:
            msg += f"│ {reasons.get(s, ''):<42}│\n"
        msg += "├──────────── 🔮 *Dự đoán tiếp* ───────────────┤\n"
        msg += f"│ 👉 *Phiên {session_id+1}:* {final_prediction:<29}│\n"
        msg += "├──────── 📈 *Hiệu suất từng chiến thuật* ──────┤\n"
        for s in strategies:
            acc = (
                strategy_stats[s]["correct"] * 100 / strategy_stats[s]["total"]
                if strategy_stats[s]["total"]
                else 0
            )
            msg += f"│ {name_map[s]:<14}: {acc:.1f}% ({strategy_stats[s]['correct']}/{strategy_stats[s]['total']}){' ' * 10}│\n"
        msg += "├──────────── 📂 *Lịch sử gần nhất* ───────────┤\n"
        msg += history
        msg += f"└──────────── 🕒 {get_vn_time()} ───────┘"

        send_telegram_message(msg.strip())
        last_prediction = final_prediction
        last_prediction_session_id = session_id
    else:
        final_prediction, _ = get_final_prediction(data)
        last_prediction = final_prediction
        last_prediction_session_id = session_id


def main():
    while True:
        check_telegram_command()
        if is_bot_enabled:
            data = fetch_data()
            if len(data) >= 2:
                predict_and_send(data)
        time.sleep(1)


if __name__ == "__main__":
    main()
