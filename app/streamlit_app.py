"""문서 분류 + 유형별 요약 서비스 화면 (Streamlit).

이 화면은 FastAPI 서버(steps/step10_serve_api.py)를 호출만 한다. 먼저 API 서버를 띄운 뒤 실행한다.

uv run python steps/step10_serve_api.py          # 터미널 1
uv run streamlit run app/streamlit_app.py        # 터미널 2
"""

import json
import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DEMO_CSV = ROOT / "data" / "demo" / "batch_sample.csv"
DEFAULT_API = os.getenv("SUMMARY_API_URL", "http://127.0.0.1:8000")
AUTO = "자동 (라우팅)"
LABEL_KO = {"announcement": "공지", "manual": "매뉴얼", "minutes": "회의록"}
FIELD_KO = {"summary": "요약", "audience": "대상", "schedule": "일정", "actions": "대상자가 할 일",
            "changes": "달라진 점", "prerequisites": "사전 준비", "steps": "절차",
            "cautions": "주의 사항", "discussion": "논의", "decisions": "결정", "owners": "담당",
            "follow_ups": "미정 사항"}
REQUEST_TIMEOUT = 300  # 콜드 스타트와 대체 경로까지 기다린다

st.set_page_config(page_title="문서 분류·요약", layout="wide")


@st.cache_data(ttl=30, show_spinner=False)
def fetch_health(api_url):
    response = httpx.get(f"{api_url}/health", timeout=5)
    response.raise_for_status()
    return response.json()


def summarize(api_url, text, target, confidential):
    """API를 호출해 (HTTP 상태, 응답 본문)을 돌려준다. 연결 자체가 안 되면 상태 0."""
    payload = {"text": text, "confidential": confidential}
    if target != AUTO:
        payload["target"] = target
    try:
        response = httpx.post(f"{api_url}/summarize", json=payload, timeout=REQUEST_TIMEOUT)
        return response.status_code, response.json()
    except httpx.HTTPError as exc:
        return 0, {"error": {"code": "api_unreachable",
                             "message": f"API 서버에 연결할 수 없습니다 ({type(exc).__name__})."}}


def model_options(health):
    targets = health.get("targets", {})
    return [AUTO] + [f"{name}" for name in targets]


def render_summary(summary):
    for key, value in summary.items():
        name = FIELD_KO.get(key, key)
        if value is None:
            st.markdown(f"**{name}**: :gray[미기재]")
        elif isinstance(value, list):
            st.markdown(f"**{name}**" + (":" if value else ": :gray[(없음)]"))
            for item in value:
                st.markdown(f"- {item}")
        else:
            st.markdown(f"**{name}**: {value}")


def render_result(status, body):
    if status == 0 or "error" in body:
        error = body.get("error", {})
        st.error(f"[{status or '연결 실패'}] {error.get('message', '알 수 없는 오류')} "
                 f"(`{error.get('code', '-')}`)")
        if "classification" in body:
            label = body["classification"]["label"]
            st.caption(f"분류 결과는 받았습니다: {LABEL_KO.get(label, label)} "
                       f"(확신도 {body['classification']['confidence']:.2f})")
        attempts = (body.get("routing") or {}).get("attempts", [])
        if attempts:
            st.caption("시도한 생성 모델: " + " → ".join(
                f"{a['target']}({a['error'] or '성공'})" for a in attempts))
        if body.get("detail"):
            st.json(body["detail"], expanded=False)
        return
    classification, generation, routing = body["classification"], body["generation"], body["routing"]
    columns = st.columns(4)
    columns[0].metric("문서 유형", LABEL_KO.get(classification["label"], classification["label"]))
    columns[1].metric("분류 확신도", f"{classification['confidence']:.2f}")
    columns[2].metric("생성 모델", generation["target"])
    columns[3].metric("응답 시간", f"{body['elapsed_sec']:.1f}초")
    route_note = "기밀 문서 → 로컬 모델만" if routing["confidential"] else "일반 문서"
    st.caption(f"라우팅: {'자동' if routing['mode'] == 'auto' else '직접 지정'} · {route_note} · "
               f"후보 순서 {' → '.join(routing['plan'])}"
               + (f" · API 비용 ${generation['api_cost_usd']:.5f}"
                  if generation.get("api_cost_usd") else ""))
    for warning in body.get("warnings", []):
        st.warning(warning)
    st.subheader(f"{LABEL_KO.get(classification['label'], '')} 요약")
    render_summary(generation["summary"])


# ---------- 사이드바: API 연결 ----------
with st.sidebar:
    st.header("API 연결")
    api_url = st.text_input("API 주소", DEFAULT_API).rstrip("/")
    try:
        health = fetch_health(api_url)
        st.success("API 연결됨")
        st.caption(f"분류기: {health['classifier']['run_name']}")
        status_ko = {"provisional": "잠정", "final": "확정", "default": "기본값"}
        st.caption(f"기본 생성 모델: {health['default_target']} "
                   f"({status_ko.get(health['default_target_status'], health['default_target_status'])})")
        st.caption(f"기밀 문서용: {' → '.join(health['routing_plans']['confidential'])}")
        st.caption(f"Ollama: {'연결됨' if health['ollama_reachable'] else '연결 안 됨'}")
    except httpx.HTTPError:
        health = None
        st.error("API 서버에 연결할 수 없습니다.")
        st.code("uv run python steps/step10_serve_api.py", language="powershell")

st.title("문서 분류 + 유형별 요약")
st.caption("직접 학습한 분류기로 공지·매뉴얼·회의록을 구분하고, 유형에 맞는 형식으로 생성 모델이 요약합니다.")
if health is None:
    st.stop()

single_tab, batch_tab = st.tabs(["문서 하나 요약", "CSV 일괄 처리"])

# ---------- 탭 1: 문서 하나 ----------
with single_tab:
    samples = pd.read_csv(DEMO_CSV, encoding="utf-8-sig") if DEMO_CSV.exists() else None
    if samples is not None:
        picked = st.selectbox("예시 문서 불러오기 (선택)", ["직접 입력"] + samples["id"].tolist())
        default_text = "" if picked == "직접 입력" else \
            samples.loc[samples["id"] == picked, "text"].iloc[0]
    else:
        default_text = ""
    text = st.text_area("문서 원문", default_text, height=160, max_chars=4000)
    left, right = st.columns(2)
    confidential = left.checkbox("기밀 문서 (외부 전송 금지 → 로컬 모델만 사용)", key="single_conf")
    target = right.selectbox("생성 모델", model_options(health), key="single_target")
    if st.button("요약하기", type="primary", disabled=not text.strip()):
        with st.spinner("분류하고 요약하는 중입니다… (로컬 모델 첫 요청은 몇 초 더 걸릴 수 있습니다)"):
            st.session_state.single_result = summarize(api_url, text, target, confidential)
    if "single_result" in st.session_state:
        render_result(*st.session_state.single_result)

# ---------- 탭 2: CSV 일괄 처리 ----------
with batch_tab:
    st.markdown("`text` 열이 있는 CSV를 올리면 문서마다 분류·요약합니다. `id` 열은 있으면 결과에 함께 남기고, "
                "다른 열은 쓰지 않습니다.")
    uploaded = st.file_uploader("CSV 파일", type=["csv"])
    use_demo = st.checkbox(f"예시 파일 사용 ({DEMO_CSV.relative_to(ROOT).as_posix()})",
                           value=uploaded is None)
    frame = None
    if uploaded is not None and not use_demo:
        frame = pd.read_csv(uploaded, encoding="utf-8-sig")
    elif use_demo and DEMO_CSV.exists():
        frame = pd.read_csv(DEMO_CSV, encoding="utf-8-sig")
    if frame is not None:
        if "text" not in frame.columns:
            st.error("CSV에 `text` 열이 없습니다.")
            st.stop()
        frame = frame.dropna(subset=["text"])
        if "id" not in frame.columns:
            frame.insert(0, "id", [f"row-{i + 1}" for i in range(len(frame))])
        st.caption(f"{len(frame)}건")
        st.dataframe(frame[["id", "text"]], width="stretch", height=180)
        left, right = st.columns(2)
        batch_confidential = left.checkbox("모두 기밀 문서로 처리 (로컬 모델만)", key="batch_conf")
        batch_target = right.selectbox("생성 모델", model_options(health), key="batch_target")
        uses_cloud = (batch_target == "openai") or (batch_target == AUTO and not batch_confidential
                                                    and health["targets"].get(
                                                        health["default_target"]) == "openai")
        if uses_cloud:
            st.info(f"상용 모델을 쓰므로 {len(frame)}건만큼 API 요금이 발생합니다"
                    f"(이 데이터 기준 1,000건에 약 $0.16).")
        if st.button("일괄 처리 시작", type="primary"):
            progress = st.progress(0.0, text="처리 중…")
            rows = []
            for index, record in enumerate(frame.itertuples(index=False), start=1):
                status, body = summarize(api_url, str(record.text), batch_target,
                                         batch_confidential)
                ok = status == 200
                routing = body.get("routing") or {}
                rows.append({
                    "id": record.id, "ok": ok, "http_status": status,
                    "error_code": "" if ok else body.get("error", {}).get("code", ""),
                    "label": (body.get("classification") or {}).get("label", ""),
                    "confidence": round((body.get("classification") or {}).get("confidence", 0), 3),
                    "used_model": routing.get("used_target", ""),
                    "fallback_used": routing.get("fallback_used", False),
                    "body_extracted": (body.get("preprocessing") or {}).get("body_extracted",
                                                                            False),
                    "elapsed_sec": round(body.get("elapsed_sec", 0), 2),
                    "summary": (body.get("generation") or {}).get("summary", {}).get("summary", "")
                    if ok else "",
                    "summary_json": json.dumps((body.get("generation") or {}).get("summary"),
                                               ensure_ascii=False) if ok else "",
                    "text": record.text,
                })
                progress.progress(index / len(frame), text=f"{index}/{len(frame)} 처리")
            st.session_state.batch_result = pd.DataFrame(rows)
            progress.empty()
    if "batch_result" in st.session_state:
        result = st.session_state.batch_result
        success = int(result["ok"].sum())
        columns = st.columns(4)
        columns[0].metric("성공", f"{success} / {len(result)}")
        columns[1].metric("대체 경로 사용", int(result["fallback_used"].sum()))
        columns[2].metric("평균 응답 시간", f"{result['elapsed_sec'].mean():.1f}초")
        counts = result[result["ok"]]["label"].map(LABEL_KO).value_counts()
        columns[3].metric("유형 분포", " · ".join(f"{k} {v}" for k, v in counts.items()) or "-")
        st.dataframe(result.drop(columns=["summary_json", "text"]), width="stretch")
        st.download_button("결과 CSV 내려받기",
                           result.to_csv(index=False).encode("utf-8-sig"),
                           file_name="summary_results.csv", mime="text/csv")
