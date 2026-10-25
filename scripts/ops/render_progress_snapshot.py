"""Render a read-only progress page and PDF from the checked-in WBS snapshot."""

import argparse
import csv
import json
from collections import Counter
from html import escape
from pathlib import Path
from textwrap import fill
from urllib.parse import quote_from_bytes

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

STATUS = {
    "DONE": "완료",
    "IN_PROGRESS": "진행",
    "READY": "준비",
    "VERIFY": "재검증",
    "BLOCKED": "대기",
    "PLANNED": "계획",
}
INK, MUTED, BLUE = "#16243A", "#526276", "#255CC2"


def render_html(model, output):
    template = Path(__file__).with_name("progress_snapshot_template.html").read_text()
    rows = model["rows"]
    counts = Counter(row["status"] for row in rows)
    if dict(counts) != model["counts"] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("WBS snapshot status counts or IDs do not match")
    table_rows = []
    for row in rows:
        table_rows.append(
            f'<tr data-status="{escape(row["status"])}">'
            f'<td class="task-id">{escape(row["id"])}</td>'
            f'<td><strong>{escape(row["title"])}</strong><details>'
            f'<summary>완료 기준 · 선행 작업</summary><p>{escape(row["criteria"])}</p>'
            f'<p>선행: {escape(row["dependencies"])} · 초기 예상 {escape(row["effort_days"])}D</p>'
            "</details></td>"
            f'<td><span class="badge {escape(row["status"])}">{STATUS[row["status"]]}</span></td></tr>'
        )
    replacements = {
        "version": escape(model["version"]),
        "updated_date": escape(model["updated_date"]),
        "done": str(counts["DONE"]),
        "total": str(len(rows)),
        "percent": f'{100 * counts["DONE"] / len(rows):.1f}',
        "counts": " · ".join(f"{STATUS[s]} {counts[s]}" for s in STATUS),
        "rows": "\n".join(table_rows),
        # 정답 성적과 실행 관측을 같은 데이터에서 보여 주되 서로 다른 열에 둔다.
        "model_rows": "\n".join(
            "<tr>" + "".join(f"<td>{escape(row[key])}</td>" for key in
                             ("name", "scope", "runtime", "quality", "remaining")) + "</tr>"
            for row in model.get("model_results", [])
        ),
    }
    for key, value in replacements.items():
        template = template.replace("{{" + key + "}}", value)
    with output.with_name("model_test_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        # 생성 CSV도 저장소의 LF 규칙을 따라 결과 갱신 시 공백 오류를 막는다.
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["모델", "시험 조건", "실행·지연", "품질 근거", "판정·남은 검증"])
        writer.writerows(
            [row[key] for key in ("name", "scope", "runtime", "quality", "remaining")]
            for row in model.get("model_results", [])
        )
    # 문서 서버의 허용 경로를 늘리지 않고 브라우저가 표의 CSV를 바로 저장한다.
    template = template.replace(
        "{{model_csv_href}}",
        "data:text/csv;charset=utf-8," + quote_from_bytes(output.with_name("model_test_results.csv").read_bytes()),
    )
    output.with_name("index.html").write_text(template, encoding="utf-8")


def new_page(title, subtitle, page):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.065, 0.946, title, fontsize=23, color=INK, weight="bold", va="top")
    ax.text(0.065, 0.898, subtitle, fontsize=10.5, color=MUTED, va="top")
    ax.plot([0.065, 0.935], [0.865, 0.865], color="#DCE3EE", linewidth=1)
    ax.text(
        0.065,
        0.035,
        "CCTV · AIoT  |  개발 관리용 · 실제 장비 합격과 구분",
        fontsize=9,
        color=MUTED,
    )
    ax.text(0.935, 0.035, str(page), fontsize=9, color=MUTED, ha="right")
    return fig, ax


def block(ax, top, title, lines, *, color=BLUE):
    ax.text(0.065, top, title, fontsize=14, color=color, weight="bold", va="top")
    for index, line in enumerate(lines):
        ax.text(
            0.075, top - 0.033 - index * 0.025, line, fontsize=10.5, color=INK, va="top"
        )


def render(model, output):
    font_manager.fontManager.addfont(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    )
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "pdf.fonttype": 3})
    rows, counts = model["rows"], model["counts"]
    # 사양서 추가 작업도 집계하되 빈 작업판과 상태 합계 불일치는 거부한다.
    if not rows or sum(counts.values()) != len(rows):
        raise ValueError("WBS snapshot counts do not match")
    output.parent.mkdir(parents=True, exist_ok=True)
    render_html(model, output)
    # 1차 범위는 확정했지만 앞당긴 현장 날짜는 제안이므로 확정 일정으로 표시하지 않는다.
    subtitle = (
        f"{model['updated_date']}  ·  문서 v{model['version']}  ·  11월 1차 POC / 범위 확정·현장일 미정"
    )
    with PdfPages(output) as pdf:
        fig, ax = new_page("CCTV · AIoT 개발 진행 현황", subtitle, 1)
        ratio = counts["DONE"] / len(rows)
        ax.text(
            0.065,
            0.824,
            f"{counts['DONE']} / {len(rows)}",
            fontsize=34,
            color=BLUE,
            weight="bold",
            va="top",
        )
        ax.text(0.45, 0.817, f"WBS 완료 {ratio:.1%}", fontsize=19, color=INK, va="top")
        ax.text(
            0.45,
            0.779,
            "작업 수 기준입니다. 전체 구현률·남은 공수 비율은 아닙니다.",
            fontsize=9,
            color=MUTED,
            va="top",
        )
        ax.add_patch(Rectangle((0.065, 0.731), 0.87, 0.014, color="#E7EDF6"))
        ax.add_patch(Rectangle((0.065, 0.731), 0.87 * ratio, 0.014, color=BLUE))
        ax.text(
            0.065,
            0.711,
            "  ·  ".join(f"{STATUS[s]} {counts.get(s,0)}" for s in STATUS),
            fontsize=11,
            color=INK,
            va="top",
        )
        block(
            ax,
            0.657,
            # 합성 미리보기 통합과 실제 B GPU·연속 영상 실측을 구분한다.
            "O5 담당 영상 미리보기 통합 PASS",
            [
                "Python134·JS7·Firefox1 PASS · 합성 JPEG",
                "A→B 영상 전환·오래된 프레임/이전 실행 차단",
                "2초 미리보기 · 연속 영상/B GPU·무선 잔여",
            ],
        )
        block(
            ax,
            0.518,
            "11월 1차 POC · 범위 확정/현장 일정 미정",
            [
                "11/20 준비 → 11/23~27 현장 제안 · 전체 요구사항 유지",
                "준비150h 가정 중 작업120h·여유30h · 이후 가용시간 미확정",
                "미완료28행 · 오케스트레이션 추가 공수 재배정 필요 · 12/31 목표 유지",
            ],
        )
        block(
            ax,
            0.379,
            "주요 이슈·결정 대기",
            [
                # 배포 설정의 로컬 검증을 실제 NX 설치·성능 합격으로 표시하지 않는다.
                "I68: 새 B GPU·무선 DDS·실제 지연/패킷 손실·기관 계약",
                "30W·systemd 정상 확인 · 부팅 timeout 재발 검증 I43 OPEN",
                "I15 센서 실수신·실물 가동 대기 · 대상/ID 수정 운영 반영 잔여",
                "AGX/NX 후보 구성·복구92시험 통과 · 실장비 전환/성능 대기",
                "I56 CPU 보완·GPU 지연 개선 미확인 · I64 운영 미반영",
            ],
            color="#9D3C22",
        )
        block(
            ax,
            0.213,
            "최근 근거와 갱신 방식",
            [
                "다음: 새 통합본 B GPU·무선 DDS/영상 인계 실측",
                "기존 H264 3분29.97FPS · 원격/장기·낙상·센서 검증 잔여",
                "매 작업 종료 시 원본 문서와 사내 HTML/PDF를 같은 주소로 갱신",
                "실시간 자동 동기화는 아님. 기준일과 버전을 확인하세요",
            ],
        )
        pdf.savefig(fig)
        fig.savefig(output.with_suffix(".png"), dpi=130)
        plt.close(fig)
        # 작업 증가 시 페이지를 나눠 마지막 행이 하단 안내와 겹치지 않게 한다.
        subsets = [rows[start:start + 17] for start in range(0, len(rows), 17)]
        for page, subset in enumerate(subsets, start=2):
            fig, ax = new_page(f"전체 WBS · {page-1}/{len(subsets)}", subtitle, page)
            ax.text(0.065, 0.835, "ID", fontsize=11, weight="bold", color=INK)
            ax.text(0.19, 0.835, "작업·산출물", fontsize=11, weight="bold", color=INK)
            ax.text(
                0.885, 0.835, "상태", fontsize=11, weight="bold", color=INK, ha="center"
            )
            for index, row in enumerate(subset):
                top = 0.805 - index * 0.039
                if index % 2 == 0:
                    ax.add_patch(
                        Rectangle(
                            (0.06, top - 0.029), 0.88, 0.037, color="#F4F7FB", zorder=0
                        )
                    )
                ax.text(0.065, top, row["id"], fontsize=10.5, color=INK, va="top")
                ax.text(0.19, top, row["title"], fontsize=10.2, color=INK, va="top")
                color = {
                    "DONE": "#207448",
                    "IN_PROGRESS": BLUE,
                    "BLOCKED": "#AC3E26",
                }.get(row["status"], MUTED)
                ax.text(
                    0.885,
                    top,
                    STATUS[row["status"]],
                    fontsize=10.5,
                    color=color,
                    va="top",
                    ha="center",
                )
            ax.text(
                0.065,
                0.09,
                "완료는 각 행의 계약/내부 검증 범위입니다. 재검증은 완료로 합산하지 않습니다.",
                fontsize=9,
                color=MUTED,
            )
            pdf.savefig(fig)
            fig.savefig(output.with_name(f"{output.stem}_page{page}.png"), dpi=130)
            plt.close(fig)

        if model.get("model_results"):
            # 사양서 반영으로 기존 모델 시험을 새로 통과한 것처럼 표시하지 않는다.
            model_page = len(subsets) + 2
            fig, ax = new_page("전체 모델 결과 · 기존 검증 이력", subtitle, model_page)
            ax.text(0.065, 0.835, "실행 확인과 정확도 합격은 별도입니다. P=정밀도, R=재현율, F1=조화평균.",
                    fontsize=9, color=MUTED)
            values = [[fill(row[key], width=18) for key in ("name", "runtime", "quality", "remaining")]
                      for row in model["model_results"]]
            table = ax.table(cellText=values, colLabels=["모델", "실행·지연", "품질 근거", "판정·남은 일"],
                             cellLoc="left", colLoc="left", bbox=[0.06, 0.13, 0.88, 0.67],
                             colWidths=[0.18, 0.27, 0.29, 0.26])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            for (row_index, _), cell in table.get_celld().items():
                cell.set_edgecolor("#DCE3EE")
                cell.set_facecolor("#EAF1FF" if row_index == 0 else "white" if row_index % 2 else "#F4F7FB")
                cell.set_text_props(color=INK)
            ax.text(0.065, 0.09, "기존 시험 이력입니다. 싸움·월담의 새 품질 근거나 KISA 합격 성적은 포함하지 않습니다.",
                    fontsize=8, color=MUTED)
            pdf.savefig(fig)
            fig.savefig(output.with_name(f"{output.stem}_page{model_page}.png"), dpi=130)
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(json.loads(args.data.read_text()), args.output)
