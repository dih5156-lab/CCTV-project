"""Render a read-only progress page and PDF from the checked-in WBS snapshot."""

import argparse
import json
from collections import Counter
from html import escape
from pathlib import Path

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
    }
    for key, value in replacements.items():
        template = template.replace("{{" + key + "}}", value)
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
    if len(rows) != 33 or sum(counts.values()) != len(rows):
        raise ValueError("WBS snapshot counts do not match")
    output.parent.mkdir(parents=True, exist_ok=True)
    render_html(model, output)
    subtitle = (
        f"{model['updated_date']}  ·  문서 v{model['version']}  ·  목표 종료 2026-12-31"
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
            # 소유권·종료 통과와 재진입 관측을 native 멈춤의 원인 확정으로 확대하지 않는다.
            "이번 검증: 실제 AI 후처리·버퍼 소유권5조건",
            [
                "CPU 영상3565개 독립 소유 · 중지20요청 반환 · SIGTERM 후2.18~14.65초 종료",
                "flush 제외 대조: 같은 스레드 stop() 중첩1건 · I40 원 실패 인과 미확정",
                "제품206파일/SDK/운영 변경 없음 · P5.1 진행 중 · I40 OPEN",
            ],
        )
        block(
            ax,
            0.518,
            "다음 작업과 진행 중인 검증",
            [
                "P5.1: 종료 중 SIGTERM 재진입의 중복 정리·자원 수명 영향 재현",
                "P2.4 BLOCKED: 내부 복구 시험 보존, 실제 부하·보관/지연 기준 합의 대기",
                "P3 장치 큐·격리·출력 결과 구현은 미착수. 실물 전광판·스피커 시험 후속",
            ],
        )
        block(
            ax,
            0.379,
            "주요 이슈·결정 대기",
            [
                # 별도 Jetson 보유와 설치 합격을 구분하고 사용자 지정 검증 시점을 표시한다.
                "I40 OPEN: native NULL 대기는2초 join 제한 밖 · 단위 시험 범위와 구분",
                "I32/I37/I38·I46 배포 정합 잔여 · I45 DB 경합·I34 CUDA OPEN",
                "별도 Jetson 설치 시험은 기능 완성 후·정식 배포 전(사용자 결정)",
                "I43 재부팅·장시간 검증 및 D02/D04/D06 운영 기준은 별도 잔여",
            ],
            color="#9D3C22",
        )
        block(
            ax,
            0.213,
            "최근 근거와 갱신 방식",
            [
                "다중 부하 P6.2, 추적 품질 P5.2~5.5/P6.2, 장비 P3/P4 후속",
                "현재 디스크 보존은 시작 당시 메모리/전체 복구본과 구분",
                "매 작업 종료 시 원본 문서와 사내 HTML/PDF를 같은 주소로 갱신",
                "실시간 자동 동기화는 아님. 기준일과 버전을 확인하세요",
            ],
        )
        pdf.savefig(fig)
        fig.savefig(output.with_suffix(".png"), dpi=130)
        plt.close(fig)
        for page, subset in enumerate((rows[:17], rows[17:]), start=2):
            fig, ax = new_page(f"전체 WBS · {page-1}/2", subtitle, page)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(json.loads(args.data.read_text()), args.output)
