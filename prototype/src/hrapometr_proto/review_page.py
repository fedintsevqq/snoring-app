"""HTML-страница для ручной разметки окон. Одна страница, без серверов и внешних скриптов."""

from __future__ import annotations

import json

LABELS = [
    ("snore", "Храп"),
    ("breathing", "Дыхание"),
    ("speech", "Речь"),
    ("noise", "Шум или тишина"),
    ("unsure", "Не уверен"),
]

_TEMPLATE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Проверка детектора</title>
<style>
  :root { color-scheme: light dark; --accent: #1B3A6B; --ok: #1a7f37; }
  body { font: 16px/1.5 system-ui, sans-serif; max-width: 760px; margin: 0 auto; padding: 16px; }
  header { position: sticky; top: 0; background: Canvas; padding: 8px 0; border-bottom: 1px solid #8884; }
  .card { border: 1px solid #8886; border-radius: 12px; padding: 12px; margin: 12px 0; }
  .card.done { border-color: var(--ok); }
  .buttons { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
  button { font: inherit; padding: 6px 12px; border-radius: 8px; border: 1px solid #8886; cursor: pointer; background: ButtonFace; }
  button.picked { background: var(--accent); color: #fff; border-color: var(--accent); }
  audio { width: 100%; }
  .hint { color: #888; font-size: 14px; }
</style>
</head>
<body>
<header>
  <strong id="progress"></strong>
  <button id="download">Скачать labels.json</button>
</header>
<p>Каждое окно — 30 секунд записи. Выберите <b>«Храп»</b>, если в окне слышен хотя бы один храп (даже на фоне речи или дыхания).
Если храпа нет, отметьте, что звучит главным. Сомневаетесь — «Не уверен». Откуда окно, детектор его пометил или нет, не показываем нарочно.</p>
<div id="list"></div>
<script>
const RUN = __RUN__;
const TOKEN = __TOKEN__;
const CLIPS = __CLIPS__;
const LABELS = __LABELS__;
// токен выборки в ключе: после нового prepare старые ответы не прилипнут к другим клипам
const STORAGE_KEY = "hrapometr-review-" + RUN + "-" + TOKEN;
let answers = {};
try { answers = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); } catch (e) { answers = {}; }

function save() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(answers)); } catch (e) { /* без сохранения тоже работает */ }
  const done = Object.keys(answers).length;
  document.getElementById("progress").textContent = "Размечено " + done + " из " + CLIPS.length;
  document.querySelectorAll(".card").forEach(card => {
    card.classList.toggle("done", card.dataset.id in answers);
    card.querySelectorAll("button").forEach(b => b.classList.toggle("picked", answers[card.dataset.id] === b.dataset.label));
  });
}

const list = document.getElementById("list");
CLIPS.forEach((clip, i) => {
  const card = document.createElement("div");
  card.className = "card";
  card.dataset.id = clip.id;
  card.innerHTML = '<div class="hint">Окно ' + (i + 1) + ' из ' + CLIPS.length + '</div>' +
    '<audio controls preload="none" src="' + clip.file + '"></audio><div class="buttons"></div>';
  const buttons = card.querySelector(".buttons");
  LABELS.forEach(([value, title]) => {
    const b = document.createElement("button");
    b.textContent = title;
    b.dataset.label = value;
    b.addEventListener("click", () => {
      answers[clip.id] = value;
      save();
      const next = card.nextElementSibling;
      if (next) next.scrollIntoView({ behavior: "smooth", block: "center" });
    });
    buttons.appendChild(b);
  });
  list.appendChild(card);
});

document.getElementById("download").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify({ run: RUN, token: TOKEN, labels: answers }, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "labels.json";
  a.click();
});
save();
</script>
</body>
</html>
"""


def render_page(run_name: str, token: str, clips: list[dict]) -> str:
    """clips — список {"id": "c001", "file": "clips/c001.wav"} в порядке показа.

    token — метка этой выборки: попадает в ключ localStorage и в labels.json.
    """

    def js(value) -> str:
        # '</' экранируем, чтобы данные не закрыли тег script
        return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")

    return (
        _TEMPLATE.replace("__RUN__", js(run_name))
        .replace("__TOKEN__", js(token))
        .replace("__CLIPS__", js(clips))
        .replace("__LABELS__", js(LABELS))
    )
