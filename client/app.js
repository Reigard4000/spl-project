"use strict";

const API_STORAGE_KEY = "spl_api_url";
const DEFAULT_API_URL = "http://127.0.0.1:8000";
const CHECK_TIMEOUT_MS = 180000;

const EXAMPLES = {
    clean: "Здравствуйте! Какие документы нужны для поступления на бюджет?",
    adversarial: "К@к1е д0кумеHты Hy)KHы для п0стyп/\\еHия?",
    anomaly:
        "ПОБЕДИТЕЛЬ №5! Вы выиграли 8000 рублей! Срочно перейдите по ссылке " +
        "http://example.com/prize/5 и укажите данные банковской карты.",
};

const REASON_LABELS = {
    adversarial: "Adversarial-атака",
    anomaly: "Аномалия",
};

const apiInput = document.getElementById("api-url");
const healthButton = document.getElementById("health-btn");
const healthStatus = document.getElementById("health-status");
const requestText = document.getElementById("request-text");
const sendButton = document.getElementById("send-btn");
const clearButton = document.getElementById("clear-btn");
const requestError = document.getElementById("request-error");
const decision = document.getElementById("decision");
const decisionDetails = document.getElementById("decision-details");
const rawDetails = document.getElementById("raw-details");
const rawJson = document.getElementById("raw-json");
const logsButton = document.getElementById("logs-btn");
const logsError = document.getElementById("logs-error");
const logsBody = document.querySelector("#logs-table tbody");

function apiBase() {
    let value = apiInput.value.trim().replace(/\/+$/, "");
    if (value && !/^https?:\/\//i.test(value)) {
        value = `http://${value}`;
    }
    return value || DEFAULT_API_URL;
}

function setHealth(text, kind, title = "") {
    healthStatus.textContent = text;
    healthStatus.className = `badge ${kind}`;
    healthStatus.title = title;
}

function showError(element, message) {
    element.textContent = message;
    element.hidden = !message;
}

function describeError(data) {
    if (typeof data === "string") {
        return data;
    }
    if (data && typeof data === "object") {
        if (Array.isArray(data.detail)) {
            return data.detail
                .map((item) => item.msg || JSON.stringify(item))
                .join("; ");
        }
        if (data.detail) {
            return String(data.detail);
        }
        if (data.error) {
            return String(data.error);
        }
        return JSON.stringify(data);
    }
    return String(data);
}

async function fetchJson(path, options = {}, timeoutMs = 30000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        const response = await fetch(`${apiBase()}${path}`, {
            ...options,
            signal: controller.signal,
            headers: {
                Accept: "application/json",
                ...(options.headers || {}),
            },
        });
        const text = await response.text();
        let data = null;
        if (text) {
            try {
                data = JSON.parse(text);
            } catch {
                data = text;
            }
        }
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${describeError(data)}`);
        }
        return data;
    } catch (error) {
        if (error.name === "AbortError") {
            throw new Error("Превышено время ожидания ответа API.");
        }
        throw error;
    } finally {
        clearTimeout(timer);
    }
}

async function checkHealth() {
    healthButton.disabled = true;
    setHealth("проверяю...", "muted");
    try {
        const data = await fetchJson("/health", {}, 10000);
        if (data && data.status === "ok") {
            setHealth("API доступен", "ok");
        } else {
            setHealth("неожиданный ответ", "warn", JSON.stringify(data));
        }
    } catch (error) {
        setHealth("API недоступен", "bad", error.message);
    } finally {
        healthButton.disabled = false;
    }
}

function reasonLabel(reason) {
    if (!reason) {
        return "—";
    }
    return REASON_LABELS[reason] || reason;
}

function formatTime(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return String(value || "—");
    }
    return date.toLocaleString("ru-RU");
}

function addDetailBlock(container, title) {
    const block = document.createElement("div");
    block.className = "detail-block";
    const heading = document.createElement("strong");
    heading.textContent = title;
    block.append(heading);
    container.append(block);
    return block;
}

function addDefinition(block, definitions) {
    const list = document.createElement("dl");
    for (const [term, value] of definitions) {
        const dt = document.createElement("dt");
        dt.textContent = term;
        const dd = document.createElement("dd");
        dd.textContent = value;
        list.append(dt, dd);
    }
    block.append(list);
}

function addScores(block, scores) {
    const entries = Object.entries(scores || {}).sort(
        (left, right) => Number(right[1]) - Number(left[1]),
    );
    if (entries.length === 0) {
        return;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "scores";
    for (const [label, rawValue] of entries) {
        const value = Number(rawValue);
        const safeValue = Number.isFinite(value) ? value : 0;

        const row = document.createElement("div");
        row.className = "score-row";

        const name = document.createElement("span");
        name.textContent = label;

        const bar = document.createElement("div");
        bar.className = "score-bar";
        const fill = document.createElement("div");
        fill.className = "score-fill";
        fill.style.width = `${Math.max(0, Math.min(100, safeValue * 100))}%`;
        bar.append(fill);

        const shown = document.createElement("span");
        shown.className = "score-value";
        shown.textContent = `${(safeValue * 100).toFixed(1)}%`;

        row.append(name, bar, shown);
        wrapper.append(row);
    }
    block.append(wrapper);
}

function renderDecision(payload) {
    const allowed = Boolean(payload && payload.allowed);
    decision.className = `decision ${allowed ? "allowed" : "blocked"}`;
    decision.replaceChildren();

    const title = document.createElement("p");
    title.className = "decision-title";
    title.textContent = allowed ? "Пропущено к модели" : "Заблокировано";
    decision.append(title);

    const subtitle = document.createElement("p");
    subtitle.className = "decision-subtitle";
    if (allowed) {
        const prediction = payload && payload.prediction;
        const category = prediction && prediction.category;
        subtitle.textContent = category
            ? `Категория модели: ${category}`
            : "Запрос прошёл проверку.";
    } else {
        subtitle.textContent = `Причина: ${reasonLabel(payload && payload.reason)}`;
    }
    decision.append(subtitle);

    decisionDetails.replaceChildren();
    const allowedPrediction = payload && payload.prediction;
    if (allowed && allowedPrediction) {
        const categoryBlock = addDetailBlock(decisionDetails, "Предсказание модели");
        addDefinition(categoryBlock, [
            ["Категория", String(allowedPrediction.category || "—")],
        ]);
        addScores(categoryBlock, allowedPrediction.scores);
    }

    if (!allowed && payload && payload.explanation) {
        const explanationBlock = addDetailBlock(
            decisionDetails,
            "Объяснение детектора",
        );
        const explanation = document.createElement("p");
        explanation.textContent = payload.explanation;
        explanationBlock.append(explanation);
    }

    if (!allowed && !(payload && payload.explanation)) {
        const noteBlock = addDetailBlock(decisionDetails, "Объяснение детектора");
        const note = document.createElement("p");
        note.className = "hint";
        note.textContent = "Для этой блокировки объяснение не предусмотрено.";
        noteBlock.append(note);
    }

    decisionDetails.hidden = decisionDetails.childElementCount === 0;

    rawJson.textContent = JSON.stringify(payload, null, 2);
    rawDetails.hidden = false;
}

function renderLogs(entries) {
    logsBody.replaceChildren();
    if (!Array.isArray(entries) || entries.length === 0) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 5;
        cell.className = "empty";
        cell.textContent = "Журнал пуст. Отправь запрос через форму.";
        row.append(cell);
        logsBody.append(row);
        return;
    }

    for (const entry of entries) {
        const row = document.createElement("tr");
        row.title = "Нажми, чтобы подставить текст в форму";

        const timeCell = document.createElement("td");
        timeCell.textContent = formatTime(entry.timestamp);

        const statusCell = document.createElement("td");
        statusCell.textContent =
            entry.status === "allowed" ? "Пропущено" : "Заблокировано";
        statusCell.className =
            entry.status === "allowed" ? "status-allowed" : "status-blocked";

        const reasonCell = document.createElement("td");
        reasonCell.textContent = reasonLabel(entry.reason);

        const categoryCell = document.createElement("td");
        categoryCell.textContent =
            entry.prediction && entry.prediction.category
                ? entry.prediction.category
                : "—";

        const textCell = document.createElement("td");
        textCell.textContent = entry.text;

        row.append(timeCell, statusCell, reasonCell, categoryCell, textCell);
        row.addEventListener("click", () => {
            requestText.value = entry.text;
            requestText.focus();
        });
        logsBody.append(row);
    }
}

async function loadLogs() {
    logsButton.disabled = true;
    showError(logsError, "");
    try {
        const entries = await fetchJson("/logs?limit=20", {}, 30000);
        renderLogs(entries);
    } catch (error) {
        showError(logsError, error.message);
    } finally {
        logsButton.disabled = false;
    }
}

async function sendRequest() {
    const text = requestText.value.trim();
    if (!text) {
        showError(requestError, "Введите текст запроса.");
        requestText.focus();
        return;
    }

    showError(requestError, "");
    sendButton.disabled = true;
    sendButton.textContent = "Проверяю...";
    decision.className = "decision muted";
    decision.textContent = "Запрос отправлен, ждём ответ API...";
    decisionDetails.hidden = true;
    rawDetails.hidden = true;

    try {
        const payload = await fetchJson(
            "/check",
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text }),
            },
            CHECK_TIMEOUT_MS,
        );
        renderDecision(payload);
        await loadLogs();
    } catch (error) {
        decision.className = "decision blocked";
        decision.textContent = "Не удалось получить решение";
        showError(
            requestError,
            `${error.message}. Проверь, что API запущен и адрес указан верно.`,
        );
    } finally {
        sendButton.disabled = false;
        sendButton.textContent = "Проверить";
    }
}

function resetResult() {
    decision.className = "decision muted";
    decision.textContent = "Отправь запрос, чтобы увидеть результат.";
    decisionDetails.replaceChildren();
    decisionDetails.hidden = true;
    rawDetails.hidden = true;
    rawJson.textContent = "";
    showError(requestError, "");
}

function saveApiUrl() {
    const value = apiInput.value.trim();
    if (value) {
        localStorage.setItem(API_STORAGE_KEY, value);
    } else {
        localStorage.removeItem(API_STORAGE_KEY);
    }
    setHealth("не проверен", "muted");
}

function init() {
    const storedUrl = localStorage.getItem(API_STORAGE_KEY);
    if (storedUrl) {
        apiInput.value = storedUrl;
    }

    apiInput.addEventListener("change", saveApiUrl);
    healthButton.addEventListener("click", checkHealth);
    sendButton.addEventListener("click", sendRequest);
    logsButton.addEventListener("click", loadLogs);
    clearButton.addEventListener("click", () => {
        requestText.value = "";
        resetResult();
        requestText.focus();
    });

    document.querySelectorAll("[data-example]").forEach((button) => {
        button.addEventListener("click", () => {
            const key = button.dataset.example;
            requestText.value = EXAMPLES[key] || "";
            requestText.focus();
        });
    });

    requestText.addEventListener("keydown", (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
            event.preventDefault();
            sendRequest();
        }
    });

    resetResult();
    checkHealth();
    loadLogs();
}

init();
