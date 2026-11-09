# Веб-клиент SPL

Отдельный HTML/JS-клиент для проверки API SPL. Умеет:

- проверять доступность API (`GET /health`);
- отправлять текст в `POST /check`;
- показывать решение, причину, категорию, scores и SHAP-объяснение;
- загружать последние записи журнала (`GET /logs`);
- хранить адрес API в `localStorage` браузера.

Клиент не требует сторонних JS-библиотек. Для запуска нужен Python 3.10+
(используется стандартная библиотека), современный браузер (Edge, Chrome или
Firefox) и запущенный API SPL.

## Запуск API

Из корня проекта:

```bash
./run.sh api
```

На Windows:

```cmd
run.cmd api
```

## Запуск клиента

### Linux/macOS

```bash
./client/run_client.sh
```

### Windows 10/11

```cmd
client\run_client.cmd
```

Или через PowerShell:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\client\run_client.ps1
```

`run_client.cmd` уже обходит Execution Policy и вызывает `run_client.ps1`.

После запуска откроется браузер:

```text
http://127.0.0.1:3000/
```

Если браузер не открылся, перейди по адресу вручную. На Linux/macOS можно
запустить и через Makefile:

```bash
make run-client
```

## Параметры

Linux/macOS:

```bash
./client/run_client.sh --no-browser
./client/run_client.sh --port 3000
```

Windows:

```powershell
.\client\run_client.ps1 -NoBrowser
.\client\run_client.ps1 -Port 3000
.\client\run_client.ps1 -PythonBin C:\Python312\python.exe
```

## Важно

- API должен быть запущен до отправки запросов.
- Клиент специально открывается на порту `3000`: этот origin разрешён в CORS
  API (`http://127.0.0.1:3000` и `http://localhost:3000`).
- Если открыть `index.html` как файл через `file://`, браузер заблокирует
  запросы к API из-за CORS. Используй скрипт запуска.
- Первый чистый запрос может занять время: API скачивает модель GeRaCl
  (около 600 МБ).
- Если порт 3000 занят, закрой старый экземпляр клиента.
- Если меняешь порт клиента, добавь новый origin в `ALLOWED_ORIGINS` в
  `src/presentation/api/app.py`, иначе браузер заблокирует запросы из-за CORS.

## Ручной запуск сервера

```bash
python client/server.py --no-browser
```

Сервер отдаёт статические файлы из папки `client/`.
