"""Генерация реалистичных чистых, adversarial- и аномальных текстов для SPL."""

import csv
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
CLEAN_PATH = BASE_DIR / "clean_texts.txt"
ANOMALY_PATH = BASE_DIR / "anomaly_texts.txt"
TRAIN_PATH = BASE_DIR / "detector_train.csv"

RANDOM_SEED = 42
CLEAN_TARGET = 1000
HARD_NEGATIVE_TARGET = 250
SHORT_TARGET = 300
MEDIUM_TARGET = 400
LONG_TARGET = 300
ANOMALY_TARGET = 200
ANOMALY_PER_CATEGORY = 40

@dataclass(frozen=True)
class CleanSample:
    """Чистый текст с устойчивым base_id и категорией."""

    text: str
    category: str
    base_id: str


TOPIC_QUESTIONS: dict[str, list[str]] = {
    "documents": [
        "какие документы нужны для поступления на бюджет",
        "можно ли подать документы через личный кабинет абитуриента",
        "нужно ли заверять копии паспорта и аттестата у нотариуса",
        "как отправить документы в приёмную комиссию по электронной почте",
        "что делать, если оригинал аттестата находится в другом городе",
        "принимаете ли вы результаты ЕГЭ из другого региона",
        "нужна ли медицинская справка для поступления на психологию",
        "как восстановиться в университете после отчисления",
        "можно ли подать документы сразу на две специальности",
        "где получить справку об обучении для военкомата",
        "какие документы нужны для перевода из другого вуза",
        "можно ли приложить портфолио к документам на дизайн",
    ],
    "transfer": [
        "как перевестись из другого вуза на второй курс",
        "какие документы нужны для перевода из другого университета",
        "сохранится ли стипендия после перевода из другого вуза",
        "можно ли перевестись с платного отделения на бюджет",
        "как перезачесть уже изученные дисциплины после перевода",
        "нужна ли академическая справка для перевода в ваш университет",
        "кто принимает решение о переводе из другого вуза",
        "можно ли перевестись в середине учебного года",
        "как получить справку о периоде обучения для перевода",
        "куда обратиться по поводу перевода на другой факультет",
        "обязательно ли согласие деканата старого вуза для перевода",
        "можно ли перевестись на заочную форму обучения из другого университета",
    ],
    "it": [
        "есть ли на кафедре информатики лаборатории по машинному обучению",
        "какие языки программирования изучают на IT-направлении",
        "можно ли выбрать Python вместо C++ на первом курсе",
        "как получить доступ к университетскому GitLab",
        "где найти расписание консультаций по базам данных",
        "есть ли курсы по кибербезопасности и защите информации",
        "как записаться на олимпиаду по программированию",
        "нужен ли собственный ноутбук для обучения на IT-направлении",
        "какие требования к проектной работе по веб-разработке",
        "можно ли пройти стажировку в IT-компании вместо летней практики",
        "какие технологии используются в лаборатории искусственного интеллекта",
        "можно ли изучать мобильную разработку на втором курсе",
    ],
    "dorm": [
        "как получить место в общежитии на время учёбы",
        "какие документы нужны для заселения в студенческое общежитие",
        "есть ли в общежитии комнаты для семейных студентов",
        "сколько стоит проживание в общежитии в месяц",
        "можно ли выбрать соседа по комнате при заселении",
        "есть ли в общежитии кухня и стиральные машины",
        "как добраться от общежития до главного корпуса",
        "предоставляется ли постельное бельё при заселении",
        "можно ли продлить проживание в общежитии на летние каникулы",
        "есть ли в общежитии интернет и рабочие места для учёбы",
        "как написать заявление на замену комнаты в общежитии",
        "есть ли льготы на оплату общежития для малоимущих студентов",
    ],
    "certificates": [
        "как получить справку об обучении для военкомата",
        "где заказать академическую справку для перевода в другой вуз",
        "сколько дней делается справка о составе семьи",
        "можно ли получить справку об обучении в электронном виде",
        "какие документы нужны для оформления справки о стипендии",
        "как получить выписку из зачётной книжки",
        "где найти образец заявления на выдачу справки",
        "нужно ли платить за повторную выдачу студенческого билета",
        "можно ли получить справку 2-НДФЛ в деканате",
        "как подтвердить статус студента для банка",
        "где взять справку о том, что я проживаю в общежитии",
        "сколько стоит изготовление копии зачётной книжки",
    ],
    "payment": [
        "как оплатить обучение за первый семестр через банк",
        "можно ли оплачивать обучение частями в течение года",
        "где найти реквизиты для перевода оплаты за обучение",
        "есть ли скидка на обучение для отличников",
        "когда нужно внести оплату, чтобы не быть отчисленным",
        "как получить квитанцию об оплате обучения",
        "можно ли вернуть часть денег при отчислении по собственному желанию",
        "предусмотрена ли рассрочка для студентов платного отделения",
        "как изменится стоимость обучения в следующем учебном году",
        "можно ли оплатить общежитие из материнского капитала",
        "кто может получить социальную стипендию и как её оформить",
        "как оформить налоговый вычет за обучение",
    ],
    "exams": [
        "когда пройдут вступительные экзамены по математике",
        "какая минимальная сумма баллов ЕГЭ нужна для поступления на экономику",
        "можно ли пересдать экзамен, если не набрал проходной балл",
        "как подать апелляцию на результат вступительного испытания",
        "где посмотреть расписание экзаменов для первокурсников",
        "сколько времени даётся на выполнение вступительного испытания",
        "можно ли взять на экзамен калькулятор и справочные материалы",
        "как узнать результаты экзамена по русскому языку",
        "нужно ли сдавать профильную математику на направление информатики",
        "как перенести экзамен по уважительной причине",
        "будут ли учитываться олимпиадные достижения при поступлении",
        "где найти демонстрационные варианты вступительных испытаний",
    ],
    "schedule": [
        "до какого числа можно подать документы на очную форму обучения",
        "когда начинается приём заявлений на заочное отделение",
        "подскажите сроки проведения вступительных испытаний в этом году",
        "когда будет опубликован приказ о зачислении на бюджет",
        "до какой даты нужно принести оригинал аттестата",
        "когда начинаются занятия у первого курса",
        "какие сроки пересдачи экзаменационной сессии",
        "когда заканчивается приём заявок на общежитие",
        "подскажите дату собрания для поступивших на юридический факультет",
        "до какого времени работает приёмная комиссия в субботу",
        "когда начнётся заселение в общежитие для первокурсников",
        "когда будут выдавать студенческие билеты",
    ],
}

TOPIC_CONTEXTS: dict[str, list[str]] = {
    "documents": [
        "Я заканчиваю школу и хочу заранее подготовить полный пакет документов, "
        "чтобы не потерять время в последний момент.",
        "У меня уже есть аттестат и результаты ЕГЭ, но я не уверен, какие именно "
        "копии и справки потребуются приёмной комиссии.",
    ],
    "transfer": [
        "Сейчас я учусь в другом университете, но рассматриваю перевод в ваш вуз "
        "и хочу понять все требования заранее.",
        "После переезда в другой город мне важно сохранить место в вузе, поэтому "
        "я внимательно изучаю порядок перевода и перезачёта дисциплин.",
    ],
    "it": [
        "Я планирую поступать на IT-направление и внимательно сравниваю учебные "
        "программы, практику и технические требования.",
        "Меня интересует не только поступление, но и то, как организована "
        "проектная работа, стажировки и доступ к современным инструментам.",
    ],
    "dorm": [
        "Я приехал из другого города и рассчитываю на общежитие, поэтому хочу "
        "заранее понять условия проживания и порядок оплаты.",
        "Мне важно заранее узнать, какие вещи можно взять в общежитие и как "
        "устроен пропускной режим для гостей.",
    ],
    "certificates": [
        "Мне часто нужны разные справки для работы и военкомата, и я хочу "
        "заранее узнать порядок их получения.",
        "Работодатель просит официальное подтверждение обучения, поэтому мне "
        "нужно понять, какие документы и сроки действуют в университете.",
    ],
    "payment": [
        "Я рассматриваю платную форму обучения и хочу заранее понять стоимость, "
        "сроки оплаты и возможные льготы.",
        "В моей семье доход нестабильный, поэтому мне важно узнать о рассрочке, "
        "стипендиях и другой финансовой поддержке студентов.",
    ],
    "exams": [
        "Я готовлюсь к вступительным испытаниям и хочу уточнить их формат, "
        "расписание и правила проведения.",
        "У меня есть результаты ЕГЭ, но я не до конца понимаю, как учитываются "
        "олимпиады и дополнительные достижения.",
    ],
    "schedule": [
        "Я пытаюсь спланировать учебный год и хочу заранее узнать важные даты "
        "и режим работы деканата.",
        "Мне нужно согласовать поездку и работу с родителями, поэтому я хочу "
        "заранее понять сроки заселения, сессии и каникул.",
    ],
}

SHORT_OPENINGS = {
    "official": [
        "Здравствуйте!",
        "Добрый день!",
        "Уважаемые сотрудники приёмной комиссии!",
        "Здравствуйте, уважаемая приёмная комиссия!",
    ],
    "colloquial": [
        "Привет!",
        "Всем привет!",
        "Здравствуйте, ребята!",
        "Добрый день!",
    ],
}

SHORT_CLOSINGS = {
    "official": [
        "Заранее благодарю за подробный ответ.",
        "Буду признателен за разъяснение.",
        "Прошу вас ответить в удобное для вас время.",
        "С уважением, абитуриент.",
    ],
    "colloquial": [
        "Спасибо!",
        "Заранее спасибо за ответ.",
        "Буду очень благодарен за помощь.",
        "Жду ответа, спасибо.",
    ],
}

SHORT_TEMPLATES = [
    "{opening} Подскажите, пожалуйста, {question}? {closing}",
    "{opening} Хотелось бы уточнить, {question}. {closing}",
    "{opening} Меня интересует, {question}. {closing}",
    "{opening} {question}? {closing}",
    "{opening} Не могли бы вы объяснить, {question}? {closing}",
    "{opening} Уточните, пожалуйста, {question}. {closing}",
]

MEDIUM_TEMPLATES = [
    "{opening} {context} Подскажите, пожалуйста, {question1}. "
    "Также хочу уточнить: {question2}? {closing}",
    "{opening} {context} Меня интересует, {question1}. "
    "И ещё один вопрос: {question2}? {closing}",
    "{opening} {context} Не могли бы вы объяснить, {question1}? "
    "Отдельно уточню: {question2}. {closing}",
    "{opening} {context} Скажите, пожалуйста, {question1}. "
    "Кроме того, подскажите, {question2}? {closing}",
    "{opening} {context} Хотелось бы понять, {question1}. "
    "И ещё: {question2}? {closing}",
]

LONG_INTROS = [
    "Я сейчас заканчиваю школу и внимательно выбираю университет, поэтому хочу "
    "заранее понять, как устроены поступление, обучение и студенческая жизнь, "
    "чтобы спокойно подготовить все документы и не упустить важные детали.",
    "Моя ситуация требует более подробной консультации: я рассматриваю несколько "
    "вузов, сравниваю условия обучения и хочу понять, какие шаги нужно предпринять "
    "именно в моём случае.",
    "Я уже определился с направлением, но у меня осталось много организационных "
    "вопросов, связанных с документами, сроками и будущей студенческой жизнью, "
    "поэтому хотел бы получить развёрнутую консультацию.",
    "Я переезжаю из другого региона и хочу заранее спланировать поступление, "
    "проживание и финансовые расходы, чтобы первая неделя в университете прошла "
    "без лишнего стресса.",
    "Мне важно не просто узнать короткий ответ, а понять общий порядок действий, "
    "официальные сроки и людей, к которым можно обратиться с похожим вопросом "
    "в дальнейшем.",
]

LONG_ADDITIONS = [
    "Подскажите, пожалуйста, на какие официальные страницы и документы можно "
    "ориентироваться, чтобы получать только актуальную информацию.",
    "Если возможно, опишите порядок действий по шагам, потому что я впервые "
    "сталкиваюсь с этой процедурой и не хочу ничего перепутать.",
    "Уточните, пожалуйста, сроки рассмотрения обращения и способ получения "
    "ответа: по электронной почте, в личном кабинете или при личном визите.",
    "Буду признателен, если вы перечислите документы, которые нужно подготовить "
    "заранее, и объясните, какие копии потребуются в обязательном порядке.",
    "Скажите, пожалуйста, куда именно обращаться с этим вопросом: в приёмную "
    "комиссию, деканат, учебный отдел или студенческий офис.",
    "Мне также важно понять, что делать, если какой-то документ будет оформлен "
    "позже остальных и можно ли принести его дополнительно.",
    "Не могли бы вы подсказать, какие ошибки чаще всего допускают абитуриенты "
    "в такой ситуации и как их избежать.",
    "Если вопрос решается не сразу, объясните, пожалуйста, как отслеживать "
    "статус заявления и куда обращаться при возникновении проблем.",
    "Хотелось бы заранее понять, отличается ли порядок действий для иногородних "
    "студентов и есть ли дополнительные требования для иностранных граждан.",
    "Пожалуйста, уточните, действуют ли эти правила в текущем учебном году и "
    "планируются ли изменения в следующем приёмном сезоне.",
]

LONG_CLOSINGS = [
    "Заранее благодарю за подробный ответ и потраченное время.",
    "Спасибо большое за помощь, буду ждать вашего ответа.",
    "Буду очень признателен за разъяснение и ссылки на официальные документы.",
    "С уважением, абитуриент, который хочет подготовиться заранее.",
    "Спасибо за внимание к моему вопросу и за любую полезную информацию.",
]

HOMOGLYPHS = {
    "а": "a",
    "о": "o",
    "е": "e",
    "с": "c",
    "р": "p",
    "х": "x",
    "у": "y",
    "к": "k",
    "м": "m",
}

KEYBOARD_LAYOUT = {
    "й": "q",
    "ц": "w",
    "у": "e",
    "к": "r",
    "е": "t",
    "н": "y",
    "г": "u",
    "ш": "i",
    "щ": "o",
    "з": "p",
    "х": "[",
    "ъ": "]",
    "ф": "a",
    "ы": "s",
    "в": "d",
    "а": "f",
    "п": "g",
    "р": "h",
    "о": "j",
    "л": "k",
    "д": "l",
    "ж": ";",
    "э": "'",
    "я": "z",
    "ч": "x",
    "с": "c",
    "м": "v",
    "и": "b",
    "т": "n",
    "ь": "m",
    "б": ",",
    "ю": ".",
}

EMOJIS = [
    "😀",
    "😁",
    "😂",
    "🤣",
    "😊",
    "😍",
    "🤩",
    "😎",
    "🥳",
    "😱",
    "🔥",
    "💥",
    "✨",
    "🎉",
    "🎁",
    "👍",
    "💯",
    "🚀",
    "⭐",
    "❤️",
]

CODE_TEMPLATES = [
    "def process_{i}(items): return [item * {n} for item in items if item]",
    "SELECT id, name FROM students WHERE course = {n} ORDER BY name;",
    "<div class='block-{i}'>error: timeout, code: {n}</div>",
    "for (let i = 0; i < {n}; i++) {{ console.log(i * {i_minus}); }}",
    "import numpy as np; x = np.random.rand({n}, 3); print(x.mean(axis=0))",
    "git checkout -b feature/{i}; npm install; npm run build",
    "class User{i}: pass",
    "curl -X POST https://example.com/api/{i} -d 'id={n}'",
    "if (!user.isActive) throw new Error('inactive-{i}');",
    "BEGIN TRANSACTION; UPDATE accounts SET balance = balance - {n} WHERE id = {i}; COMMIT;",
]

SPAM_TEMPLATES = [
    "ПОБЕДИТЕЛЬ №{i}! Вы выиграли {n} рублей! Срочно перейдите по ссылке "
    "http://example.com/prize/{i} и укажите данные банковской карты.",
    "ТОЛЬКО СЕГОДНЯ! Кредит без отказа за {n} минут! Звоните прямо сейчас, "
    "количество предложений ограничено!",
    "Ваш аккаунт заблокирован! Чтобы восстановить доступ, отправьте пароль и код "
    "из СМС на адрес security-{i}@example.com.",
    "РАСПРОДАЖА! Скидка {n}% на всё! Закажите сейчас, и мы подарим второй товар "
    "бесплатно! Переходите на http://example.com/sale/{i}.",
    "Уважаемый клиент! Вам одобрен займ на сумму {n} рублей. Для получения денег "
    "нужно только оплатить комиссию и сообщить CVV.",
    "ЗАРАБОТОК БЕЗ ВЛОЖЕНИЙ! Доход {n} рублей в день! Пишите в мессенджер, "
    "и мы расскажем секрет успеха!",
    "Ваша посылка задержана! Оплатите пошлину {n} рублей по ссылке "
    "http://example.com/parcel/{i}, иначе посылка будет уничтожена.",
    "Поздравляем! Ваш номер выбран среди {n} участников. Получите приз, "
    "отправив паспортные данные на prize-{i}@example.com.",
    "СРОЧНО! Горящие туры со скидкой {n}%! Только сегодня и только для вас! "
    "Звоните, количество мест ограничено!",
    "Инвестируйте {n} рублей и получите гарантированную прибыль за неделю! "
    "Напишите нам, чтобы узнать подробности.",
]

ENGLISH_SENTENCES = [
    "I would like to know more about the admission requirements for international students.",
    "Could you please explain how the application deadline works for this program?",
    "The documentation is incomplete, and I need to understand what to submit next.",
    "Our team is developing a new machine learning service for university applicants.",
    "Please review the attached transcript and confirm whether it meets the requirements.",
    "I am interested in the internship opportunities available during the summer semester.",
    "The dormitory application form requires additional information about my financial situation.",
    "We need to schedule a meeting to discuss the transfer procedure and credit recognition.",
    "This message contains technical details about the web application and its deployment.",
    "The support team will respond within five business days after receiving the request.",
]

REPEATED_PHRASES = [
    "очень срочно",
    "нужно очень быстро",
    "пожалуйста ответьте",
    "это очень важно",
    "спасибо большое",
    "повторяю ещё раз",
]

HARD_NEGATIVE_CORES = [
    "Подскажите, где скачать Python для выполнения лабораторных по программированию?",
    "На сайте https://example.org/schedule указано, что занятия перенесены; "
    "это официальная информация?",
    "Я отправил скан паспорта на priem@example.org, нужно ли приносить оригинал?",
    "В заявлении написано: «прошу перевести меня на второй курс», что это означает?",
    "Можно ли сдать задание по SQL через университетский GitLab?",
    "В личном кабинете статус заявления изменился на In Progress, когда ждать ответа?",
    "Для оплаты обучения я использую реквизиты с https://example.org/payment, всё верно?",
    "Можно ли подключить ноутбук с Windows 11 к Wi-Fi Eduroam в общежитии?",
    "Где получить доступ к API учебного расписания для студенческого проекта?",
    "В приказе написано: «студенту предоставляется общежитие», нужно ли отдельное заявление?",
    "Как отправить документы на transfer@example.org, если файл больше 10 MB?",
    "Подскажите, где найти инструкцию по MS Office и Excel для первокурсников.",
    "В расписании на https://example.org/timetable указана аудитория B-201, это главный корпус?",
    "Можно ли оплатить обучение через СБП по QR-коду с сайта университета?",
    "Я указал email student@example.org в заявлении, туда придёт подтверждение?",
    "В договоре написано: «оплата производится двумя платежами», можно ли внести всю сумму сразу?",
    "Где посмотреть список литературы в формате PDF по курсу Data Science?",
    "Можно ли использовать LaTeX для оформления курсовой работы?",
    "На странице https://example.org/faq написано, что справка готова; где её забрать?",
    "Кто отвечает за доступ к личному кабинету, если не работает вход через SSO?",
    "Можно ли перевести документы в PDF и отправить на documents@example.org?",
    "В чате поддержки мне ответили: «обращение зарегистрировано», сколько ждать?",
    "Подскажите, где найти логин и пароль для университетского Moodle.",
    "Можно ли сдать экзамен по английскому в формате online, если я на больничном?",
    "В инструкции сказано: «студент обязан подтвердить email», где это сделать?",
    "У меня не открывается файл с расширением .xlsx, можно ли отправить его в PDF?",
    "На странице https://example.org/it-support описан сброс пароля через SMS, это безопасно?",
    "Можно ли указать в заявке ссылку на GitHub вместо бумажного портфолио?",
    "В ответе деканата написано: «перезачёт дисциплин согласован», когда появится оценка?",
    "Где найти инструкцию по установке Microsoft Teams на macOS?",
]


def count_words(text: str) -> int:
    """Посчитать слова в тексте."""
    return len(text.split())


def generate_short_text(rng: random.Random, topic: str) -> str:
    """Сгенерировать короткое обращение на 10–25 слов."""
    style = rng.choice(("official", "colloquial"))
    question = rng.choice(TOPIC_QUESTIONS[topic])
    template = rng.choice(SHORT_TEMPLATES)
    closing = rng.choice(SHORT_CLOSINGS[style])
    text = template.format(
        opening=rng.choice(SHORT_OPENINGS[style]),
        question=question,
        closing=closing,
    )
    while count_words(text) < 10:
        text = f"{text} {rng.choice(SHORT_CLOSINGS[style])}"
    return text


def generate_medium_text(rng: random.Random, topic: str) -> str:
    """Сгенерировать обращение среднего размера."""
    style = rng.choice(("official", "colloquial"))
    questions = rng.sample(TOPIC_QUESTIONS[topic], k=2)
    template = rng.choice(MEDIUM_TEMPLATES)
    text = template.format(
        opening=rng.choice(SHORT_OPENINGS[style]),
        context=rng.choice(TOPIC_CONTEXTS[topic]),
        question1=questions[0],
        question2=questions[1],
        closing=rng.choice(SHORT_CLOSINGS[style]),
    )
    while count_words(text) < 20:
        text = f"{text} {rng.choice(LONG_ADDITIONS)}"
    return text


def generate_long_text(rng: random.Random, topic: str) -> str:
    """Сгенерировать длинное обращение на 100+ слов."""
    style = rng.choice(("official", "colloquial"))
    questions = rng.sample(TOPIC_QUESTIONS[topic], k=4)
    prefixes = [
        "Кроме того, хочу уточнить:",
        "Также меня интересует:",
        "Отдельный вопрос:",
        "И ещё:",
    ]
    parts = [
        rng.choice(SHORT_OPENINGS[style]),
        rng.choice(LONG_INTROS),
        rng.choice(TOPIC_CONTEXTS[topic]),
    ]
    for question, prefix in zip(questions, prefixes, strict=True):
        parts.append(f"{prefix} {question}?")
    parts.extend(rng.sample(LONG_ADDITIONS, k=2))
    parts.append(rng.choice(LONG_CLOSINGS))

    text = " ".join(parts)
    additions = list(LONG_ADDITIONS)
    rng.shuffle(additions)
    while count_words(text) < 100:
        parts.append(additions.pop() if additions else rng.choice(LONG_ADDITIONS))
        text = " ".join(parts)
    return text


def generate_hard_negatives(rng: random.Random) -> list[CleanSample]:
    """Сгенерировать нормальные тексты с латиницей, URL, email и цитатами."""
    openings = ["Здравствуйте!", "Добрый день!", "Всем привет!", ""]
    closings = [
        "Заранее спасибо.",
        "Буду благодарен за помощь.",
        "Спасибо за ответ.",
        "Помогите, пожалуйста, разобраться.",
    ]
    seen: set[str] = set()
    samples: list[CleanSample] = []
    while len(samples) < HARD_NEGATIVE_TARGET:
        core = rng.choice(HARD_NEGATIVE_CORES)
        opening = rng.choice(openings)
        closing = rng.choice(closings)
        parts = [part for part in (opening, core, closing) if part]
        text = " ".join(parts)
        if text in seen:
            continue
        seen.add(text)
        samples.append(CleanSample(text=text, category="hard_negative", base_id=""))
    return samples


def generate_clean_samples() -> list[CleanSample]:
    """Сгенерировать чистые тексты всех категорий и назначить base_id."""
    rng = random.Random(RANDOM_SEED)
    topics = list(TOPIC_QUESTIONS)
    seen: set[str] = set()
    samples: list[CleanSample] = []

    def add_generated(generator: Callable[[random.Random, str], str], target: int) -> None:
        added = 0
        while added < target:
            topic = rng.choice(topics)
            text = generator(rng, topic)
            if text in seen:
                continue
            seen.add(text)
            samples.append(CleanSample(text=text, category=topic, base_id=""))
            added += 1

    add_generated(generate_short_text, SHORT_TARGET)
    add_generated(generate_medium_text, MEDIUM_TARGET)
    add_generated(generate_long_text, LONG_TARGET)
    samples.extend(generate_hard_negatives(rng))
    rng.shuffle(samples)

    return [
        CleanSample(text=sample.text, category=sample.category, base_id=f"base_{index:05d}")
        for index, sample in enumerate(samples, start=1)
    ]


def write_clean_texts(samples: list[CleanSample]) -> None:
    """Сохранить чистые тексты в текстовый файл."""
    CLEAN_PATH.write_text(
        "\n".join(sample.text for sample in samples) + "\n",
        encoding="utf-8",
    )
    logger.info("Создан %s (%s текстов)", CLEAN_PATH, len(samples))


def apply_homoglyphs(text: str, rng: random.Random) -> str:
    """Заменить часть кириллических букв визуально похожими латинскими."""
    result: list[str] = []
    replaced = 0
    for character in text:
        replacement = HOMOGLYPHS.get(character.lower())
        if replacement is not None and rng.random() < 0.35:
            result.append(replacement.upper() if character.isupper() else replacement)
            replaced += 1
        else:
            result.append(character)

    if replaced == 0:
        positions = [
            index for index, character in enumerate(text) if character.lower() in HOMOGLYPHS
        ]
        if positions:
            index = rng.choice(positions)
            character = text[index]
            replacement = HOMOGLYPHS[character.lower()]
            result[index] = replacement.upper() if character.isupper() else replacement
    return "".join(result)


def apply_keyboard_layout(text: str, rng: random.Random) -> str:
    """Перевести русские буквы в латинские по раскладке QWERTY."""
    del rng
    result: list[str] = []
    for character in text:
        replacement = KEYBOARD_LAYOUT.get(character.lower())
        if replacement is None:
            result.append(character)
        else:
            result.append(replacement.upper() if character.isupper() else replacement)
    return "".join(result)


def apply_invisible_characters(text: str, rng: random.Random) -> str:
    """Вставить невидимые символы внутрь слов."""
    words = text.split(" ")
    changed = False
    result: list[str] = []
    for word in words:
        if len(word) >= 2 and rng.random() < 0.35:
            position = rng.randint(1, len(word) - 1)
            invisible = "\u200b" if rng.random() < 0.5 else "\u00ad"
            word = f"{word[:position]}{invisible}{word[position:]}"
            changed = True
        result.append(word)

    if not changed:
        for index, word in enumerate(result):
            if len(word) >= 2:
                position = rng.randint(1, len(word) - 1)
                result[index] = f"{word[:position]}\u200b{word[position:]}"
                break
    return " ".join(result)


def apply_typos(text: str, rng: random.Random) -> str:
    """Переставить соседние буквы, имитируя случайные опечатки."""
    words = text.split(" ")
    changed = False
    result: list[str] = []
    for word in words:
        characters = list(word)
        positions = [
            index
            for index in range(len(characters) - 1)
            if characters[index].isalpha() and characters[index + 1].isalpha()
        ]
        if positions and rng.random() < 0.25:
            index = rng.choice(positions)
            characters[index], characters[index + 1] = (
                characters[index + 1],
                characters[index],
            )
            changed = True
        result.append("".join(characters))

    if not changed:
        for word_index, word in enumerate(result):
            characters = list(word)
            positions = [
                index
                for index in range(len(characters) - 1)
                if characters[index].isalpha() and characters[index + 1].isalpha()
            ]
            if positions:
                index = rng.choice(positions)
                characters[index], characters[index + 1] = (
                    characters[index + 1],
                    characters[index],
                )
                result[word_index] = "".join(characters)
                break
    return " ".join(result)


ATTACK_FUNCTIONS: dict[str, Callable[[str, random.Random], str]] = {
    "homoglyph": apply_homoglyphs,
    "keyboard_layout": apply_keyboard_layout,
    "invisible": apply_invisible_characters,
    "typo": apply_typos,
}


def generate_corrupted_rows(
    clean_samples: list[CleanSample],
    rng: random.Random,
) -> list[dict[str, str | int]]:
    """Сгенерировать по одному искажённому тексту на каждый базовый текст."""
    attack_types = list(ATTACK_FUNCTIONS)
    selected_types = [
        attack_types[index % len(attack_types)]
        for index in range(len(clean_samples))
    ]
    rng.shuffle(selected_types)

    rows: list[dict[str, str | int]] = []
    for sample, attack_type in zip(clean_samples, selected_types, strict=True):
        attack_function = ATTACK_FUNCTIONS[attack_type]
        corrupted = attack_function(sample.text, rng)
        if corrupted == sample.text:
            corrupted = apply_keyboard_layout(sample.text, rng)
        rows.append(
            {
                "text": corrupted,
                "label": 1,
                "attack_type": attack_type,
                "base_id": sample.base_id,
                "category": sample.category,
            }
        )
    return rows


def generate_anomaly_texts(rng: random.Random) -> list[str]:
    """Сгенерировать 200 нетипичных, но не adversarial-текстов."""
    seen: set[str] = set()

    def normalize(text: str) -> str:
        return " ".join(text.replace("\n", " ").split())

    code_texts: list[str] = []
    while len(code_texts) < ANOMALY_PER_CATEGORY:
        index = len(code_texts) + 1
        template = rng.choice(CODE_TEMPLATES)
        text = normalize(template.format(i=index, n=index * 7 + 3, i_minus=index + 1))
        if text not in seen:
            seen.add(text)
            code_texts.append(text)

    spam_texts: list[str] = []
    while len(spam_texts) < ANOMALY_PER_CATEGORY:
        index = len(spam_texts) + 1
        template = rng.choice(SPAM_TEMPLATES)
        text = normalize(template.format(i=index, n=(index + 3) * 1000))
        if text not in seen:
            seen.add(text)
            spam_texts.append(text)

    english_texts: list[str] = []
    while len(english_texts) < ANOMALY_PER_CATEGORY:
        index = len(english_texts) + 1
        sentences = rng.sample(ENGLISH_SENTENCES, k=3)
        text = normalize(" ".join(sentences) + f" Case reference {index}.")
        if text not in seen:
            seen.add(text)
            english_texts.append(text)

    emoji_texts: list[str] = []
    while len(emoji_texts) < ANOMALY_PER_CATEGORY:
        index = len(emoji_texts) + 1
        emoji_text = " ".join(rng.choice(EMOJIS) for _ in range(60))
        text = normalize(f"{emoji_text} СРОЧНО!!! Сообщение номер {index}.")
        if text not in seen:
            seen.add(text)
            emoji_texts.append(text)

    repeated_texts: list[str] = []
    while len(repeated_texts) < ANOMALY_PER_CATEGORY:
        index = len(repeated_texts) + 1
        phrase = rng.choice(REPEATED_PHRASES)
        repeats = 40 + index % 20
        text = normalize(f"{' '.join([phrase] * repeats)} номер {index}")
        if text not in seen:
            seen.add(text)
            repeated_texts.append(text)

    return code_texts + spam_texts + english_texts + emoji_texts + repeated_texts


def write_anomaly_texts(rng: random.Random) -> list[str]:
    """Сгенерировать и сохранить anomaly_texts.txt."""
    texts = generate_anomaly_texts(rng)
    ANOMALY_PATH.write_text("\n".join(texts) + "\n", encoding="utf-8")
    logger.info("Создан %s (%s текстов)", ANOMALY_PATH, len(texts))
    return texts


def write_train_csv(
    clean_samples: list[CleanSample],
    corrupted_rows: list[dict[str, str | int]],
) -> None:
    """Сохранить чистые и искажённые тексты в detector_train.csv."""
    clean_rows: list[dict[str, str | int]] = [
        {
            "text": sample.text,
            "label": 0,
            "attack_type": "",
            "base_id": sample.base_id,
            "category": sample.category,
        }
        for sample in clean_samples
    ]
    rows = clean_rows + corrupted_rows
    fieldnames = ["text", "label", "attack_type", "base_id", "category"]

    with TRAIN_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Создан %s: %s строк", TRAIN_PATH, len(rows))


def main() -> None:
    """Сгенерировать все данные и вывести статистику."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    rng = random.Random(RANDOM_SEED)

    clean_samples = generate_clean_samples()
    base_samples = [
        sample for sample in clean_samples if sample.category != "hard_negative"
    ]
    corrupted_rows = generate_corrupted_rows(base_samples, rng)
    anomaly_texts = write_anomaly_texts(rng)
    write_clean_texts(clean_samples)
    write_train_csv(clean_samples, corrupted_rows)

    length_counts = {"short": 0, "medium": 0, "long": 0}
    for sample in clean_samples:
        words = count_words(sample.text)
        if words < 20:
            length_counts["short"] += 1
        elif words < 100:
            length_counts["medium"] += 1
        else:
            length_counts["long"] += 1

    category_counts: dict[str, int] = {}
    for sample in clean_samples:
        category_counts[sample.category] = category_counts.get(sample.category, 0) + 1

    attack_counts: dict[str, int] = {}
    for row in corrupted_rows:
        attack_type = str(row["attack_type"])
        attack_counts[attack_type] = attack_counts.get(attack_type, 0) + 1

    logger.info("Статистика:")
    logger.info("  чистых текстов: %s", len(clean_samples))
    logger.info("  по категориям: %s", category_counts)
    logger.info(
        "  по длине: коротких=%s, средних=%s, длинных=%s",
        length_counts["short"],
        length_counts["medium"],
        length_counts["long"],
    )
    logger.info("  испорченных текстов: %s", len(corrupted_rows))
    logger.info("  по типам атак: %s", attack_counts)
    logger.info("  аномальных текстов: %s", len(anomaly_texts))
    logger.info(
        "  всего строк в CSV: %s",
        len(clean_samples) + len(corrupted_rows),
    )


if __name__ == "__main__":
    main()
