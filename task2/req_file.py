import requests
from bs4 import BeautifulSoup
import os

# Список страниц для скачивания
pages = {
    "Гарри Поттер": "https://harrypotter.fandom.com/ru/wiki/Гарри_Поттер",
    "Гермиона Грейнджер": "https://harrypotter.fandom.com/ru/wiki/Гермиона_Грейнджер",
    "Рон Уизли": "https://harrypotter.fandom.com/ru/wiki/Рон_Уизли",
    "Северус Снегг": "https://harrypotter.fandom.com/ru/wiki/Северус_Снегг",
    "Беллатриса Лестрейндж": "https://harrypotter.fandom.com/ru/wiki/Беллатриса_Лестрейндж",
    "Сириус Блэк": "https://harrypotter.fandom.com/ru/wiki/Сириус_Блэк",
    "Орден Феникса": "https://harrypotter.fandom.com/ru/wiki/Категория:Орден_Феникса",
    "Альбус Дамблдор": "https://harrypotter.fandom.com/ru/wiki/Альбус_Дамблдор",
    "Билл Уизли": "https://harrypotter.fandom.com/ru/wiki/Билл_Уизли",
    "Римус Люпин": "https://harrypotter.fandom.com/ru/wiki/Римус_Люпин",
    "Нимфадора Тонкс": "https://harrypotter.fandom.com/ru/wiki/Нимфадора_Тонкс",
    "Лили Поттер": "https://harrypotter.fandom.com/ru/wiki/Лили_Поттер",
    "Джеймс Поттер": "https://harrypotter.fandom.com/ru/wiki/Джеймс_Поттер",
    "Азкабан": "https://harrypotter.fandom.com/ru/wiki/Азкабан",
    "Нурменгард": "https://harrypotter.fandom.com/ru/wiki/Нурменгард",
    "Заклинание": "https://harrypotter.fandom.com/ru/wiki/Заклинание",
    "Локомотор": "https://harrypotter.fandom.com/ru/wiki/Локомотор",
    "Манящие чары": "https://harrypotter.fandom.com/ru/wiki/Манящие_чары",
    "Заклятие исчезновения": "https://harrypotter.fandom.com/ru/wiki/Заклятие_исчезновения",
    "Репаро": "https://harrypotter.fandom.com/ru/wiki/Репаро",
    "Адское пламя": "https://harrypotter.fandom.com/ru/wiki/Адское_пламя",
    "Убивающее заклятие": "https://harrypotter.fandom.com/ru/wiki/Убивающее_заклятие",
    "Пророчество": "https://harrypotter.fandom.com/ru/wiki/Пророчество",
    "Визенгамот": "https://harrypotter.fandom.com/ru/wiki/Визенгамот",
    "Свадьба Флёр Делакур и Билла Уизли": "https://harrypotter.fandom.com/ru/wiki/Свадьба_Флёр_Делакур_и_Билла_Уизли",
    "Побеги из Азкабана": "https://harrypotter.fandom.com/ru/wiki/Побеги_из_Азкабана",
    "Распределение": "https://harrypotter.fandom.com/ru/wiki/Распределение",
    "Тотальное освобождение из Азкабана": "https://harrypotter.fandom.com/ru/wiki/Тотальное_освобождение_из_Азкабана",
    "Ноябрьское происшествие в Лондоне": "https://harrypotter.fandom.com/ru/wiki/Ноябрьское_происшествие_в_Лондоне",
    "Турнир Трёх Волшебников": "https://harrypotter.fandom.com/ru/wiki/Турнир_Трёх_Волшебников",
    "Заклинание призыва молнии Геллерта Грин-де-Вальда": "https://harrypotter.fandom.com/ru/wiki/Заклинание_призыва_молнии_Геллерта_Грин-де-Вальда",
    "Обряд возрождения": "https://harrypotter.fandom.com/ru/wiki/Обряд_возрождения",
    "Протего Диаболика": "https://harrypotter.fandom.com/ru/wiki/Протего_Диаболика",
    "Мортмордре": "https://harrypotter.fandom.com/ru/wiki/Мортмордре",
}

# Папка для сохранения файлов
os.makedirs("hp_docs", exist_ok=True)

for name, url in pages.items():
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Убираем скрипты и стили
    for script in soup(["script", "style"]):
        script.decompose()

    # Получаем основной текст статьи
    content_div = soup.find("div", {"class": "mw-parser-output"})
    if not content_div:
        print(f"Не найден контент для {name}")
        continue

    paragraphs = content_div.find_all("p")
    text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

    # Сохраняем в файл
    filename = f"./task2/hp_docs/{name.replace(' ', '_')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Сохранено: {filename}")
