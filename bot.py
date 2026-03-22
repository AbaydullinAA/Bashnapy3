import logging
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ConversationHandler
)
from recommendations import (
    get_collab_recommendations, plot_user_type_similarity,
    user_types_list, all_watches, user_item_matrix, centered_matrix
)

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Состояния
MODE, COLLAB_RATING, CONTENT_INPUT = range(3)

# Загружаем данные о часах (с фото)
watches_df = pd.read_csv('watches_processed.csv')
# Предполагаем, что есть столбец 'image_url' или 'image_path'
# Если нет, создаём заглушку
if 'image_url' not in watches_df.columns:
    watches_df['image_url'] = 'https://via.placeholder.com/300x200?text=No+Image'

# Создаём словарь для быстрого поиска: название -> данные о часах
watches_dict = {row['watch_name']: row.to_dict() for _, row in watches_df.iterrows()}
# Список всех названий для поиска
all_titles = list(watches_dict.keys())

# Функция нечёткого поиска
def fuzzy_search(query, max_distance=6):
    query = query.lower().strip()
    matches = []
    for title in all_titles:
        dist = levenshtein(query, title)
        if dist <= max_distance:
            matches.append((title, dist))
    matches.sort(key=lambda x: x[1])  # сортируем по расстоянию
    return [m[0] for m in matches]

# Функция для отправки фото часов
async def send_watch_photo(update, context, watch_name):
    watch = watches_dict.get(watch_name)
    if not watch:
        await update.message.reply_text("Информация о часах не найдена.")
        return False
    photo_url = watch.get('image_url')
    if photo_url:
        await update.message.reply_photo(photo=photo_url, caption=f"<b>{watch_name}</b>\nБренд: {watch['brand_name']}\nРейтинг: {watch['rating']}", parse_mode='HTML')
    else:
        await update.message.reply_text(f"<b>{watch_name}</b>\nБренд: {watch['brand_name']}\nРейтинг: {watch['rating']}", parse_mode='HTML')
    return True

# Старт
async def start(update: Update, context):
    keyboard = [
        [InlineKeyboardButton("Коллаборативная рекомендация (оценить часы)", callback_data='collab')],
        [InlineKeyboardButton("Контентная рекомендация (похожие часы)", callback_data='content')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Привет! Я помогу подобрать часы.\nВыберите режим:",
        reply_markup=reply_markup
    )
    return MODE

# Обработка выбора режима
async def mode_choice(update: Update, context):
    query = update.callback_query
    await query.answer()
    choice = query.data
    if choice == 'collab':
        context.user_data['ratings'] = {}
        await query.edit_message_text(
            "Режим коллаборативной рекомендации.\n"
            "Оцените несколько часов от 1 до 5.\n"
            "Введите название часов (можно с опечатками).\n"
            "Для завершения введите /done"
        )
        return COLLAB_RATING
    elif choice == 'content':
        await query.edit_message_text(
            "Режим контентной рекомендации.\n"
            "Введите название часов, и я найду похожие:"
        )
        return CONTENT_INPUT
    else:
        await query.edit_message_text("Неизвестный выбор.")
        return MODE

# Коллаборативный режим: получение названия
async def collab_get_title(update: Update, context):
    user_input = update.message.text.strip()
    if user_input.lower() == '/done':
        # Завершаем сбор оценок
        user_ratings = context.user_data.get('ratings', {})
        if not user_ratings:
            await update.message.reply_text("Вы не оценили ни одних часов. Попробуйте снова /start")
            return ConversationHandler.END
        # Здесь вызываем коллаборативные рекомендации
        # Сохраняем временно оценки для дальнейшей обработки
        context.user_data['final_ratings'] = user_ratings
        await update.message.reply_text("Спасибо! Ищу похожих пользователей...")
        # Генерируем рекомендации
        try:
            recommendations, sim_scores = get_collab_recommendations(user_ratings, top_k=10, top_n=5)
            # Сохраняем sim_scores для графика
            context.user_data['sim_scores'] = sim_scores
            if not recommendations:
                await update.message.reply_text("Не удалось найти рекомендации. Попробуйте другие часы.")
            else:
                msg = "Рекомендуемые часы (коллаборативная фильтрация):\n"
                for i, (title, score) in enumerate(recommendations, 1):
                    msg += f"{i}. {title} (прогноз: {score:.2f})\n"
                await update.message.reply_text(msg)
            # Строим график сходства с типами (используем сохранённый sim_scores)
            await build_and_send_similarity_plot(update, context)
        except Exception as e:
            logger.error(f"Ошибка при получении рекомендаций: {e}")
            await update.message.reply_text("Произошла ошибка при расчёте рекомендаций.")
        return ConversationHandler.END

    # Ищем часы (сначала точное совпадение подстроки, потом нечёткий поиск)
    matching = [title for title in all_titles if user_input.lower() in title.lower()]
    if not matching:
        # Нечёткий поиск
        matching = fuzzy_search(user_input, max_distance=6)
    if not matching:
        await update.message.reply_text("Часы не найдены. Попробуйте другое название.")
        return COLLAB_RATING
    if len(matching) > 1:
        # Предлагаем выбрать из списка
        keyboard = []
        for i, title in enumerate(matching[:5]):  # ограничим 5 вариантами
            keyboard.append([InlineKeyboardButton(title, callback_data=f"watch_{i}")])
        # Добавляем кнопку отмены
        keyboard.append([InlineKeyboardButton("Отмена", callback_data="cancel")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        context.user_data['pending_matches'] = matching
        await update.message.reply_text(
            "Найдено несколько вариантов. Выберите нужные часы:",
            reply_markup=reply_markup
        )
        return COLLAB_RATING
    else:
        # Один вариант
        title = matching[0]
        context.user_data['pending_title'] = title
        await send_watch_photo(update, context, title)
        # Спрашиваем, те ли это часы
        keyboard = [
            [InlineKeyboardButton("Да", callback_data="yes")],
            [InlineKeyboardButton("Нет", callback_data="no")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Это те часы?", reply_markup=reply_markup)
        return COLLAB_RATING

# Обработка выбора из списка
async def collab_choose_watch(update: Update, context):
    query = update.callback_query
    await query.answer()
    data = query.data
    if data == "cancel":
        await query.edit_message_text("Ввод отменён. Введите другое название или /done")
        return COLLAB_RATING
    if data.startswith("watch_"):
        idx = int(data.split("_")[1])
        matching = context.user_data.get('pending_matches', [])
        if 0 <= idx < len(matching):
            title = matching[idx]
            context.user_data['pending_title'] = title
            # Отправляем фото
            await query.edit_message_text("Вы выбрали:")
            await send_watch_photo(query.message, context, title)
            # Спрашиваем подтверждение
            keyboard = [
                [InlineKeyboardButton("Да", callback_data="yes")],
                [InlineKeyboardButton("Нет", callback_data="no")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.message.reply_text("Это те часы?", reply_markup=reply_markup)
            return COLLAB_RATING
    return COLLAB_RATING

# Подтверждение выбранных часов
async def confirm_watch(update: Update, context):
    query = update.callback_query
    await query.answer()
    if query.data == "yes":
        title = context.user_data.get('pending_title')
        if title:
            context.user_data['pending_title'] = None
            context.user_data['waiting_rating'] = title
            await query.edit_message_text(f"Введите оценку для {title} (от 1 до 5):")
            return COLLAB_RATING
    else:
        await query.edit_message_text("Введите другое название или /done")
        return COLLAB_RATING

# Получение оценки
async def collab_get_rating(update: Update, context):
    try:
        rating = float(update.message.text.strip())
        if rating < 1 or rating > 5:
            await update.message.reply_text("Оценка должна быть от 1 до 5.")
            return COLLAB_RATING
        title = context.user_data.get('waiting_rating')
        if title:
            user_ratings = context.user_data.get('ratings', {})
            user_ratings[title] = rating
            context.user_data['ratings'] = user_ratings
            context.user_data['waiting_rating'] = None
            await update.message.reply_text(f"Добавлено: {title} — {rating}")
        else:
            await update.message.reply_text("Произошла ошибка. Попробуйте снова.")
        # Снова запрашиваем следующее название
        await update.message.reply_text("Введите следующее название (или /done для завершения):")
        return COLLAB_RATING
    except ValueError:
        await update.message.reply_text("Введите число.")
        return COLLAB_RATING

# Построение графика сходства и отправка
async def build_and_send_similarity_plot(update, context):
    sim_scores = context.user_data.get('sim_scores')
    if sim_scores is None:
        return
    # plot_user_type_similarity создаёт файл 'user_similarity_to_types.png'
    # Ваша функция должна принимать sim_scores и user_types_list
    # Если она не сохраняет файл, допишите
    plot_user_type_similarity(sim_scores, user_types_list)  # предполагается, что сохраняет картинку
    with open('user_similarity_to_types.png', 'rb') as f:
        await update.message.reply_photo(photo=f, caption="Ваше сходство с типами пользователей")

# Контентный режим: получение названия
async def content_input(update: Update, context):
    user_input = update.message.text.strip()
    # Нечёткий поиск
    matching = fuzzy_search(user_input, max_distance=6)
    if not matching:
        await update.message.reply_text("Часы не найдены. Попробуйте другое название.")
        return CONTENT_INPUT
    # Для простоты берём первый вариант
    title = matching[0]
    await send_watch_photo(update, context, title)
    # Получаем похожие часы (вызываем вашу функцию get_content_recommendations)
    from recommendations import get_content_recommendations
    target, recs = get_content_recommendations(title, top_n=5)
    if target is None:
        await update.message.reply_text(recs)
    else:
        msg = f"Похожие часы на {target}:\n"
        for i, (rec_title, sim) in enumerate(recs, 1):
            msg += f"{i}. {rec_title} (сходство: {sim:.3f})\n"
        await update.message.reply_text(msg)
    return ConversationHandler.END

# Отмена
async def cancel(update: Update, context):
    await update.message.reply_text("Действие отменено. Введите /start для начала.")
    return ConversationHandler.END

# Обработка ошибок (если пользователь вводит текст не в том состоянии)
async def unknown(update: Update, context):
    await update.message.reply_text("Я вас не понял. Используйте /start для начала или /cancel для отмены.")

def main():
    TOKEN = "..."
    application = Application.builder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            MODE: [CallbackQueryHandler(mode_choice, pattern='^(collab|content)$')],
            COLLAB_RATING: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, collab_get_title),
                CallbackQueryHandler(collab_choose_watch, pattern='^(watch_|cancel)'),
                CallbackQueryHandler(confirm_watch, pattern='^(yes|no)$'),
                MessageHandler(filters.Regex(r'^\d+(\.\d+)?$'), collab_get_rating)  # числа с точкой
            ],
            CONTENT_INPUT: [MessageHandler(filters.TEXT & ~filters.COMMAND, content_input)]
        },
        fallbacks=[CommandHandler('cancel', cancel), MessageHandler(filters.ALL, unknown)],
        per_user=True,
        per_chat=True
    )
    application.add_handler(conv_handler)

    application.run_polling()

if __name__ == '__main__':
    main()


