# Базовые политики

Конфигурация — medium-1.0. Версия генератора — 1.0.1.

Каждая строка усредняет эпизоды с одинаковым весом. Разброс награды — стандартное отклонение по всей указанной выборке. Шаги включают submit. Время измерено от загрузки открытых таблиц до терминальной оценки, включая запись траектории и обмен с оценщиком. Генерация и запуск процессов исключены.

| Выборка | Политика | N | P | R | F1 | Награда ± σ | Шаги | Время, с |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | flag_all | 10 | 0.000 | 0.000 | 0.000 | -1.000 ± 0.000 | 3.0 | 0.007 |
| train | oracle | 10 | 0.600 | 0.600 | 1.000 | 0.990 ± 0.012 | 6.1 | 0.007 |
| train | random_explorer | 10 | 0.000 | 0.000 | 0.100 | -0.812 ± 0.564 | 61.0 | 0.011 |
| train | reconciler | 10 | 0.104 | 0.286 | 0.118 | -0.777 ± 0.497 | 16.7 | 0.009 |
| dev | flag_all | 10 | 0.000 | 0.000 | 0.000 | -1.000 ± 0.000 | 3.0 | 0.007 |
| dev | oracle | 10 | 0.400 | 0.400 | 1.000 | 0.991 ± 0.016 | 5.7 | 0.006 |
| dev | random_explorer | 10 | 0.000 | 0.000 | 0.000 | -1.000 ± 0.000 | 61.0 | 0.010 |
| dev | reconciler | 10 | 0.061 | 0.250 | 0.086 | -0.840 ± 0.250 | 15.6 | 0.009 |
| fresh | flag_all | 20 | 0.000 | 0.000 | 0.000 | -1.000 ± 0.000 | 3.0 | 0.005 |
| fresh | oracle | 20 | 0.650 | 0.650 | 1.000 | 0.987 ± 0.013 | 7.3 | 0.004 |
| fresh | random_explorer | 20 | 0.000 | 0.000 | 0.100 | -0.812 ± 0.564 | 61.0 | 0.010 |
| fresh | reconciler | 20 | 0.113 | 0.321 | 0.137 | -0.741 ± 0.376 | 16.5 | 0.007 |

P и R пустого правильного ответа равны 0, F1 равен 1 по §9.4. Поэтому средние P и R даже у oracle могут быть меньше единицы.

## Порядок наград

- train — oracle 0.990, reconciler -0.777, random_explorer -0.812, flag_all -1.000.
- dev — oracle 0.991, reconciler -0.840, flag_all -1.000, random_explorer -1.000.
- fresh — oracle 0.987, reconciler -0.741, random_explorer -0.812, flag_all -1.000.

У random_explorer пустая сдача на чистой книге встретилась 3/40 раз. После 60 инспекций такой эпизод приносит 0.88. Эта ветвь §9.4 может поднять random выше flag_all, даже если ни одна случайная находка не засчитана.

Невалидных пакетов находок — 0/160. Результат каждого эпизода повторно рассчитан по терминальному действию.

## По типам ошибок

Счётчики объединены внутри выборки. P, R и F1 рассчитаны из этих счётчиков. При отсутствии примеров типа в обеих колонках метрика помечена прочерком.

| Выборка | Политика | Тип | TP | FP | FN | P | R | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| train | flag_all | duplicate_invoice | 0 | 80 | 4 | 0.000 | 0.000 | 0.000 |
| train | flag_all | payment_misallocation | 0 | 80 | 8 | 0.000 | 0.000 | 0.000 |
| train | flag_all | fx_rate_error | 0 | 70 | 4 | 0.000 | 0.000 | 0.000 |
| train | flag_all | cutoff_error | 0 | 70 | 3 | 0.000 | 0.000 | 0.000 |
| train | oracle | duplicate_invoice | 4 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| train | oracle | payment_misallocation | 8 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| train | oracle | fx_rate_error | 4 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| train | oracle | cutoff_error | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| train | random_explorer | duplicate_invoice | 0 | 3 | 4 | 0.000 | 0.000 | 0.000 |
| train | random_explorer | payment_misallocation | 0 | 1 | 8 | 0.000 | 0.000 | 0.000 |
| train | random_explorer | fx_rate_error | 0 | 5 | 4 | 0.000 | 0.000 | 0.000 |
| train | random_explorer | cutoff_error | 0 | 4 | 3 | 0.000 | 0.000 | 0.000 |
| train | reconciler | duplicate_invoice | 0 | 0 | 4 | 0.000 | 0.000 | 0.000 |
| train | reconciler | payment_misallocation | 7 | 18 | 1 | 0.280 | 0.875 | 0.424 |
| train | reconciler | fx_rate_error | 2 | 92 | 2 | 0.021 | 0.500 | 0.041 |
| train | reconciler | cutoff_error | 1 | 27 | 2 | 0.036 | 0.333 | 0.065 |
| dev | flag_all | duplicate_invoice | 0 | 80 | 4 | 0.000 | 0.000 | 0.000 |
| dev | flag_all | payment_misallocation | 0 | 80 | 6 | 0.000 | 0.000 | 0.000 |
| dev | flag_all | fx_rate_error | 0 | 70 | 3 | 0.000 | 0.000 | 0.000 |
| dev | flag_all | cutoff_error | 0 | 70 | 6 | 0.000 | 0.000 | 0.000 |
| dev | oracle | duplicate_invoice | 4 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| dev | oracle | payment_misallocation | 6 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| dev | oracle | fx_rate_error | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| dev | oracle | cutoff_error | 6 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| dev | random_explorer | duplicate_invoice | 0 | 0 | 4 | 0.000 | 0.000 | 0.000 |
| dev | random_explorer | payment_misallocation | 0 | 2 | 6 | 0.000 | 0.000 | 0.000 |
| dev | random_explorer | fx_rate_error | 0 | 3 | 3 | 0.000 | 0.000 | 0.000 |
| dev | random_explorer | cutoff_error | 0 | 6 | 6 | 0.000 | 0.000 | 0.000 |
| dev | reconciler | duplicate_invoice | 0 | 0 | 4 | 0.000 | 0.000 | 0.000 |
| dev | reconciler | payment_misallocation | 6 | 15 | 0 | 0.286 | 1.000 | 0.444 |
| dev | reconciler | fx_rate_error | 0 | 77 | 3 | 0.000 | 0.000 | 0.000 |
| dev | reconciler | cutoff_error | 4 | 24 | 2 | 0.143 | 0.667 | 0.235 |
| fresh | flag_all | duplicate_invoice | 0 | 160 | 11 | 0.000 | 0.000 | 0.000 |
| fresh | flag_all | payment_misallocation | 0 | 160 | 13 | 0.000 | 0.000 | 0.000 |
| fresh | flag_all | fx_rate_error | 0 | 140 | 15 | 0.000 | 0.000 | 0.000 |
| fresh | flag_all | cutoff_error | 0 | 140 | 11 | 0.000 | 0.000 | 0.000 |
| fresh | oracle | duplicate_invoice | 11 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| fresh | oracle | payment_misallocation | 13 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| fresh | oracle | fx_rate_error | 15 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| fresh | oracle | cutoff_error | 11 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| fresh | random_explorer | duplicate_invoice | 0 | 6 | 11 | 0.000 | 0.000 | 0.000 |
| fresh | random_explorer | payment_misallocation | 0 | 7 | 13 | 0.000 | 0.000 | 0.000 |
| fresh | random_explorer | fx_rate_error | 0 | 8 | 15 | 0.000 | 0.000 | 0.000 |
| fresh | random_explorer | cutoff_error | 0 | 4 | 11 | 0.000 | 0.000 | 0.000 |
| fresh | reconciler | duplicate_invoice | 0 | 0 | 11 | 0.000 | 0.000 | 0.000 |
| fresh | reconciler | payment_misallocation | 13 | 40 | 0 | 0.245 | 1.000 | 0.394 |
| fresh | reconciler | fx_rate_error | 4 | 167 | 11 | 0.023 | 0.267 | 0.043 |
| fresh | reconciler | cutoff_error | 7 | 56 | 4 | 0.111 | 0.636 | 0.189 |
