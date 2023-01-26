# Библиотека для графовых нейронных сетей [graph-nn]

## Краткое описание

Эта библиотека содержит инструменты для работы с графовыми нейронными сетями, а также вспомогательные модули и алгоритмы, которые, в совокупности, позволяют создавать, обучать и использовать модели, слои и наборы данных, работающие с данными в графовом представлении.

Библиотека находится в стадии активного развития с конечной целью решения задач прогнозной аналитики в области анализа социальных сетей и построения карьерных траекторий для студентов и выпускников университетов, а также для компаний, заинтересованных в развитии своих сотрудников и подборе персонала.

Для этого уже на текущем этапе разработки, помимо базовых моделей графовых нейронных сетей, примеров и инструментов для создания унаследованных решений, в библиотеку включен анализатор связей социальной сети "ВКонтакте" и биржи труда HeadHunter, а также алгоритмы поиска кратчайшего пути во взвешенном графе с различными типами соединений и вершин.

Все это вместе взятое дает исследователям и разработчикам основу для создания собственных решений в области графовых нейронных сетей для решения сложных социальных и технических проблем.

## Композиция репозитория

### Наборы данных (datasets)

Библиотека содержит определения для работы с [datasets](./gns/dataset), созданными в соответствии с принципами наследования от базового класса.
Базовый класс набора данных задается в [соответствующей части модуля набора данных](./gns/dataset/dataset.py).
Библиотека также определяет частные реализации наборов данных (социальные сети) для разработки примеров и тестов (в частности, [Cora dataset](./gns/dataset/cora.py) для примеров с анализом цитирования сообщений в социальных сетях), а также пример датасета для индустриального применения в плане поиска вакансий [SfeduDataset](./gns/dataset/sfedu_dataset.py) и специальный датасет для загрузки данных из графовой базы данных [ArangoDataset](./gns/dataset/arango_dataset.py).

### Загрузчики (loaders)

Для загрузки наборов данных с сервера было решено реализовать специальный [SingleLoader](./gns/loaders/single_loader.py) для реализованных элементов графовой нейронной сети и несколько примеров. Кроме того, добавлен [BatchLoader](./gns/loaders/batch_loader.py) для пакетной загрузки данных (butch) и [DisjointLoader](./gns/loaders/disjoint_loader.py) для раздельной.

### Граф (graph)

Основная работа графовой нейронной сети определяется базовым классом [Graph](./gns/graph/graph.py), который является контейнером для данных.
Контейнер работает со следующими параметрами:
- `x`: для представления особенностей узлов,
- `a`: для представления матрицы смежности,
- `e`: для представления атрибутов ребер графа,
- `y`: для представления узлов или меток графа.

Дополнительно реализован алгоритм нахождения кратчайшего расстояния Беллмана-Форда, представленный соответствующим [классом](./gns/bellman_ford/bellman_ford_original.py ) и [измененным алгоритмом](./gns/bellman_ford/bellman_ford_modified.py ).

### Слои нейронной сети (layers)

Для основной работы библиотеки были созданы следующие слои нейронной сети:

- Сверточный слой [Чебышева](./gns/layer/cheb.py) для графовой нейронной сети.
- Основной (базовый) класс для [сверточного слоя](./gns/layer/convolution.py) Графическая нейронная сеть.
- [Сверточный](./gns/layer/gcn_convolution.py ) слой графовой нейронной сети.
- [Специальный сверточный слой](./gns/layer/gcs_convolution.py) с пропусками во время обучения.
- Основной (базовый) класс слоя для [GlobalPool](./gns/layer/global_pool.py).
- [Глобальная сумма](./gns/layer/global_sum_pool.py ) является реализацией базового класса GlobalPoolLayer.
- Основной слой с алгоритмом [GraphSAGE](./gns/layer/graphsage.py).

### Отправка сообщений (message passing)

Для реализации алгоритма продвижения информации по графовой нейронной сети был реализован алгоритм с помощью 
[Базового класса](./gns/message/generic_message_passing.py) для передачи сообщений в графовой нейронной сети (для алгоритма GraphSage).

### Модели (model)

[Основная модель](./gns/model/gcn.py) создана для сверточной нейронной сети, дополняя модель Tensorflow/Karas и специальная модель для индустриального примера [SfeduModel](./gns/model/sfedu_conv_model.py).

### Модели рассеяния (разреживания, scatter)

Для базовой функции передачи сообщений `Generic Message Passing`, а также для вспомогательной библиотеки функций были реализованы модели рассеяния:
- [scatter_max](./gns/scatter/scatter_max.py): Уменьшает количество сообщений.
- [scatter_mean](./gns/scatter/scatter_mean.py): Усредняет сообщения.
- [scatter_min](./gns/scatter/scatter_min.py): Уменьшает количество сообщений.
- [scatter_prod](./gns/scatter/scatter_prod.py): Умножает сообщения.
- [сумма рассеяния](./gns/scatter/scatter_sum.py): Суммирует сообщения.

### Предварительные преобразования (preprocess)

Определяется базовым классом преобразования [LayerPreprocess](./gns/transformation/layer_process.py). Он реализует функцию предварительной обработки в сверточном слое для матрицы смежности.

### Библиотека вспомогательных функций

Библиотека обладает достаточным количеством утилит и вспомогательных функций, необходимых для работы:
- [add_self_loops](./gns/utils/add_self_loops.py): Добавляет циклы к заданной матрице смежности.
- [batch_generator](./gns/utils/batch_generator.py): Перебирает данные с заданным количеством эпох, возвращает генератор python по одному пакету за раз.
- [chebyshev_filter](./gns/utils/chebyshev_filter.py): Реализация фильтра Чебышева для заданной матрицы смежности.
- [chebyshev_polynomial](./gns/utils/chebyshev_polynomial.py): Вычисляет полиномы Чебышева от X до порядка k.
- [check_dtypes](./gns/utils/check_dtypes.py): Проверка типа набора данных.
- [check_dtypes_decorator](./gns/utils/check_dtypes_decorator.py): Декоратор для автоматической проверки типа.
- [collate_labels_disjoint](./gns/utils/collate_labels_disjoint.py): Определяет соответствие списка меток для непересекающегося (disjoint) режима.
- [degree_power](./gns/utils/degree_power.py): Вычисляет отклонение.
- [deserialize_kwarg](./gns/utils/deserialize_kwarg.py): Десериализация аргументов.
- [deserialize_scatter](./gns/utils/deserialize_scatter.py): Десериализация рассеяния (`scatter`)
- [dot_production](./gns/utils/dot_production.py): Вычисляет умножение `a @ b` для `a` и `b` одного ранга (оба 2-го или оба 3-го ранга).
- [gcn_filter](./gns/utils/gcn_filter.py): Фильтры для графа.
- [get_spec](./gns/utils/get_spec.py): Возвращает спецификацию (описание или метаданные) для тензора типа `tensorflow.Tensor`.
- [idx_to_mask](./gns/utils/idx_to_mask.py): Возвращает маску по индексам.
- [load_binary_file](./gns/utils/load_binary_file.py): Загружает значение из файла, сериализованного модулем pickle.
- [mask_to_float_weights](./gns/utils/mask_to_float_weights.py): Преобразует битовую маску в простые веса для вычисления средних потерь по узлам сети.
- [mask_to_simple_weights](./gns/utils/mask_to_simple_weights.py): Преобразует битовую маску в простые веса для вычисления средних потерь по узлам сети.
- [dot_production_in_mixed_mode](./gns/utils/dot_production_in_mixed_mode.py): Вычисляет эквивалент функции `tf.einsum('ij, bjk->bik', a, b)`.
- [dot_production_modal](./gns/utils/dot_production_modal.py): Вычисляет матричное умножение для `a` и `b`.
- [normalized_adjacency_matrix](./gns/utils/normalized_adjacency_matrix.py): Нормализует заданную матрицу смежности.
- [normalized_laplacian](./gns/utils/normalized_laplacian.py): Вычисляет нормализованный лапласиан заданной матрицы смежности.
- [preprocess_features](./gns/utils/preprocess_features.py): Вычислительные возможности.
- [read_file](./gns/utils/read_file.py): Чтение файла с данными.
- [rescale_laplacian лапласиан](./gns/utils/rescale_laplacian.py): Масштабирует собственные значения Лапласа до `[-1,1]`.
- [reshape](./gns/utils/reshape.py): Изменяет форму в соответствии с формой, автоматически справляясь с разрежением.
- [serialize_kwarg](./gns/utils/serialize_kwarg.py): Сериализация атрибутов.
- [serialize_scatter](./gns/utils/serialize_scatter.py): Сериализация разброса.
- [shuffle_inplace](./gns/utils/shuffle_inplace.py): Перемешивание `np.random.shuffle`.
- [sparse_matrices_to_sparse_tensors](./gns/utils/sparse_matrices_to_sparse_tensors.py): Преобразование разреженных матриц Scipy в тензор.
- [sparse_matrix_to_sparse_tensor](./gns/utils/sparse_matrix_to_sparse_tensor.py): Преобразует разреженную матрицу Scipy в разреженный тензор.
- [convert_node_objects_to_disjoint](./gns/utils/convert_node_objects_to_disjoint.py): Преобразует списки узловых объектов, матриц смежности и граничных объектов в непересекающийся режим.
- [to_tensorflow_signature](./gns/utils/to_tensorflow_signature.py): Преобразует сигнатуру набора данных в сигнатуру тензорного потока.
- [transpose](./gns/utils/transpose.py): Переносит параметр `a`, автоматически справляясь с разреженностью с помощью перегруженных функций тензорного потока.

### Конфигурация, параметры и настройки

#### Конфигурация библиотеки устанавливает множество файлов в каталоге [config](./gns/config).

Основной состав (именованные параметры):
- методы агрегирования,
- свойства и атрибуты,
- константы приложения,
- типы данных,
- наборы данных,
- папки,
- именованные функции,
- инициализаторы,
- модели,
- имена,
- ссылки.

#### Как использовать

Может быть использован следующим образом.

Настройка envs

```sh
cp .env.dist .env
```

Создать виртуальную среду
```sh
virtualenv -p <path_to_python>
источник venv venv/bin/активировать
```

Установка пакетов

установка 
```sh 
pip install -r requirements.txt
```

Если вы измените некоторые пакеты, вы можете заморозить это с помощью команды
```sh
pip freeze > requirements.txt
```

### Дополнительные инструменты

### Парсер HH

Определяет генератор набора данных вакансий /ключевых слов из HH.ru .

Коллекция простых скриптов для обхода вакансий с сайта HH.ru через API для генерации CSV-файла по полям данных, 
таким как: имя, описание и ключевые навыки.

Это помогает сгенерировать CSV-файл в следующем формате:
``csv
"$name1 & $description1","ключевые навыки1"
"$name2 & $description2","ключевые навыки2"
"$name3 & $description3","ключевые навыки3"
...
```

Скрипты протестированы на python 3.10, но должны работать и на предыдущих версиях.

#### Получение страниц

Измените поле "текст" в `download.py` к твоему:

```py
text = 'НАЗВАНИЕ:Наука о данных'
```

Затем запустите скрипт

```sh
cd ./gns/crawlers/hh
python download.py
```

Этот скрипт загрузит результаты сохранения из API в папку `./gns/crawlers/docs/pagination` в формате JSON.

#### Получение подробной информации о вакансиях

На следующем шаге нам нужно загрузить расширенную информацию о вакансиях:

```sh
python parse.py
```

Скрипт вызовет API и сохранит ответы в папку "./gns/crawlers/docs/vacancies".

#### Сгенерировать CSV

```sh
python generate.py
```

Результат будет сохранен в папке `./gns/crawlers/docs/csv`.

### Поисковик API VK

#### Как использовать

```sh
cd ./gns/crawlers/vk
python main.py <vk_nickname_or_id>
```

### Makefile

Для автоматизации некоторых задач предоставляется Makefile. Доступные команды:
- install: установка пакетов.
- freeze: фиксация установленных пакетов.
- clear: очистка.
- serve: управление библиотекой:
  - автоматическое форматирование,
  - автоматическая сортировка импортов,
  - проверка типов.
- test: запуск тестов.

## Примеры

Примеры приведены в каталоге [examples](./examples):
- [Тестовый пример](./examples/example_citation_gcn.py ) для набора данных `Cora` (анализ графf цитирования сообщений в социальных сетях).
- [Тестовый пример](./examples/example_citation_cheb.py ) для набора данных `Cora` для сверточного слоя Чебышева (анализ граф цитирования сообщений социальной сети).
- [Простой тестовый пример](./примеры/example_simple_citation.py) для набора данных Cora (анализ графа цитирования сообщений в социальных сетях).
- Примеры нахождения кратчайшего расстояния на графе для алгоритма [Беллмана-Форда](./examples/example_bellman_ford_original.py) и [модифицированного алгоритма Беллмана-Форда](./examples/example_bellman_ford_modified.py).
- Индустриальный пример [подбора вакансий персонала для производства](./examples/example_industry_gsn_model_training.py). Тренировка модели на базе данных полученных при помощи класса SfeduDataset, модель представляет из себя GraphSAGE решение, в её задачи входит поиск похожего графа описывающего вакансию на граф описывающий пользователя.
- Индустриальный пример [генерации большого графа](./gns/examples/example_industry_gcn_dataset_create.py). Данный пример демонстрирует способ генерации большого графа совместимого с библиотекой GNS при помощи класса TechDataset, в результате работы скрипта создаётся два файла содержащие 249 (полный - jd_data) и 177 (сжатый - jd_data2) вершин (в зависимости от набора исходных данных). Данные берутся из источника https://github.com/ZhongTr0n/JD_Analysis
- Индустриальный пример [тренировки модели для класса TechDataset](./gns/examples/example_industry_gcn_model_training_simple.py). Это пример тренировки модели с использованием данных полученных при помощи класса TechDataset. Модель представляет из себя стандартную Graph Convolutional Network, в её задачи входит поиск  возможных связей между узлами описывающими граф пользователя: карьерная траектория.
- Индустриальный пример [ренировки модели Graph Convolutional Network](./gns/examples/example_industry_gcn_model_training.py). Пример тренировки модели Graph Convolutional Network с усложнённой логикой проверки и с применением продвинутых методов использования штатного инструментария библиотеки GNS. Пример также демонстрирует процедуру загрузки датасета, его применение на уровне модели, далее происходит тренировка после чего отображается точность полученной модели. Основная задача модели это поиск возможных связей между узлами описывающими граф пользователя: карьерная траектория.
- Индустриальный пример [генерации серии графов](./gns/examples/example_industry_gsn_dataset_create.py). Данный пример демонстрирует способ генерации серии небольших графов совместимых с библиотекой GNS при помощи класса SfeduDataset, из неё выбираются графы описывающие пользователей и вакансии, после чего происходит преобразование данных в необходимый формат.

Для визуализации примеров данные загружаются в ArangoDB, для этого был использован [docker-compose файл](./docker-compose.yml) с [монтированием раздела](./arangodb_data).

Визуализация полного графа технологий.

![полный граф технологий в ArangoDB](./docs/images/map.png)

Визуализация графа компетенций PHP-разработчика.

![граф компетенций PHP-разработчика](./docs/images/user1-php-backend.png)

Визуализация графа компетенций devops-разработчика.

![граф компетенций devops-разработчика](./docs/images/user2-dba-devops.png)

Визуализация графа компетенций JS-разработчика.

![граф компетенций JS-разработчика](./docs/images/user3-js-fullstack.png)

### Разработка поддерживается исследовательским центром «Сильный искусственный интеллект в промышленности» Университета ИТМО.

![](https://gitlab.actcognitive.org/itmo-sai-code/organ/-/raw/main/docs/AIM-Strong_Sign_Norm-01_Colors.svg)