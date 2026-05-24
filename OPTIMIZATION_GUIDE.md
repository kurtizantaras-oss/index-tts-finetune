# Оптимизация производительности IndexTTS2 для Tesla V100

## Быстрый старт

Для максимальной скорости инференса на Tesla V100 используйте:

```bash
python infer_vi_optimized.py --text "Xin chào, đây là bản thử nghiệm." --output output.wav
```

По умолчанию включены все оптимизации:
- ✅ FP16 (полуточная точность)
- ✅ torch.compile (компиляция графа)
- ✅ CUDA ядра BigVGAN

## Ключевые оптимизации для V100

### 1. **FP16 (Half Precision)** - +30-50% скорости
Tesla V100 имеет специализированные тензорные ядра для FP16:
```bash
python infer_vi_optimized.py --text "..." --fp16
```

### 2. **torch.compile** - +20-40% скорости
Компилирует граф вычислений модели:
```bash
python infer_vi_optimized.py --text "..." --use-torch-compile
```

### 3. **CUDA Kernels для BigVGAN** - +15-25% скорости
Специализированные ядра для вокодера:
```bash
python infer_vi_optimized.py --text "..." --use-cuda-kernel
```

## Параметры для балансировки скорость/качество

### Уменьшить размер сегментов (быстрее, меньше памяти):
```bash
python infer_vi_optimized.py --text "..." --max-text-tokens 80
```

### Уменьшить beam search (быстрее):
```bash
python infer_vi_optimized.py --text "..." --num-beams 1
```

### Полный набор для максимальной скорости:
```bash
python infer_vi_optimized.py \
  --text "Xin chào thế giới" \
  --output fast_output.wav \
  --fp16 \
  --use-torch-compile \
  --use-cuda-kernel \
  --max-text-tokens 80 \
  --num-beams 1 \
  --temperature 0.8
```

## Сравнение производительности

| Конфигурация | Относительная скорость | Использование памяти |
|--------------|------------------------|----------------------|
| Базовая (FP32) | 1.0x | 100% |
| + FP16 | ~1.4x | ~70% |
| + FP16 + torch.compile | ~1.7x | ~65% |
| + Все оптимизации | ~2.0x+ | ~60% |

## Дополнительные флаги

```bash
# DeepSpeed (может помочь с памятью)
--use-deepspeed

# Accelerate для GPT2
--use-accel

# Подробные логи
--verbose

# Изменить устройство
--device cuda:0
```

## Примеры использования

### Синтез из файла:
```bash
python infer_vi_optimized.py --text-file prompt.txt --output result.wav
```

### С эмоциями:
```bash
python infer_vi_optimized.py \
  --text "Tôi rất vui!" \
  --emo-audio emotion_ref.wav \
  --emo-alpha 0.8
```

### Пакетная обработка:
```bash
for text in "Câu một" "Câu hai" "Câu ba"; do
  python infer_vi_optimized.py --text "$text" --output "out_$i.wav"
done
```

## Решение проблем

### Ошибка при загрузке CUDA ядер BigVGAN:
```bash
# Убедитесь, что CUDA установлена правильно
nvcc --version

# Или отключите CUDA ядра
python infer_vi_optimized.py --text "..." --use-cuda-kernel=False
```

### Недостаточно памяти GPU:
```bash
# Уменьшите размер сегментов
python infer_vi_optimized.py --text "..." --max-text-tokens 60

# Отключите torch.compile (требует больше памяти при компиляции)
python infer_vi_optimized.py --text "..." --use-torch-compile=False
```

### Первый запуск медленный:
Первый запуск с `torch.compile` включает компиляцию графа (1-2 минуты).
Последующие запуски будут быстрее благодаря кэшированию.

## Аппаратные требования

- **Минимум**: 16GB VRAM (Tesla V100 16GB)
- **Рекомендуется**: 32GB VRAM для длинных текстов
- **CUDA**: 11.0+
