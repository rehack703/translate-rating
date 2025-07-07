-----

# 🌐 Advanced AI Translation Quality Evaluation System 🌐

This repository contains a Python-based system designed to evaluate the quality of machine translations. It employs a multi-faceted approach, assessing translations based on **BLEU score**, **completeness (word count)**, **naturalness (grammar and fluency)**, **semantic similarity**, and the detection of **common translation errors**. The system then consolidates these metrics into an **overall quality score and grade**, offering detailed insights and suggestions for improvement.

## ✨ Features

  * **Multi-Lingual Tokenization**: Supports Korean (KoNLP: Okt, Komoran, Kkma), Japanese (MeCab), and English (NLTK) for accurate text segmentation.
  * **Enhanced BLEU Score Calculation**: Utilizes `nltk.translate.bleu_score` with smoothing and `sacrebleu` for robust evaluation, incorporating multiple reference translations (Google Translate, MarianMT).
  * **Completeness Check**: Compares the word count of original and translated texts, considering language-specific token ratios, to identify potential omissions or excessive additions.
  * **Naturalness Assessment**:
      * **Grammar Check**: Simple rule-based grammar checks for English, Korean, and Japanese to detect common grammatical errors (e.g., subject-verb agreement, particle misuse).
      * **Fluency Analysis**: Evaluates sentence length and word repetition to assess overall readability and naturalness.
  * **Semantic Similarity**: Employs a pre-trained `SentenceTransformer` model (`distiluse-base-multilingual-cased-v1`) to calculate the cosine similarity between the embeddings of the original and translated sentences, ensuring meaning preservation.
  * **Error Detection**: Identifies specific translation errors such as numerical mismatches, missing brackets, and unpreserved email addresses or URLs.
  * **Comprehensive Scoring & Grading**: Combines all evaluation metrics into a weighted overall score (out of 100) and assigns a letter grade (A+ to F), along with actionable suggestions for improvement.

-----

## 🚀 Getting Started

### Prerequisites

This system requires several Python libraries and some system-level packages for Korean and Japanese morphological analysis.

If you are running this in a **Google Colab environment**, you can run the following commands to install everything:

```bash
!apt-get update
!apt-get install -y mecab libmecab-dev mecab-ipadic-utf8
!pip install mecab-python3
!pip install konlpy
!pip install sacrebleu
!pip install googletrans==3.1.0a0
!pip install torch transformers
!pip install sentence-transformers
!pip install nltk
!ln -s /etc/mecabrc /usr/local/etc/mecabrc
!pip install khaiii
```

For **local environments**, you'll need to install `MeCab` (for Japanese) and its dictionaries manually first. Then, install the Python bindings and other libraries:

```bash
# For MeCab on Ubuntu/Debian, you might use:
# sudo apt-get update
# sudo apt-get install -y mecab libmecab-dev mecab-ipadic-utf8

pip install mecab-python3
pip install konlpy
pip install sacrebleu
pip install googletrans==3.1.0a0
pip install torch transformers
pip install sentence-transformers
pip install nltk
```

Additionally, for NLTK's functionalities, you need to download the `punkt` tokenizer data:

```python
import nltk
nltk.download('punkt')
```

-----

### Usage

1.  **Save the code**: Save the provided Python script (excluding the installation commands) as `translation_evaluator.py`.

2.  **Run the script**: Execute the script from your terminal:

    ```bash
    python translation_evaluator.py
    ```

3.  **Follow the prompts**: The program will ask you to input the source language, target language, the original text, and the translated text.

    ```
    🌐 고급 AI 번역 품질 평가 시스템 🌐
    --------------------------------------------------
    원문 언어를 입력하세요 (ja: 일본어, ko: 한국어, en: 영어): en
    번역 언어를 입력하세요 (ja: 일본어, ko: 한국어, en: 영어): ko

    원문을 입력하세요:
    Hello, this is a test translation to assess the quality of the translation system. There are 123 important points. (Note: This is a sample text.)

    번역문을 입력하세요:
    안녕하세요, 이것은 번역 시스템의 품질을 평가하기 위한 테스트 번역입니다. 123개의 중요한 사항이 있습니다. (참고: 이것은 샘플 텍스트입니다.)

    평가를 시작합니다...
    ```

4.  **View Results**: The system will then display a detailed evaluation report, including scores, detected errors, and suggestions for improvement.

-----
