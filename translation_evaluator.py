import MeCab
from konlpy.tag import Okt, Kkma, Komoran
import sacrebleu
from googletrans import Translator
import torch
import torch.nn.functional as F
from transformers import MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import re

# NLTK 필요 데이터 다운로드 (Ensure this is run once, e.g., at the start of your script or in a setup phase)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# 언어별 토크나이저 설정 개선
def tokenize_text(original, translated, src_lang, dest_lang):
    # 소스 언어 토큰화
    if src_lang == "ja":  # 일본어
        mecab = MeCab.Tagger("-Owakati")
        src_tokens = mecab.parse(original).strip().split()
    elif src_lang == "ko":  # 한국어 - 여러 토크나이저 결합
        okt = Okt()
        # Komoran and Kkma instances are memory-intensive if created repeatedly.
        # It's better to instantiate them once if possible, or handle their lifecycle.
        # For this script, we'll keep them here for simplicity per function call.
        komoran = Komoran()
        kkma = Kkma()

        # 각 토크나이저의 결과를 조합하여 더 정확한 토큰화
        okt_tokens = okt.morphs(original)
        komoran_tokens = komoran.morphs(original)
        kkma_tokens = kkma.morphs(original)

        # 가장 세분화된 토큰화 결과 선택 (일반적으로 Kkma가 가장 세분화)
        src_tokens = kkma_tokens
    else:  # 영어 및 기타 언어
        src_tokens = nltk.word_tokenize(original)

    # 대상 언어 토큰화
    if dest_lang == "ko":  # 한국어
        kkma = Kkma()  # 더 세분화된 한국어 토크나이저 사용
        dest_tokens = kkma.morphs(translated)
    elif dest_lang == "ja":  # 일본어
        mecab = MeCab.Tagger("-Owakati")
        dest_tokens = mecab.parse(translated).strip().split()
    else:  # 영어 및 기타 언어
        dest_tokens = nltk.word_tokenize(translated)

    return src_tokens, dest_tokens

# 향상된 BLEU 점수 계산
def calculate_bleu(original, translated, src_lang, dest_lang):
    translator = Translator()

    # 여러 번역 모델을 통해 다양한 참조 번역 생성
    references = []

    # Google 번역
    try:
        google_ref = translator.translate(original, src=src_lang, dest=dest_lang).text
        references.append(google_ref)
    except Exception as e:
        print(f"Google Translate error: {e}")
        pass

    # Marian MT 모델 사용 - 다양한 참조 번역을 위해
    try:
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{dest_lang}"
        marian_tokenizer = MarianTokenizer.from_pretrained(model_name)
        marian_model = MarianMTModel.from_pretrained(model_name)

        inputs = marian_tokenizer(original, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = marian_model.generate(**inputs)
        marian_ref = marian_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        references.append(marian_ref)
    except Exception as e:
        # 모델이 없는 언어 조합일 경우 건너뜀
        print(f"MarianMT error for {model_name}: {e}")
        pass

    # 참조 번역이 하나도 없을 경우 (최소한 Google Translate 결과라도 사용)
    if not references:
        # Fallback to Google Translate if MarianMT fails and no initial reference was added
        try:
            references = [translator.translate(original, src=src_lang, dest=dest_lang).text]
        except Exception as e:
            print(f"Failed to get any reference translations: {e}")
            return 0.0 # Return 0 if no reference can be generated

    # BLEU 계산 방법 개선 - SmoothingFunction 적용
    # 토큰화 함수 정의 (dest_lang에 맞춰 토큰화)
    def get_tokens_for_bleu(text, lang):
        if lang == "ko":
            return Kkma().morphs(text)
        elif lang == "ja":
            return MeCab.Tagger("-Owakati").parse(text).strip().split()
        else:
            return nltk.word_tokenize(text)

    candidate = get_tokens_for_bleu(translated, dest_lang)
    references_tokenized = [get_tokens_for_bleu(ref, dest_lang) for ref in references]

    # NLTK BLEU (문장 단위로 더 세밀한 평가)
    weights = (0.25, 0.25, 0.25, 0.25)  # 1-gram, 2-gram, 3-gram, 4-gram
    smoothing = SmoothingFunction().method4  # smoothing 적용

    nltk_bleu = sentence_bleu(references_tokenized, candidate, weights=weights, smoothing_function=smoothing) * 100

    # sacrebleu (말뭉치 단위 평가)
    # sacrebleu expects references as a list of lists of strings, where each inner list is a reference for a sentence.
    # Since we're doing sentence-level evaluation, the structure is a bit different.
    # We provide `[translated]` as the hypotheses and `[[ref1], [ref2], ...]` as references for sacrebleu.
    sacre_bleu = sacrebleu.corpus_bleu([translated], references).score


    # 두 BLEU 점수의 가중 평균 (문장 단위 BLEU에 더 높은 가중치)
    final_bleu = (nltk_bleu * 0.7) + (sacre_bleu * 0.3)

    return final_bleu

# 향상된 단어 수 체크
def check_missing(original_tokens, translated_tokens, src_lang, dest_lang):
    # 언어별 토큰 비율 계산 (언어마다 토큰화 시 비율이 다름)
    lang_ratio = {
        "ko-en": 1.3,  # 한국어가 영어보다 토큰 수가 적을 수 있음
        "en-ko": 0.7,  # 영어가 한국어보다 토큰 수가 적을 수 있음
        "ja-en": 1.2,  # 일본어가 영어보다 토큰 수가 적을 수 있음
        "en-ja": 0.8,  # 영어가 일본어보다 토큰 수가 적을 수 있음
        "ko-ja": 1.1,  # 한국어가 일본어보다 토큰 수가 적을 수 있음
        "ja-ko": 0.9,  # 일본어가 한국어보다 토큰 수가 적을 수 있음
    }

    lang_pair = f"{src_lang}-{dest_lang}"
    ratio = lang_ratio.get(lang_pair, 1.0)  # 기본값은 1.0

    expected_tokens = len(original_tokens) * ratio
    actual_tokens = len(translated_tokens)

    # 토큰 수 차이 계산
    difference = abs(expected_tokens - actual_tokens)
    difference_ratio = difference / expected_tokens if expected_tokens > 0 else 0

    if difference_ratio > 0.3:  # 예상되는 토큰 수와 30% 이상 차이나면 경고
        if expected_tokens > actual_tokens:
            return f"번역에서 단어가 누락되었을 가능성이 높습니다. (예상: {expected_tokens:.1f}, 실제: {actual_tokens})"
        else:
            return f"번역에서 단어가 과도하게 추가되었을 가능성이 있습니다. (예상: {expected_tokens:.1f}, 실제: {actual_tokens})"
    elif difference_ratio > 0.15:  # 15~30% 차이
        if expected_tokens > actual_tokens:
            return f"번역에서 일부 단어가 누락되었을 가능성이 있습니다. (예상: {expected_tokens:.1f}, 실제: {actual_tokens})"
        else:
            return f"번역에서 일부 단어가 추가되었을 가능성이 있습니다. (예상: {expected_tokens:.1f}, 실제: {actual_tokens})"
    else:
        return f"단어 수 적절. (예상: {expected_tokens:.1f}, 실제: {actual_tokens})"

# 문법 오류 점검 기능 추가
def check_grammar(translated, dest_lang):
    # 간단한 규칙 기반 문법 검사
    grammar_errors = 0
    error_messages = []

    if dest_lang == "en":
        # 영어 문법 규칙 검사
        # 주어-동사 일치 검사 (간단한 예)
        subject_verb_errors = len(re.findall(r'\b(he|she|it) (are|am|were)\b', translated.lower()))
        grammar_errors += subject_verb_errors
        if subject_verb_errors > 0:
            error_messages.append("주어-동사 불일치 오류")

        # 관사 사용 검사 (간단한 예) - 'a' before a vowel sound
        article_errors = len(re.findall(r'\b(a) [aeiou]', translated.lower()))
        grammar_errors += article_errors
        if article_errors > 0:
            error_messages.append("관사 사용 오류")

    elif dest_lang == "ko":
        # 한국어 문법 규칙 검사 (조사 오용 등)
        # 이 부분은 매우 복잡하며, 간단한 정규식으로는 한계가 있습니다.
        # 예시로 '을를' '은는' '이가' '와과' 가 연속으로 나오거나 어색하게 사용된 경우를 찾아볼 수 있습니다.
        particle_errors = len(re.findall(r'[가-힣]+(을를|은는|이가|와과)\b', translated)) # This is a very simplistic check
        grammar_errors += particle_errors
        if particle_errors > 0:
            error_messages.append("조사 오용 가능성")

    elif dest_lang == "ja":
        # 일본어 문법 규칙 검사 (조사 오용 등)
        # 마찬가지로 간단한 정규식으로는 한계가 있습니다.
        particle_errors = len(re.findall(r'[ぁ-んァ-ン]+(はが|をに|にを)\b', translated)) # Simplistic check
        grammar_errors += particle_errors
        if particle_errors > 0:
            error_messages.append("조사 오용 가능성")

    grammar_score = max(0, 1 - (grammar_errors * 0.2)) # Each error reduces score by 0.2, min 0

    return grammar_score, ", ".join(error_messages) if error_messages else "문법 오류 없음"

# 향상된 자연스러움 점수
def evaluate_naturalness(translated, lang):
    # 언어별 문법 검사
    grammar_score, grammar_errors = check_grammar(translated, lang)

    # 문장 구조 복잡성 분석
    sentences = re.split(r'[.!?。！？]+', translated)
    sentences = [s.strip() for s in sentences if s.strip()] # Remove empty strings

    sentence_lengths = [len(s.split()) for s in sentences] # Word count per sentence
    num_sentences = len(sentences)
    avg_sentence_length = sum(sentence_lengths) / num_sentences if num_sentences > 0 else 0

    # 평균 문장 길이에 따른 점수 조정 (너무 길거나 짧으면 감점)
    length_score = 1.0
    if lang == "en":
        if avg_sentence_length < 5 or avg_sentence_length > 35:
            length_score = 0.8
    elif lang == "ko":
        if avg_sentence_length < 10 or avg_sentence_length > 60:
            length_score = 0.8
    elif lang == "ja":
        if avg_sentence_length < 10 or avg_sentence_length > 50:
            length_score = 0.8

    # 반복 단어/구문 검사
    tokens = nltk.word_tokenize(translated) if lang == "en" else tokenize_text("", translated, lang, lang)[1]
    if tokens:
        unique_tokens = set(tokens)
        repetition_ratio = len(unique_tokens) / len(tokens)
        repetition_score = min(1.0, repetition_ratio * 1.5)  # Repetition reduces score
    else:
        repetition_score = 1.0 # No tokens, so no repetition

    # 전체 자연스러움 점수 계산 (가중치 적용)
    naturalness_score = (grammar_score * 0.5) + (length_score * 0.3) + (repetition_score * 0.2)

    return naturalness_score, grammar_errors

# 향상된 의미 유사도 평가
def evaluate_similarity(original, translated, src_lang, dest_lang):
    # 언어별 문장 임베딩 모델
    try:
        # 다국어 sentence-transformers 모델 사용
        model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

        # 원본과 번역문의 임베딩 생성
        orig_embedding = model.encode([original])[0]
        trans_embedding = model.encode([translated])[0]

        # 코사인 유사도 계산
        similarity = F.cosine_similarity(
            torch.tensor(orig_embedding).unsqueeze(0),
            torch.tensor(trans_embedding).unsqueeze(0)
        ).item()

        # 키워드 보존 검사 (중요 명사 등이 보존되었는지)
        src_tokens, _ = tokenize_text(original, "", src_lang, "")
        _, dest_tokens = tokenize_text("", translated, "", dest_lang)

        # 고유 명사 및 중요 키워드 (숫자, 영문 등) 추출
        # This part requires more sophisticated NLP for actual "keywords"
        # For now, a simple heuristic for alphanumeric tokens
        src_keywords = set([t for t in src_tokens if re.match(r'[A-Za-z0-9]', t) and len(t) > 1])
        dest_keywords = set([t for t in dest_tokens if re.match(r'[A-Za-z0-9]', t) and len(t) > 1])

        # Convert destination keywords to lowercase for comparison, as case might change
        dest_keywords_lower = {k.lower() for k in dest_keywords}

        # Keyword preservation score
        keyword_score = 1.0
        if src_keywords:
            preserved_count = 0
            for k in src_keywords:
                # Check if the keyword (or its lowercase version) is present in the translated tokens
                if any(k.lower() == d_lower for d_lower in dest_keywords_lower):
                    preserved_count += 1
                # Simple check for transliteration/translation (e.g., Apple -> 애플)
                # This is very basic and would ideally need a translation dictionary or more advanced NLP
                elif src_lang == 'en' and dest_lang == 'ko' and k.lower() == 'apple' and '애플' in translated:
                    preserved_count += 1
                elif src_lang == 'en' and dest_lang == 'ja' and k.lower() == 'apple' and 'アップル' in translated:
                    preserved_count += 1

            keyword_score = preserved_count / len(src_keywords)
        else:
            keyword_score = 1.0 # No keywords to check

        # 최종 의미 유사도 (임베딩 유사도 70%, 키워드 보존 30%)
        final_similarity = (similarity * 0.7) + (keyword_score * 0.3)

        return final_similarity
    except Exception as e:
        print(f"유사도 평가 중 오류 발생: {e}")
        # 오류 발생 시 기본 유사도 반환
        return 0.5

# 주요 번역 오류 감지
def detect_translation_errors(original, translated, src_lang, dest_lang):
    errors = []

    # 숫자 불일치 검사
    src_numbers = re.findall(r'\d+', original)
    dest_numbers = re.findall(r'\d+', translated)

    if len(src_numbers) != len(dest_numbers):
        errors.append("숫자 개수 불일치")
    else:
        for i, num in enumerate(src_numbers):
            if i < len(dest_numbers) and num != dest_numbers[i]:
                errors.append(f"숫자 값 불일치: {num} → {dest_numbers[i]}")

    # 특수 기호 불일치 검사 (예: 괄호, 인용부호 등)
    # This might be too strict for some cases where punctuation can change due to language conventions
    src_punctuation = re.findall(r'[\[\]\(\)\{\}\"\'\«\»\„\‟\‹\›]', original)
    dest_punctuation = re.findall(r'[\[\]\(\)\{\}\"\'\«\»\„\‟\‹\›]', translated)

    if len(src_punctuation) != len(dest_punctuation):
        errors.append("특수 기호 개수 불일치 (괄호, 인용부호 등)")

    # 이메일, URL 누락 검사
    src_emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', original)
    dest_emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', translated)

    if len(src_emails) != len(dest_emails):
        errors.append("이메일 주소 누락 또는 변형")
    else:
        # Check if actual email addresses are preserved (case-insensitive for comparison)
        src_emails_lower = {e.lower() for e in src_emails}
        dest_emails_lower = {e.lower() for e in dest_emails}
        if src_emails_lower != dest_emails_lower:
            errors.append("이메일 주소 값 불일치")


    src_urls = re.findall(r'https?://\S+', original)
    dest_urls = re.findall(r'https?://\S+', translated)

    if len(src_urls) != len(dest_urls):
        errors.append("URL 누락 또는 변형")
    else:
        # Check if actual URLs are preserved
        src_urls_set = set(src_urls)
        dest_urls_set = set(dest_urls)
        if src_urls_set != dest_urls_set:
            errors.append("URL 값 불일치")

    return errors

# 종합 품질 점수 계산
def calculate_overall_score(bleu_score, completeness_message, naturalness_score, similarity_score, errors):
    # 각 항목별 가중치 설정
    weights = {
        'bleu': 0.2,
        'completeness': 0.2,
        'naturalness': 0.3,
        'similarity': 0.3
    }

    # 완전성 점수 계산 (누락 메시지에서 점수 추출)
    if "적절" in completeness_message:
        completeness_score = 1.0
    elif "일부" in completeness_message:
        completeness_score = 0.7
    else: # "높습니다" or "과도하게"
        completeness_score = 0.4

    # 오류 페널티
    # More errors lead to a higher penalty, up to a certain maximum.
    error_penalty = min(0.2, len(errors) * 0.05) # Max 20% penalty for 4 or more errors

    # BLEU 점수 정규화 (0-100 -> 0-1)
    normalized_bleu = bleu_score / 100

    # 종합 점수 계산
    overall_score = (
        weights['bleu'] * normalized_bleu +
        weights['completeness'] * completeness_score +
        weights['naturalness'] * naturalness_score +
        weights['similarity'] * similarity_score
    ) * (1 - error_penalty) # Apply penalty at the end

    # 최종 100점 만점으로 변환
    final_score = overall_score * 100

    # 점수에 따른 등급 부여
    if final_score >= 90:
        grade = "A+ (전문 번역사 수준)"
    elif final_score >= 80:
        grade = "A (매우 우수)"
    elif final_score >= 70:
        grade = "B (우수)"
    elif final_score >= 60:
        grade = "C (양호)"
    elif final_score >= 50:
        grade = "D (개선 필요)"
    else:
        grade = "F (불량)"

    return round(final_score, 1), grade

# 메인 평가 함수
def evaluate_translation(original, translated, src_lang, dest_lang):
    print("\n평가 진행 중...")

    # 토큰화
    src_tokens, dest_tokens = tokenize_text(original, translated, src_lang, dest_lang)

    # BLEU 점수 계산
    bleu_score = calculate_bleu(original, translated, src_lang, dest_lang)
    print("- BLEU 점수 계산 완료")

    # 완전성 검사
    completeness_message = check_missing(src_tokens, dest_tokens, src_lang, dest_lang)
    print("- 완전성 검사 완료")

    # 문법 및 자연스러움 평가
    naturalness_score, grammar_errors = evaluate_naturalness(translated, dest_lang)
    print("- 자연스러움 평가 완료")

    # 의미 유사도 평가
    similarity_score = evaluate_similarity(original, translated, src_lang, dest_lang)
    print("- 의미 유사도 평가 완료")

    # 번역 오류 감지
    translation_errors = detect_translation_errors(original, translated, src_lang, dest_lang)
    print("- 번역 오류 감지 완료")

    # 종합 점수 계산
    overall_score, grade = calculate_overall_score(
        bleu_score,
        completeness_message,
        naturalness_score,
        similarity_score,
        translation_errors
    )

    return {
        "bleu_score": bleu_score,
        "completeness": completeness_message,
        "naturalness": naturalness_score,
        "grammar_errors": grammar_errors,
        "similarity": similarity_score,
        "translation_errors": translation_errors,
        "overall_score": overall_score,
        "grade": grade
    }

# 결과 시각적 표시
def display_results(results):
    print("\n" + "="*50)
    print("📊 번역 품질 평가 결과")
    print("="*50)

    print(f"\n🏆 종합 점수: {results['overall_score']}/100 ({results['grade']})")

    print("\n📈 세부 평가:")
    print(f"- BLEU 점수: {results['bleu_score']:.2f}/100")
    print(f"- 완전성: {results['completeness']}")
    print(f"- 자연스러움: {results['naturalness']:.2f}/1.0")
    print(f"- 문법 검사: {results['grammar_errors']}")
    print(f"- 의미 유사도: {results['similarity']:.2f}/1.0")

    if results['translation_errors']:
        print("\n⚠️ 감지된 오류:")
        for error in results['translation_errors']:
            print(f"  • {error}")
    else:
        print("\n✅ 중대한 번역 오류가 감지되지 않았습니다.")

    print("\n💡 개선 제안:")
    if results['overall_score'] < 60:
        print("  • 번역의 정확성과 완전성을 높이는 것이 필요합니다.")
        print("  • 원문의 의미를 더 잘 전달할 수 있도록 번역을 재검토하세요.")
    elif results['overall_score'] < 80:
        print("  • 문법과 자연스러움을 개선하면 번역 품질이 향상될 것입니다.")
        print("  • 일부 문장 구조를 대상 언어에 맞게 수정해보세요.")
    else:
        print("  • 높은 품질의 번역입니다. 미세한 조정만 필요합니다.")

    print("="*50)

# 메인 함수
def main():
    print("\n🌐 고급 AI 번역 품질 평가 시스템 🌐")
    print("-"*50)

    src_lang = input("원문 언어를 입력하세요 (ja: 일본어, ko: 한국어, en: 영어): ").strip().lower()
    dest_lang = input("번역 언어를 입력하세요 (ja: 일본어, ko: 한국어, en: 영어): ").strip().lower()

    print("\n원문을 입력하세요:")
    original = input().strip()

    print("\n번역문을 입력하세요:")
    translated = input().strip()

    print("\n평가를 시작합니다...")
    results = evaluate_translation(original, translated, src_lang, dest_lang)
    display_results(results)

if __name__ == "__main__":
    main()
