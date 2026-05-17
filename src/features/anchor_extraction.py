"""
陈述句 anchor 抽取模块。

为 True-False 陈述句抽取 subject / relation / tail 的 token 级锚点。
优先使用简单规则（系表结构、常见关系短语），最后回退到启发式 fallback。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 关系短语模式库
# ---------------------------------------------------------------------------

# 系动词 / 系表结构
COPULA_PATTERN = re.compile(
    r"\b(is|are|was|were|has|have)\b",
    re.IGNORECASE,
)

# 常见关系短语（长度降序排列以优先长匹配）
RELATION_PATTERNS = [
    "is located in",
    "are located in",
    "was founded by",
    "were founded by",
    "was invented by",
    "were invented by",
    "was discovered by",
    "were discovered by",
    "is made of",
    "are made of",
    "is part of",
    "are part of",
    "belongs to",
    "consists of",
    "is known for",
    "is a",
    "are a",
    "is the",
    "are the",
    "was the",
    "were the",
    "is an",
    "are an",
    "was an",
    "were an",
    "was a",
    "were a",
]


def _build_relation_regex() -> re.Pattern:
    """构建关系短语正则（按长度降序，忽略大小写）。"""
    sorted_patterns = sorted(RELATION_PATTERNS, key=len, reverse=True)
    escaped = [re.escape(p) for p in sorted_patterns]
    return re.compile("|".join(escaped), re.IGNORECASE)


_RELATION_REGEX = _build_relation_regex()


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class AnchorSpans:
    """陈述句的 token 级锚点信息。"""

    statement: str
    subject_char_span: tuple[int, int] | None = None
    relation_char_span: tuple[int, int] | None = None
    tail_char_span: tuple[int, int] | None = None
    subject_token_indices: list[int] = field(default_factory=list)
    relation_token_indices: list[int] = field(default_factory=list)
    tail_token_indices: list[int] = field(default_factory=list)
    last_token_index: int = -1
    rule_name: str = "none"
    valid: bool = False
    fallback_reason: str | None = None


# ---------------------------------------------------------------------------
# 字符级 span 抽取
# ---------------------------------------------------------------------------

def extract_char_spans(statement: str) -> dict:
    """从陈述句中抽取 subject / relation / tail 的字符级 span。

    返回:
        {"subject": (start, end) | None,
         "relation": (start, end) | None,
         "tail": (start, end) | None,
         "rule_name": str,
         "fallback_reason": str | None}
    """
    text = statement.strip()

    # ---- 规则 1: 系表结构 ------------------------------------------------
    copula_match = COPULA_PATTERN.search(text)
    if copula_match:
        copula_start = copula_match.start()
        copula_end = copula_match.end()

        subject_span = (0, copula_start) if copula_start > 0 else None
        relation_span = (copula_start, copula_end)
        tail_span = (copula_end + 1, len(text)) if copula_end + 1 < len(text) else None

        # 清理尾部空白
        if subject_span:
            s_start, s_end = subject_span
            s_start = _lstrip_idx(text, s_start, s_end)
            s_end = _rstrip_idx(text, s_start, s_end)
            if s_start < s_end:
                subject_span = (s_start, s_end)
            else:
                subject_span = None

        if tail_span:
            t_start, t_end = tail_span
            t_start = _lstrip_idx(text, t_start, t_end)
            t_end = _rstrip_idx(text, t_start, t_end)
            if t_start < t_end:
                tail_span = (t_start, t_end)
            else:
                tail_span = None

        return {
            "subject": subject_span,
            "relation": relation_span,
            "tail": tail_span,
            "rule_name": "copula",
            "fallback_reason": None,
        }

    # ---- 规则 2: 常见关系短语 --------------------------------------------
    phrase_match = _RELATION_REGEX.search(text)
    if phrase_match:
        phrase_start = phrase_match.start()
        phrase_end = phrase_match.end()

        subject_span = (0, phrase_start) if phrase_start > 0 else None
        relation_span = (phrase_start, phrase_end)
        tail_span = (phrase_end + 1, len(text)) if phrase_end + 1 < len(text) else None

        # 清理空白
        if subject_span:
            s_start, s_end = subject_span
            s_start = _lstrip_idx(text, s_start, s_end)
            s_end = _rstrip_idx(text, s_start, s_end)
            subject_span = (s_start, s_end) if s_start < s_end else None

        if tail_span:
            t_start, t_end = tail_span
            t_start = _lstrip_idx(text, t_start, t_end)
            t_end = _rstrip_idx(text, t_start, t_end)
            tail_span = (t_start, t_end) if t_start < t_end else None

        return {
            "subject": subject_span,
            "relation": relation_span,
            "tail": tail_span,
            "rule_name": "relation_phrase",
            "fallback_reason": None,
        }

    # ---- 规则 3: fallback -----------------------------------------------
    return _fallback_char_spans(text)


def _fallback_char_spans(text: str) -> dict:
    """启发式 fallback：前 N 词 = subject, 中间动词 = relation, 后 N 词 = tail。"""
    words = text.split()
    n = len(words)

    if n <= 2:
        # 极短句: subject = 全部, relation/tail = None
        return {
            "subject": (0, len(text)),
            "relation": None,
            "tail": None,
            "rule_name": "fallback",
            "fallback_reason": "too_short",
        }

    # 前 1-3 个词为 subject
    subj_word_count = min(3, max(1, n // 3))
    # 后 1-3 个词为 tail
    tail_word_count = min(3, max(1, n // 3))

    # 在剩余中间区段寻找动词
    mid_start = subj_word_count
    mid_end = n - tail_word_count

    relation_span = None
    if mid_start < mid_end:
        mid_words = words[mid_start:mid_end]
        # 简单动词检测
        verb_idx_in_mid = _find_first_verb_index(mid_words)
        if verb_idx_in_mid is not None:
            abs_idx = mid_start + verb_idx_in_mid
            # 找到该词在原文中的位置
            prefix_len = len(" ".join(words[:abs_idx]))
            if abs_idx > 0:
                prefix_len += 1  # space
            verb_word = words[abs_idx]
            relation_span = (prefix_len, prefix_len + len(verb_word))

    # 计算 subject 和 tail 在原文中的位置
    subject_end_pos = len(" ".join(words[:subj_word_count]))
    subject_span = (0, subject_end_pos)

    tail_start_pos = len(" ".join(words[: n - tail_word_count]))
    if n - tail_word_count > 0:
        tail_start_pos += 1  # space
    tail_span = (tail_start_pos, len(text)) if tail_word_count > 0 else None

    # 清理
    if subject_span:
        s_start, s_end = subject_span
        s_start = _lstrip_idx(text, s_start, s_end)
        s_end = _rstrip_idx(text, s_start, s_end)
        subject_span = (s_start, s_end) if s_start < s_end else None

    if tail_span:
        t_start, t_end = tail_span
        t_start = _lstrip_idx(text, t_start, t_end)
        t_end = _rstrip_idx(text, t_start, t_end)
        tail_span = (t_start, t_end) if t_start < t_end else None

    return {
        "subject": subject_span,
        "relation": relation_span,
        "tail": tail_span,
        "rule_name": "fallback",
        "fallback_reason": "no_relation_pattern_matched",
    }


def _find_first_verb_index(words: list[str]) -> int | None:
    """在词列表中查找第一个可能的动词（简单启发式）。"""
    common_verbs = {
        "is", "are", "was", "were", "has", "have", "had",
        "do", "does", "did", "can", "could", "will", "would",
        "may", "might", "shall", "should", "must",
        "made", "found", "known", "used", "called", "named",
        "located", "situated", "discovered", "invented",
        "contains", "includes", "belongs", "consists",
        "became", "become", "being", "been",
    }
    for i, w in enumerate(words):
        if w.lower() in common_verbs:
            return i
    return None


def _lstrip_idx(text: str, start: int, end: int) -> int:
    """将 start 向右移动以跳过前导空白。"""
    while start < end and text[start].isspace():
        start += 1
    return start


def _rstrip_idx(text: str, start: int, end: int) -> int:
    """将 end 向左移动以跳过尾部空白。"""
    while end > start and text[end - 1].isspace():
        end -= 1
    return end


# ---------------------------------------------------------------------------
# Token 对齐
# ---------------------------------------------------------------------------

def align_char_span_to_token_indices(
    tokenizer,
    statement: str,
    char_span: tuple[int, int] | None,
) -> list[int]:
    """使用 tokenizer 的 offset_mapping 将字符 span 映射到 token 索引列表。

    参数:
        tokenizer: HuggingFace tokenizer。
        statement: 原始文本。
        char_span: (start_char, end_char) 或 None。

    返回:
        token 索引列表（若 char_span 为 None 则返回空列表）。
    """
    if char_span is None:
        return []

    encoded = tokenizer(
        statement,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    offsets = encoded.get("offset_mapping", [])
    if not offsets:
        return []

    char_start, char_end = char_span
    selected: list[int] = []

    for idx, (tok_start, tok_end) in enumerate(offsets):
        # 跳过特殊 token（offset 为 (0, 0)）
        if tok_start == tok_end:
            continue
        # 检查 token 与 char span 是否有重叠
        if tok_end > char_start and tok_start < char_end:
            selected.append(idx)

    return selected


# ---------------------------------------------------------------------------
# 顶层接口
# ---------------------------------------------------------------------------

def extract_anchors(tokenizer, statement: str) -> AnchorSpans:
    """对单条陈述句抽取 token 级 anchor。

    参数:
        tokenizer: HuggingFace tokenizer。
        statement: 陈述句文本。

    返回:
        AnchorSpans 实例。
    """
    text = statement.strip()

    # Step 1: 字符级 span 抽取
    spans = extract_char_spans(text)

    # Step 2: token 对齐
    subject_indices = align_char_span_to_token_indices(
        tokenizer, text, spans["subject"]
    )
    relation_indices = align_char_span_to_token_indices(
        tokenizer, text, spans["relation"]
    )
    tail_indices = align_char_span_to_token_indices(
        tokenizer, text, spans["tail"]
    )

    # Step 3: 确定 last token index
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"][0]
    # 找到最后一个非 padding 的位置（Qwen tokenizer 通常有 eos）
    last_idx = len(input_ids) - 1
    # 确保不是 padding
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        while last_idx >= 0 and input_ids[last_idx].item() == tokenizer.pad_token_id:
            last_idx -= 1

    # 验证有效性
    valid = len(subject_indices) > 0  # 至少有 subject
    fallback_reason = spans.get("fallback_reason")

    return AnchorSpans(
        statement=text,
        subject_char_span=spans["subject"],
        relation_char_span=spans["relation"],
        tail_char_span=spans["tail"],
        subject_token_indices=subject_indices,
        relation_token_indices=relation_indices,
        tail_token_indices=tail_indices,
        last_token_index=last_idx,
        rule_name=spans["rule_name"],
        valid=valid,
        fallback_reason=fallback_reason,
    )
