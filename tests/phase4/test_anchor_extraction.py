"""测试 anchor extraction 模块。"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from src.features.anchor_extraction import (
    extract_char_spans,
    align_char_span_to_token_indices,
    extract_anchors,
    AnchorSpans,
)


# ---------------------------------------------------------------------------
# 字符级 span 抽取测试
# ---------------------------------------------------------------------------

class TestExtractCharSpans:
    """测试 extract_char_spans 函数。"""

    def test_copula_is(self):
        result = extract_char_spans("Paris is the capital of France.")
        assert result["rule_name"] == "copula"
        assert result["subject"] is not None
        assert result["relation"] is not None
        assert result["tail"] is not None

    def test_copula_are(self):
        result = extract_char_spans("Dogs are friendly animals.")
        assert result["rule_name"] == "copula"
        assert result["subject"] is not None

    def test_copula_was(self):
        result = extract_char_spans("Einstein was a physicist.")
        assert result["rule_name"] == "copula"
        assert result["subject"] is not None

    def test_relation_phrase(self):
        result = extract_char_spans(
            "The Eiffel Tower is located in Paris."
        )
        # is located in → "is" 是 copula 先匹配
        assert result["rule_name"] in ("copula", "relation_phrase")

    def test_relation_phrase_invented(self):
        # 无系动词的陈述句
        result = extract_char_spans(
            "Thomas Edison invented the light bulb."
        )
        # "invented" 不在 copula 词表中，可能进入 fallback 或 relation_phrase
        assert result["rule_name"] in ("copula", "relation_phrase", "fallback")

    def test_fallback_short(self):
        result = extract_char_spans("Hello.")
        assert result["rule_name"] in ("fallback", "copula")

    def test_fallback_no_pattern(self):
        result = extract_char_spans("The cat sat on a mat and slept.")
        assert result["rule_name"] in ("fallback", "copula")

    def test_subject_not_empty_for_valid_sentence(self):
        result = extract_char_spans("London is a big city.")
        assert result["subject"] is not None
        s_start, s_end = result["subject"]
        assert s_start < s_end

    def test_empty_string(self):
        result = extract_char_spans("")
        assert result["rule_name"] in ("fallback", "copula")


# ---------------------------------------------------------------------------
# Token 对齐测试
# ---------------------------------------------------------------------------

class TestTokenAlignment:
    """测试 token 对齐功能。"""

    def _make_mock_tokenizer(self, offset_mapping):
        """创建 mock tokenizer。"""
        tokenizer = MagicMock()
        tokenizer.return_value = {"offset_mapping": offset_mapping}
        return tokenizer

    def test_simple_alignment(self):
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "offset_mapping": [
                (0, 0),   # special token
                (0, 5),   # "Paris"
                (5, 6),   # space
                (6, 8),   # "is"
                (8, 9),   # space
                (9, 12),  # "the"
                (12, 13), # space
                (13, 20), # "capital"
            ]
        }
        result = align_char_span_to_token_indices(tokenizer, "Paris is the capital", (0, 5))
        # token 1 覆盖 (0,5)
        assert 1 in result

    def test_null_span(self):
        tokenizer = MagicMock()
        result = align_char_span_to_token_indices(tokenizer, "test", None)
        assert result == []

    def test_empty_offsets(self):
        tokenizer = MagicMock()
        tokenizer.return_value = {"offset_mapping": []}
        result = align_char_span_to_token_indices(tokenizer, "test", (0, 4))
        assert result == []

    def test_overlap_detection(self):
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "offset_mapping": [
                (0, 0),
                (0, 4),   # "test"
                (4, 5),   # space
                (5, 10),  # "phrase"
            ]
        }
        # 覆盖 "test phrase" 的前 6 个字符（包含 "test " 部分）
        result = align_char_span_to_token_indices(tokenizer, "test phrase", (0, 6))
        # token 1 (0,4) 重叠，(4,5) 是空格跳过
        assert 1 in result


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------

class TestExtractAnchors:
    """测试 extract_anchors 顶层接口。"""

    def _make_simple_tokenizer(self):
        """创建一个行为类似 HuggingFace tokenizer 的简单 mock。

        extract_anchors 调用 tokenizer 两次：
        1. tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
        2. tokenizer(text, return_tensors="pt", add_special_tokens=True)
        """
        import torch

        def tokenize_impl(text, return_offsets_mapping=False, return_tensors=None, add_special_tokens=True, **kwargs):
            words = text.split() if text.strip() else []
            offsets = [(0, 0)]  # BOS token
            token_count = 1
            for i, w in enumerate(words):
                if i > 0:
                    offsets.append((0, 0))  # 简化空格 token
                    token_count += 1
                start = len(" ".join(words[:i]))
                if i > 0:
                    start += 1
                end = start + len(w)
                offsets.append((start, end))
                token_count += 1

            ids = list(range(token_count))
            result = {}
            if return_offsets_mapping:
                result["offset_mapping"] = offsets

            if return_tensors == "pt":
                t = torch.tensor([ids], dtype=torch.long)
                result["input_ids"] = t
                result["attention_mask"] = torch.ones(1, token_count, dtype=torch.long)
            else:
                result["input_ids"] = torch.tensor(ids, dtype=torch.long)

            return result

        tokenizer = MagicMock()
        tokenizer.side_effect = tokenize_impl
        tokenizer.pad_token_id = None
        return tokenizer

    def test_extract_anchors_returns_valid(self):
        tokenizer = self._make_simple_tokenizer()
        anchor = extract_anchors(tokenizer, "Paris is the capital of France.")
        assert isinstance(anchor, AnchorSpans)
        # 至少应有 subject
        assert anchor.valid or anchor.fallback_reason is not None

    def test_extract_anchors_has_last_token(self):
        tokenizer = self._make_simple_tokenizer()
        anchor = extract_anchors(tokenizer, "This is a test.")
        assert anchor.last_token_index >= 0

    def test_extract_anchors_no_crash_empty(self):
        tokenizer = self._make_simple_tokenizer()
        anchor = extract_anchors(tokenizer, "")
        assert isinstance(anchor, AnchorSpans)

    def test_extract_anchors_no_crash_special_chars(self):
        tokenizer = self._make_simple_tokenizer()
        anchor = extract_anchors(tokenizer, "!!!")
        assert isinstance(anchor, AnchorSpans)
