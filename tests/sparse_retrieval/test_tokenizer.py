from rag.sparse_retrieval.tokenizer import tokenize


def test_tokenize_lowercases_text():

    assert tokenize("Refund POLICY") == [
        "refund",
        "policy",
    ]


def test_tokenize_removes_punctuation():

    assert tokenize(
        "API endpoint: /documents/{id}"
    ) == [
        "api",
        "endpoint",
        "documents",
        "id",
    ]


def test_tokenize_handles_empty_text():

    assert tokenize("") == []