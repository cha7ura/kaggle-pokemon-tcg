from tools.fetch_papers import clean_markdown, arxiv_pdf_url


def test_arxiv_pdf_url():
    assert arxiv_pdf_url("2402.01118") == "https://arxiv.org/pdf/2402.01118"
    assert arxiv_pdf_url("2112.03178") == "https://arxiv.org/pdf/2112.03178"


def test_clean_markdown_collapses_whitespace_and_strips_control():
    raw = "Title\n\n\n\nbody\ttext\x0c with   spaces\n\n\n\nend"
    out = clean_markdown(raw)
    assert "\x0c" not in out            # form feed (pdf page break) removed
    assert "\n\n\n" not in out          # no 3+ consecutive newlines
    assert "with spaces" in out         # runs of spaces collapsed
    assert out.startswith("Title")
    assert out.endswith("end")
