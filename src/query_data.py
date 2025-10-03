import requests, pathlib, sys, os, argparse
from xml.etree import ElementTree as ET
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def fetch_pmc_ids(term: str = "cancer", limit: int =1) -> list[str]:
    """
    Use NCBI ESearch to get PMC IDs for a query term.
    Returns a list like ['PMC12345', ...] up to `limit`.
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pmc",
        "term": term,
        "retmax": limit,
        "retmode": "json",
        "sort": "relevance",
    }
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    raw_ids = data.get("esearchresult", {}).get("idlist", [])
    pmcids = ["PMC" + rid for rid in raw_ids]
    return pmcids

def fetch_xml(pmcid: str) -> str:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pmc",
        "id": pmcid,
        "rettype": "full",
        "retmode": "xml",
    }
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.text

def pretty_print_xml(xml_text: str) -> str:
    # Pretty-print with stdlib (fast + no extra deps)
    # Remove DOCTYPE first (ElementTree can choke on it)
    import re
    xml_text = re.sub(r"<!DOCTYPE[^>]*>", "", xml_text, flags=re.IGNORECASE)
    root = ET.fromstring(xml_text)
    # ET.tostring doesn't pretty print; add newlines between tags as a simple legibility hack
    rough = ET.tostring(root, encoding="unicode")
    # insert a newline between close><open to avoid mega-lines
    pretty = rough.replace("><", ">\n<")
    return pretty

def extract_plain_text(xml_text: str,
                       lower: bool = False,
                       remove_numeric_citations: bool = True,
                       remove_tables: bool = True,
                       remove_tabular_lines: bool = True) -> str:
    """
    Extract readable plain text from a PMC article XML.
    Parameters:
        lower: lowercase final text
        remove_numeric_citations: remove [1], [2,3], etc.
        remove_tables: drop XML table content
        remove_tabular_lines: drop residual lines that look like flattened tables
    """
    import re
    xml_text = re.sub(r"<!DOCTYPE[^>]*>", "", xml_text, flags=re.IGNORECASE)
    root = ET.fromstring(xml_text)

    # strip namespaces
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]

    ignore_tags = set()
    if remove_tables:
        ignore_tags.update({"table", "table-wrap", "thead", "tbody", "tfoot",
                            "tr", "td", "th"})

    def text_of(node):
        if node.tag in ignore_tags:
            return ""
        parts = []
        if node.text:
            parts.append(node.text)
        for c in list(node):
            parts.append(text_of(c))
            if c.tail:
                parts.append(c.tail)
        return "".join(parts)

    blocks = []

    # Title
    t = root.find(".//front/article-meta/title-group/article-title")
    if t is not None:
        title = text_of(t).strip()
        if title:
            blocks.append(title)

    # Abstract(s)
    for ab in root.findall(".//abstract"):
        head = ab.find("./title")
        head_txt = text_of(head).strip() if head is not None else "Abstract"
        paras = []
        for child in list(ab):
            if child.tag == "title":
                continue
            if child.tag == "p":
                txt = text_of(child).strip()
                if txt:
                    paras.append(txt)
            elif child.tag == "sec":
                sec_title = child.find("./title")
                sec_head = text_of(sec_title).strip() if sec_title is not None else ""
                sec_paras = [text_of(p).strip() for p in child.findall("./p")]
                sec_paras = [p for p in sec_paras if p]
                if sec_paras:
                    block = (sec_head + "\n") if sec_head else ""
                    block += "\n".join(sec_paras)
                    paras.append(block)
        if paras:
            blocks.append(head_txt + "\n" + "\n".join(paras))

    # Body
    body = root.find(".//body")

    def walk_sec(sec, level=1):
        title_el = sec.find("./title")
        if title_el is not None:
            head = text_of(title_el).strip()
            if head:
                blocks.append(("#" * min(level, 6)) + " " + head)
        for p in sec.findall("./p"):
            txt = text_of(p).strip()
            if txt:
                blocks.append(txt)
        for lst in sec.findall("./list"):
            for li in lst.findall("./list-item"):
                li_txt = text_of(li).strip()
                if li_txt:
                    blocks.append("- " + li_txt)
        for sub in sec.findall("./sec"):
            walk_sec(sub, level + 1)

    if body is not None:
        for top in body.findall("./sec"):
            walk_sec(top, 1)

    text = "\n\n".join(blocks)

    # Normalize whitespace early
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove typical numeric citation brackets
    if remove_numeric_citations:
        text = re.sub(r"\[(?:\d+(?:\s*,\s*\d+)*)\]", "", text)

    if remove_tabular_lines:
        filtered_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                filtered_lines.append(line)
                continue
            # Count numbers & commas
            digit_count = sum(ch.isdigit() for ch in stripped)
            comma_numbers = re.findall(r"\b\d{1,3}(?:,\d{3})+\b", stripped)
            plain_numbers = re.findall(r"\b\d+\b", stripped)
            digit_ratio = digit_count / max(1, len(stripped))
            # Heuristics: looks tabular if many large comma numbers or high digit density
            if (len(comma_numbers) >= 3 and len(plain_numbers) >= 6) or digit_ratio > 0.45:
                continue
            # Also skip lines that start with a cancer site code pattern followed by dense numbers
            if re.match(r"^[A-Z][a-z].*\b(C\d{2}(?:-C\d{2})?)\b.*\d", stripped) and digit_ratio > 0.35:
                continue
            filtered_lines.append(line)
        text = "\n".join(filtered_lines)

    # Final punctuation spacing cleanup
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n +", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    if lower:
        text = text.lower()

    return text.strip()

def main():
    outdir = pathlib.Path(config.OUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    for sub in ["raw", "pretty", "plain"]:
        (outdir / sub).mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {outdir}")
    print(f"Searching PMC for term='{config.TERM}' (limit={config.LIMIT}) …")
    try:
        pmcids = fetch_pmc_ids(config.TERM, config.LIMIT)
    except Exception as e:
        print(f"Failed to fetch PMC ID list: {e}")
        sys.exit(1)

    if not pmcids:
        print("No PMC IDs returned. Exiting.")
        return

    id_list_path = outdir / f"pmcids_{config.TERM}.txt"
    id_list_path.write_text("\n".join(pmcids), encoding="utf-8")
    print(f"Got {len(pmcids)} PMCIDs. Saved list to {id_list_path}")

    last_raw = last_pretty = last_plain = None

    for pmcid in pmcids:
        try:
            print(f"Fetching {pmcid} …")
            xml_text = fetch_xml(pmcid)

            raw_xml_path = outdir / f"raw/{pmcid}.xml"
            raw_xml_path.write_text(xml_text, encoding="utf-8")
            last_raw = raw_xml_path

            pretty = pretty_print_xml(xml_text)
            pretty_path = outdir / f"pretty/{pmcid}.xml"
            pretty_path.write_text(pretty, encoding="utf-8")
            last_pretty = pretty_path

            plain = extract_plain_text(xml_text)
            txt_path = outdir / f"plain/{pmcid}.plain.txt"
            txt_path.write_text(plain, encoding="utf-8")
            last_plain = txt_path

            print(f"  Saved {pmcid}")
        except Exception as e:
            print(f"  ERROR {pmcid}: {e}")

    if last_raw and last_pretty and last_plain:
        print("Saved examples:")
        print(f"  - raw XML   → {last_raw}")
        print(f"  - pretty XML→ {last_pretty}")
        print(f"  - plain text→ {last_plain}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        sys.exit(1)
