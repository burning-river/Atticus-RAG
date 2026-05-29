# from wtpsplit import SaT

# text_splitter = SaT("sat-3l")

from wtpsplit import WTPSplit

text_splitter = WTPSplit("sat-3l-sm") 

def split_text_to_paragraphs(text: str) -> list[list[str]]:
    return text_splitter.split(text, do_paragraph_segmentation=True)


def extract_chunks(paragraphs: list[list[str]]) -> list[str]:
    chunks = []
    for paragraph in paragraphs:
        chunk = ""
        for sentence in paragraph:
            chunk += sentence + " "
        chunks.append(chunk.strip())

    return chunks
